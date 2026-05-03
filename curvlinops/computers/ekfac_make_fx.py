"""EKFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxEKFACComputer``, which extends
``MakeFxKFACComputer`` with FX-based eigenvalue correction, using
:class:`LayerIO` instead of forward/backward hooks.

The standalone factory :func:`make_compute_ekfac_eigencorrection_batch`
returns a traced per-batch eigencorrection closure for users who want a
functional API without going through :class:`MakeFxEKFACComputer` (mirrors
:func:`make_compute_kfac_batch`).
"""

from collections.abc import Callable, MutableMapping

from torch import Tensor, empty, no_grad

from curvlinops.computers._base import ParamGroup, ParamGroupKey, _EKFACMixin
from curvlinops.computers.ekfac_hooks import (
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.computers.io_collector import LayerIO
from curvlinops.computers.kfac_make_fx import MakeFxKFACComputer
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _make_fx, fork_rng_with_seed


def make_compute_ekfac_eigencorrection_batch(
    model_func: Callable,
    loss_func: Callable,
    params: dict[str, Tensor],
    X: Tensor | MutableMapping,
    y: Tensor,
    fisher_type: FisherType = FisherType.MC,
    mc_samples: int = 1,
    separate_weight_and_bias: bool = True,
    batch_size_fn: Callable[[Tensor | MutableMapping], int] | None = None,
    output_check_fn: Callable[[Tensor, Tensor], object] | None = None,
) -> tuple[
    Callable[
        [
            dict[str, Tensor],
            Tensor | MutableMapping,
            Tensor,
            dict[ParamGroupKey, Tensor],
            dict[ParamGroupKey, Tensor],
        ],
        dict[ParamGroupKey, Tensor],
    ],
    list[ParamGroup],
]:
    """Set up and trace per-batch EKFAC eigenvalue correction computation.

    Builds a :class:`LayerIO` (pinned to :attr:`KFACType.EXPAND` because
    :func:`compute_eigenvalue_correction_linear_weight_sharing` consumes
    EXPAND-format ``(a, g)`` regardless of the operator's ``kfac_approx``)
    and traces the per-batch eigencorrection reduction with
    :func:`_make_fx`.

    Args:
        model_func: Functional model ``(params, X) -> prediction``.
        loss_func: Loss function.
        params: Named parameter dict.
        X: Example input tensor.
        y: Example target tensor.
        fisher_type: Type of Fisher information. Defaults to
            ``FisherType.MC``.
        mc_samples: Number of Monte-Carlo samples. Defaults to ``1``.
        separate_weight_and_bias: Whether to treat weights and biases
            separately. Defaults to ``True``.
        batch_size_fn: Function to extract batch size from ``X``.
            Defaults to ``X.shape[0]``.
        output_check_fn: Optional ``(output, y) -> object`` callback
            forwarded to :class:`LayerIO`; raise inside it to reject
            unsupported output/target shapes.

    Returns:
        Tuple of ``(traced_fn, mapping)`` where ``traced_fn`` is a compiled
        function ``(params, X, y, input_eigvecs, gradient_eigvecs) ->
        eigencorrections`` (dict mapping parameter group keys to tensors)
        and ``mapping`` is the list of parameter groups.
    """
    io = LayerIO(
        model_func,
        loss_func,
        params,
        X,
        fisher_type=fisher_type,
        mc_samples=mc_samples,
        kfac_approx=KFACType.EXPAND,
        separate_weight_and_bias=separate_weight_and_bias,
        batch_size_fn=batch_size_fn,
        output_check_fn=output_check_fn,
    )

    def compute_eigencorrection_batch(
        params: dict[str, Tensor],
        X: Tensor | MutableMapping,
        y: Tensor,
        input_eigvecs: dict[ParamGroupKey, Tensor],
        gradient_eigvecs: dict[ParamGroupKey, Tensor],
    ) -> dict[ParamGroupKey, Tensor]:
        """Compute per-batch eigenvalue corrections for all groups.

        Args:
            params: Named model parameters.
            X: Input batch.
            y: Target batch.
            input_eigvecs: Input covariance eigenvectors per parameter group.
            gradient_eigvecs: Gradient covariance eigenvectors per parameter group.

        Returns:
            Dict mapping parameter group keys to eigencorrection tensors.
        """
        layer_inputs, layer_output_grads = io.populate(params, X, y)
        snap = io.snapshot(layer_inputs, layer_output_grads)

        eigencorrections: dict[ParamGroupKey, Tensor] = {}
        for group in io.mapping:
            group_key = tuple(group.values())
            a, g = snap.standardized_io(group)
            aaT_eigvecs = input_eigvecs[group_key] if "W" in group else None
            eigencorrections[group_key] = (
                compute_eigenvalue_correction_linear_weight_sharing(
                    g, gradient_eigvecs[group_key], a, aaT_eigvecs
                )
            )
        return eigencorrections

    # Example eigvec shapes are derived from params:
    # ``d_out = W.shape[0]``; ``d_in = W[0].numel() (+1 for bias pad)``.
    example_input_eigvecs: dict[ParamGroupKey, Tensor] = {}
    example_gradient_eigvecs: dict[ParamGroupKey, Tensor] = {}
    for group in io.mapping:
        group_key = tuple(group.values())
        p1 = params[next(iter(group.values()))]
        d_out = p1.shape[0]
        example_gradient_eigvecs[group_key] = empty(
            d_out, d_out, dtype=p1.dtype, device=p1.device
        )
        if "W" in group:
            d_in = p1.numel() // p1.shape[0] + ("b" in group)
            example_input_eigvecs[group_key] = empty(
                d_in, d_in, dtype=p1.dtype, device=p1.device
            )

    with io.enable_param_grads(params):
        traced_fn = _make_fx(compute_eigencorrection_batch)(
            params, X, y, example_input_eigvecs, example_gradient_eigvecs
        )

    return traced_fn, io.mapping


class MakeFxEKFACComputer(_EKFACMixin, MakeFxKFACComputer):
    """EKFAC computer that uses FX graph tracing for eigenvalue correction.

    Extends :class:`MakeFxKFACComputer` with eigencorrection tracing via
    :func:`make_compute_ekfac_eigencorrection_batch`. The 2d-output
    restriction is enforced inside the trace by the parent's
    ``output_check_fn`` hook (overridden below).
    """

    def _output_check_fn(self) -> Callable[[Tensor, Tensor], object]:
        """Return EKFAC's 2d-output guard.

        Returns:
            The :meth:`_rearrange_for_larger_than_2d_output` static method,
            which raises :class:`ValueError` for non-2d outputs.
        """
        return self._rearrange_for_larger_than_2d_output

    def _trace_eigencorrection_batch_functions(
        self,
    ) -> tuple[dict[int, Callable], list[ParamGroup]]:
        """Trace per-batch eigencorrection for all batch sizes in the data.

        Returns:
            Tuple of ``(traced_fns, mapping)``.
        """
        traced_fns: dict[int, Callable] = {}
        mapping: list[ParamGroup] | None = None

        for X, y in self._loop_over_data(desc="Eigencorrection tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size in traced_fns:
                continue
            traced_fns[batch_size], mapping = make_compute_ekfac_eigencorrection_batch(
                self._model_func,
                self._loss_func,
                self._params,
                X,
                y,
                fisher_type=self._fisher_type,
                mc_samples=self._mc_samples,
                separate_weight_and_bias=self._separate_weight_and_bias,
                batch_size_fn=self._batch_size_fn,
                output_check_fn=self._output_check_fn(),
            )

        return traced_fns, mapping

    def compute(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor],
        dict[ParamGroupKey, Tensor],
        dict[ParamGroupKey, Tensor],
        list[ParamGroup],
    ]:
        """Compute eigenvalue-corrected Kronecker factors.

        Uses traced batch functions for both KFAC factor computation and
        eigenvalue correction.

        Returns:
            Tuple of ``(input_covariance_eigenvectors,
            gradient_covariance_eigenvectors, corrected_eigenvalues, mapping)``.
        """
        traced_batch = self._trace_batch_functions()
        eigencorrection_fns, _ = self._trace_eigencorrection_batch_functions()

        input_covariances, gradient_covariances, mapping = (
            self._compute_kronecker_factors(traced_batch)
        )
        input_covariances, gradient_covariances = self._eigenvectors_(
            input_covariances, gradient_covariances
        )

        corrected_eigenvalues: dict[ParamGroupKey, Tensor] = {}

        # Seed only for stochastic fisher types. fork_rng_with_seed isolates
        # the seed from the caller's global RNG state.
        seed = self._seed if self._fisher_type == FisherType.MC else None
        with fork_rng_with_seed(seed), no_grad():
            for X, y in self._loop_over_data(desc="Eigenvalue correction"):
                batch_size = self._batch_size_fn(X)
                eigcorrs = eigencorrection_fns[batch_size](
                    self._params, X, y, input_covariances, gradient_covariances
                )

                is_averaged = self._loss_func.reduction == "mean"
                weight = batch_size / self._N_data if is_averaged else 1.0
                for key, eigcorr in eigcorrs.items():
                    self._set_or_add_(corrected_eigenvalues, key, eigcorr.mul_(weight))

        return input_covariances, gradient_covariances, corrected_eigenvalues, mapping
