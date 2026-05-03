"""EKFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxEKFACComputer``, which extends
``MakeFxKFACComputer`` with FX-based eigenvalue correction, using the IO
collector (``with_kfac_io``) instead of forward/backward hooks.
"""

from collections.abc import Callable, MutableMapping

from torch import Tensor, empty, no_grad

from curvlinops.computers._base import ParamGroup, ParamGroupKey, _EKFACMixin
from curvlinops.computers.ekfac_hooks import (
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.computers.io_collector import LayerIO
from curvlinops.computers.kfac_make_fx import (
    MakeFxKFACComputer,
    make_compute_kfac_io_batch,
    make_group_gatherers,
)
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _enable_requires_grad, _make_fx, fork_rng_with_seed


def make_compute_ekfac_eigencorrection_batch(
    model_func: Callable,
    loss_func: Callable,
    params: dict[str, Tensor],
    X: Tensor | MutableMapping,
    y: Tensor,
    fisher_type: FisherType = FisherType.MC,
    mc_samples: int = 1,
    separate_weight_and_bias: bool = True,
    output_check_fn: Callable[[Tensor], None] | None = None,
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

    Builds on :func:`make_compute_kfac_io_batch` by adding eigencorrection
    computation (weight-sharing format conversion + rotated per-example
    gradient squaring) and tracing the entire pipeline with ``make_fx``.

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
        output_check_fn: Passed to :func:`make_compute_kfac_io_batch`.

    Returns:
        Tuple of ``(traced_fn, mapping)`` where ``traced_fn`` is a compiled
        function ``(params, X, y, input_eigvecs, gradient_eigvecs) ->
        eigencorrections`` (dict mapping parameter group keys to tensors)
        and ``mapping`` is the list of parameter groups.
    """
    inputs_and_grad_outputs_batch, mapping, io_groups, io_param_names, layer_hparams = (
        make_compute_kfac_io_batch(
            model_func,
            loss_func,
            params,
            X,
            fisher_type,
            mc_samples,
            separate_weight_and_bias,
            output_check_fn,
        )
    )

    group_inputs, group_grads = make_group_gatherers(
        io_groups, io_param_names, layer_hparams, KFACType.EXPAND
    )

    def compute_eigencorrection_batch(
        params: dict[str, Tensor],
        X: Tensor | MutableMapping,
        y: Tensor,
        input_eigvecs: dict[ParamGroupKey, Tensor],
        gradient_eigvecs: dict[ParamGroupKey, Tensor],
    ) -> dict[tuple[str, ...], Tensor]:
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
        layer_inputs, layer_output_grads = inputs_and_grad_outputs_batch(params, X, y)

        eigencorrections: dict[tuple[str, ...], Tensor] = {}
        for group in mapping:
            group_key = tuple(group.values())

            g = group_grads(group, layer_output_grads)
            ggT_eigvecs = gradient_eigvecs[group_key]

            a = None
            aaT_eigvecs = None
            if "W" in group:
                a = group_inputs(group, layer_inputs)
                aaT_eigvecs = input_eigvecs[group_key]

            eigencorrections[group_key] = (
                compute_eigenvalue_correction_linear_weight_sharing(
                    g, ggT_eigvecs, a, aaT_eigvecs
                )
            )

        return eigencorrections

    # Create example eigvecs with correct shapes for tracing.
    # Shapes derived from params: D_out = W.shape[0], D_in = W[0].numel() (+1 for bias pad).
    example_input_eigvecs: dict[ParamGroupKey, Tensor] = {}
    example_gradient_eigvecs: dict[ParamGroupKey, Tensor] = {}
    for group in mapping:
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

    # ``make_fx`` traces the autograd.grad call inside the closure into
    # explicit backward aten ops. ``params`` must therefore be differentiable
    # at trace time; the wrap is local to this call so any prior
    # ``requires_grad`` state on the user's tensors is restored after tracing.
    with _enable_requires_grad(list(params.values())):
        traced_fn = _make_fx(compute_eigencorrection_batch)(
            params, X, y, example_input_eigvecs, example_gradient_eigvecs
        )

    return traced_fn, mapping


class MakeFxEKFACComputer(_EKFACMixin, MakeFxKFACComputer):
    """EKFAC computer that uses FX graph tracing for eigenvalue correction.

    Extends ``MakeFxKFACComputer`` (which already drives KFAC factor tracing
    through :class:`LayerIO`) with eigenvalue correction tracing via the
    same abstraction. The 2d-output restriction is enforced inside the
    trace via :class:`LayerIO`'s ``output_check_fn`` hook.
    """

    def _trace_batch_functions(
        self,
    ) -> tuple[dict[int, Callable], list[ParamGroup]]:
        """Trace KFAC factor computation with EKFAC's 2d-output check.

        Overrides the parent only to thread ``output_check_fn`` into the
        :class:`LayerIO` constructor.

        Returns:
            Tuple of ``(traced_fns, mapping)``.
        """
        traced_fns: dict[int, Callable] = {}
        io: LayerIO | None = None

        for X, y in self._loop_over_data(desc="FX tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size in traced_fns:
                continue

            if io is None:
                io = LayerIO(
                    self._model_func,
                    self._loss_func,
                    self._params,
                    X,
                    fisher_type=self._fisher_type,
                    mc_samples=self._mc_samples,
                    kfac_approx=self._kfac_approx,
                    separate_weight_and_bias=self._separate_weight_and_bias,
                    batch_size_fn=self._batch_size_fn,
                    output_check_fn=self._rearrange_for_larger_than_2d_output,
                )

            traced_fns[batch_size] = self._trace_one_batch(io, X, y)

        return traced_fns, io.mapping

    def _trace_eigencorrection_batch_functions(
        self,
    ) -> tuple[dict[int, Callable], list[ParamGroup]]:
        """Trace per-batch eigencorrection for all batch sizes in the data.

        Returns:
            Tuple of ``(traced_fns, mapping)``.
        """
        traced_fns: dict[int, Callable] = {}
        io: LayerIO | None = None

        for X, y in self._loop_over_data(desc="Eigencorrection tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size in traced_fns:
                continue

            if io is None:
                # ``kfac_approx`` is hardcoded to :attr:`KFACType.EXPAND`:
                # :func:`compute_eigenvalue_correction_linear_weight_sharing`
                # consumes EXPAND-format ``(a, g)`` regardless of the operator's
                # ``kfac_approx`` (which only controls factor accumulation).
                io = LayerIO(
                    self._model_func,
                    self._loss_func,
                    self._params,
                    X,
                    fisher_type=self._fisher_type,
                    mc_samples=self._mc_samples,
                    kfac_approx=KFACType.EXPAND,
                    separate_weight_and_bias=self._separate_weight_and_bias,
                    batch_size_fn=self._batch_size_fn,
                    output_check_fn=self._rearrange_for_larger_than_2d_output,
                )

            traced_fns[batch_size] = self._trace_eigencorrection_one_batch(io, X, y)

        return traced_fns, io.mapping

    def _trace_eigencorrection_one_batch(
        self, io: LayerIO, X: Tensor | MutableMapping, y: Tensor
    ) -> Callable:
        """Trace per-batch eigencorrection for one example ``(X, y)`` pair.

        Args:
            io: The :class:`LayerIO` (must already have an ``io_fn`` cached
                for ``X``'s batch size).
            X: Example input batch (shape baked into the trace).
            y: Example target batch.

        Returns:
            Traced callable
            ``(params, X, y, input_eigvecs, gradient_eigvecs) -> eigencorrections``.
        """

        def compute_eigencorrection_batch(
            params: dict[str, Tensor],
            X: Tensor | MutableMapping,
            y: Tensor,
            input_eigvecs: dict[ParamGroupKey, Tensor],
            gradient_eigvecs: dict[ParamGroupKey, Tensor],
        ) -> dict[tuple[str, ...], Tensor]:
            layer_inputs, layer_output_grads = io.populate(params, X, y)
            snap = io.snapshot(layer_inputs, layer_output_grads)

            eigencorrections: dict[tuple[str, ...], Tensor] = {}
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
            p1 = self._params[next(iter(group.values()))]
            d_out = p1.shape[0]
            example_gradient_eigvecs[group_key] = empty(
                d_out, d_out, dtype=p1.dtype, device=p1.device
            )
            if "W" in group:
                d_in = p1.numel() // p1.shape[0] + ("b" in group)
                example_input_eigvecs[group_key] = empty(
                    d_in, d_in, dtype=p1.dtype, device=p1.device
                )

        with io.enable_param_grads(self._params):
            return _make_fx(compute_eigencorrection_batch)(
                self._params, X, y, example_input_eigvecs, example_gradient_eigvecs
            )

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
