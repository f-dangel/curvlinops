"""KFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxKFACComputer``, which computes Kronecker factors
by tracing the model's forward+backward pass with ``torch.fx`` via
:class:`LayerIO`, collecting layer inputs/outputs rather than using
forward/backward hooks. The entire per-batch computation (IO collection,
backward pass, and covariance einsums) is traced with ``make_fx``, allowing
``torch.compile`` to optimize the full per-batch kernel.

The standalone factory :func:`make_compute_kfac_batch` returns a traced
per-batch closure for users who want a functional API without going through
:class:`MakeFxKFACComputer`.
"""

from collections.abc import Callable, MutableMapping

from einops import einsum
from torch import Tensor, eye, no_grad

from curvlinops.computers._base import ParamGroup, ParamGroupKey, _BaseKFACComputer
from curvlinops.computers.io_collector import LayerIO
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _make_fx, fork_rng_with_seed


def _make_kfac_closure(io: LayerIO) -> Callable:
    """Build the per-batch KFAC reduction closure operating on a shared ``io``.

    Returns:
        A closure ``(params, X, y) -> (input_covs, gradient_covs)``.
    """

    def compute_batch(
        params: dict[str, Tensor], X: Tensor | MutableMapping, y: Tensor
    ) -> tuple[dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor]]:
        layer_inputs, layer_output_grads = io.populate(params, X, y)
        snap = io.snapshot(layer_inputs, layer_output_grads)

        input_covs: dict[ParamGroupKey, Tensor] = {}
        gradient_covs: dict[ParamGroupKey, Tensor] = {}
        for group in io.mapping:
            a, g = snap.standardized_io(group)
            group_key = tuple(group.values())
            if a is not None:
                aaT = einsum(a, a, "batch shared i, batch shared j -> i j")
                input_covs[group_key] = aaT.div_(a.shape[0] * a.shape[1])
            if io.fisher_type == FisherType.FORWARD_ONLY:
                W = params[next(iter(group.values()))]
                ggT = eye(W.shape[0], dtype=W.dtype, device=W.device)
            else:
                ggT = einsum(g, g, "v batch shared i, v batch shared j -> i j")
            gradient_covs[group_key] = ggT
        return input_covs, gradient_covs

    return compute_batch


def make_compute_kfac_batch(
    model_func: Callable,
    loss_func: Callable,
    params: dict[str, Tensor],
    X: Tensor | MutableMapping,
    y: Tensor,
    fisher_type: FisherType = FisherType.MC,
    mc_samples: int = 1,
    kfac_approx: str = KFACType.EXPAND,
    separate_weight_and_bias: bool = True,
    batch_size_fn: Callable[[Tensor | MutableMapping], int] | None = None,
    output_check_fn: Callable[[Tensor, Tensor], object] | None = None,
) -> tuple[
    Callable[
        [dict[str, Tensor], Tensor | MutableMapping, Tensor],
        tuple[dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor]],
    ],
    list[ParamGroup],
]:
    """Set up and trace per-batch KFAC Kronecker factor computation.

    Builds a :class:`LayerIO` configured for the given Fisher/KFAC settings,
    wraps the per-batch reduction (IO collection, backward pass, per-group
    covariance einsums) in :func:`_make_fx`, and returns the traced function
    along with the parameter-group mapping. Each call produces a
    shape-specific traced function (the ``X`` shape is baked in).

    Args:
        model_func: Functional model ``(params, X) -> prediction``.
        loss_func: Loss function (``MSELoss``, ``CrossEntropyLoss``, or
            ``BCEWithLogitsLoss``).
        params: Named parameter dict.
        X: Example input tensor.
        y: Example target tensor.
        fisher_type: Type of Fisher information. Defaults to
            ``FisherType.MC``.
        mc_samples: Number of Monte-Carlo samples (only used when
            ``fisher_type=FisherType.MC``). Defaults to ``1``.
        kfac_approx: KFAC approximation type (``KFACType.EXPAND`` or
            ``KFACType.REDUCE``). Defaults to ``KFACType.EXPAND``.
        separate_weight_and_bias: Whether to treat weights and biases
            separately. Defaults to ``True``.
        batch_size_fn: Function to extract batch size from ``X``.
            Defaults to ``X.shape[0]``.
        output_check_fn: Optional ``(output, y) -> object`` callback
            forwarded to :class:`LayerIO`; raise inside it to reject
            unsupported output/target shapes.

    Returns:
        Tuple of ``(traced_fn, mapping)`` where ``traced_fn`` is a compiled
        function ``(params, X, y) -> (input_covs, gradient_covs)`` (each a
        dict mapping parameter group keys to tensors) and ``mapping`` is the
        list of parameter groups.
    """
    io = LayerIO(
        model_func,
        loss_func,
        params,
        X,
        fisher_type=fisher_type,
        mc_samples=mc_samples,
        kfac_approx=kfac_approx,
        separate_weight_and_bias=separate_weight_and_bias,
        batch_size_fn=batch_size_fn,
        output_check_fn=output_check_fn,
    )
    closure = _make_kfac_closure(io)
    with io.enable_param_grads(params):
        traced_fn = _make_fx(closure)(params, X, y)
    return traced_fn, io.mapping


class MakeFxKFACComputer(_BaseKFACComputer):
    """KFAC computer that uses FX graph tracing instead of hooks.

    Supports plain callable ``model_func``.
    """

    _output_check_fn: Callable[[Tensor, Tensor], object] | None = None

    def _make_layer_io(self, X: Tensor | MutableMapping, kfac_approx: str) -> LayerIO:
        """Build a :class:`LayerIO` configured for this operator.

        Args:
            X: Bootstrap input batch.
            kfac_approx: Value for ``LayerIO``'s ``kfac_approx``.

        Returns:
            The configured :class:`LayerIO`.
        """
        return LayerIO(
            self._model_func,
            self._loss_func,
            self._params,
            X,
            fisher_type=self._fisher_type,
            mc_samples=self._mc_samples,
            kfac_approx=kfac_approx,
            separate_weight_and_bias=self._separate_weight_and_bias,
            batch_size_fn=self._batch_size_fn,
            output_check_fn=self._output_check_fn,
        )

    def _trace_per_batch_size(
        self,
        make_closure: Callable[[LayerIO], Callable],
        desc: str,
        kfac_approx: str,
        *,
        make_extra_args: Callable[[LayerIO], tuple] | None = None,
    ) -> tuple[dict[int, Callable], list[ParamGroup]]:
        """Trace ``make_closure(io)`` once per unique batch size, sharing one ``LayerIO``.

        Builds one :class:`LayerIO` lazily on the first batch and reuses it
        for subsequent batch sizes via :meth:`LayerIO.ensure_io_fn`, so the
        bootstrap trace and the cross-batch-size ``io_fn`` cache are shared.

        Args:
            make_closure: Given the shared :class:`LayerIO`, returns the
                closure traced as ``_make_fx(closure)(params, X, y, *extra)``.
            desc: Progress description for ``_loop_over_data``.
            kfac_approx: Value for ``LayerIO``'s ``kfac_approx``.
            make_extra_args: Given the shared :class:`LayerIO`, returns extra
                positional trace args appended after ``(params, X, y)``.
                ``None`` (default) means no extras.

        Returns:
            Tuple of ``(traced_fns, mapping)`` keyed by batch size.
        """
        traced_fns: dict[int, Callable] = {}
        io: LayerIO | None = None
        for X, y in self._loop_over_data(desc=desc):
            bs = self._batch_size_fn(X)
            if bs in traced_fns:
                continue
            if io is None:
                io = self._make_layer_io(X, kfac_approx)
                closure = make_closure(io)
                extra_args = () if make_extra_args is None else make_extra_args(io)
            io.ensure_io_fn(X, self._params)
            with io.enable_param_grads(self._params):
                traced_fns[bs] = _make_fx(closure)(self._params, X, y, *extra_args)
        return traced_fns, io.mapping

    def _trace_batch_functions(
        self,
    ) -> tuple[dict[int, Callable], list[ParamGroup]]:
        """Trace per-batch KFAC computation for all batch sizes in the data.

        Returns:
            Tuple of ``(traced_fns, mapping)`` keyed by batch size.
        """
        return self._trace_per_batch_size(
            _make_kfac_closure, "FX tracing", self._kfac_approx
        )

    def compute(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Compute KFAC's Kronecker factors.

        Traces IO collection and batch computation in a single data pass,
        then accumulates factors in a second pass.

        Returns:
            Tuple of ``(input_covariances, gradient_covariances, mapping)``.
        """
        return self._compute_kronecker_factors(self._trace_batch_functions())

    def _compute_kronecker_factors(
        self,
        traced_batch: tuple[dict[int, Callable], list[ParamGroup]],
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Accumulate KFAC's Kronecker factors using pre-traced batch functions.

        Runs the pre-traced per-batch functions and accumulates input and
        gradient covariances across all batches.

        Args:
            traced_batch: Pre-traced batch functions from
                :meth:`_trace_batch_functions`.

        Returns:
            Tuple of (input_covariances, gradient_covariances, mapping).
        """
        traced_fns, mapping = traced_batch

        input_covariances: dict[ParamGroupKey, Tensor] = {}
        gradient_covariances: dict[ParamGroupKey, Tensor] = {}

        # Seed only for stochastic fisher types. fork_rng_with_seed isolates
        # the seed from the caller's global RNG state.
        # no_grad: the traced graph already contains explicit backward ops from
        # make_fx; disabling the outer autograd avoids retaining intermediates.
        seed = self._seed if self._fisher_type == FisherType.MC else None
        with fork_rng_with_seed(seed), no_grad():
            for X, y in self._loop_over_data(desc="KFAC matrices"):
                batch_size = self._batch_size_fn(X)
                input_covs, gradient_covs = traced_fns[batch_size](self._params, X, y)

                # The traced batch function returns per-batch averages.
                # Accumulate with batch_size / N_data weighting.
                weight = batch_size / self._N_data
                for key, aaT in input_covs.items():
                    self._set_or_add_(input_covariances, key, aaT.mul_(weight))

                is_averaged = (
                    self._loss_func.reduction == "mean"
                    or self._fisher_type == FisherType.FORWARD_ONLY
                )
                grad_weight = weight if is_averaged else 1.0
                for key, ggT in gradient_covs.items():
                    self._set_or_add_(gradient_covariances, key, ggT.mul_(grad_weight))

        return input_covariances, gradient_covariances, mapping
