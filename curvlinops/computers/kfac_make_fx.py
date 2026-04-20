"""KFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxKFACComputer``, which computes Kronecker factors
by tracing the model's forward+backward pass with ``torch.fx`` via the IO
collector, collecting layer inputs/outputs rather than using forward/backward
hooks. The entire per-batch computation (IO collection, backward pass, and
covariance einsums) is traced with ``make_fx``, allowing ``torch.compile``
to optimize the full per-batch kernel.
"""

from collections import UserDict, defaultdict
from collections.abc import Callable, MutableMapping
from math import sqrt

from einops import einsum
from torch import Tensor, autograd, cat, eye, no_grad
from torch.nn import CrossEntropyLoss

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _BaseKFACComputer
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac_math import (
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _make_fx, fork_rng_with_seed


def _build_param_groups_from_io(
    io_param_names: dict[str, dict[str, str]],
    separate_weight_and_bias: bool,
) -> tuple[list[ParamGroup], dict[ParamGroupKey, list[str]]]:
    """Build parameter groups and IO-layer mapping from IO collector detections.

    Groups IO layers by weight name to form virtual modules. Each virtual
    module produces one group (joint) or separate W/b groups (separate).
    Also maps each group key to its contributing IO layer names.

    Args:
        io_param_names: Per-IO-layer parameter names from the IO collector.
        separate_weight_and_bias: Whether to treat weight and bias separately.

    Returns:
        Tuple of (parameter groups, IO-layer mapping).

    Raises:
        ValueError: If joint treatment and a weight has conflicting biases.
    """
    # Build virtual modules: one per unique weight (or standalone bias).
    # Each has a ParamGroup and a list of contributing IO layer names.
    modules: dict[str, ParamGroup] = {}
    module_io: dict[str, list[str]] = defaultdict(list)

    for io_name, pnames in io_param_names.items():
        w, b = pnames.get("W"), pnames.get("b")
        if w is not None:
            modules.setdefault(w, {"W": w})
            module_io[w].append(io_name)
            if b is not None:
                existing = modules[w].get("b")
                if not separate_weight_and_bias and existing and existing != b:
                    raise ValueError(
                        f"Weight '{w}' is used with conflicting biases "
                        f"'{existing}' and '{b}' under joint treatment. "
                        f"Use separate_weight_and_bias=True."
                    )
                modules[w]["b"] = b
        elif b is not None:
            modules[b] = {"b": b}
            module_io[b].append(io_name)

    # Convert virtual modules to parameter groups and IO-layer mapping
    groups: list[ParamGroup] = []
    io_groups: dict[ParamGroupKey, list[str]] = {}

    for mod_key, mod in modules.items():
        io = module_io[mod_key]
        param_dicts = (
            [{r: n} for r, n in mod.items()] if separate_weight_and_bias else [mod]
        )
        for pd in param_dicts:
            groups.append(pd)
            key = tuple(pd.values())
            # Bias-only groups: only IO layers that actually use the bias
            io_groups[key] = (
                [n for n in io if "b" in io_param_names[n]] if "W" not in pd else io
            )

    return groups, io_groups


def _bias_pad(has_joint_wb: bool, io_layer_params: dict[str, str]) -> int | None:
    """Determine bias padding for a specific IO layer usage.

    Args:
        has_joint_wb: Whether the parameter group has joint weight+bias.
        io_layer_params: Parameter roles for this IO layer.

    Returns:
        ``1`` if bias active, ``0`` if joint but bias inactive, ``None`` otherwise.
    """
    if not has_joint_wb:
        return None
    return 1 if "b" in io_layer_params else 0


def make_group_gatherers(
    io_groups: dict[ParamGroupKey, list[str]],
    io_param_names: dict[str, dict[str, str]],
    layer_hparams: dict[str, dict],
    kfac_approx: str = KFACType.EXPAND,
) -> tuple[
    Callable[[ParamGroup, dict[str, Tensor]], Tensor],
    Callable[[ParamGroup, dict[str, Tensor]], Tensor],
]:
    """Create closures that gather per-group layer inputs/gradients in weight sharing format.

    Args:
        io_groups: Mapping from parameter group keys to IO layer names.
        io_param_names: Per-IO-layer parameter name mappings.
        layer_hparams: Per-IO-layer hyperparameter dicts.
        kfac_approx: KFAC approximation type. Defaults to ``KFACType.EXPAND``.

    Returns:
        Tuple of ``(group_inputs, group_grads)`` closures.
    """

    def group_inputs(group: ParamGroup, layer_inputs: dict[str, Tensor]) -> Tensor:
        """Gather a group's layer inputs in weight sharing format.

        Args:
            group: Parameter group dict.
            layer_inputs: Raw layer inputs keyed by IO layer name.

        Returns:
            Concatenated tensor of shape ``[batch, shared, d_in]``.
        """
        group_key = tuple(group.values())
        io_names = io_groups[group_key]
        has_joint_wb = "b" in group
        xs = [
            input_to_weight_sharing_format(
                layer_inputs[n].data.detach(),
                kfac_approx,
                layer_hyperparams=layer_hparams[n],
                bias_pad=_bias_pad(has_joint_wb, io_param_names[n]),
            )
            for n in io_names
        ]
        return cat(xs, dim=1)

    def group_grads(group: ParamGroup, layer_output_grads: dict[str, Tensor]) -> Tensor:
        """Gather a group's layer output gradients in weight sharing format.

        Args:
            group: Parameter group dict.
            layer_output_grads: Batched output gradients keyed by IO layer name.

        Returns:
            Concatenated tensor of shape ``[v, batch, shared, d_out]``.
        """
        group_key = tuple(group.values())
        io_names = io_groups[group_key]
        gs = [
            grad_to_weight_sharing_format(
                layer_output_grads[n].data.detach(),
                kfac_approx,
                layer_hyperparams=layer_hparams[n],
                num_leading_dims=2,
            )
            for n in io_names
        ]
        return cat(gs, dim=2)

    return group_inputs, group_grads


def make_compute_kfac_io_batch(
    model_func: Callable,
    loss_func: Callable,
    params: dict[str, Tensor],
    X: Tensor | MutableMapping,
    fisher_type: FisherType = FisherType.MC,
    mc_samples: int = 1,
    separate_weight_and_bias: bool = True,
    output_check_fn: Callable[[Tensor], None] | None = None,
    intermediate_as_batch: bool = True,
) -> tuple[
    Callable[
        [dict[str, Tensor], Tensor | MutableMapping, Tensor],
        tuple[dict[str, Tensor], dict[str, Tensor]],
    ],
    list[ParamGroup],
    dict[ParamGroupKey, list[str]],
    dict[str, dict[str, str]],
    dict[str, dict],
]:
    """Set up per-batch IO collection and backward pass for KFAC.

    Returns an **untraced** closure that, given ``(params, X, y)``, runs the
    forward pass with IO collection and backpropagates the Fisher-type-specific
    gradient outputs to produce per-layer inputs and batched output gradients.

    Args:
        model_func: Functional model ``(params, X) -> prediction``.
        loss_func: Loss function (``MSELoss``, ``CrossEntropyLoss``, or
            ``BCEWithLogitsLoss``).
        params: Named parameter dict (example for IO tracing).
        X: Example input tensor (shapes determine the traced graph).
        fisher_type: Type of Fisher information. Defaults to
            ``FisherType.MC``.
        mc_samples: Number of Monte-Carlo samples (only used when
            ``fisher_type=FisherType.MC``). Defaults to ``1``.
        separate_weight_and_bias: Whether to treat weights and biases
            separately. Defaults to ``True``.
        output_check_fn: Optional callback ``(output) -> None`` called with
            the model output during setup. Raise inside this callback to
            reject unsupported output shapes (e.g., EKFAC's 2d restriction).
        intermediate_as_batch: Whether to treat the model output's
            intermediate (non-batch, non-class) axes as additional batch
            samples. ``True`` (default) reproduces KFAC-expand
            (Eschenhagen et al., 2023): per-token grad_outputs are computed
            on a flattened ``[B*prod(D), C]`` view of the model output.
            ``False`` keeps the intermediate axes separate so each
            ``(*D, C)`` slice is treated as one per-sample output, and the
            per-sample decomposition ``sum_{v,n,t} g g^T (otimes) a a^T``
            reconstructs the exact per-layer GGN block (for position-wise
            layers with ``FisherType.TYPE2``). Not supported with
            ``FisherType.EMPIRICAL``.

    Returns:
        Tuple of ``(inputs_and_grad_outputs_batch_fn, mapping, io_groups, io_param_names,
        layer_hparams)`` where ``inputs_and_grad_outputs_batch_fn(params, X, y)`` returns
        ``(layer_inputs, layer_output_grads)`` dicts keyed by IO layer name.
        For mean reduction, ``layer_output_grads`` is scaled so that
        ``sum_v grad_outputs grad_outputs^T`` equals the batch loss Hessian
        (i.e., the ``1/N`` reduction factor is folded in once as ``1/sqrt(N)``
        per vector). Downstream consumers therefore do not need to apply any
        additional reduction-dependent correction.

    Raises:
        ValueError: If ``intermediate_as_batch=False`` is combined with
            ``FisherType.EMPIRICAL`` (the empirical per-datum loss helper
            assumes a 1d prediction shape).
    """
    if not intermediate_as_batch and fisher_type == FisherType.EMPIRICAL:
        raise ValueError(
            "intermediate_as_batch=False is not supported with "
            "FisherType.EMPIRICAL because the per-datum loss helper assumes "
            "a 1d prediction shape."
        )

    grad_outputs_computer = _BaseKFACComputer._set_up_grad_outputs_computer(
        loss_func, fisher_type, mc_samples
    )
    io_fn, io_param_names, layer_hparams = with_kfac_io(
        model_func, X, params, fisher_type
    )
    mapping, io_groups = _build_param_groups_from_io(
        io_param_names, separate_weight_and_bias
    )

    def inputs_and_grad_outputs_batch(
        params: dict[str, Tensor], X: Tensor | MutableMapping, y: Tensor
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Run forward pass with IO collection and backpropagate grad outputs.

        Args:
            params: Named model parameters.
            X: Input batch.
            y: Target batch.

        Returns:
            ``(layer_inputs, layer_output_grads)`` dicts keyed by IO layer
            name. ``layer_output_grads`` is empty when the IO collector
            did not store outputs (e.g. ``FORWARD_ONLY``).
        """
        output, layer_inputs, layer_outputs = io_fn(params, X)

        if output_check_fn is not None:
            output_check_fn(output)

        if fisher_type == FisherType.FORWARD_ONLY:
            return layer_inputs, {}

        if intermediate_as_batch:
            # CrossEntropyLoss expects class dim second; other losses last.
            if isinstance(loss_func, CrossEntropyLoss):
                output_for_grad = output.movedim(1, -1).flatten(0, -2)
                y_for_grad = y.flatten()
            else:
                output_for_grad = output.flatten(0, -2)
                y_for_grad = y.flatten(0, -2)
        else:
            output_for_grad = output
            y_for_grad = y
        grad_outputs = grad_outputs_computer(output_for_grad.detach(), y_for_grad, None)
        # Equivalent to the hooks backend's two-step scaling (pre-multiply
        # ``grad_outputs`` by ``1/num_loss_terms``, then apply
        # ``compute_loss_correction`` on ``ggT``): combining both into a single
        # ``1/sqrt(N)`` per vector squares to the same ``1/N`` on ``ggT``.
        mean_scale = 1.0 / sqrt(output_for_grad.shape[0])
        grad_outputs.mul_({"sum": 1.0, "mean": mean_scale}[loss_func.reduction])

        io_layer_names = list(layer_outputs)
        output_tensors = list(layer_outputs.values())
        layer_output_grads_list = autograd.grad(
            output_for_grad,
            output_tensors,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
        )
        layer_output_grads = dict(zip(io_layer_names, layer_output_grads_list))
        return layer_inputs, layer_output_grads

    return (
        inputs_and_grad_outputs_batch,
        mapping,
        io_groups,
        io_param_names,
        layer_hparams,
    )


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
    output_check_fn: Callable[[Tensor], None] | None = None,
) -> tuple[
    Callable[
        [dict[str, Tensor], Tensor | MutableMapping, Tensor],
        tuple[list[Tensor], list[Tensor]],
    ],
    list[ParamGroup],
]:
    """Set up and trace the per-batch KFAC Kronecker factor computation.

    Builds on :func:`make_compute_kfac_io_batch` by adding the covariance
    computation (weight-sharing format conversion + einsum) and tracing the
    entire pipeline with ``make_fx`` into a single FX graph with zero graph
    breaks.

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
        output_check_fn: Passed to :func:`make_compute_kfac_io_batch`.

    Returns:
        Tuple of ``(traced_fn, mapping)`` where ``traced_fn`` is a compiled
        function ``(params, X, y) -> (input_covs, gradient_covs)`` (each a
        dict mapping parameter group keys to tensors) and ``mapping`` is the
        list of parameter groups.
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

    if batch_size_fn is None:

        def batch_size_fn(X):
            return X.shape[0]

    group_inputs, group_grads = make_group_gatherers(
        io_groups, io_param_names, layer_hparams, kfac_approx
    )

    def compute_batch(
        params: dict[str, Tensor], X: Tensor | MutableMapping, y: Tensor
    ) -> tuple[dict[tuple[str, ...], Tensor], dict[tuple[str, ...], Tensor]]:
        """Compute per-batch input and gradient covariances for all groups.

        Args:
            params: Named model parameters.
            X: Input batch.
            y: Target batch.

        Returns:
            Tuple of ``(input_covs, gradient_covs)`` dicts.
        """
        layer_inputs, layer_output_grads = inputs_and_grad_outputs_batch(params, X, y)
        batch_size = batch_size_fn(X)

        input_covs: dict[tuple[str, ...], Tensor] = {}
        for group in mapping:
            if "W" not in group:
                continue
            x = group_inputs(group, layer_inputs)
            group_key = tuple(group.values())
            xxT = einsum(x, x, "batch shared i, batch shared j -> i j")
            input_covs[group_key] = xxT.div_(batch_size * x.shape[1])

        gradient_covs: dict[tuple[str, ...], Tensor] = {}
        for group in mapping:
            group_key = tuple(group.values())
            if fisher_type == FisherType.FORWARD_ONLY:
                W = params[next(iter(group.values()))]
                gradient_covs[group_key] = eye(
                    W.shape[0], dtype=W.dtype, device=W.device
                )
                continue
            g = group_grads(group, layer_output_grads)
            gradient_covs[group_key] = einsum(
                g, g, "v batch shared i, v batch shared j -> i j"
            )

        return input_covs, gradient_covs

    traced_fn = _make_fx(compute_batch)(params, X, y)

    return traced_fn, mapping


class MakeFxKFACComputer(_BaseKFACComputer):
    """KFAC computer that uses FX graph tracing instead of hooks.

    Uses the IO collector (``with_kfac_io``) to detect affine layers and collect
    their inputs/outputs via ``torch.fx``, then computes Kronecker factors from
    these collected values. The entire per-batch computation (forward pass with
    IO collection, backward pass, and covariance einsums) is traced with
    ``make_fx``, allowing ``torch.compile`` to optimize the full per-batch
    kernel.

    Supports plain callable ``model_func``.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and enable gradients on params for autograd.grad."""
        super().__init__(*args, **kwargs)
        for p in self._params.values():
            p.requires_grad_(True)

    def _trace_batch_functions(
        self,
    ) -> tuple[dict[int, Callable], list[ParamGroup]]:
        """Trace per-batch KFAC computation for all batch sizes in the data.

        Iterates over the data once, tracing one FX graph per unique batch
        size.

        Returns:
            Tuple of ``(traced_fns, mapping)``.
        """
        traced_fns: dict[int, Callable] = {}
        mapping: list[ParamGroup] | None = None

        for X, y in self._loop_over_data(desc="FX tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size not in traced_fns:
                if isinstance(X, UserDict):
                    _register_userdict_as_pytree()
                traced_fns[batch_size], mapping = make_compute_kfac_batch(
                    self._model_func,
                    self._loss_func,
                    self._params,
                    X,
                    y,
                    self._fisher_type,
                    self._mc_samples,
                    self._kfac_approx,
                    self._separate_weight_and_bias,
                    self._batch_size_fn,
                )

        return traced_fns, mapping

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
                for key, xxT in input_covs.items():
                    self._set_or_add_(input_covariances, key, xxT.mul_(weight))

                is_averaged = (
                    self._loss_func.reduction == "mean"
                    or self._fisher_type == FisherType.FORWARD_ONLY
                )
                grad_weight = weight if is_averaged else 1.0
                for key, ggT in gradient_covs.items():
                    self._set_or_add_(gradient_covariances, key, ggT.mul_(grad_weight))

        return input_covariances, gradient_covariances, mapping
