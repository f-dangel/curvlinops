"""KFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxKFACComputer``, which computes Kronecker factors
by tracing the model's forward+backward pass with ``torch.fx`` via the IO
collector, collecting layer inputs/outputs rather than using forward/backward
hooks. The entire per-batch computation (IO collection, backward pass, and
covariance einsums) is traced with ``make_fx``, allowing ``torch.compile``
to optimize the full per-batch kernel.
"""

from collections import UserDict, defaultdict
from collections.abc import Callable
from typing import Any

from einops import einsum
from torch import Tensor, autograd, cat, manual_seed, no_grad

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _BaseKFACComputer
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _make_fx, make_functional_call


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


def _make_batch_fn(
    io_fn: Callable,
    io_param_names: dict[str, dict[str, str]],
    layer_hparams: dict[str, dict[str, Any]],
    mapping: list[ParamGroup],
    io_groups: dict[ParamGroupKey, list[str]],
    kfac_approx: Any,
    fisher_type: FisherType,
    loss_reduction: str,
    num_per_example_loss_terms: int | None,
    grad_outputs_computer: Callable,
    rearrange_fn: Callable,
) -> Callable[[dict[str, Tensor], Tensor, Tensor], tuple[list[Tensor], list[Tensor]]]:
    """Build the per-batch KFAC factor computation function (internal).

    This is the low-level builder used by :func:`make_compute_kfac_batch` and
    by the EKFAC backend (which needs separate access to IO functions).

    Args:
        io_fn: Traced IO-collecting forward function from ``with_kfac_io``.
        io_param_names: Layer parameter name mappings.
        layer_hparams: Layer hyperparameter dicts.
        mapping: List of parameter groups.
        io_groups: Maps group keys to contributing IO layer names.
        kfac_approx: KFAC approximation type.
        fisher_type: Fisher type (``TYPE2``, ``MC``, ``EMPIRICAL``, etc.).
        loss_reduction: Loss reduction mode (``"sum"`` or ``"mean"``).
        num_per_example_loss_terms: Number of loss terms per example.
        grad_outputs_computer: Function computing batched gradient outputs
            ``(prediction, labels, generator) -> grad_outputs``.
        rearrange_fn: Function ``(output, y) -> (output_rearranged, y_rearranged)``
            for larger-than-2d outputs.

    Returns:
        Function ``(params, X, y) -> (input_covs, gradient_covs)`` where each
        is a list of tensors (one per group with a weight / one per group).
    """
    # Pre-compute group structure as lists for stable ordering in the trace
    weight_group_info = []
    for group in mapping:
        if "W" not in group:
            continue
        group_key = tuple(group.values())
        io_names = io_groups.get(group_key, [])
        has_joint_wb = "b" in group
        bias_pads = [_bias_pad(has_joint_wb, io_param_names[n]) for n in io_names]
        hparams = [layer_hparams[n] for n in io_names]
        weight_group_info.append((io_names, bias_pads, hparams))

    grad_group_info = []
    for group in mapping:
        group_key = tuple(group.values())
        io_names = io_groups.get(group_key, [])
        hparams = [layer_hparams[n] for n in io_names]
        grad_group_info.append((io_names, hparams))

    is_forward_only = fisher_type == FisherType.FORWARD_ONLY

    def compute_batch(
        params: dict[str, Tensor], X: Tensor, y: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        output, layer_inputs, layer_outputs = io_fn(params, X)

        # Input covariances
        input_covs = []
        for io_names, bias_pads, hparams in weight_group_info:
            xs = [
                input_to_weight_sharing_format(
                    layer_inputs[n].data.detach(),
                    kfac_approx,
                    layer_hyperparams=hp,
                    bias_pad=bp,
                )
                for n, bp, hp in zip(io_names, bias_pads, hparams)
            ]
            x = cat(xs, dim=1)
            scale = x.shape[1]
            xxT = einsum(x, x, "batch shared i, batch shared j -> i j")
            input_covs.append(xxT.div_(scale))

        if is_forward_only:
            return input_covs, []

        # Backward pass
        output_local, y_local = rearrange_fn(output, y)
        grad_outputs = grad_outputs_computer(output_local.detach(), y_local, None)
        num_loss_terms = output_local.shape[0]
        scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[loss_reduction]
        grad_outputs.mul_(scale)

        io_layer_names = list(layer_outputs)
        output_tensors = list(layer_outputs.values())
        layer_output_grads_list = autograd.grad(
            output_local,
            output_tensors,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
        )
        layer_output_grads = dict(zip(io_layer_names, layer_output_grads_list))

        # Gradient covariances
        gradient_covs = []
        for io_names, hparams in grad_group_info:
            gs = [
                grad_to_weight_sharing_format(
                    layer_output_grads[n].data.detach(),
                    kfac_approx,
                    layer_hyperparams=hp,
                    num_leading_dims=2,
                )
                for n, hp in zip(io_names, hparams)
            ]
            g = cat(gs, dim=2)
            correction = compute_loss_correction(
                g.shape[1],
                num_per_example_loss_terms,
                loss_reduction,
            )
            ggT = einsum(g, g, "v batch shared i, v batch shared j -> i j").mul_(
                correction
            )
            gradient_covs.append(ggT)

        return input_covs, gradient_covs

    return compute_batch


def make_compute_kfac_batch(
    model_func: Callable,
    loss_func: Callable,
    params: dict[str, Tensor],
    X: Tensor,
    y: Tensor,
    fisher_type: FisherType = FisherType.MC,
    mc_samples: int = 1,
    kfac_approx: str = KFACType.EXPAND,
    separate_weight_and_bias: bool = True,
) -> tuple[
    Callable[[dict[str, Tensor], Tensor, Tensor], tuple[list[Tensor], list[Tensor]]],
    list[ParamGroup],
    list[ParamGroupKey],
    list[ParamGroupKey],
]:
    """Set up and trace the per-batch KFAC Kronecker factor computation.

    Analogous to :func:`make_batch_hessian_vector_product` but for KFAC:
    detects affine layers via the IO collector, builds the per-batch Kronecker
    factor computation, and traces it with ``make_fx`` into a single FX graph
    with zero graph breaks.

    The interface mirrors :class:`MakeFxKFACComputer` minus the accumulation-
    related arguments (``data``, ``seed``, ``num_data``, etc.).

    ``params``, ``X``, and ``y`` serve as example inputs for tracing (their
    shapes and device determine the traced graph). The returned function
    accepts any tensors with matching shapes.

    Args:
        model_func: Functional model ``(params, X) -> prediction``, or an
            ``nn.Module`` (converted internally via ``make_functional_call``).
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

    Returns:
        Tuple of ``(traced_fn, mapping, weight_group_keys, all_group_keys)``
        where ``traced_fn`` is a compiled function
        ``(params, X, y) -> (input_covs, gradient_covs)`` (each a list of
        tensors), ``mapping`` is the list of parameter groups, and the keys
        index the covariance lists.
    """
    from torch.nn import CrossEntropyLoss, Module

    if isinstance(model_func, Module):
        model_func = make_functional_call(model_func)

    # Infer num_per_example_loss_terms from (y, loss_func)
    batch_size = y.shape[0]
    if isinstance(loss_func, CrossEntropyLoss):
        num_per_example_loss_terms = y.numel() // batch_size
    else:
        num_per_example_loss_terms = y.shape[:-1].numel() // batch_size

    # Set up grad_outputs_computer and rearrange_fn from (loss_func, fisher_type)
    grad_outputs_computer = _BaseKFACComputer._set_up_grad_outputs_computer(
        loss_func, fisher_type, mc_samples
    )

    if isinstance(loss_func, CrossEntropyLoss):

        def rearrange_fn(output: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
            return output.movedim(1, -1).flatten(0, -2), y.flatten()

    else:

        def rearrange_fn(output: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
            return output.flatten(0, -2), y.flatten(0, -2)

    # Trace IO collection and build parameter groups
    io_fn, io_param_names, layer_hparams = with_kfac_io(
        model_func, X, params, fisher_type
    )
    mapping, io_groups = _build_param_groups_from_io(
        io_param_names, separate_weight_and_bias
    )
    weight_group_keys = [tuple(g.values()) for g in mapping if "W" in g]
    all_group_keys = [tuple(g.values()) for g in mapping]

    # Build and trace the per-batch computation
    batch_fn = _make_batch_fn(
        io_fn,
        io_param_names,
        layer_hparams,
        mapping,
        io_groups,
        kfac_approx,
        fisher_type,
        loss_func.reduction,
        num_per_example_loss_terms,
        grad_outputs_computer,
        rearrange_fn,
    )
    traced_fn = _make_fx(batch_fn)(params, X, y)

    return traced_fn, mapping, weight_group_keys, all_group_keys


class MakeFxKFACComputer(_BaseKFACComputer):
    """KFAC computer that uses FX graph tracing instead of hooks.

    Uses the IO collector (``with_kfac_io``) to detect affine layers and collect
    their inputs/outputs via ``torch.fx``, then computes Kronecker factors from
    these collected values. The entire per-batch computation (forward pass with
    IO collection, backward pass, and covariance einsums) is traced with
    ``make_fx``, allowing ``torch.compile`` to optimize the full per-batch
    kernel.

    Supports both ``nn.Module`` and plain callable ``model_func``.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and enable gradients on params for autograd.grad."""
        super().__init__(*args, **kwargs)
        for p in self._params.values():
            p.requires_grad_(True)

    def _trace_batch_functions(
        self,
    ) -> tuple[
        dict[int, Callable],
        list[ParamGroup],
        list[ParamGroupKey],
        list[ParamGroupKey],
    ]:
        """Trace per-batch KFAC computation for all batch sizes in the data.

        Iterates over the data once, tracing one FX graph per unique batch
        size.

        Returns:
            Tuple of ``(traced_fns, mapping, weight_group_keys, all_group_keys)``.
        """
        traced_fns: dict[int, Callable] = {}
        mapping: list[ParamGroup] | None = None
        weight_group_keys: list[ParamGroupKey] | None = None
        all_group_keys: list[ParamGroupKey] | None = None

        for X, y in self._loop_over_data(desc="FX tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size not in traced_fns:
                if isinstance(X, UserDict):
                    _register_userdict_as_pytree()
                io_fn, io_param_names, layer_hparams = with_kfac_io(
                    self._model_func, X, self._params, self._fisher_type
                )
                mapping, io_groups = _build_param_groups_from_io(
                    io_param_names, self._separate_weight_and_bias
                )
                weight_group_keys = [tuple(g.values()) for g in mapping if "W" in g]
                all_group_keys = [tuple(g.values()) for g in mapping]
                batch_fn = _make_batch_fn(
                    io_fn,
                    io_param_names,
                    layer_hparams,
                    mapping,
                    io_groups,
                    self._kfac_approx,
                    self._fisher_type,
                    self._loss_func.reduction,
                    self._num_per_example_loss_terms,
                    self._grad_outputs_computer,
                    self._rearrange_for_larger_than_2d_output,
                )
                traced_fns[batch_size] = _make_fx(batch_fn)(self._params, X, y)

        return traced_fns, mapping, weight_group_keys, all_group_keys

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
        traced_batch: tuple[
            dict[int, Callable],
            list[ParamGroup],
            list[ParamGroupKey],
            list[ParamGroupKey],
        ],
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Accumulate KFAC's Kronecker factors using pre-traced batch functions.

        Runs the pre-traced per-batch functions and accumulates input and
        gradient covariances across all batches.

        Args:
            traced_batch: Pre-traced batch functions from
                :func:`trace_kfac_batch`.

        Returns:
            Tuple of (input_covariances, gradient_covariances, mapping).
        """
        traced_fns, mapping, weight_group_keys, all_group_keys = traced_batch

        grad_normalization = {
            "sum": 1.0,
            "mean": 1.0 / self._N_data,
        }[self._loss_func.reduction]

        input_covariances: dict[ParamGroupKey, Tensor] = {}
        gradient_covariances: dict[ParamGroupKey, Tensor] = {}

        manual_seed(self._seed)

        # no_grad: the traced graph already contains explicit backward ops from
        # make_fx; disabling the outer autograd avoids retaining intermediates.
        with no_grad():
            for X, y in self._loop_over_data(desc="KFAC matrices"):
                batch_size = self._batch_size_fn(X)
                input_covs, gradient_covs = traced_fns[batch_size](self._params, X, y)

                for key, xxT in zip(weight_group_keys, input_covs):
                    self._set_or_add_(input_covariances, key, xxT.div_(self._N_data))

                for key, ggT in zip(all_group_keys, gradient_covs):
                    self._set_or_add_(
                        gradient_covariances, key, ggT.mul_(grad_normalization)
                    )

        if self._fisher_type == FisherType.FORWARD_ONLY:
            self._set_gradient_covariances_to_identity(gradient_covariances, mapping)

        return input_covariances, gradient_covariances, mapping
