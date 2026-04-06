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
from torch import Tensor, autograd, cat, manual_seed
from torch.fx.experimental.proxy_tensor import make_fx

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _BaseKFACComputer
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import FisherType


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

    def _trace_io_functions(
        self,
    ) -> tuple[
        dict[int, Callable],
        dict[str, dict[str, str]],
        dict[str, dict[str, Any]],
    ]:
        """Pre-trace IO collection functions for all batch sizes in the data.

        Iterates over the data once, calling ``with_kfac_io`` for each unique
        batch size. This separates the (expensive) FX tracing step from the
        factor computation, and makes it independently measurable.

        Returns:
            Tuple of ``(traced_io_fns, io_param_names, layer_hparams)`` where
            ``traced_io_fns`` maps batch sizes to traced IO-collecting callables,
            ``io_param_names`` maps layer names to parameter name dicts, and
            ``layer_hparams`` maps layer names to hyperparameter dicts.
        """
        traced_io_fns: dict[int, Callable] = {}
        io_param_names: dict[str, dict[str, str]] | None = None
        layer_hparams: dict[str, dict[str, Any]] | None = None

        for X, _ in self._loop_over_data(desc="FX tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size not in traced_io_fns:
                if isinstance(X, UserDict):
                    _register_userdict_as_pytree()
                traced_io_fns[batch_size], io_param_names, layer_hparams = with_kfac_io(
                    self._model_func, X, self._params, self._fisher_type
                )

        return traced_io_fns, io_param_names, layer_hparams

    def _trace_forward_backward_and_covariances(
        self,
        io_fn: Callable,
        io_param_names: dict[str, dict[str, str]],
        layer_hparams: dict[str, dict[str, Any]],
        mapping: list[ParamGroup],
        io_groups: dict[ParamGroupKey, list[str]],
        X: Tensor,
        y: Tensor,
    ) -> Callable:
        """Trace the full per-batch KFAC computation with ``make_fx``.

        Traces forward pass (IO collection), backward pass (batched gradients),
        and covariance einsums into a single FX graph. This captures
        ``autograd.grad`` as forward ops, enabling ``torch.compile``
        optimization of the entire per-batch kernel.

        Args:
            io_fn: Traced IO-collecting forward function.
            io_param_names: Layer parameter name mappings.
            layer_hparams: Layer hyperparameter dicts.
            mapping: List of parameter groups.
            io_groups: Maps group keys to contributing IO layer names.
            X: Example input for tracing.
            y: Example target for tracing.

        Returns:
            Traced function with signature
            ``(params, X, y) -> (input_covs, gradient_covs)`` where each
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
            bias_pads = [
                _bias_pad(has_joint_wb, io_param_names[n]) for n in io_names
            ]
            hparams = [layer_hparams[n] for n in io_names]
            weight_group_info.append((io_names, bias_pads, hparams))

        grad_group_info = []
        for group in mapping:
            group_key = tuple(group.values())
            io_names = io_groups.get(group_key, [])
            hparams = [layer_hparams[n] for n in io_names]
            grad_group_info.append((io_names, hparams))

        is_forward_only = self._fisher_type == FisherType.FORWARD_ONLY
        kfac_approx = self._kfac_approx
        grad_outputs_computer = self._grad_outputs_computer
        rearrange_fn = self._rearrange_for_larger_than_2d_output
        loss_reduction = self._loss_func.reduction
        num_per_example_loss_terms = self._num_per_example_loss_terms

        def forward_backward_and_covariances(
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
            grad_outputs = grad_outputs_computer(
                output_local.detach(), y_local, None
            )
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
                ggT = einsum(
                    g, g, "v batch shared i, v batch shared j -> i j"
                ).mul_(correction)
                gradient_covs.append(ggT)

            return input_covs, gradient_covs

        return make_fx(forward_backward_and_covariances)(self._params, X, y)

    def compute(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Compute KFAC's Kronecker factors, tracing IO functions first.

        Overrides the base class to separate FX tracing from factor computation.

        Returns:
            Tuple of ``(input_covariances, gradient_covariances, mapping)``.
        """
        traced_io = self._trace_io_functions()
        return self._compute_kronecker_factors(traced_io)

    def _compute_kronecker_factors(
        self,
        traced_io: tuple[
            dict[int, Callable],
            dict[str, dict[str, str]],
            dict[str, dict[str, Any]],
        ],
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Compute KFAC's Kronecker factors from pre-traced IO functions.

        The full per-batch computation (IO collection, backward pass, and
        covariance einsums) is traced with ``make_fx`` and cached by batch
        size. The outer loop accumulates results across batches.

        Args:
            traced_io: Pre-traced IO functions from :meth:`_trace_io_functions`.

        Returns:
            Tuple of (input_covariances, gradient_covariances, mapping).
        """
        traced_io_fns, io_param_names, layer_hparams = traced_io

        grad_normalization = {
            "sum": 1.0,
            "mean": 1.0 / self._N_data,
        }[self._loss_func.reduction]

        mapping, io_groups = _build_param_groups_from_io(
            io_param_names, self._separate_weight_and_bias
        )

        # Build group keys in stable order (matching the traced function's output)
        weight_group_keys = [
            tuple(g.values()) for g in mapping if "W" in g
        ]
        all_group_keys = [tuple(g.values()) for g in mapping]

        # Trace the full per-batch computation (forward + backward + covariances)
        # for each unique batch size, cached by batch size.
        traced_fns: dict[int, Callable] = {}

        input_covariances: dict[ParamGroupKey, Tensor] = {}
        gradient_covariances: dict[ParamGroupKey, Tensor] = {}

        manual_seed(self._seed)

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            batch_size = self._batch_size_fn(X)
            if batch_size not in traced_fns:
                traced_fns[batch_size] = (
                    self._trace_forward_backward_and_covariances(
                        traced_io_fns[batch_size],
                        io_param_names,
                        layer_hparams,
                        mapping,
                        io_groups,
                        X,
                        y,
                    )
                )

            input_covs, gradient_covs = traced_fns[batch_size](self._params, X, y)

            for key, xxT in zip(weight_group_keys, input_covs):
                self._set_or_add_(
                    input_covariances, key, xxT.div_(self._N_data)
                )

            for key, ggT in zip(all_group_keys, gradient_covs):
                self._set_or_add_(
                    gradient_covariances, key, ggT.mul_(grad_normalization)
                )

        if self._fisher_type == FisherType.FORWARD_ONLY:
            self._set_gradient_covariances_to_identity(gradient_covariances, mapping)

        return input_covariances, gradient_covariances, mapping
