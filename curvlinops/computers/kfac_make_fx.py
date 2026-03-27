"""KFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxKFACComputer``, which computes Kronecker factors
by tracing the model's forward pass with ``torch.fx`` via the IO collector,
collecting layer inputs/outputs rather than using forward/backward hooks.
Only the forward pass is traced with ``make_fx``; the backward pass runs
eagerly to avoid tracing issues with ``torch._C.Generator``.
"""

from collections import UserDict, defaultdict
from collections.abc import Callable
from typing import Any

from einops import einsum
from torch import Tensor, autograd, cat

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _BaseKFACComputer
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import FisherType
from curvlinops.utils import _seed_generator


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
    these collected values. The IO-collecting forward pass is traced with
    ``make_fx`` (inside ``with_kfac_io``) and cached by batch size; the backward
    pass (MC sampling + gradient covariances) runs eagerly.

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

    def _compute_kronecker_factors(
        self,
        traced_io: tuple[
            dict[int, Callable],
            dict[str, dict[str, str]],
            dict[str, dict[str, Any]],
        ]
        | None = None,
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Compute KFAC's Kronecker factors using FX graph tracing.

        The IO-collecting forward pass (from ``with_kfac_io``) is cached by
        batch size. Input covariances are computed from the collected layer
        inputs. The backward pass (MC sampling + gradient covariances) runs
        eagerly to avoid ``make_fx`` tracing issues with ``torch._C.Generator``.

        Args:
            traced_io: Pre-traced IO functions from :meth:`_trace_io_functions`.
                If ``None``, tracing is performed automatically.

        Returns:
            Tuple of (input_covariances, gradient_covariances, mapping).
        """
        if traced_io is None:
            traced_io = self._trace_io_functions()
        traced_io_fns, io_param_names, layer_hparams = traced_io

        # N_data normalization is applied eagerly here, outside the traced forward
        # pass, rather than inside the per-batch computation (as the hooks backend
        # does). This keeps the traced function purely per-batch, with the global
        # normalization applied after each batch completes.
        grad_normalization = {
            "sum": 1.0,
            "mean": 1.0 / self._N_data,
        }[self._loss_func.reduction]

        # Build parameter groups from IO detections (handles weight
        # tying correctly by grouping by weight name)
        mapping, io_groups = _build_param_groups_from_io(
            io_param_names, self._separate_weight_and_bias
        )

        input_covariances: dict[ParamGroupKey, Tensor] = {}
        gradient_covariances: dict[ParamGroupKey, Tensor] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            batch_size = self._batch_size_fn(X)

            # Forward pass with IO collection
            io_fn = traced_io_fns[batch_size]
            output, layer_inputs, layer_outputs = io_fn(self._params, X)

            # Compute input/gradient covariances one parameter group at a time
            # (bounds memory for CNNs with patch extraction)
            for group in mapping:
                group_key = tuple(group.values())
                io_names = io_groups.get(group_key, [])

                # Input covariance (only for groups with a weight)
                if "W" in group:
                    has_joint_wb = "b" in group
                    xs = [
                        input_to_weight_sharing_format(
                            layer_inputs[n].data.detach(),
                            self._kfac_approx,
                            layer_hyperparams=layer_hparams[n],
                            bias_pad=_bias_pad(has_joint_wb, io_param_names[n]),
                        )
                        for n in io_names
                    ]
                    x = cat(xs, dim=1)
                    scale = x.shape[1]
                    xxT = einsum(x, x, "batch shared i, batch shared j -> i j")
                    self._set_or_add_(
                        input_covariances, group_key, xxT.div_(scale * self._N_data)
                    )

            # Forward-only KFAC does not require gradient covariances
            if self._fisher_type == FisherType.FORWARD_ONLY:
                continue

            # Backward pass: compute per-layer output gradients
            layer_output_grads = self._compute_layer_output_grads(
                output, y, layer_outputs
            )
            for group in mapping:
                group_key = tuple(group.values())
                io_names = io_groups.get(group_key, [])
                gs = [
                    grad_to_weight_sharing_format(
                        layer_output_grads[n].data.detach(),
                        self._kfac_approx,
                        layer_hyperparams=layer_hparams[n],
                        num_leading_dims=2,
                    )
                    for n in io_names
                ]
                g = cat(gs, dim=2)
                correction = compute_loss_correction(
                    g.shape[1],
                    self._num_per_example_loss_terms,
                    self._loss_func.reduction,
                )
                ggT = einsum(g, g, "v batch shared i, v batch shared j -> i j").mul_(
                    correction
                )
                self._set_or_add_(
                    gradient_covariances, group_key, ggT.mul_(grad_normalization)
                )

        # Handle FORWARD_ONLY case
        if self._fisher_type == FisherType.FORWARD_ONLY:
            self._set_gradient_covariances_to_identity(gradient_covariances, mapping)

        return input_covariances, gradient_covariances, mapping

    def _compute_layer_output_grads(
        self,
        output: Tensor,
        y: Tensor,
        layer_outputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute scaled batched gradients for all tracked layers.

        Rearranges the output for >2d, computes Fisher-type-specific
        ``grad_outputs``, scales by loss reduction, then backpropagates all
        gradient vectors in parallel via ``autograd.grad(is_grads_batched=True)``.

        Args:
            output: Model output tensor.
            y: Target tensor.
            layer_outputs: Collected layer outputs from the IO function.

        Returns:
            Dictionary mapping IO layer names to batched gradient tensors.
        """
        output, y = self._rearrange_for_larger_than_2d_output(output, y)

        grad_outputs = self._grad_outputs_computer(output.detach(), y, self._generator)
        num_loss_terms = output.shape[0]
        scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[self._loss_func.reduction]
        grad_outputs.mul_(scale)

        io_layer_names = list(layer_outputs)
        output_tensors = list(layer_outputs.values())
        layer_output_grads = autograd.grad(
            output,
            output_tensors,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
        )
        return dict(zip(io_layer_names, layer_output_grads))
