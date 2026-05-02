"""Parameter-group construction and per-group IO standardization.

Helpers that turn :func:`with_kfac_io`'s detected layer metadata into
parameter groups and per-group ``(inputs, output_grads)`` gatherers in the
weight-sharing format consumed by KFAC-style operators.

These helpers are pure post-processing on ``with_kfac_io``'s outputs; they
contain no KFAC-specific reduction logic and no FX tracing.
"""

from collections import defaultdict
from collections.abc import Callable

from torch import Tensor, cat

from curvlinops.computers._base import ParamGroup, ParamGroupKey
from curvlinops.computers.kfac_math import (
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import KFACType


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
