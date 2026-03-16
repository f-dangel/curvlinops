"""KFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxKFACComputer``, which computes Kronecker factors
by tracing the model's forward pass with ``torch.fx`` via the IO collector,
collecting layer inputs/outputs rather than using forward/backward hooks.
Only the forward pass is traced with ``make_fx``; the backward pass runs
eagerly to avoid tracing issues with ``torch._C.Generator``.
"""

from collections import UserDict, defaultdict
from collections.abc import Callable, MutableMapping
from typing import Any

from einops import einsum
from torch import Tensor, autograd, cat
from torch.func import functional_call

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac import KFACComputer, ParamGroup
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import FisherType
from curvlinops.utils import _seed_generator


def _trace_io(
    f: Callable[[Tensor | MutableMapping, dict[str, Tensor]], Tensor],
    x: Tensor | MutableMapping,
    params: dict[str, Tensor],
    fisher_type: FisherType,
) -> tuple[Callable, dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    """Trace f and return an IO-collecting function with layer metadata.

    Args:
        f: Function with signature ``f(x, params) -> output``.
        x: Example input tensor or mapping for tracing.
        params: Dictionary mapping parameter names to tensors.
        fisher_type: Type of Fisher information computation.

    Returns:
        Tuple of ``(io_fn, io_param_names, layer_hparams)``.
    """
    if isinstance(x, UserDict):
        _register_userdict_as_pytree()

    return with_kfac_io(f, x, params, fisher_type)


def _build_param_groups_from_io(  # noqa: C901
    io_param_names: dict[str, dict[str, str]],
    separate_weight_and_bias: bool,
) -> tuple[list[ParamGroup], dict[tuple[str, ...], list[str]]]:
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
    # Collect one (W, optional b) per unique weight, plus standalone biases.
    # Also track which IO layers contribute to each weight.
    virtual_modules: dict[str, ParamGroup] = {}
    weight_io_layers: dict[str, dict[str, None]] = defaultdict(dict)
    standalone_biases: list[tuple[str, str]] = []  # (bias_name, io_name)

    for io_name, pnames in io_param_names.items():
        w, b = pnames.get("W"), pnames.get("b")
        if w is not None:
            if w not in virtual_modules:
                virtual_modules[w] = {"W": w}
            weight_io_layers[w][io_name] = None
            if b is not None:
                existing_b = virtual_modules[w].get("b")
                if (
                    not separate_weight_and_bias
                    and existing_b is not None
                    and existing_b != b
                ):
                    raise ValueError(
                        f"Weight '{w}' is used with conflicting biases "
                        f"'{existing_b}' and '{b}' under joint treatment. "
                        f"Use separate_weight_and_bias=True."
                    )
                virtual_modules[w]["b"] = b
        elif b is not None:
            standalone_biases.append((b, io_name))

    # Build parameter groups and IO-layer mapping
    groups: list[ParamGroup] = []
    io_groups: dict[tuple[str, ...], list[str]] = {}

    for w, module_params in virtual_modules.items():
        io_names = list(weight_io_layers[w])
        if separate_weight_and_bias:
            groups.append({"W": w})
            io_groups[(w,)] = io_names
            b = module_params.get("b")
            if b is not None:
                groups.append({"b": b})
                # Bias IO layers: only those that actually have the bias
                io_groups[(b,)] = [n for n in io_names if io_param_names[n].get("b")]
        else:
            groups.append(module_params)
            io_groups[tuple(module_params.values())] = io_names

    for b, io_name in standalone_biases:
        groups.append({"b": b})
        io_groups[(b,)] = [io_name]

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


class MakeFxKFACComputer(KFACComputer):
    """KFAC computer that uses FX graph tracing instead of hooks.

    Uses the IO collector (``with_kfac_io``) to detect affine layers and collect
    their inputs/outputs via ``torch.fx``, then computes Kronecker factors from
    these collected values. The IO-collecting forward pass is traced with
    ``make_fx`` (inside ``with_kfac_io``) and cached by batch size; the backward
    pass (MC sampling + gradient covariances) runs eagerly.
    """

    def _compute_kronecker_factors(  # noqa: C901
        self,
    ) -> tuple[dict[tuple[str, ...], Tensor], dict[tuple[str, ...], Tensor]]:
        """Compute KFAC's Kronecker factors using FX graph tracing.

        The IO-collecting forward pass (from ``with_kfac_io``) is cached by
        batch size. Input covariances are computed from the collected layer
        inputs. The backward pass (MC sampling + gradient covariances) runs
        eagerly to avoid ``make_fx`` tracing issues with ``torch._C.Generator``.

        Returns:
            Tuple containing (input_covariances, gradient_covariances) dictionaries
            keyed by parameter group tuples.
        """

        def f(x, params: dict[str, Tensor]) -> Tensor:
            return functional_call(self._model_func, params, (x,))

        grad_normalization = {
            "sum": 1.0,
            "mean": 1.0 / self._N_data,
        }[self._loss_func.reduction]

        # Cache for IO functions: make_fx bakes in tensor shapes (e.g. from
        # nn.Flatten), so different batch sizes need separate traces
        traced_io_fns: dict[int, Callable] = {}

        # Layer metadata (identical across batch sizes), populated on first trace
        io_param_names: dict[str, dict[str, str]] | None = None
        layer_hparams: dict[str, dict[str, Any]] | None = None
        # IO groups: maps parameter group key → list of IO layer names.
        # Built once from io_param_names and self._mapping.
        io_groups: dict[tuple[str, ...], list[str]] | None = None

        input_covariances: dict[tuple[str, ...], Tensor] = {}
        gradient_covariances: dict[tuple[str, ...], Tensor] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            # Maybe trace for current batch size and set up layer metadata
            if (batch_size := self._batch_size_fn(X)) not in traced_io_fns:
                traced_io_fns[batch_size], io_param_names, layer_hparams = _trace_io(
                    f, X, self._params, self._fisher_type
                )

            # Forward pass with IO collection
            io_fn = traced_io_fns[batch_size]
            output, layer_inputs, layer_outputs = io_fn(X, self._params)

            if io_groups is None:
                # Build parameter groups from IO detections (handles weight
                # tying correctly by grouping by weight name)
                self._mapping, io_groups = _build_param_groups_from_io(
                    io_param_names, self._separate_weight_and_bias
                )

            # Compute input/gradient covariances one parameter group at a time
            # (bounds memory for CNNs with patch extraction)
            for group in self._mapping:
                group_key = tuple(group.values())
                io_names = io_groups.get(group_key, [])

                # Input covariance (only for groups with a weight)
                if "W" not in group:
                    names_with_input = []
                else:
                    names_with_input = [n for n in io_names if n in layer_inputs]
                if names_with_input:
                    has_joint_wb = "b" in group and "W" in group
                    xs = [
                        input_to_weight_sharing_format(
                            layer_inputs[n].data.detach(),
                            self._kfac_approx,
                            layer_hyperparams=layer_hparams[n],
                            bias_pad=_bias_pad(has_joint_wb, io_param_names[n]),
                        )
                        for n in names_with_input
                    ]
                    x = cat(xs, dim=1) if len(xs) > 1 else xs[0]
                    scale = x.shape[1]
                    xxT = einsum(x, x, "batch shared i, batch shared j -> i j")
                    self._set_or_add_(
                        input_covariances, group_key, xxT.div_(scale * self._N_data)
                    )

            if self._fisher_type == FisherType.FORWARD_ONLY:
                continue

            # Backward pass: compute per-layer output gradients
            layer_output_grads = self._compute_layer_output_grads(
                output, y, layer_outputs
            )
            for group in self._mapping:
                group_key = tuple(group.values())
                io_names = io_groups.get(group_key, [])
                names_with_grad = [n for n in io_names if n in layer_output_grads]
                if not names_with_grad:
                    continue
                gs = [
                    grad_to_weight_sharing_format(
                        layer_output_grads[n].data.detach(),
                        self._kfac_approx,
                        layer_hyperparams=layer_hparams[n],
                        num_leading_dims=2,
                    )
                    for n in names_with_grad
                ]
                g = cat(gs, dim=2) if len(gs) > 1 else gs[0]
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

        if self._fisher_type == FisherType.FORWARD_ONLY:
            self._set_gradient_covariances_to_identity(gradient_covariances)

        return input_covariances, gradient_covariances

    def _compute_layer_output_grads(
        self,
        output: Tensor,
        y: Tensor,
        layer_outputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute scaled batched gradients for all tracked layers.

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
