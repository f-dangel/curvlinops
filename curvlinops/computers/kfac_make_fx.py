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
from curvlinops.computers.kfac import (
    KFACComputer,
    ParameterUsage,
    _module_name_from_param,
)
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import FisherType, _has_joint_weight_and_bias
from curvlinops.utils import _seed_generator


def _trace_io(
    f: Callable[[Tensor | MutableMapping, dict[str, Tensor]], Tensor],
    x: Tensor | MutableMapping,
    params: dict[str, Tensor],
    fisher_type: FisherType,
) -> tuple[Callable, dict[str, str], dict[str, dict[str, Any]]]:
    """Trace f and return an IO-collecting function with layer metadata.

    Wraps ``with_kfac_io`` and derives the IO-to-module name mapping from
    the detected parameter names (e.g. ``"0.weight"`` → ``"0"``).

    Args:
        f: Function with signature ``f(x, params) -> output``.
        x: Example input tensor or mapping for tracing.
        params: Dictionary mapping parameter names to tensors.
        fisher_type: Type of Fisher information computation.

    Returns:
        Tuple of ``(io_fn, io_to_module, layer_hparams)``.
    """
    if isinstance(x, UserDict):
        _register_userdict_as_pytree()

    io_fn, layer_param_names, layer_hparams = with_kfac_io(f, x, params, fisher_type)
    io_to_module = {
        io_name: _module_name_from_param(next(iter(pnames.values())))
        for io_name, pnames in layer_param_names.items()
    }
    return io_fn, io_to_module, layer_hparams


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
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Compute KFAC's Kronecker factors using FX graph tracing.

        The IO-collecting forward pass (from ``with_kfac_io``) is cached by
        batch size. Input covariances are computed from the collected layer
        inputs. The backward pass (MC sampling + gradient covariances) runs
        eagerly to avoid ``make_fx`` tracing issues with ``torch._C.Generator``.

        Returns:
            Tuple containing (input_covariances, gradient_covariances) dictionaries.
        """

        def f(x, params: dict[str, Tensor]) -> Tensor:
            return functional_call(self._model_func, params, (x,))

        # N_data normalization is applied eagerly here, outside the traced forward
        # pass, rather than inside the per-batch computation (as the hooks backend
        # does). This keeps the traced function purely per-batch, with the global
        # normalization applied after each batch completes.
        grad_normalization = {
            "sum": 1.0,
            "mean": 1.0 / self._N_data,
        }[self._loss_func.reduction]

        # Cache for IO functions: make_fx bakes in tensor shapes (e.g. from nn.Flatten),
        # so different batch sizes need separate traces
        traced_io_fns: dict[int, Callable] = {}

        # Layer metadata (identical across batch sizes), populated on first trace
        io_to_module: dict[str, str] | None = None
        io_to_usage: dict[str, ParameterUsage] | None = None
        layer_hparams: dict[str, dict[str, Any]] | None = None

        # Set up dictionaries for the covariances that will be populated
        input_covariances: dict[str, Tensor] = {}
        gradient_covariances: dict[str, Tensor] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            # Maybe trace for current batch size and set up layer metadata
            if (batch_size := self._batch_size_fn(X)) not in traced_io_fns:
                traced_io_fns[batch_size], io_to_module, layer_hparams = _trace_io(
                    f, X, self._params, self._fisher_type
                )

            # Forward pass with IO collection
            io_fn = traced_io_fns[batch_size]
            output, layer_inputs, layer_outputs = io_fn(X, self._params)

            # Build lookup from IO collector names to ParameterUsage objects
            if io_to_usage is None:
                io_to_usage = {
                    io_name: self._usage_by_module[mod_name]
                    for io_name, mod_name in io_to_module.items()
                }

            # Group IO layer names by parameter usage (for weight tying,
            # multiple IO layers map to the same usage and get concatenated)
            io_groups: dict[str, list[str]] = defaultdict(list)
            for io_name in io_to_usage:
                io_groups[io_to_usage[io_name].name].append(io_name)

            # Compute input covariances (one parameter group at a time to
            # bound memory for CNNs with patch extraction)
            for usage_name, io_names in io_groups.items():
                names_with_input = [n for n in io_names if n in layer_inputs]
                if not names_with_input:
                    continue
                usage = io_to_usage[names_with_input[0]]
                has_joint_wb = _has_joint_weight_and_bias(
                    self._separate_weight_and_bias, usage.params
                )
                xs = [
                    input_to_weight_sharing_format(
                        layer_inputs[n].data.detach(),
                        self._kfac_approx,
                        layer_hyperparams=layer_hparams[n],
                        bias_pad=1 if has_joint_wb else None,
                    )
                    for n in names_with_input
                ]
                x = cat(xs, dim=1) if len(xs) > 1 else xs[0]
                scale = x.shape[1]
                xxT = einsum(x, x, "batch shared i, batch shared j -> i j")
                self._set_or_add_(
                    input_covariances, usage_name, xxT.div_(scale * self._N_data)
                )

            # Forward-only KFAC does not require gradient covariances
            if self._fisher_type == FisherType.FORWARD_ONLY:
                continue

            # Compute gradient covariances (one parameter group at a time)
            layer_output_grads = self._compute_layer_output_grads(
                output, y, layer_outputs, io_to_module
            )
            for usage_name, io_names in io_groups.items():
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
                ggT = einsum(
                    g, g, "v batch shared i, v batch shared j -> i j"
                ).mul_(correction)
                self._set_or_add_(
                    gradient_covariances, usage_name, ggT.mul_(grad_normalization)
                )

        # Handle FORWARD_ONLY case
        if self._fisher_type == FisherType.FORWARD_ONLY:
            self._set_gradient_covariances_to_identity(gradient_covariances)

        return input_covariances, gradient_covariances

    def _compute_layer_output_grads(
        self,
        output: Tensor,
        y: Tensor,
        layer_outputs: dict[str, Tensor],
        io_to_module: dict[str, str],
    ) -> dict[str, Tensor]:
        """Compute scaled batched gradients for all tracked layers.

        Rearranges the output for >2d, computes Fisher-type-specific
        ``grad_outputs``, scales by loss reduction, then backpropagates all
        gradient vectors in parallel via ``autograd.grad(is_grads_batched=True)``.

        Args:
            output: Model output tensor.
            y: Target tensor.
            layer_outputs: Collected layer outputs from the IO function.
            io_to_module: Mapping from IO collector layer names to module names.

        Returns:
            Dictionary mapping IO layer names to batched gradient tensors.
        """
        output, y = self._rearrange_for_larger_than_2d_output(output, y)

        grad_outputs = self._grad_outputs_computer(output.detach(), y, self._generator)
        num_loss_terms = output.shape[0]
        scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[self._loss_func.reduction]
        grad_outputs.mul_(scale)

        io_layer_names = [n for n in io_to_module if n in layer_outputs]
        output_tensors = [layer_outputs[n] for n in io_layer_names]
        layer_output_grads = autograd.grad(
            output,
            output_tensors,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
        )
        return dict(zip(io_layer_names, layer_output_grads))
