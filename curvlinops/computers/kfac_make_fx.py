"""KFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxKFACComputer``, which computes Kronecker factors
by tracing the model's forward pass with ``torch.fx`` via the IO collector,
collecting layer inputs/outputs rather than using forward/backward hooks.
Only the forward pass is traced with ``make_fx``; the backward pass runs
eagerly to avoid tracing issues with ``torch._C.Generator``.
"""

from collections.abc import Callable
from typing import Any

from einops import einsum
from torch import Tensor, autograd, eye
from torch.func import functional_call

from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac import KFACComputer
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import FisherType, _has_joint_weight_and_bias
from curvlinops.utils import _seed_generator, identify_free_parameters


class MakeFxKFACComputer(KFACComputer):
    """KFAC computer that uses FX graph tracing instead of hooks.

    Uses the IO collector (``with_kfac_io``) to detect affine layers and collect
    their inputs/outputs via ``torch.fx``, then computes Kronecker factors from
    these collected values. The IO-collecting forward pass is traced with
    ``make_fx`` (inside ``with_kfac_io``) and cached by batch size; the backward
    pass (MC sampling + gradient covariances) runs eagerly.
    """

    def _compute_kronecker_factors(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Compute KFAC's Kronecker factors using FX graph tracing.

        The IO-collecting forward pass (from ``with_kfac_io``) is cached by
        batch size. Input covariances are computed from the collected layer
        inputs. The backward pass (MC sampling + gradient covariances) runs
        eagerly to avoid ``make_fx`` tracing issues with ``torch._C.Generator``.

        Returns:
            Tuple containing (input_covariances, gradient_covariances) dictionaries.
        """
        # Build functional model: identify free params by name, wrap in f(x, params)
        named_params = identify_free_parameters(self._model_func, self._params)

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

        traced_io_fns: dict[int, Callable] = {}
        io_to_module: dict[str, str] | None = None
        layer_hparams: dict[str, dict[str, Any]] | None = None

        # Set up dictionaries for the covariances that will be populated
        input_covariances: dict[str, Tensor] = {}
        gradient_covariances: dict[str, Tensor] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="KFAC matrices"):

            # TODO
            batch_size = self._batch_size_fn(X)
            if batch_size not in traced_io_fns:
                f_io, layer_param_names, layer_hparams = with_kfac_io(
                    f, X, named_params, self._fisher_type
                )
                # Extract layer info on first trace
                if io_to_module is None:
                    io_to_module = self._build_io_to_module(layer_param_names)
                traced_io_fns[batch_size] = f_io

            # Phase 1: Forward pass (IO collection) + input covariances
            output, layer_inputs, layer_outputs = traced_io_fns[batch_size](
                X, named_params
            )

            for io_layer_name, x in layer_inputs.items():
                mod_name = io_to_module[io_layer_name]
                has_joint_wb = _has_joint_weight_and_bias(
                    self._separate_weight_and_bias, self._mapping[mod_name]
                )
                x = input_to_weight_sharing_format(
                    x.data.detach(),
                    self._kfac_approx,
                    layer_hyperparams=layer_hparams[io_layer_name],
                    append_ones_for_bias=has_joint_wb,
                )
                scale = x.shape[1]
                xxT = einsum(x, x, "batch shared i, batch shared j -> i j")
                self._set_or_add_(
                    input_covariances, mod_name, xxT.div_(scale * self._N_data)
                )

            # Phase 2: Eager backward + gradient covariances
            if self._fisher_type != FisherType.FORWARD_ONLY:
                io_layer_names, batched_grads = self._compute_batched_grads(
                    output, y, layer_outputs, io_to_module
                )
                for io_layer_name, g in zip(io_layer_names, batched_grads):
                    mod_name = io_to_module[io_layer_name]
                    g = grad_to_weight_sharing_format(
                        g.data.detach(),
                        self._kfac_approx,
                        layer_hyperparams=layer_hparams[io_layer_name],
                        num_leading_dims=2,
                    )
                    correction = compute_loss_correction(
                        g.shape[1],
                        self._num_per_example_loss_terms,
                        self._loss_func.reduction,
                    )
                    ggT = einsum(
                        g, g, "v batch shared i, v batch shared j -> i j"
                    ).mul_(correction)
                    self._set_or_add_(
                        gradient_covariances, mod_name, ggT.mul_(grad_normalization)
                    )

        # Handle FORWARD_ONLY case
        if self._fisher_type == FisherType.FORWARD_ONLY:
            for mod_name, param_pos in self._mapping.items():
                param = self._params[next(iter(param_pos.values()))]
                gradient_covariances[mod_name] = eye(
                    param.shape[0], dtype=param.dtype, device=self.device
                )

        return input_covariances, gradient_covariances

    @staticmethod
    def _build_io_to_module(
        layer_param_names: dict[str, dict[str, str]],
    ) -> dict[str, str]:
        """Build mapping from IO collector layer names to module names.

        Derives the module name from any parameter name in the layer:
        ``"0.weight"`` → ``"0"``, ``"layer.sub.weight"`` → ``"layer.sub"``,
        ``"weight"`` → ``""`` (bare module, no prefix).

        Args:
            layer_param_names: Maps IO layer names to parameter name dicts.

        Returns:
            Mapping from IO collector layer names to module names.
        """
        io_to_module: dict[str, str] = {}
        for io_layer_name, pnames in layer_param_names.items():
            any_param_name = next(iter(pnames.values()))
            if "." in any_param_name:
                io_to_module[io_layer_name] = any_param_name.rsplit(".", 1)[0]
            else:
                io_to_module[io_layer_name] = ""
        return io_to_module

    def _compute_batched_grads(
        self,
        output: Tensor,
        y: Tensor,
        layer_outputs: dict[str, Tensor],
        io_to_module: dict[str, str],
    ) -> tuple[list[str], tuple[Tensor, ...]]:
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
            Tuple of (io_layer_names_with_outputs, batched_grads).
        """
        output, y = self._rearrange_for_larger_than_2d_output(output, y)

        io_layer_names = [n for n in io_to_module if n in layer_outputs]
        output_tensors = [layer_outputs[n] for n in io_layer_names]

        grad_outputs = self._grad_outputs_computer(output.detach(), y, self._generator)
        num_loss_terms = output.shape[0]
        scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[self._loss_func.reduction]
        grad_outputs.mul_(scale)

        batched_grads = autograd.grad(
            output,
            output_tensors,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
        )
        return io_layer_names, batched_grads
