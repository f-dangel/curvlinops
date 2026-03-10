"""KFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxKFACComputer``, which computes Kronecker factors
by tracing the model's forward pass with ``torch.fx`` via the IO collector,
collecting layer inputs/outputs rather than using forward/backward hooks.
Only the forward pass is traced with ``make_fx``; the backward pass runs
eagerly to avoid tracing issues with ``torch._C.Generator``.
"""

from collections import UserDict
from collections.abc import Callable, MutableMapping
from typing import Any

from einops import einsum
from torch import Tensor, autograd, eye
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac import KFACComputer
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import FisherType, _has_joint_weight_and_bias
from curvlinops.utils import _seed_generator


class MakeFxKFACComputer(KFACComputer):
    """KFAC computer that uses FX graph tracing instead of hooks.

    Uses the IO collector (``with_kfac_io``) to detect affine layers and collect
    their inputs/outputs via ``torch.fx``, then computes Kronecker factors from
    these collected values. Only the forward pass (IO collection + input
    covariances) is traced with ``make_fx`` and cached; the backward pass
    (MC sampling + gradient covariances) runs eagerly.
    """

    def _input_covariance_from_io(
        self,
        io_layer_name: str,
        x: Tensor,
        module_name: str,
        layer_hyperparams: dict[str, dict[str, Any]],
    ) -> tuple[str, Tensor]:
        """Compute the input covariance for one layer from collected IO.

        Returns the unnormalized covariance (not divided by ``_N_data``).

        Args:
            io_layer_name: IO collector layer name.
            x: The collected layer input tensor.
            module_name: Module name in ``self._mapping``.
            layer_hyperparams: Hyperparameters from IO collector.

        Returns:
            Tuple of (module_name, covariance).
        """
        x = x.data.detach()

        params = self._mapping[module_name]
        has_joint_wb = _has_joint_weight_and_bias(
            self._separate_weight_and_bias, params
        )

        x = input_to_weight_sharing_format(
            x,
            self._kfac_approx,
            layer_hyperparams=layer_hyperparams[io_layer_name],
            append_ones_for_bias=has_joint_wb,
        )
        scale = x.shape[1]
        covariance = einsum(x, x, "batch shared i, batch shared j -> i j").div_(scale)
        return module_name, covariance

    def _gradient_covariance_from_io(
        self,
        io_layer_name: str,
        g: Tensor,
        module_name: str,
        layer_hyperparams: dict[str, dict[str, Any]],
    ) -> tuple[str, Tensor]:
        """Compute the gradient covariance for one layer from batched VJPs.

        Expects ``g`` to have shape ``[num_vectors, batch, ...]`` from a batched
        backward pass (``is_grads_batched=True``). Sums the outer products over
        both the vector and batch dimensions.

        Returns the unnormalized covariance (not divided by ``_N_data``).
        The batch-local correction for ``"mean"`` reduction uses
        ``num_loss_terms² / num_per_example_loss_terms`` (without ``_N_data``).

        Args:
            io_layer_name: IO collector layer name.
            g: Batched gradients at the layer's output with shape
                ``[num_vectors, batch, ...]``.
            module_name: Module name in ``self._mapping``.
            layer_hyperparams: Hyperparameters from IO collector.

        Returns:
            Tuple of (module_name, gradient_covariance).
        """
        g = g.data.detach()
        batch_size = g.shape[1]

        g = grad_to_weight_sharing_format(
            g,
            self._kfac_approx,
            layer_hyperparams=layer_hyperparams[io_layer_name],
            num_leading_dims=2,
        )

        correction = compute_loss_correction(
            batch_size, self._num_per_example_loss_terms, self._loss_func.reduction
        )

        ggT = einsum(g, g, "v batch shared i, v batch shared j -> i j").mul_(correction)
        return module_name, ggT

    def _setup_model(
        self,
    ) -> tuple[
        Callable,
        dict[str, Tensor],
        dict[str, str],
        dict[str, dict[str, Any]],
    ]:
        """Build the functional wrapper and extract static layer info.

        Traces the model once with the IO collector to detect affine layers and
        extract their parameter names and hyperparameters. Results are cached
        so repeated calls (e.g. KFAC factors then EKFAC correction) reuse the
        same trace.

        Returns:
            Tuple of (f, named_params, io_to_module, layer_hyperparams).
        """
        if hasattr(self, "_setup_cache"):
            return self._setup_cache

        # Build named_params dict by matching self._params against model parameters
        param_ids = {p.data_ptr() for p in self._params}
        named_params: dict[str, Tensor] = {}
        for name, param in self._model_func.named_parameters():
            if param.data_ptr() in param_ids:
                named_params[name] = param

        # Create functional wrapper
        model = self._model_func

        # Get example input from the first data batch
        x_example = next(iter(self._data))[0]
        if isinstance(x_example, MutableMapping):
            if isinstance(x_example, UserDict):
                _register_userdict_as_pytree()

            def f(x: MutableMapping, params: dict[str, Tensor]) -> Tensor:
                return functional_call(model, params, (x,))

        else:
            # Move to model device (data may be on CPU while model is on GPU)
            x_example = x_example.to(self.device)

            def f(x: Tensor, params: dict[str, Tensor]) -> Tensor:
                return functional_call(model, params, (x,))

        # Trace once with example data to extract static layer info
        f_with_kfac_io = with_kfac_io(f, x_example, named_params, self._fisher_type)
        _, _, _, layer_param_names, layer_hyperparams = f_with_kfac_io(
            x_example, named_params
        )

        # Build mapping from IO collector layer names to module names
        io_to_module: dict[str, str] = {}
        for io_layer_name, pnames in layer_param_names.items():
            any_param_name = next(iter(pnames.values()))
            # "0.weight" -> "0", "layer.sub.weight" -> "layer.sub",
            # "weight" -> "" (bare module, no prefix)
            if "." in any_param_name:
                io_to_module[io_layer_name] = any_param_name.rsplit(".", 1)[0]
            else:
                io_to_module[io_layer_name] = ""

        self._setup_cache = (
            f,
            named_params,
            io_to_module,
            layer_hyperparams,
        )
        return self._setup_cache

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
            Tuple of (io_names_with_outputs, batched_grads).
        """
        output, y = self._rearrange_for_larger_than_2d_output(output, y)

        io_names = [n for n in io_to_module if n in layer_outputs]
        output_tensors = [layer_outputs[n] for n in io_names]

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
        return io_names, batched_grads

    def _make_forward_fn(
        self,
        f_with_kfac_io: Callable,
        io_to_module: dict[str, str],
        layer_hparams: dict[str, dict[str, Any]],
    ) -> Callable:
        """Build a function for the traced forward pass.

        The returned function runs the forward pass (IO collection) and computes
        input covariances. It is traced with ``make_fx`` and cached by batch
        size. It also returns the model output and layer outputs so that the
        caller can run the backward pass eagerly.

        Args:
            f_with_kfac_io: IO-collecting function for a specific batch size.
            io_to_module: Mapping from IO collector layer names to module names.
            layer_hparams: Hyperparameters from IO collector.

        Returns:
            A function ``(params, X) -> (input_covs, output, layer_outputs)``.
        """

        def _forward(
            params: dict[str, Tensor], X: Tensor
        ) -> tuple[dict[str, Tensor], Tensor, dict[str, Tensor]]:
            output, layer_inputs, layer_outputs, _, _ = f_with_kfac_io(X, params)

            input_covs: dict[str, Tensor] = {}
            for io_name, x in layer_inputs.items():
                mod_name, cov = self._input_covariance_from_io(
                    io_name,
                    x,
                    io_to_module[io_name],
                    layer_hparams,
                )
                self._set_or_add_(input_covs, mod_name, cov)

            return input_covs, output, layer_outputs

        return _forward

    def _compute_kronecker_factors(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Compute KFAC's Kronecker factors using FX graph tracing.

        The forward pass (IO collection + input covariances) is traced with
        ``make_fx`` and cached by batch size. The backward pass (MC sampling +
        gradient covariances) runs eagerly to avoid ``make_fx`` tracing issues
        with ``torch._C.Generator``.

        Returns:
            Tuple containing (input_covariances, gradient_covariances) dictionaries.
        """
        f, named_params, io_to_module, layer_hparams = self._setup_model()

        if not hasattr(self, "_traced_forward_fns"):
            self._traced_forward_fns: dict[int, Callable] = {}

        input_covariances: dict[str, Tensor] = {}
        gradient_covariances: dict[str, Tensor] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        # N_data normalization is applied eagerly here, outside the traced forward
        # pass, rather than inside the per-batch computation (as the hooks backend
        # does). This keeps the traced function purely per-batch, with the global
        # normalization applied after each batch completes.
        grad_normalization = {
            "sum": 1.0,
            "mean": 1.0 / self._N_data,
        }[self._loss_func.reduction]

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            batch_size = self._batch_size_fn(X)
            if batch_size not in self._traced_forward_fns:
                f_io = with_kfac_io(f, X, named_params, self._fisher_type)
                forward_fn = self._make_forward_fn(f_io, io_to_module, layer_hparams)
                self._traced_forward_fns[batch_size] = make_fx(forward_fn)(
                    named_params, X
                )

            # Phase 1: Traced forward + input covariances
            traced_forward = self._traced_forward_fns[batch_size]
            input_covs, output, layer_outputs = traced_forward(named_params, X)

            for key, val in input_covs.items():
                self._set_or_add_(input_covariances, key, val.div_(self._N_data))

            # Phase 2: Eager backward + gradient covariances
            if self._fisher_type != FisherType.FORWARD_ONLY:
                io_names, batched_grads = self._compute_batched_grads(
                    output, y, layer_outputs, io_to_module
                )
                for io_name, batched_g in zip(io_names, batched_grads):
                    mod_name, ggT = self._gradient_covariance_from_io(
                        io_name,
                        batched_g,
                        io_to_module[io_name],
                        layer_hparams,
                    )
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
