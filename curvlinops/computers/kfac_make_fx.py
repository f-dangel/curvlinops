"""KFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxKFACComputer``, which computes Kronecker factors
by tracing the model with ``torch.fx`` via the IO collector and collecting
layer inputs/outputs, rather than using forward/backward hooks.
"""

from collections.abc import Callable, MutableMapping
from typing import Any

from einops import einsum, rearrange, reduce
from torch import Tensor, autograd, cat, eye
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx

from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac import FisherType, KFACComputer, KFACType
from curvlinops.kfac_utils import extract_averaged_patches, extract_patches
from curvlinops.utils import _seed_generator

# Type alias for the batch computation function returned by _make_compute_kfac_batch
ComputeKfacBatchFn = Callable[
    [dict[str, Tensor], Tensor, Tensor],
    tuple[dict[str, Tensor], dict[str, Tensor]],
]


class MakeFxKFACComputer(KFACComputer):
    """KFAC computer that uses FX graph tracing instead of hooks.

    Uses the IO collector (``with_kfac_io``) to detect affine layers and collect
    their inputs/outputs via ``torch.fx``, then computes Kronecker factors from
    these collected values. This is a functional alternative to the hook-based
    ``KFACComputer``.
    """

    def _input_covariance_from_io(
        self,
        io_layer_name: str,
        x: Tensor,
        module_name: str,
        layer_param_names: dict[str, dict[str, str]],
        layer_hyperparams: dict[str, dict[str, Any]],
        named_params: dict[str, Tensor],
    ) -> tuple[str, Tensor]:
        """Compute the input covariance for one layer from collected IO.

        Returns the unnormalized covariance (not divided by ``_N_data``).

        Args:
            io_layer_name: IO collector layer name.
            x: The collected layer input tensor.
            module_name: Module name in ``self._mapping``.
            layer_param_names: Parameter name mapping from IO collector.
            layer_hyperparams: Hyperparameters from IO collector.
            named_params: Named parameter tensors.

        Returns:
            Tuple of (module_name, covariance).
        """
        x = x.data.detach()

        hyperparams = layer_hyperparams[io_layer_name]
        if hyperparams:
            weight_name = layer_param_names[io_layer_name]["weight"]
            kernel_size = named_params[weight_name].shape[2:]
            patch_extractor_fn = {
                KFACType.EXPAND: extract_patches,
                KFACType.REDUCE: extract_averaged_patches,
            }[self._kfac_approx]
            x = patch_extractor_fn(
                x,
                kernel_size,
                hyperparams["stride"],
                hyperparams["padding"],
                hyperparams["dilation"],
                hyperparams["groups"],
            )

        if self._kfac_approx == KFACType.EXPAND:
            scale = x.shape[1:-1].numel()
            x = rearrange(x, "batch ... d_in -> (batch ...) d_in")
        else:
            scale = 1.0
            x = reduce(x, "batch ... d_in -> batch d_in", "mean")

        params = self._mapping[module_name]
        if not self._separate_weight_and_bias and {"weight", "bias"} == set(
            params.keys()
        ):
            x = cat([x, x.new_ones(x.shape[0], 1)], dim=1)

        covariance = einsum(x, x, "b i,b j -> i j").div_(scale)
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

        if layer_hyperparams[io_layer_name]:
            g = rearrange(g, "v batch c o1 o2 -> v batch o1 o2 c")

        if self._kfac_approx == KFACType.EXPAND:
            g = rearrange(g, "v batch ... d_out -> v (batch ...) d_out")
        else:
            g = reduce(g, "v batch ... d_out -> v batch d_out", "sum")

        num_loss_terms = batch_size * self._num_per_example_loss_terms
        correction = {
            "sum": 1.0,
            "mean": num_loss_terms**2 / self._num_per_example_loss_terms,
        }[self._loss_func.reduction]

        ggT = einsum(g, g, "v b i, v b j -> i j").mul_(correction)
        return module_name, ggT

    def _setup_model(
        self,
    ) -> tuple[
        Callable,
        dict[str, Tensor],
        dict[str, str],
        dict[str, dict[str, str]],
        dict[str, dict[str, Any]],
    ]:
        """Build the functional wrapper and extract static layer info.

        Traces the model once with the IO collector to detect affine layers and
        extract their parameter names and hyperparameters. Results are cached
        so repeated calls (e.g. KFAC factors then EKFAC correction) reuse the
        same trace.

        Returns:
            Tuple of (f, named_params, io_to_module, layer_param_names,
            layer_hyperparams).

        Raises:
            ValueError: If the data uses ``MutableMapping`` inputs.
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

        def f(x: Tensor, params: dict[str, Tensor]) -> Tensor:
            return functional_call(model, params, (x,))

        # Get example input from the first data batch
        x_example = next(iter(self._data))[0]
        if isinstance(x_example, MutableMapping):
            raise ValueError(
                "The make_fx backend does not support MutableMapping inputs. "
                "Use the hooks backend instead."
            )

        # Trace once with example data to extract static layer info
        f_with_kfac_io = with_kfac_io(f, x_example, named_params, self._fisher_type)
        _, _, _, layer_param_names, layer_hyperparams = f_with_kfac_io(
            x_example, named_params
        )

        # Build mapping from IO collector layer names to module names
        io_to_module: dict[str, str] = {}
        for io_layer_name, pnames in layer_param_names.items():
            any_param_name = next(iter(pnames.values()))
            # "0.weight" -> "0", "layer.sub.weight" -> "layer.sub"
            io_to_module[io_layer_name] = any_param_name.rsplit(".", 1)[0]

        self._setup_cache = (
            f,
            named_params,
            io_to_module,
            layer_param_names,
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

    def _make_compute_kfac_batch(
        self,
        f_with_kfac_io: Callable,
        io_to_module: dict[str, str],
        layer_param_names: dict[str, dict[str, str]],
        layer_hparams: dict[str, dict[str, Any]],
    ) -> ComputeKfacBatchFn:
        """Build a function that computes KFAC factors for a single batch.

        Args:
            f_with_kfac_io: IO-collecting function for a specific batch size.
            io_to_module: Mapping from IO collector layer names to module names.
            layer_param_names: Parameter name mapping from IO collector.
            layer_hparams: Hyperparameters from IO collector.

        Returns:
            A function ``(params, X, y) -> (input_covs, grad_covs)``.
        """

        def _compute_kfac_batch(
            params: dict[str, Tensor], X: Tensor, y: Tensor
        ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
            output, layer_inputs, layer_outputs, _, _ = f_with_kfac_io(X, params)

            input_covs: dict[str, Tensor] = {}
            for io_name, x in layer_inputs.items():
                mod_name, cov = self._input_covariance_from_io(
                    io_name,
                    x,
                    io_to_module[io_name],
                    layer_param_names,
                    layer_hparams,
                    params,
                )
                self._set_or_add_(input_covs, mod_name, cov)

            grad_covs: dict[str, Tensor] = {}
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
                    self._set_or_add_(grad_covs, mod_name, ggT)

            return input_covs, grad_covs

        return _compute_kfac_batch

    def _compute_kronecker_factors(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Compute KFAC's Kronecker factors using FX graph tracing.

        Maintains a cache of traced batch functions keyed by batch size. On
        the first encounter of a new batch size, the batch function is traced
        with ``make_fx`` into an ATen-level graph. Subsequent batches of the
        same size reuse the cached traced function.

        Returns:
            Tuple containing (input_covariances, gradient_covariances) dictionaries.
        """
        f, named_params, io_to_module, layer_param_names, layer_hparams = (
            self._setup_model()
        )

        if not hasattr(self, "_traced_batch_fns"):
            self._traced_batch_fns: dict[int, ComputeKfacBatchFn] = {}

        input_covariances: dict[str, Tensor] = {}
        gradient_covariances: dict[str, Tensor] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        grad_normalization = {
            "sum": 1.0,
            "mean": 1.0 / self._N_data,
        }[self._loss_func.reduction]

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            batch_size = X.shape[0]
            if batch_size not in self._traced_batch_fns:
                f_io = with_kfac_io(f, X, named_params, self._fisher_type)
                batch_fn = self._make_compute_kfac_batch(
                    f_io, io_to_module, layer_param_names, layer_hparams
                )
                self._traced_batch_fns[batch_size] = make_fx(batch_fn)(
                    named_params, X, y
                )

            batch_input_covs, batch_grad_covs = self._traced_batch_fns[batch_size](
                named_params, X, y
            )

            # Input covariances estimate E[aaT], normalize by 1/N_data
            for key, val in batch_input_covs.items():
                self._set_or_add_(input_covariances, key, val.div_(self._N_data))

            # Gradient covariances: 1/N_data for "mean" loss, raw sum for "sum" loss
            for key, val in batch_grad_covs.items():
                self._set_or_add_(
                    gradient_covariances, key, val.mul_(grad_normalization)
                )

        # Handle FORWARD_ONLY case
        if self._fisher_type == FisherType.FORWARD_ONLY:
            for mod_name, param_pos in self._mapping.items():
                param = self._params[next(iter(param_pos.values()))]
                gradient_covariances[mod_name] = eye(
                    param.shape[0], dtype=param.dtype, device=self.device
                )

        return input_covariances, gradient_covariances
