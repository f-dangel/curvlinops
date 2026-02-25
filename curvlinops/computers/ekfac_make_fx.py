"""EKFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxEKFACComputer``, which extends ``EKFACComputer``
with FX-based eigenvalue correction, using the IO collector (``with_kfac_io``)
instead of forward/backward hooks.
"""

from collections.abc import Callable
from typing import Any

from einops import rearrange
from torch import Tensor, autograd, cat
from torch.fx.experimental.proxy_tensor import make_fx

from curvlinops.computers.ekfac import (
    EKFACComputer,
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac_make_fx import MakeFxKFACComputer
from curvlinops.kfac_utils import extract_patches
from curvlinops.utils import _seed_generator

# Type alias for the batch correction function
ComputeCorrectionBatchFn = Callable[
    [dict[str, Tensor], Tensor, Tensor],
    dict[str, Tensor],
]


class MakeFxEKFACComputer(EKFACComputer, MakeFxKFACComputer):
    """EKFAC computer that uses FX graph tracing for eigenvalue correction.

    Extends ``EKFACComputer`` with FX-based eigenvalue correction computation.
    Kronecker factor computation is inherited from ``MakeFxKFACComputer``.
    The eigenvalue correction pass uses the IO collector (``with_kfac_io``)
    to collect layer inputs/outputs and ``autograd.grad`` with
    ``is_grads_batched=True`` for efficient batched backward passes.
    """

    def compute_eigenvalue_correction(
        self,
        input_covariances_eigenvectors: dict[str, Tensor],
        gradient_covariances_eigenvectors: dict[str, Tensor],
    ) -> dict[str, Tensor | dict[int, Tensor]]:
        """Compute eigenvalue corrections using FX graph tracing.

        Maintains a cache of traced correction functions keyed by batch size.
        On the first encounter of a new batch size, the correction function is
        traced with ``make_fx``. Subsequent batches of the same size reuse the
        cached traced function.

        Args:
            input_covariances_eigenvectors: Dictionary mapping module names to
                input covariance eigenvectors.
            gradient_covariances_eigenvectors: Dictionary mapping module names to
                gradient covariance eigenvectors.

        Returns:
            Dictionary containing corrected eigenvalues for each module.
        """
        f, named_params, io_to_module, layer_param_names, layer_hyperparams = (
            self._setup_model()
        )

        if not hasattr(self, "_traced_correction_fns"):
            self._traced_correction_fns: dict[int, ComputeCorrectionBatchFn] = {}

        corrected_eigenvalues: dict[str, Tensor | dict[int, Tensor]] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="Eigenvalue correction"):
            batch_size = X.shape[0]
            if batch_size not in self._traced_correction_fns:
                self._traced_correction_fns[batch_size] = (
                    self._trace_correction_batch_fn(
                        f,
                        named_params,
                        io_to_module,
                        layer_param_names,
                        layer_hyperparams,
                        input_covariances_eigenvectors,
                        gradient_covariances_eigenvectors,
                        X,
                        y,
                    )
                )

            batch_corrections = self._traced_correction_fns[batch_size](
                named_params, X, y
            )

            # Accumulate corrections into the output structure.
            # Keys with "::" are separate weight/bias: "mod_name::pos"
            # Keys without "::" are joint weight+bias: "mod_name"
            for key, val in batch_corrections.items():
                if "::" in key:
                    mod_name, pos_str = key.split("::")
                    pos = int(pos_str)
                    if mod_name not in corrected_eigenvalues:
                        corrected_eigenvalues[mod_name] = {}
                    self._set_or_add_(corrected_eigenvalues[mod_name], pos, val)
                else:
                    self._set_or_add_(corrected_eigenvalues, key, val)

        return corrected_eigenvalues

    def _trace_correction_batch_fn(
        self,
        f: Callable,
        named_params: dict[str, Tensor],
        io_to_module: dict[str, str],
        layer_param_names: dict[str, dict[str, str]],
        layer_hyperparams: dict[str, dict[str, Any]],
        input_cov_eigvecs: dict[str, Tensor],
        gradient_cov_eigvecs: dict[str, Tensor],
        X: Tensor,
        y: Tensor,
    ) -> ComputeCorrectionBatchFn:
        """Create and trace a correction batch function for a specific batch size.

        Creates a per-batch-size IO function via ``with_kfac_io``, builds the
        correction computation closure, then traces it with ``make_fx``.

        Args:
            f: The plain functional model wrapper.
            named_params: Named parameter tensors.
            io_to_module: Mapping from IO collector layer names to module names.
            layer_param_names: Parameter name mapping from IO collector.
            layer_hyperparams: Hyperparameters from IO collector.
            input_cov_eigvecs: Input covariance eigenvectors per module.
            gradient_cov_eigvecs: Gradient covariance eigenvectors per module.
            X: Example input tensor (determines batch size for tracing).
            y: Example target tensor.

        Returns:
            A traced function ``(params, X, y) -> corrections``.
        """
        f_with_kfac_io = with_kfac_io(f, X, named_params, self._fisher_type)
        compute_correction_batch = self._make_compute_correction_batch(
            f_with_kfac_io,
            io_to_module,
            layer_param_names,
            layer_hyperparams,
            input_cov_eigvecs,
            gradient_cov_eigvecs,
        )
        return make_fx(compute_correction_batch)(named_params, X, y)

    def _make_compute_correction_batch(
        self,
        f_with_kfac_io: Callable,
        io_to_module: dict[str, str],
        layer_param_names: dict[str, dict[str, str]],
        layer_hyperparams: dict[str, dict[str, Any]],
        input_cov_eigvecs: dict[str, Tensor],
        gradient_cov_eigvecs: dict[str, Tensor],
    ) -> ComputeCorrectionBatchFn:
        """Build a function that computes eigenvalue corrections for one batch.

        Creates a closure around the given IO function and eigenvectors.

        Args:
            f_with_kfac_io: IO-collecting function for a specific batch size.
            io_to_module: Mapping from IO collector layer names to module names.
            layer_param_names: Parameter name mapping from IO collector.
            layer_hyperparams: Hyperparameters from IO collector.
            input_cov_eigvecs: Input covariance eigenvectors per module.
            gradient_cov_eigvecs: Gradient covariance eigenvectors per module.

        Returns:
            A function ``(params, X, y) -> corrections``.
        """

        def _compute_correction_batch(
            params: dict[str, Tensor], X: Tensor, y: Tensor
        ) -> dict[str, Tensor]:
            """Compute eigenvalue corrections for a single batch.

            Args:
                params: Named parameter tensors for the model.
                X: Input tensor for the batch.
                y: Target tensor for the batch.

            Returns:
                Dictionary of eigenvalue corrections for this batch.
            """
            output, layer_inputs, layer_outputs, _, _ = f_with_kfac_io(X, params)

            output, y = self._rearrange_for_larger_than_2d_output(output, y)

            corrections: dict[str, Tensor] = {}

            io_names_with_outputs = [n for n in io_to_module if n in layer_outputs]
            output_tensors = [layer_outputs[n] for n in io_names_with_outputs]

            # Compute grad_outputs (Fisher-type-specific)
            grad_outputs = self._grad_outputs_computer(
                output.detach(), y, self._generator
            )
            num_loss_terms = output.shape[0]
            scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[
                self._loss_func.reduction
            ]
            grad_outputs.mul_(scale)

            # Backpropagate all gradient vectors in parallel (uses vmap)
            batched_grads = autograd.grad(
                output,
                output_tensors,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
            )

            for io_name, batched_g in zip(io_names_with_outputs, batched_grads):
                mod_name = io_to_module[io_name]
                x = layer_inputs.get(io_name)
                self._eigenvalue_correction_from_io(
                    io_name,
                    batched_g,
                    x,
                    mod_name,
                    layer_param_names,
                    layer_hyperparams,
                    params,
                    input_cov_eigvecs,
                    gradient_cov_eigvecs,
                    corrections,
                )

            return corrections

        return _compute_correction_batch

    def _eigenvalue_correction_from_io(
        self,
        io_layer_name: str,
        batched_g: Tensor,
        x: Tensor | None,
        module_name: str,
        layer_param_names: dict[str, dict[str, str]],
        layer_hyperparams: dict[str, dict[str, Any]],
        named_params: dict[str, Tensor],
        input_cov_eigvecs: dict[str, Tensor],
        gradient_cov_eigvecs: dict[str, Tensor],
        corrections: dict[str, Tensor],
    ) -> None:
        """Compute eigenvalue correction for one layer from IO data.

        Processes batched gradients (from ``is_grads_batched=True``) by
        flattening the ``[num_vectors, batch]`` dimensions into a single
        batch dimension and expanding the layer input to match. Then calls
        ``compute_eigenvalue_correction_linear_weight_sharing``.

        Args:
            io_layer_name: IO collector layer name.
            batched_g: Batched gradients at the layer's output with shape
                ``[num_vectors, batch, ...]``.
            x: Layer input tensor with shape ``[batch, ...]`` or ``None``.
            module_name: Module name in ``self._mapping``.
            layer_param_names: Parameter name mapping from IO collector.
            layer_hyperparams: Hyperparameters from IO collector.
            named_params: Named parameter tensors.
            input_cov_eigvecs: Input covariance eigenvectors per module.
            gradient_cov_eigvecs: Gradient covariance eigenvectors per module.
            corrections: Dictionary to store computed corrections (mutated).
        """
        g = batched_g.data.detach()  # [num_vectors, batch, ...]
        num_vectors = g.shape[0]
        batch_size = g.shape[1]

        hyperparams = layer_hyperparams[io_layer_name]
        if hyperparams:  # Conv2d
            g = rearrange(g, "v batch c o1 o2 -> v batch o1 o2 c")
        g = rearrange(g, "v batch ... d_out -> (v batch) (...) d_out")

        param_pos = self._mapping[module_name]
        a_required = "weight" in param_pos

        a = None
        if a_required and x is not None:
            a = x.data.detach()  # [batch, ...]
            if hyperparams:  # Conv2d
                weight_name = layer_param_names[io_layer_name]["weight"]
                kernel_size = named_params[weight_name].shape[2:]
                a = extract_patches(
                    a,
                    kernel_size,
                    hyperparams["stride"],
                    hyperparams["padding"],
                    hyperparams["dilation"],
                    hyperparams["groups"],
                )
            a = rearrange(a, "batch ... d_in -> batch (...) d_in")
            # Expand to match [num_vectors, batch, S, D2] then flatten
            a = a.unsqueeze(0).expand(num_vectors, -1, -1, -1)
            a = rearrange(a, "v batch s d -> (v batch) s d")

        # Loss scaling correction
        num_loss_terms = batch_size * self._num_per_example_loss_terms
        correction = {
            "sum": 1.0,
            "mean": num_loss_terms**2
            / (self._N_data * self._num_per_example_loss_terms),
        }[self._loss_func.reduction]

        aaT_eigvecs = input_cov_eigvecs.get(module_name)
        ggT_eigvecs = gradient_cov_eigvecs[module_name]

        if not self._separate_weight_and_bias and {"weight", "bias"} == set(
            param_pos.keys()
        ):
            a_augmented = cat([a, a.new_ones(*a.shape[:-1], 1)], dim=-1)
            eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                g, ggT_eigvecs, a_augmented, aaT_eigvecs
            )
            corrections[module_name] = eigencorrection.mul_(correction)
        else:
            for p_name, pos in param_pos.items():
                eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                    g,
                    ggT_eigvecs,
                    None if p_name == "bias" else a,
                    None if p_name == "bias" else aaT_eigvecs,
                )
                key = f"{module_name}::{pos}"
                corrections[key] = eigencorrection.mul_(correction)
