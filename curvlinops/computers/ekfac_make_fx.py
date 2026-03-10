"""EKFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxEKFACComputer``, which extends ``EKFACComputer``
with FX-based eigenvalue correction, using the IO collector (``with_kfac_io``)
instead of forward/backward hooks. Only the forward pass is traced with
``make_fx``; the backward pass runs eagerly.
"""

from collections.abc import Callable
from typing import Any

from einops import rearrange
from torch import Tensor, cat
from torch.fx.experimental.proxy_tensor import make_fx

from curvlinops.computers.ekfac import (
    EKFACComputer,
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac_make_fx import MakeFxKFACComputer
from curvlinops.computers.kfac_math import compute_loss_correction, prepare_io_for_ekfac
from curvlinops.kfac_utils import _has_joint_weight_and_bias
from curvlinops.utils import _seed_generator


class MakeFxEKFACComputer(EKFACComputer, MakeFxKFACComputer):
    """EKFAC computer that uses FX graph tracing for eigenvalue correction.

    Extends ``EKFACComputer`` with FX-based eigenvalue correction computation.
    Kronecker factor computation is inherited from ``MakeFxKFACComputer``.
    Only the forward pass (IO collection) is traced with ``make_fx``; the
    backward pass and eigenvalue correction computation run eagerly.
    """

    def compute_eigenvalue_correction(
        self,
        input_covariances_eigenvectors: dict[str, Tensor],
        gradient_covariances_eigenvectors: dict[str, Tensor],
    ) -> dict[str, Tensor | dict[int, Tensor]]:
        """Compute eigenvalue corrections using FX graph tracing.

        The forward pass (IO collection) is traced with ``make_fx`` and cached
        by batch size. The backward pass (MC sampling + gradient computation)
        and eigenvalue correction run eagerly.

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

        if not hasattr(self, "_traced_forward_io_fns"):
            self._traced_forward_io_fns: dict[int, Callable] = {}

        corrected_eigenvalues: dict[str, Tensor | dict[int, Tensor]] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="Eigenvalue correction"):
            batch_size = X.shape[0]
            if batch_size not in self._traced_forward_io_fns:
                f_io = with_kfac_io(f, X, named_params, self._fisher_type)
                forward_io = self._make_forward_io_fn(f_io)
                self._traced_forward_io_fns[batch_size] = make_fx(forward_io)(
                    named_params, X
                )

            # Phase 1: Traced forward (IO collection)
            traced_forward = self._traced_forward_io_fns[batch_size]
            output, layer_inputs, layer_outputs = traced_forward(named_params, X)

            # Phase 2: Eager backward + eigenvalue correction
            io_names, batched_grads = self._compute_batched_grads(
                output, y, layer_outputs, io_to_module
            )

            for io_name, batched_g in zip(io_names, batched_grads):
                mod_name = io_to_module[io_name]
                x = layer_inputs.get(io_name)
                self._eigenvalue_correction_from_io(
                    io_name,
                    batched_g,
                    x,
                    mod_name,
                    layer_param_names,
                    layer_hyperparams,
                    named_params,
                    input_covariances_eigenvectors,
                    gradient_covariances_eigenvectors,
                    corrected_eigenvalues,
                )

        return corrected_eigenvalues

    @staticmethod
    def _make_forward_io_fn(
        f_with_kfac_io: Callable,
    ) -> Callable:
        """Build a function for the traced forward pass (IO collection only).

        Args:
            f_with_kfac_io: IO-collecting function for a specific batch size.

        Returns:
            A function ``(params, X) -> (output, layer_inputs, layer_outputs)``.
        """

        def _forward_io(
            params: dict[str, Tensor], X: Tensor
        ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
            output, layer_inputs, layer_outputs, _, _ = f_with_kfac_io(X, params)
            return output, layer_inputs, layer_outputs

        return _forward_io

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
        corrected_eigenvalues: dict[str, Tensor | dict[int, Tensor]],
    ) -> None:
        """Compute eigenvalue correction for one layer from IO data.

        Processes batched gradients (from ``is_grads_batched=True``) by
        flattening the ``[num_vectors, batch]`` dimensions into a single
        batch dimension and expanding the layer input to match. Then calls
        ``compute_eigenvalue_correction_linear_weight_sharing``.

        Accumulates results directly into ``corrected_eigenvalues``, matching
        the structure used by the hooks backend in
        ``EKFACComputer._accumulate_corrected_eigenvalues``.

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
            corrected_eigenvalues: Dictionary to accumulate corrections (mutated).
        """
        g = batched_g.data.detach()  # [num_vectors, batch, ...]
        num_vectors = g.shape[0]
        batch_size = g.shape[1]

        # Flatten [v, batch] -> (v batch) for g, expand and flatten a to match
        g = rearrange(g, "v batch ... -> (v batch) ...")

        param_pos = self._mapping[module_name]
        a_required = "weight" in param_pos

        a = None
        if a_required and x is not None:
            a = x.data.detach()  # [batch, ...]
            a = a.unsqueeze(0).expand(num_vectors, *(-1,) * a.ndim)
            a = rearrange(a, "v batch ... -> (v batch) ...")

        is_conv2d, kernel_size, hyperparams = self._conv_info(
            io_layer_name, layer_hyperparams, layer_param_names, named_params
        )

        g, a = prepare_io_for_ekfac(
            g,
            a,
            is_conv2d=is_conv2d,
            kernel_size=kernel_size,
            stride=hyperparams.get("stride") if is_conv2d else None,
            padding=hyperparams.get("padding") if is_conv2d else None,
            dilation=hyperparams.get("dilation") if is_conv2d else None,
            groups=hyperparams.get("groups") if is_conv2d else None,
        )

        correction = compute_loss_correction(
            batch_size,
            self._num_per_example_loss_terms,
            self._loss_func.reduction,
            self._N_data,
        )

        aaT_eigvecs = input_cov_eigvecs.get(module_name)
        ggT_eigvecs = gradient_cov_eigvecs[module_name]

        if _has_joint_weight_and_bias(self._separate_weight_and_bias, param_pos):
            a_augmented = cat([a, a.new_ones(*a.shape[:-1], 1)], dim=-1)
            eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                g, ggT_eigvecs, a_augmented, aaT_eigvecs
            )
            self._set_or_add_(
                corrected_eigenvalues, module_name, eigencorrection.mul_(correction)
            )
        else:
            if module_name not in corrected_eigenvalues:
                corrected_eigenvalues[module_name] = {}
            for p_name, pos in param_pos.items():
                eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                    g,
                    ggT_eigvecs,
                    None if p_name == "bias" else a,
                    None if p_name == "bias" else aaT_eigvecs,
                )
                self._set_or_add_(
                    corrected_eigenvalues[module_name],
                    pos,
                    eigencorrection.mul_(correction),
                )
