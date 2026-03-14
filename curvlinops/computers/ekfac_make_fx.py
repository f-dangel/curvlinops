"""EKFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxEKFACComputer``, which extends ``EKFACComputer``
with FX-based eigenvalue correction, using the IO collector (``with_kfac_io``)
instead of forward/backward hooks. Only the forward pass is traced with
``make_fx``; the backward pass runs eagerly.
"""

from collections.abc import Callable
from typing import Any

from torch import Tensor
from torch.func import functional_call

from curvlinops.computers.ekfac import (
    EKFACComputer,
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.computers.kfac_make_fx import MakeFxKFACComputer, _trace_io
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import KFACType, _has_joint_weight_and_bias
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
    ) -> dict[str, Tensor | dict[str, Tensor]]:
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

        def f(x, params: dict[str, Tensor]) -> Tensor:
            return functional_call(self._model_func, params, (x,))

        # Cache for IO functions: make_fx bakes in tensor shapes (e.g. from nn.Flatten),
        # so different batch sizes need separate traces
        traced_io_fns: dict[int, Callable] = {}

        # Layer metadata (identical across batch sizes), populated on first trace
        io_to_module: dict[str, str] | None = None
        layer_hparams: dict[str, dict[str, Any]] | None = None

        corrected_eigenvalues: dict[str, Tensor | dict[str, Tensor]] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="Eigenvalue correction"):
            # Maybe trace for current batch size and set up layer metadata
            if (batch_size := self._batch_size_fn(X)) not in traced_io_fns:
                traced_io_fns[batch_size], io_to_module, layer_hparams = _trace_io(
                    f, X, self._params, self._fisher_type
                )

            # Forward pass with IO collection
            io_fn = traced_io_fns[batch_size]
            output, layer_inputs, layer_outputs = io_fn(X, self._params)

            # Backward pass: compute per-layer output gradients
            layer_output_grads = self._compute_layer_output_grads(
                output, y, layer_outputs, io_to_module
            )

            # Accumulate eigenvalue corrections per layer
            for io_layer_name, batched_g in layer_output_grads.items():
                mod_name = io_to_module[io_layer_name]
                x = layer_inputs.get(io_layer_name)
                self._eigenvalue_correction_from_io(
                    batched_g,
                    x,
                    mod_name,
                    layer_hparams[io_layer_name],
                    input_covariances_eigenvectors,
                    gradient_covariances_eigenvectors,
                    corrected_eigenvalues,
                )

        return corrected_eigenvalues

    def _eigenvalue_correction_from_io(
        self,
        batched_g: Tensor,
        x: Tensor | None,
        module_name: str,
        layer_hparams: dict[str, Any],
        input_cov_eigvecs: dict[str, Tensor],
        gradient_cov_eigvecs: dict[str, Tensor],
        corrected_eigenvalues: dict[str, Tensor | dict[str, Tensor]],
    ) -> None:
        """Compute eigenvalue correction for one layer from IO data.

        Converts gradients and inputs to weight sharing format, then calls
        ``compute_eigenvalue_correction_linear_weight_sharing`` which handles
        the ``[V, N, S, D]`` gradient shape natively.

        Accumulates results directly into ``corrected_eigenvalues``, matching
        the structure used by the hooks backend in
        ``EKFACComputer._accumulate_corrected_eigenvalues``.

        Args:
            batched_g: Batched gradients at the layer's output with shape
                ``[num_vectors, batch, ...]``.
            x: Layer input tensor with shape ``[batch, ...]`` or ``None``.
            module_name: Module name in ``self._mapping``.
            layer_hparams: Hyperparameters from IO collector for this layer.
            input_cov_eigvecs: Input covariance eigenvectors per module.
            gradient_cov_eigvecs: Gradient covariance eigenvectors per module.
            corrected_eigenvalues: Dictionary to accumulate corrections (mutated).
        """
        g = batched_g.data.detach()  # [v, batch, ...]
        batch_size = g.shape[1]

        # Convert g to weight sharing format: [v, batch, ...] -> [v, batch, shared, d_out]
        g = grad_to_weight_sharing_format(
            g, KFACType.EXPAND, layer_hparams, num_leading_dims=2
        )

        param_pos = self._mapping[module_name]
        a_required = "weight" in param_pos
        has_joint_wb = _has_joint_weight_and_bias(
            self._separate_weight_and_bias, param_pos
        )

        # Convert a to weight sharing format: [batch, ...] -> [batch, shared, d_in]
        a = None
        if a_required and x is not None:
            a = input_to_weight_sharing_format(
                x.data.detach(),
                KFACType.EXPAND,
                layer_hparams,
                append_ones_for_bias=has_joint_wb,
            )

        correction = compute_loss_correction(
            batch_size,
            self._num_per_example_loss_terms,
            self._loss_func.reduction,
            self._N_data,
        )

        aaT_eigvecs = input_cov_eigvecs.get(module_name)
        ggT_eigvecs = gradient_cov_eigvecs[module_name]

        if has_joint_wb:
            eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                g, ggT_eigvecs, a, aaT_eigvecs
            )
            self._set_or_add_(
                corrected_eigenvalues,
                module_name,
                eigencorrection.mul_(correction),
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
