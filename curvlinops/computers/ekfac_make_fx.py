"""EKFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxEKFACComputer``, which extends ``EKFACComputer``
with FX-based eigenvalue correction, using the IO collector (``with_kfac_io``)
instead of forward/backward hooks. Only the forward pass is traced with
``make_fx``; the backward pass runs eagerly.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from torch import Tensor, cat
from torch.func import functional_call

from curvlinops.computers.ekfac import (
    EKFACComputer,
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.computers.kfac import ParameterUsage
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
        io_to_usage: dict[str, ParameterUsage] | None = None
        usage_by_name: dict[str, ParameterUsage] | None = None
        layer_hparams: dict[str, dict[str, Any]] | None = None

        corrected_eigenvalues: dict[str, Tensor | dict[str, Tensor]] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="Eigenvalue correction"):
            # Maybe trace for current batch size and set up layer metadata
            if (batch_size := self._batch_size_fn(X)) not in traced_io_fns:
                traced_io_fns[batch_size], io_to_module, layer_hparams = _trace_io(
                    f, X, self._params, self._fisher_type
                )

            # Build lookups from IO collector names to ParameterUsage objects
            if io_to_usage is None:
                io_to_usage = {
                    io_name: self._usage_by_module[mod_name]
                    for io_name, mod_name in io_to_module.items()
                }
                usage_by_name = {u.name: u for u in io_to_usage.values()}

            # Forward pass with IO collection
            io_fn = traced_io_fns[batch_size]
            output, layer_inputs, layer_outputs = io_fn(X, self._params)

            # Backward pass: compute per-layer output gradients
            layer_output_grads = self._compute_layer_output_grads(
                output, y, layer_outputs, io_to_module
            )

            # Format per-usage, concatenate tied weights, compute corrections
            formatted_grads: dict[str, list[Tensor]] = defaultdict(list)
            formatted_inputs: dict[str, list[Tensor]] = defaultdict(list)

            for io_layer_name, batched_g in layer_output_grads.items():
                usage = io_to_usage[io_layer_name]
                hparams = layer_hparams[io_layer_name]

                g = grad_to_weight_sharing_format(
                    batched_g.data.detach(),
                    KFACType.EXPAND,
                    hparams,
                    num_leading_dims=2,
                )
                formatted_grads[usage.name].append(g)

                x = layer_inputs.get(io_layer_name)
                if "W" in usage.params and x is not None:
                    has_joint_wb = _has_joint_weight_and_bias(
                        self._separate_weight_and_bias, usage.params
                    )
                    a = input_to_weight_sharing_format(
                        x.data.detach(),
                        KFACType.EXPAND,
                        hparams,
                        bias_pad=1 if has_joint_wb else None,
                    )
                    formatted_inputs[usage.name].append(a)

            for usage_name, gs in formatted_grads.items():
                g = cat(gs, dim=2) if len(gs) > 1 else gs[0]
                a = None
                if usage_name in formatted_inputs:
                    xs = formatted_inputs[usage_name]
                    a = cat(xs, dim=1) if len(xs) > 1 else xs[0]
                self._eigenvalue_correction_from_formatted(
                    g,
                    a,
                    usage_by_name[usage_name],
                    input_covariances_eigenvectors,
                    gradient_covariances_eigenvectors,
                    corrected_eigenvalues,
                )

        return corrected_eigenvalues

    def _eigenvalue_correction_from_formatted(
        self,
        g: Tensor,
        a: Tensor | None,
        usage: ParameterUsage,
        input_cov_eigvecs: dict[str, Tensor],
        gradient_cov_eigvecs: dict[str, Tensor],
        corrected_eigenvalues: dict[str, Tensor | dict[str, Tensor]],
    ) -> None:
        """Compute eigenvalue correction from pre-formatted IO tensors.

        Args:
            g: Gradient in weight sharing format ``[v, batch, shared, d_out]``.
            a: Input in weight sharing format ``[batch, shared, d_in]`` or ``None``.
            usage: Parameter usage info for this layer.
            input_cov_eigvecs: Input covariance eigenvectors per layer.
            gradient_cov_eigvecs: Gradient covariance eigenvectors per layer.
            corrected_eigenvalues: Dictionary to accumulate corrections (mutated).
        """
        batch_size = g.shape[1]
        has_joint_wb = _has_joint_weight_and_bias(
            self._separate_weight_and_bias, usage.params
        )

        correction = compute_loss_correction(
            batch_size,
            self._num_per_example_loss_terms,
            self._loss_func.reduction,
            self._N_data,
        )

        aaT_eigvecs = input_cov_eigvecs.get(usage.name)
        ggT_eigvecs = gradient_cov_eigvecs[usage.name]

        if has_joint_wb:
            eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                g, ggT_eigvecs, a, aaT_eigvecs
            )
            self._set_or_add_(
                corrected_eigenvalues,
                usage.name,
                eigencorrection.mul_(correction),
            )
        else:
            if usage.name not in corrected_eigenvalues:
                corrected_eigenvalues[usage.name] = {}
            for p_name, pos in usage.params.items():
                eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                    g,
                    ggT_eigvecs,
                    None if p_name == "b" else a,
                    None if p_name == "b" else aaT_eigvecs,
                )
                self._set_or_add_(
                    corrected_eigenvalues[usage.name],
                    pos,
                    eigencorrection.mul_(correction),
                )
