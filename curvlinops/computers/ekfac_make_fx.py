"""EKFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxEKFACComputer``, which extends
``MakeFxKFACComputer`` with FX-based eigenvalue correction, using the IO
collector (``with_kfac_io``) instead of forward/backward hooks. Only the
forward pass is traced with ``make_fx``; the backward pass runs eagerly.
"""

from collections.abc import Callable
from typing import Any

from torch import Tensor, cat

from curvlinops.computers._base import ParamGroup, ParamGroupKey, _EKFACMixin
from curvlinops.computers.ekfac_hooks import (
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.computers.kfac_make_fx import (
    MakeFxKFACComputer,
    _bias_pad,
    _build_param_groups_from_io,
)
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import KFACType
from curvlinops.utils import _seed_generator


class MakeFxEKFACComputer(_EKFACMixin, MakeFxKFACComputer):
    """EKFAC computer that uses FX graph tracing for eigenvalue correction.

    Extends ``MakeFxKFACComputer`` with eigenvalue correction computation.
    Kronecker factor computation is inherited from ``MakeFxKFACComputer``.
    Only the forward pass (IO collection) is traced with ``make_fx``; the
    backward pass and eigenvalue correction computation run eagerly.
    """

    def compute(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor],
        dict[ParamGroupKey, Tensor],
        dict[ParamGroupKey, Tensor],
        list[ParamGroup],
    ]:
        """Compute eigenvalue-corrected Kronecker factors, tracing IO functions first.

        Overrides the base class to trace IO functions once and reuse them for
        both factor computation and eigenvalue correction.

        Returns:
            Tuple of ``(input_covariance_eigenvectors,
            gradient_covariance_eigenvectors, corrected_eigenvalues, mapping)``.
        """
        traced_io = self._trace_io_functions()
        with self._computation_context():
            input_covariances, gradient_covariances, mapping = (
                self._compute_kronecker_factors(traced_io)
            )
            input_covariances = self._eigenvectors_(input_covariances)
            gradient_covariances = self._eigenvectors_(gradient_covariances)
            corrected_eigenvalues = self.compute_eigenvalue_correction(
                input_covariances, gradient_covariances, mapping, traced_io
            )
        return input_covariances, gradient_covariances, corrected_eigenvalues, mapping

    def compute_eigenvalue_correction(
        self,
        input_covariances_eigenvectors: dict[ParamGroupKey, Tensor],
        gradient_covariances_eigenvectors: dict[ParamGroupKey, Tensor],
        mapping: list[ParamGroup],
        traced_io: tuple[
            dict[int, Callable],
            dict[str, dict[str, str]],
            dict[str, dict[str, Any]],
        ],
    ) -> dict[ParamGroupKey, Tensor]:
        """Compute eigenvalue corrections using pre-traced IO functions.

        Args:
            input_covariances_eigenvectors: Input covariance eigenvectors
                per parameter group.
            gradient_covariances_eigenvectors: Gradient covariance eigenvectors
                per parameter group.
            mapping: List of parameter groups.
            traced_io: Pre-traced IO functions from :meth:`_trace_io_functions`.

        Returns:
            Dictionary mapping parameter group keys to corrected eigenvalues.
        """
        traced_io_fns, io_param_names, layer_hparams = traced_io

        io_groups: dict[ParamGroupKey, list[str]] | None = None
        if io_param_names is not None:
            _, io_groups = _build_param_groups_from_io(
                io_param_names, self._separate_weight_and_bias
            )

        corrected_eigenvalues: dict[ParamGroupKey, Tensor] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="Eigenvalue correction"):
            batch_size = self._batch_size_fn(X)

            # Forward pass with IO collection
            io_fn = traced_io_fns[batch_size]
            output, layer_inputs, layer_outputs = io_fn(self._params, X)

            # Backward pass: compute per-layer output gradients
            layer_output_grads = self._compute_layer_output_grads(
                output, y, layer_outputs
            )

            for group in mapping:
                group_key = tuple(group.values())
                io_names = io_groups.get(group_key, [])
                has_joint_wb = "b" in group and "W" in group

                gs = [
                    grad_to_weight_sharing_format(
                        layer_output_grads[n].data.detach(),
                        KFACType.EXPAND,
                        layer_hparams[n],
                        num_leading_dims=2,
                    )
                    for n in io_names
                ]
                g = cat(gs, dim=2)

                a = None
                if "W" in group:
                    xs = [
                        input_to_weight_sharing_format(
                            layer_inputs[n].data.detach(),
                            KFACType.EXPAND,
                            layer_hparams[n],
                            bias_pad=_bias_pad(has_joint_wb, io_param_names[n]),
                        )
                        for n in io_names
                    ]
                    a = cat(xs, dim=1)

                batch_size = g.shape[1]
                correction = compute_loss_correction(
                    batch_size,
                    self._num_per_example_loss_terms,
                    self._loss_func.reduction,
                    self._N_data,
                )

                aaT_eigvecs = input_covariances_eigenvectors.get(group_key)
                ggT_eigvecs = gradient_covariances_eigenvectors[group_key]

                eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                    g, ggT_eigvecs, a, aaT_eigvecs
                )
                self._set_or_add_(
                    corrected_eigenvalues,
                    group_key,
                    eigencorrection.mul_(correction),
                )

        return corrected_eigenvalues
