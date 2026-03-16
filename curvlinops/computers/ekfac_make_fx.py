"""EKFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxEKFACComputer``, which extends ``EKFACComputer``
with FX-based eigenvalue correction, using the IO collector (``with_kfac_io``)
instead of forward/backward hooks. Only the forward pass is traced with
``make_fx``; the backward pass runs eagerly.
"""

from collections.abc import Callable
from typing import Any

from torch import Tensor, cat
from torch.func import functional_call

from curvlinops.computers.ekfac import (
    EKFACComputer,
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.computers.kfac_make_fx import (
    MakeFxKFACComputer,
    _bias_pad,
    _map_param_groups_to_io_layers,
    _trace_io,
)
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import KFACType
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
        input_covariances_eigenvectors: dict[tuple[str, ...], Tensor],
        gradient_covariances_eigenvectors: dict[tuple[str, ...], Tensor],
    ) -> dict[tuple[str, ...], Tensor]:
        """Compute eigenvalue corrections using FX graph tracing.

        Args:
            input_covariances_eigenvectors: Input covariance eigenvectors
                per parameter group.
            gradient_covariances_eigenvectors: Gradient covariance eigenvectors
                per parameter group.

        Returns:
            Dictionary mapping parameter group keys to corrected eigenvalues.
        """

        def f(x, params: dict[str, Tensor]) -> Tensor:
            return functional_call(self._model_func, params, (x,))

        # Cache for IO functions: make_fx bakes in tensor shapes (e.g. from
        # nn.Flatten), so different batch sizes need separate traces
        traced_io_fns: dict[int, Callable] = {}

        # Layer metadata (identical across batch sizes), populated on first trace
        io_param_names: dict[str, dict[str, str]] | None = None
        layer_hparams: dict[str, dict[str, Any]] | None = None
        io_groups: dict[tuple[str, ...], list[str]] | None = None

        corrected_eigenvalues: dict[tuple[str, ...], Tensor] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="Eigenvalue correction"):
            # Maybe trace for current batch size and set up layer metadata
            if (batch_size := self._batch_size_fn(X)) not in traced_io_fns:
                traced_io_fns[batch_size], io_param_names, layer_hparams = _trace_io(
                    f, X, self._params, self._fisher_type
                )

            if io_groups is None:
                io_groups = _map_param_groups_to_io_layers(
                    self._mapping, io_param_names
                )

            # Forward pass with IO collection
            io_fn = traced_io_fns[batch_size]
            output, layer_inputs, layer_outputs = io_fn(X, self._params)

            # Backward pass: compute per-layer output gradients
            layer_output_grads = self._compute_layer_output_grads(
                output, y, layer_outputs
            )

            for group_key, io_names in io_groups.items():
                usage = self._mapping_by_key[group_key]
                has_joint_wb = "b" in usage.params and "W" in usage.params

                names_with_grad = [n for n in io_names if n in layer_output_grads]
                if not names_with_grad:
                    continue
                gs = [
                    grad_to_weight_sharing_format(
                        layer_output_grads[n].data.detach(),
                        KFACType.EXPAND,
                        layer_hparams[n],
                        num_leading_dims=2,
                    )
                    for n in names_with_grad
                ]
                g = cat(gs, dim=2) if len(gs) > 1 else gs[0]

                a = None
                if "W" in usage.params:
                    names_with_input = [n for n in io_names if n in layer_inputs]
                    if names_with_input:
                        xs = [
                            input_to_weight_sharing_format(
                                layer_inputs[n].data.detach(),
                                KFACType.EXPAND,
                                layer_hparams[n],
                                bias_pad=_bias_pad(has_joint_wb, io_param_names[n]),
                            )
                            for n in names_with_input
                        ]
                        a = cat(xs, dim=1) if len(xs) > 1 else xs[0]

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
