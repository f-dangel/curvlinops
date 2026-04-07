"""EKFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxEKFACComputer``, which extends
``MakeFxKFACComputer`` with FX-based eigenvalue correction, using the IO
collector (``with_kfac_io``) instead of forward/backward hooks. Only the
forward pass is traced with ``make_fx``; the backward pass runs eagerly.
"""

from collections import UserDict
from collections.abc import Callable
from typing import Any

from torch import Tensor, autograd, cat

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _EKFACMixin
from curvlinops.computers.ekfac_hooks import (
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac_make_fx import (
    MakeFxKFACComputer,
    _bias_pad,
    _build_param_groups_from_io,
    _make_batch_fn,
)
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import KFACType
from curvlinops.utils import _make_fx, _seed_generator


class MakeFxEKFACComputer(_EKFACMixin, MakeFxKFACComputer):
    """EKFAC computer that uses FX graph tracing for eigenvalue correction.

    Extends ``MakeFxKFACComputer`` with eigenvalue correction computation.
    Kronecker factor computation is inherited from ``MakeFxKFACComputer``.
    Only the forward pass (IO collection) is traced with ``make_fx``; the
    backward pass and eigenvalue correction computation run eagerly.
    """

    def _trace_io_functions(
        self,
    ) -> tuple[
        dict[int, Callable],
        dict[str, dict[str, str]],
        dict[str, dict[str, Any]],
    ]:
        """Pre-trace IO collection functions for all batch sizes in the data.

        Iterates over the data once, calling ``with_kfac_io`` for each unique
        batch size.

        Returns:
            Tuple of ``(traced_io_fns, io_param_names, layer_hparams)``.
        """
        traced_io_fns: dict[int, Callable] = {}
        io_param_names: dict[str, dict[str, str]] | None = None
        layer_hparams: dict[str, dict[str, Any]] | None = None

        for X, _ in self._loop_over_data(desc="FX tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size not in traced_io_fns:
                if isinstance(X, UserDict):
                    _register_userdict_as_pytree()
                traced_io_fns[batch_size], io_param_names, layer_hparams = with_kfac_io(
                    self._model_func, X, self._params, self._fisher_type
                )

        return traced_io_fns, io_param_names, layer_hparams

    def _trace_batch_functions(
        self,
        traced_io: tuple[
            dict[int, Callable],
            dict[str, dict[str, str]],
            dict[str, dict[str, Any]],
        ],
    ) -> tuple[
        dict[int, Callable],
        list[ParamGroup],
        list[ParamGroupKey],
        list[ParamGroupKey],
    ]:
        """Trace the full per-batch KFAC computation for each unique batch size.

        Args:
            traced_io: Pre-traced IO functions from :meth:`_trace_io_functions`.

        Returns:
            Tuple of ``(traced_fns, mapping, weight_group_keys, all_group_keys)``.
        """
        traced_io_fns, io_param_names, layer_hparams = traced_io

        mapping, io_groups = _build_param_groups_from_io(
            io_param_names, self._separate_weight_and_bias
        )
        weight_group_keys = [tuple(g.values()) for g in mapping if "W" in g]
        all_group_keys = [tuple(g.values()) for g in mapping]

        traced_fns: dict[int, Callable] = {}

        for X, y in self._loop_over_data(desc="Batch tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size not in traced_fns:
                batch_fn = _make_batch_fn(
                    traced_io_fns[batch_size],
                    io_param_names,
                    layer_hparams,
                    mapping,
                    io_groups,
                    self._kfac_approx,
                    self._fisher_type,
                    self._loss_func.reduction,
                    self._num_per_example_loss_terms,
                    self._grad_outputs_computer,
                    self._rearrange_for_larger_than_2d_output,
                )
                traced_fns[batch_size] = _make_fx(batch_fn)(self._params, X, y)

        return traced_fns, mapping, weight_group_keys, all_group_keys

    def _compute_layer_output_grads(
        self,
        output: Tensor,
        y: Tensor,
        layer_outputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute scaled batched gradients for all tracked layers.

        Args:
            output: Model output tensor.
            y: Target tensor.
            layer_outputs: Collected layer outputs from the IO function.

        Returns:
            Dictionary mapping IO layer names to batched gradient tensors.
        """
        output, y = self._rearrange_for_larger_than_2d_output(output, y)

        grad_outputs = self._grad_outputs_computer(output.detach(), y, self._generator)
        num_loss_terms = output.shape[0]
        scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[self._loss_func.reduction]
        grad_outputs.mul_(scale)

        io_layer_names = list(layer_outputs)
        output_tensors = list(layer_outputs.values())
        layer_output_grads = autograd.grad(
            output,
            output_tensors,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
        )
        return dict(zip(io_layer_names, layer_output_grads))

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
        traced_batch = self._trace_batch_functions(traced_io)
        input_covariances, gradient_covariances, mapping = (
            self._compute_kronecker_factors(traced_batch)
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

        _, io_groups = _build_param_groups_from_io(
            io_param_names, self._separate_weight_and_bias
        )

        corrected_eigenvalues: dict[ParamGroupKey, Tensor] = {}

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="Eigenvalue correction"):
            # Forward pass with IO collection
            batch_size = self._batch_size_fn(X)
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
