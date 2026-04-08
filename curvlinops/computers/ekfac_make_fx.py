"""EKFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxEKFACComputer``, which extends
``MakeFxKFACComputer`` with FX-based eigenvalue correction, using the IO
collector (``with_kfac_io``) instead of forward/backward hooks.
"""

from collections import UserDict
from collections.abc import Callable
from typing import Any

from einops import einsum
from torch import Tensor, cat, manual_seed

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _EKFACMixin
from curvlinops.computers.ekfac_hooks import (
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.computers.kfac_make_fx import (
    MakeFxKFACComputer,
    _bias_pad,
    make_compute_kfac_io_batch,
)
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import KFACType
from curvlinops.utils import _make_fx


class MakeFxEKFACComputer(_EKFACMixin, MakeFxKFACComputer):
    """EKFAC computer that uses FX graph tracing for eigenvalue correction.

    Extends ``MakeFxKFACComputer`` with eigenvalue correction computation.
    Kronecker factor computation is inherited from ``MakeFxKFACComputer``.
    Uses :func:`make_compute_kfac_io_batch` for both Kronecker factor
    computation and eigenvalue correction.
    """

    def _trace_io_batch_functions(
        self,
    ) -> tuple[
        dict[int, Callable],
        list[ParamGroup],
        dict[ParamGroupKey, list[str]],
        dict[str, dict[str, str]],
        dict[str, dict[str, Any]],
    ]:
        """Build IO batch functions for all batch sizes in the data.

        Iterates over the data once, calling :func:`make_compute_kfac_io_batch`
        for each unique batch size.

        Returns:
            Tuple of ``(io_batch_fns, mapping, io_groups, io_param_names,
            layer_hparams)`` where ``io_batch_fns`` maps batch sizes to
            IO batch callables.
        """
        io_batch_fns: dict[int, Callable] = {}
        mapping: list[ParamGroup] | None = None
        io_groups: dict[ParamGroupKey, list[str]] | None = None
        io_param_names: dict[str, dict[str, str]] | None = None
        layer_hparams: dict[str, dict[str, Any]] | None = None

        for X, y in self._loop_over_data(desc="FX tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size not in io_batch_fns:
                if isinstance(X, UserDict):
                    _register_userdict_as_pytree()
                (
                    io_batch_fns[batch_size],
                    mapping,
                    io_groups,
                    io_param_names,
                    (layer_hparams),
                ) = make_compute_kfac_io_batch(
                    self._model_func,
                    self._loss_func,
                    self._params,
                    X,
                    self._fisher_type,
                    self._mc_samples,
                    self._separate_weight_and_bias,
                    output_check_fn=lambda out: (
                        self._rearrange_for_larger_than_2d_output(out, y)
                    ),
                )

        return io_batch_fns, mapping, io_groups, io_param_names, layer_hparams

    def _trace_batch_functions(
        self,
        io_batch_fns: dict[int, Callable],
        mapping: list[ParamGroup],
        io_groups: dict[ParamGroupKey, list[str]],
        io_param_names: dict[str, dict[str, str]],
        layer_hparams: dict[str, dict[str, Any]],
    ) -> tuple[
        dict[int, Callable],
        list[ParamGroup],
        list[ParamGroupKey],
        list[ParamGroupKey],
    ]:
        """Trace the full per-batch KFAC computation for each unique batch size.

        Args:
            io_batch_fns: IO batch functions per batch size.
            mapping: Parameter groups.
            io_groups: IO-layer mapping.
            io_param_names: Layer parameter name mappings.
            layer_hparams: Layer hyperparameter dicts.

        Returns:
            Tuple of ``(traced_fns, mapping, weight_group_keys, all_group_keys)``.
        """
        weight_group_keys = [tuple(g.values()) for g in mapping if "W" in g]
        all_group_keys = [tuple(g.values()) for g in mapping]

        # Pre-compute group structure for stable ordering in the trace
        weight_group_info = []
        for group in mapping:
            if "W" not in group:
                continue
            group_key = tuple(group.values())
            io_names = io_groups.get(group_key, [])
            has_joint_wb = "b" in group
            bias_pads = [_bias_pad(has_joint_wb, io_param_names[n]) for n in io_names]
            hparams = [layer_hparams[n] for n in io_names]
            weight_group_info.append((io_names, bias_pads, hparams))

        grad_group_info = []
        for group in mapping:
            group_key = tuple(group.values())
            io_names = io_groups.get(group_key, [])
            hparams = [layer_hparams[n] for n in io_names]
            grad_group_info.append((io_names, hparams))

        kfac_approx = self._kfac_approx
        loss_reduction = self._loss_func.reduction
        num_per_example_loss_terms = self._num_per_example_loss_terms

        traced_fns: dict[int, Callable] = {}

        for X, y in self._loop_over_data(desc="Batch tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size not in traced_fns:
                io_batch = io_batch_fns[batch_size]

                def compute_batch(
                    params: dict[str, Tensor], X: Tensor, y: Tensor
                ) -> tuple[list[Tensor], list[Tensor]]:
                    layer_inputs, layer_output_grads = io_batch(params, X, y)

                    input_covs = []
                    for io_names, bias_pads, hparams in weight_group_info:
                        xs = [
                            input_to_weight_sharing_format(
                                layer_inputs[n].data.detach(),
                                kfac_approx,
                                layer_hyperparams=hp,
                                bias_pad=bp,
                            )
                            for n, bp, hp in zip(io_names, bias_pads, hparams)
                        ]
                        x = cat(xs, dim=1)
                        scale = x.shape[1]
                        xxT = einsum(x, x, "batch shared i, batch shared j -> i j")
                        input_covs.append(xxT.div_(scale))

                    if not layer_output_grads:
                        return input_covs, []

                    gradient_covs = []
                    for io_names, hparams in grad_group_info:
                        gs = [
                            grad_to_weight_sharing_format(
                                layer_output_grads[n].data.detach(),
                                kfac_approx,
                                layer_hyperparams=hp,
                                num_leading_dims=2,
                            )
                            for n, hp in zip(io_names, hparams)
                        ]
                        g = cat(gs, dim=2)
                        correction = compute_loss_correction(
                            g.shape[1],
                            num_per_example_loss_terms,
                            loss_reduction,
                        )
                        ggT = einsum(
                            g,
                            g,
                            "v batch shared i, v batch shared j -> i j",
                        ).mul_(correction)
                        gradient_covs.append(ggT)

                    return input_covs, gradient_covs

                traced_fns[batch_size] = _make_fx(compute_batch)(self._params, X, y)

        return traced_fns, mapping, weight_group_keys, all_group_keys

    def compute(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor],
        dict[ParamGroupKey, Tensor],
        dict[ParamGroupKey, Tensor],
        list[ParamGroup],
    ]:
        """Compute eigenvalue-corrected Kronecker factors.

        Uses :func:`make_compute_kfac_io_batch` to build IO batch functions
        that are reused for both Kronecker factor computation and eigenvalue
        correction.

        Returns:
            Tuple of ``(input_covariance_eigenvectors,
            gradient_covariance_eigenvectors, corrected_eigenvalues, mapping)``.
        """
        io_batch_fns, mapping, io_groups, io_param_names, layer_hparams = (
            self._trace_io_batch_functions()
        )
        traced_batch = self._trace_batch_functions(
            io_batch_fns, mapping, io_groups, io_param_names, layer_hparams
        )
        input_covariances, gradient_covariances, mapping = (
            self._compute_kronecker_factors(traced_batch)
        )
        input_covariances = self._eigenvectors_(input_covariances)
        gradient_covariances = self._eigenvectors_(gradient_covariances)
        corrected_eigenvalues = self.compute_eigenvalue_correction(
            input_covariances,
            gradient_covariances,
            mapping,
            io_batch_fns,
            io_groups,
            io_param_names,
            layer_hparams,
        )
        return input_covariances, gradient_covariances, corrected_eigenvalues, mapping

    def compute_eigenvalue_correction(
        self,
        input_covariances_eigenvectors: dict[ParamGroupKey, Tensor],
        gradient_covariances_eigenvectors: dict[ParamGroupKey, Tensor],
        mapping: list[ParamGroup],
        io_batch_fns: dict[int, Callable],
        io_groups: dict[ParamGroupKey, list[str]],
        io_param_names: dict[str, dict[str, str]],
        layer_hparams: dict[str, dict[str, Any]],
    ) -> dict[ParamGroupKey, Tensor]:
        """Compute eigenvalue corrections using IO batch functions.

        Reuses the IO batch functions from :func:`make_compute_kfac_io_batch`
        to obtain layer inputs and output gradients for each batch.

        Args:
            input_covariances_eigenvectors: Input covariance eigenvectors
                per parameter group.
            gradient_covariances_eigenvectors: Gradient covariance eigenvectors
                per parameter group.
            mapping: List of parameter groups.
            io_batch_fns: IO batch functions per batch size.
            io_groups: IO-layer mapping.
            io_param_names: Layer parameter name mappings.
            layer_hparams: Layer hyperparameter dicts.

        Returns:
            Dictionary mapping parameter group keys to corrected eigenvalues.
        """
        corrected_eigenvalues: dict[ParamGroupKey, Tensor] = {}

        manual_seed(self._seed)

        for X, y in self._loop_over_data(desc="Eigenvalue correction"):
            batch_size = self._batch_size_fn(X)
            layer_inputs, layer_output_grads = io_batch_fns[batch_size](
                self._params, X, y
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
