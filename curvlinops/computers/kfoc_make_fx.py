r"""KFOC computer and rearranged-GGN primitive.

Provides:

- :class:`_RearrangedGGNLinearOperator`: Van Loan rearrangement of a per-layer
  Gauss-Newton block, exposed as a rectangular PyTorch linear operator whose
  matvec consumes per-sample activations and output gradients.
- :func:`_top_rank_one_kron_factors`: thin wrapper that builds the rearranged
  operator and runs :func:`scipy.sparse.linalg.svds` for the top singular pair,
  reshaping the result into Kronecker factors.
- :class:`MakeFxKFOCComputer`: FX-based computer that collects per-layer
  inputs/gradients (via :func:`make_compute_kfac_io_batch`) and produces
  KFOC's Kronecker factors via per-group truncated SVD.
"""

from __future__ import annotations

from math import sqrt

from einops import einsum
from scipy.sparse.linalg import svds
from torch import Tensor, device, dtype, from_numpy

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _BaseKFACComputer
from curvlinops.computers.kfac_make_fx import (
    make_compute_kfac_io_batch,
    make_group_gatherers,
)
from curvlinops.kfac_utils import KFACType


class _RearrangedGGNLinearOperator(PyTorchLinearOperator):
    r"""Van Loan rearrangement of a per-layer Gauss-Newton block.

    For a layer with per-sample activations :math:`a_n \in \mathbb{R}^{d_\text{in}}`
    and output gradients :math:`g_n \in \mathbb{R}^{d_\text{out}}` (for
    :math:`n = 1, \dots, N`), the per-layer GGN block is

    .. math::
        B_l = \sum_n (g_n g_n^\top) \otimes (a_n a_n^\top).

    Its Van Loan rearrangement :math:`\mathcal{R}(B_l) \in \mathbb{R}^{d_\text{out}^2
    \times d_\text{in}^2}` satisfies
    :math:`\mathcal{R}(G \otimes A) = \mathrm{vec}(G)\,\mathrm{vec}(A)^\top`, so

    .. math::
        \mathcal{R}(B_l) = \sum_n \mathrm{vec}(g_n g_n^\top)\,\mathrm{vec}(a_n a_n^\top)^\top.

    Matvecs exploit the per-sample rank-1 structure: for
    :math:`V \in \mathbb{R}^{d_\text{in} \times d_\text{in}}`,

    .. math::
        \mathcal{R}(B_l)\,\mathrm{vec}(V)
        = \mathrm{vec}\!\left(\sum_n (a_n^\top V a_n)\,g_n g_n^\top\right),

    and analogously for the adjoint. Input and output tensor-list shapes are
    ``[(d_in, d_in)]`` and ``[(d_out, d_out)]``, so matvecs naturally consume
    and produce matrices.
    """

    def __init__(self, a_per_sample: Tensor, g_per_sample: Tensor):
        """Store per-sample activations and output gradients.

        Args:
            a_per_sample: Per-sample activations, shape ``(N, d_in)``.
            g_per_sample: Per-sample output gradients, shape ``(N, d_out)``.

        Raises:
            ValueError: If tensors are not 2D or sample counts disagree.
        """
        if a_per_sample.ndim != 2 or g_per_sample.ndim != 2:
            raise ValueError(
                "a_per_sample and g_per_sample must be 2D, got "
                f"{a_per_sample.shape} and {g_per_sample.shape}."
            )
        N, d_in = a_per_sample.shape
        N_g, d_out = g_per_sample.shape
        if N != N_g:
            raise ValueError(f"Sample count mismatch: a has {N} samples, g has {N_g}.")
        super().__init__(in_shape=[(d_in, d_in)], out_shape=[(d_out, d_out)])
        self._a = a_per_sample
        self._g = g_per_sample

    def _matmat(self, X: list[Tensor]) -> list[Tensor]:
        """Compute ``sum_n (a_n^T V[...,k] a_n) g_n g_n^T`` for each column ``k``.

        Args:
            X: List of one tensor of shape ``(d_in, d_in, K)``.

        Returns:
            List of one tensor of shape ``(d_out, d_out, K)``.
        """
        (V,) = X
        coeffs = einsum(self._a, V, self._a, "n i, i j k, n j -> n k")
        M = einsum(coeffs, self._g, self._g, "n k, n o, n p -> o p k")
        return [M]

    def _adjoint(self) -> _RearrangedGGNLinearOperator:
        r"""Return the adjoint by swapping ``a`` and ``g``.

        The adjoint matvec computes
        :math:`\sum_n (g_n^\top U g_n)\,a_n a_n^\top` for input
        :math:`U \in \mathbb{R}^{d_\text{out} \times d_\text{out}}`.

        Returns:
            A new ``_RearrangedGGNLinearOperator`` with ``a`` and ``g`` swapped.
        """
        return _RearrangedGGNLinearOperator(a_per_sample=self._g, g_per_sample=self._a)

    @property
    def device(self) -> device:
        """Device of the cached per-sample tensors.

        Returns:
            The device of ``a_per_sample``.
        """
        return self._a.device

    @property
    def dtype(self) -> dtype:
        """Data type of the cached per-sample tensors.

        Returns:
            The dtype of ``a_per_sample``.
        """
        return self._a.dtype


def _top_rank_one_kron_factors(
    a_per_sample: Tensor, g_per_sample: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Compute Frobenius-optimal rank-1 Kronecker factors of a per-layer GGN block.

    Builds the rearranged operator from per-sample stacks and runs SciPy's
    truncated SVD on it. Returns :math:`G^\star = \sqrt{\sigma_1}\,\mathrm{unvec}(u_1)`
    and :math:`A^\star = \sqrt{\sigma_1}\,\mathrm{unvec}(v_1)` so that
    :math:`G^\star \otimes A^\star` is the best Frobenius rank-1 Kronecker
    approximation of :math:`B_l`.

    Factors are returned raw: not symmetrized, not PSD-projected. Generically
    they are symmetric PSD up to floating-point noise; in near-degenerate cases
    they may be indefinite.

    Args:
        a_per_sample: Per-sample activations, shape ``(N, d_in)``.
        g_per_sample: Per-sample output gradients, shape ``(N, d_out)``.

    Returns:
        Tuple ``(G_star, A_star)`` with shapes ``(d_out, d_out)`` and
        ``(d_in, d_in)``.
    """
    _, d_in = a_per_sample.shape
    _, d_out = g_per_sample.shape
    op = _RearrangedGGNLinearOperator(a_per_sample, g_per_sample)
    u, s, vt = svds(op.to_scipy(), k=1)
    sigma = float(s[0])
    scale = sqrt(max(sigma, 0.0))
    G_star = (
        from_numpy(u[:, 0].copy())
        .to(device=a_per_sample.device, dtype=a_per_sample.dtype)
        .reshape(d_out, d_out)
        .mul_(scale)
    )
    A_star = (
        from_numpy(vt[0].copy())
        .to(device=a_per_sample.device, dtype=a_per_sample.dtype)
        .reshape(d_in, d_in)
        .mul_(scale)
    )
    return G_star, A_star


class MakeFxKFOCComputer(_BaseKFACComputer):
    """KFOC computer using FX-based IO collection and per-layer truncated SVD.

    Collects per-layer activations and output gradients via
    :func:`make_compute_kfac_io_batch`, then computes the Frobenius-optimal
    rank-1 Kronecker factors for each parameter group via the top singular
    pair of the Van Loan rearrangement.

    Expects ``FisherType.TYPE2``, ``KFACType.EXPAND``, and single-batch data
    (single-batch enforcement is done at the :class:`KFOCLinearOperator` level).
    """

    def __init__(self, *args, **kwargs):
        """Initialize and require gradients on parameters for autograd."""
        super().__init__(*args, **kwargs)
        for p in self._params.values():
            p.requires_grad_(True)

    def compute(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Compute KFOC's Frobenius-optimal Kronecker factors.

        Runs one forward+backward pass to collect per-layer activations and
        output gradients, then for each parameter group computes the top
        rank-1 Kronecker approximation of the rearranged GGN block via svds.

        Returns:
            Tuple of ``(input_covariances, gradient_covariances, mapping)``.
        """
        X, y = next(iter(self._data))
        (
            inputs_and_grad_outputs_batch,
            mapping,
            io_groups,
            io_param_names,
            layer_hparams,
        ) = make_compute_kfac_io_batch(
            self._model_func,
            self._loss_func,
            self._params,
            X,
            self._fisher_type,
            self._mc_samples,
            self._separate_weight_and_bias,
        )

        layer_inputs, layer_output_grads = inputs_and_grad_outputs_batch(
            self._params, X, y
        )

        group_inputs, group_grads = make_group_gatherers(
            io_groups, io_param_names, layer_hparams, KFACType.EXPAND
        )

        input_covariances: dict[ParamGroupKey, Tensor] = {}
        gradient_covariances: dict[ParamGroupKey, Tensor] = {}

        for group in mapping:
            group_key = tuple(group.values())
            g = group_grads(group, layer_output_grads)
            V, N, T, d_out = g.shape
            g_per_sample = g.flatten(0, 2)

            if "W" in group:
                x = group_inputs(group, layer_inputs)
                d_in = x.shape[-1]
                a_flat = x.flatten(0, 1)
                # Inputs don't depend on the backward index v; broadcast across V
                a_per_sample = (
                    a_flat.unsqueeze(0).expand(V, -1, -1).reshape(V * N * T, d_in)
                )
                G_star, A_star = _top_rank_one_kron_factors(a_per_sample, g_per_sample)
                input_covariances[group_key] = A_star
                gradient_covariances[group_key] = G_star
            else:
                # Bias-only group: no input factor; accumulate gradient outer products
                ggT = einsum(g_per_sample, g_per_sample, "n i, n j -> i j")
                gradient_covariances[group_key] = ggT

        return input_covariances, gradient_covariances, mapping
