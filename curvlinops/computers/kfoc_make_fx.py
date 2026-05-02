r"""KFOC computer and rearranged-GGN primitive.

Provides:

- :class:`_RearrangedGGNLinearOperator`: Van Loan rearrangement of a per-layer
  Gauss-Newton block, exposed as a rectangular PyTorch linear operator whose
  matvec consumes per-sample ``vec(W)`` gradients.
- :class:`MakeFxKFOCComputer`: FX-based computer that obtains per-sample
  activations and output gradients via :func:`make_compute_kfac_io_batch`
  with ``intermediate_as_batch=False``, forms per-sample ``vec(W)`` gradients,
  and extracts Kronecker factors per layer via truncated SVD on the
  rearranged operator.
"""

from __future__ import annotations

from math import sqrt

from einops import einsum
from numpy import eye as np_eye
from numpy.linalg import svd as np_svd
from scipy.sparse.linalg import svds
from torch import Tensor, as_tensor, device, dtype

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _BaseKFACComputer
from curvlinops.computers.kfac_make_fx import (
    make_compute_kfac_io_batch,
    make_group_gatherers,
)
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _assert_single_element


class _RearrangedGGNLinearOperator(PyTorchLinearOperator):
    r"""Van Loan rearrangement of a per-layer Gauss-Newton block.

    Operates on a stack of per-sample ``vec(W)`` gradients
    :math:`P_{v, n} \in \mathbb{R}^{d_\text{out} \times d_\text{in}}`
    with two extra axes beyond the layer's weight shape:

    - :math:`n \in \{1, \dots, N\}` indexes the sample in the batch
      (per-sample, not summed, because each sample contributes a separate
      rank-one outer product to the GGN), and
    - :math:`v \in \{1, \dots, V\}` indexes the backpropagated direction.
      :math:`V` equals the model's per-datum output count for the type-2
      Fisher (one direction per column of a per-datum loss Hessian square
      root), the number of MC samples for the MC-Fisher, and
      :math:`1` for the empirical Fisher.

    The per-layer GGN block is

    .. math::
        \mathbf{G} = \sum_{v, n} \mathrm{vec}(P_{v, n})\,
        \mathrm{vec}(P_{v, n})^\top,

    and its Van Loan rearrangement
    :math:`\mathcal{R}(\mathbf{G}) \in \mathbb{R}^{d_\text{out}^2 \times d_\text{in}^2}`
    acts on a matrix :math:`M` (respectively :math:`U` for the adjoint) as

    .. math::
        \mathcal{R}(\mathbf{G})\,\mathrm{vec}(M)
        &= \mathrm{vec}\!\left(\sum_{v, n} P_{v, n}\,M\,P_{v, n}^\top\right),
        \\
        \mathcal{R}(\mathbf{G})^\top\,\mathrm{vec}(U)
        &= \mathrm{vec}\!\left(\sum_{v, n} P_{v, n}^\top\,U\,P_{v, n}\right).

    Factors from the top singular pair reconstruct the Frobenius-optimal
    rank-one Kronecker approximation of :math:`\mathbf{G}`.
    """

    def __init__(self, per_sample_grads: Tensor, adjoint: bool = False):
        r"""Store per-sample ``vec(W)`` gradients.

        Args:
            per_sample_grads: Per-sample ``vec(W)`` gradients, shape
                ``(V, N, d_out, d_in)``.
            adjoint: Whether this instance represents :math:`\mathcal{R}(\mathbf{G})^\top`
                (``True``) or :math:`\mathcal{R}(\mathbf{G})` (``False``, default).

        Raises:
            ValueError: If ``per_sample_grads`` is not 4D.
        """
        if per_sample_grads.ndim != 4:
            raise ValueError(
                "per_sample_grads must be 4D (V, N, d_out, d_in), got shape "
                f"{tuple(per_sample_grads.shape)}."
            )
        _, _, d_out, d_in = per_sample_grads.shape
        if adjoint:
            super().__init__(in_shape=[(d_out, d_out)], out_shape=[(d_in, d_in)])
        else:
            super().__init__(in_shape=[(d_in, d_in)], out_shape=[(d_out, d_out)])
        self._P = per_sample_grads
        self._is_adjoint = adjoint

    def _matmat(self, X: list[Tensor]) -> list[Tensor]:
        """Apply the rearranged operator (or its adjoint) to a stack of matrices.

        Args:
            X: Single-element list with a tensor of shape ``(d_in, d_in, K)``
                (forward) or ``(d_out, d_out, K)`` (adjoint).

        Returns:
            Single-element list with the output matrix stack.
        """
        (M,) = X
        equation = (
            "vec batch out_row in_row, out_row out_col col, "
            "vec batch out_col in_col -> in_row in_col col"
            if self._is_adjoint
            else "vec batch out_row in_row, in_row in_col col, "
            "vec batch out_col in_col -> out_row out_col col"
        )
        return [einsum(self._P, M, self._P, equation)]

    def _adjoint(self) -> _RearrangedGGNLinearOperator:
        """Return the adjoint operator sharing the same ``P``.

        Returns:
            Adjoint operator (``R(G)^T``).
        """
        return type(self)(self._P, adjoint=not self._is_adjoint)

    @property
    def device(self) -> device:
        """Device of ``per_sample_grads``."""
        return self._P.device

    @property
    def dtype(self) -> dtype:
        """Data type of ``per_sample_grads``."""
        return self._P.dtype


def _top_rank_one_kron_factors(
    per_sample_grads: Tensor,
) -> tuple[Tensor, Tensor]:
    r"""Compute Frobenius-optimal rank-1 Kronecker factors of a per-layer GGN block.

    Runs the top-1 SVD of the rearranged GGN operator and reshapes the
    singular vectors into Kronecker factors. For scalar-input or
    scalar-output layers (``d_in == 1`` or ``d_out == 1``) ``svds`` cannot
    handle ``k = 1`` and we fall back to a dense SVD.

    Args:
        per_sample_grads: Per-sample ``vec(W)`` gradients, shape
            ``(V, N, d_out, d_in)``.

    Returns:
        Tuple ``(S_1, S_2)`` with shapes ``(d_out, d_out)`` and
        ``(d_in, d_in)`` such that ``S_1 (otimes) S_2`` is the best
        Frobenius rank-one Kronecker approximation of :math:`\mathbf{G}`.
        Factors come straight from the SVD reshape: not symmetrized,
        sign-normalized, or PSD-projected.
    """
    _, _, d_out, d_in = per_sample_grads.shape
    if not per_sample_grads.any():
        # ``G = 0`` (e.g., zero inputs or frozen upstream layer): the
        # optimal rank-one Kronecker approximation is the zero pair.
        # Short-circuit because ``svds`` raises ARPACK error -9 (zero
        # starting vector) on a zero operator.
        zeros = per_sample_grads.new_zeros
        return zeros(d_out, d_out), zeros(d_in, d_in)
    op = _RearrangedGGNLinearOperator(per_sample_grads)
    scipy_op = op.to_scipy()
    if d_out == 1 or d_in == 1:
        # ``svds`` requires ``k < min(shape)``; fall back to a dense SVD
        # materialized from a single matvec against the trivial (1×1) side.
        identity = np_eye(1, dtype=scipy_op.dtype)
        dense = scipy_op @ identity if d_in == 1 else scipy_op.rmatmat(identity).T
        u, s, vt = np_svd(dense, full_matrices=False)
        u, s, vt = u[:, :1], s[:1], vt[:1, :]
    else:
        u, s, vt = svds(scipy_op, k=1)
    scale = sqrt(float(s[0]))
    S_1 = (
        as_tensor(u[:, 0], dtype=op.dtype, device=op.device)
        .reshape(d_out, d_out)
        .mul_(scale)
    )
    S_2 = (
        as_tensor(vt[0], dtype=op.dtype, device=op.device)
        .reshape(d_in, d_in)
        .mul_(scale)
    )
    return S_1, S_2


class MakeFxKFOCComputer(_BaseKFACComputer):
    """KFOC computer: top-1 SVD on the rearranged per-layer GGN block.

    Collects per-sample activations and output gradients via the unflattened
    IO collector (``intermediate_as_batch=False``), forms per-sample
    ``vec(W)`` gradients, and extracts the Frobenius-optimal rank-1 Kronecker
    factors from the top singular pair of the rearranged operator.

    Requires ``FisherType.TYPE2``, ``KFACType.EXPAND``, and exactly one batch
    in ``data``. The single-batch check runs in :meth:`compute`.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and turn on grad tracking for the parameters."""
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
        output gradients, forms per-sample ``vec(W)`` gradients, and extracts
        the top rank-one Kronecker factors via SVD per parameter group.
        Bias-only groups reuse KFAC's gradient covariance (the formulas
        coincide when there is no input factor).

        Returns:
            Tuple of ``(input_covariances, gradient_covariances, mapping)``.

        Raises:
            ValueError: If the data loader yields more than one batch, or if
                the fisher type or KFAC approximation type are not supported.
        """
        if self._fisher_type != FisherType.TYPE2:
            raise ValueError(
                f"KFOC requires FisherType.TYPE2, got {self._fisher_type!r}."
            )
        if self._kfac_approx != KFACType.EXPAND:
            raise ValueError(
                f"KFOC requires KFACType.EXPAND, got {self._kfac_approx!r}."
            )
        _assert_single_element(self._data)
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
            fisher_type=self._fisher_type,
            mc_samples=self._mc_samples,
            separate_weight_and_bias=self._separate_weight_and_bias,
            intermediate_as_batch=False,
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
            if "W" in group:
                a = group_inputs(group, layer_inputs)
                per_sample_grads = einsum(
                    g, a, "vec batch shared out, batch shared inp -> vec batch out inp"
                )
                S_1, S_2 = _top_rank_one_kron_factors(per_sample_grads)
                input_covariances[group_key] = S_2
                gradient_covariances[group_key] = S_1
            else:
                gradient_covariances[group_key] = einsum(
                    g, g, "vec batch shared row, vec batch shared col -> row col"
                )

        return input_covariances, gradient_covariances, mapping
