r"""KFOC computer and rearranged-GGN primitive.

Provides:

- :class:`_RearrangedGGNLinearOperator`: Van Loan rearrangement of a per-layer
  Gauss-Newton block, exposed as a rectangular PyTorch linear operator whose
  matvec consumes per-sample ``vec(W)`` gradients.
- :class:`MakeFxKFOCComputer`: FX-based computer that obtains per-sample
  activations and output gradients via :class:`LayerIO` with
  ``intermediate_as_batch=False`` and extracts Kronecker factors per layer via
  truncated SVD on the rearranged operator.
"""

from __future__ import annotations

from math import sqrt

from einops import einsum
from numpy import eye as np_eye
from numpy.linalg import svd as np_svd
from scipy.sparse.linalg import ArpackError, svds
from torch import Tensor, as_tensor, device, dtype, no_grad

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _BaseKFACComputer
from curvlinops.computers.io_collector import LayerIO
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _assert_single_element, _make_fx


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
        """
        _, _, d_out, d_in = per_sample_grads.shape
        in_shape = [(d_out, d_out)] if adjoint else [(d_in, d_in)]
        out_shape = [(d_in, d_in)] if adjoint else [(d_out, d_out)]
        super().__init__(in_shape=in_shape, out_shape=out_shape)
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

    Raises:
        ArpackError: If ``svds`` fails for a reason other than the
            zero-GGN case (i.e., real ARPACK convergence failure).
    """
    _, _, d_out, d_in = per_sample_grads.shape
    op = _RearrangedGGNLinearOperator(per_sample_grads)
    scipy_op = op.to_scipy()
    meta = {"dtype": op.dtype, "device": op.device}
    if d_out == 1 or d_in == 1:
        # ``svds`` requires ``k < min(shape)``; fall back to a dense SVD
        # materialized from a single matvec against the trivial (1×1) side.
        # Dense SVD handles the ``G = 0`` case directly (zero singular values).
        identity = np_eye(1, dtype=scipy_op.dtype)
        dense = scipy_op @ identity if d_in == 1 else identity @ scipy_op
        u, s, vt = np_svd(dense, full_matrices=False)
        u, s, vt = u[:, :1], s[:1], vt[:1, :]
    else:
        try:
            u, s, vt = svds(scipy_op, k=1)
        except ArpackError:
            # ``svds`` raises ARPACK error -9 ("starting vector is zero")
            # when ``G = 0`` (zero inputs or frozen upstream layer). The
            # optimal rank-one Kronecker approximation of zero is the zero
            # pair; re-raise on a real ARPACK failure (non-zero GGN).
            if per_sample_grads.any():
                raise
            zeros = per_sample_grads.new_zeros
            return zeros(d_out, d_out), zeros(d_in, d_in)
    scale = sqrt(float(s[0]))
    S_1 = as_tensor(u[:, 0], **meta).reshape(d_out, d_out).mul_(scale)
    S_2 = as_tensor(vt[0], **meta).reshape(d_in, d_in).mul_(scale)
    return S_1, S_2


class MakeFxKFOCComputer(_BaseKFACComputer):
    """KFOC computer: top-1 SVD on the rearranged per-layer GGN block.

    Collects per-sample activations and output gradients via :class:`LayerIO`
    with ``intermediate_as_batch=False``, forms per-sample ``vec(W)`` gradients,
    and extracts the Frobenius-optimal rank-1 Kronecker factors from the top
    singular pair of the rearranged operator.

    Requires ``FisherType.TYPE2``, ``KFACType.EXPAND``, and exactly one batch
    in ``data``. All three preconditions fail at construction.
    """

    _SUPPORTED_FISHER_TYPE: tuple[FisherType, ...] = (FisherType.TYPE2,)
    _SUPPORTED_KFAC_APPROX: tuple[KFACType, ...] = (KFACType.EXPAND,)

    def __init__(self, *args, **kwargs):
        """Validate single-batch data."""
        super().__init__(*args, **kwargs)
        _assert_single_element(self._data)

    def compute(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Compute KFOC's Frobenius-optimal Kronecker factors.

        Runs one forward+backward pass to collect per-layer activations and
        output gradients, forms per-sample ``vec(W)`` gradients, and extracts
        the top rank-one Kronecker factors via SVD per parameter group. For
        bias-only groups (no Kronecker structure), stores the exact bias GGN
        block, which is the Frobenius optimum for a single-factor approximation.

        Returns:
            Tuple of ``(input_covariances, gradient_covariances, mapping)``.
        """
        # Use ``_loop_over_data`` so ``X`` / ``y`` land on ``self.device``
        # — a plain ``next(iter(self._data))`` keeps them on the loader's
        # device, which fails when the model lives on CUDA but the loader
        # yields CPU tensors.
        X, y = next(iter(self._loop_over_data()))

        io = LayerIO(
            self._model_func,
            self._loss_func,
            self._params,
            X,
            fisher_type=self._fisher_type,
            mc_samples=self._mc_samples,
            kfac_approx=self._kfac_approx,
            separate_weight_and_bias=self._separate_weight_and_bias,
            intermediate_as_batch=False,
            batch_size_fn=self._batch_size_fn,
        )
        # Trace IO collection only (the SVD per group is non-traceable due to
        # ``svds`` + ARPACK error handling), then replay under ``no_grad`` to
        # keep the autograd-using portion contained inside the FX graph.
        # The wrapper hides ``io.populate``'s bound ``self`` from ``make_fx``,
        # which otherwise counts it as a tracing argument.

        def populate(params, X, y):
            return io.populate(params, X, y)

        with io.enable_param_grads(self._params):
            traced_populate = _make_fx(populate)(self._params, X, y)
        with no_grad():
            layer_inputs, layer_output_grads = traced_populate(self._params, X, y)
        snap = io.snapshot(layer_inputs, layer_output_grads)

        # ``S_1 (otimes) S_2`` per group; positional return matches the base
        # class slots (``input_covariances``, ``gradient_covariances``).
        first_factors: dict[ParamGroupKey, Tensor] = {}
        second_factors: dict[ParamGroupKey, Tensor] = {}
        for group in io.mapping:
            group_key = tuple(group.values())
            per_sample_grads = snap.per_sample_grads(group)
            if "W" in group:
                S_1, S_2 = _top_rank_one_kron_factors(per_sample_grads)
                first_factors[group_key] = S_1
                second_factors[group_key] = S_2
            else:
                # Bias-only block: the Frobenius optimum is the exact GGN
                # block. ``per_sample_grads`` already sums over the shared
                # axis, so the outer product runs straight on ``[V, B, d_out]``.
                first_factors[group_key] = einsum(
                    per_sample_grads,
                    per_sample_grads,
                    "vec batch row, vec batch col -> row col",
                )

        return second_factors, first_factors, io.mapping
