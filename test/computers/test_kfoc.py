r"""Tests for :class:`KFOCLinearOperator` and its FX computer.

Two semantic checks plus a few guards:

1. **Factor reference** (``test_kfoc_factors_match_dense_svd``): KFOC's
   per-layer ``(S_1, S_2)`` match the top-1 dense SVD of the Van Loan
   rearrangement of the GGN block (the closed-form Frobenius minimizer).
2. **First-order optimality** (``test_kfoc_first_order_optimality``):
   KFOC's ``(S_1, S_2)`` are stationary points of
   ``||G_l - S_1 ⊗ S_2||_F²``, verified via autograd.
3. Rank-one recovery, zero handling, multi-batch rejection, scalar-layer
   fallback.
"""

from collections.abc import Iterable

from pytest import mark, raises
from torch import Tensor, float64, kron, manual_seed, rand, zeros
from torch.autograd import grad
from torch.linalg import svd as torch_svd
from torch.nn import Linear, Module, MSELoss, Sequential

from curvlinops import GGNLinearOperator, KFOCLinearOperator
from curvlinops.utils import allclose_report
from test.utils import block_diagonal, change_dtype, eye_like


def _assert_kfoc_factors_match_dense_svd(case):
    """Assert KFOC's per-layer Kron product matches the top-1 SVD of ``R(G_l)``.

    Works in canonical layout: ``K_canonical`` (the middle of ``kfoc``'s
    chain) is block-diagonal with one block per parameter group, and
    ``PT @ ggn @ P`` brings the GGN into the same basis. Block iteration
    on ``K_op`` then lines up directly with diagonal slices of
    ``ggn_canonical`` regardless of ``separate_weight_and_bias``.

    Args:
        case: Model, loss, parameters, data, and optional batch-size function.
    """
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    X, y = next(iter(data))

    kfoc = KFOCLinearOperator(
        model,
        loss_func,
        params,
        [(X, y)],
        check_deterministic=False,
        batch_size_fn=batch_size_fn,
    )
    _, K_op, PT = kfoc
    K_canonical = K_op @ eye_like(K_op)
    PT_mat = PT @ eye_like(PT)
    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        [(X, y)],
        batch_size_fn=batch_size_fn,
    )
    ggn_canonical = PT_mat @ ggn @ PT_mat.T

    offset = 0
    for block in K_op:
        if len(block) == 2:  # weight (or joint W+b) block: factors (S_1, S_2)
            d_out, d_in = block[0].shape[0], block[1].shape[0]
            n = d_out * d_in
            G_l = ggn_canonical[offset : offset + n, offset : offset + n]
            R = (
                G_l
                .reshape(d_out, d_in, d_out, d_in)
                .movedim(1, 2)
                .reshape(d_out**2, d_in**2)
            )
            U, S, Vh = torch_svd(R, full_matrices=False)
            scale = S[0].sqrt()
            S_1_ref = scale * U[:, 0].reshape(d_out, d_out)
            S_2_ref = scale * Vh[0].reshape(d_in, d_in)
            K_l = K_canonical[offset : offset + n, offset : offset + n]
            # Individual factor signs are joint-arbitrary, but their Kron product is not.
            assert allclose_report(K_l, kron(S_1_ref, S_2_ref))
            offset += n
        else:  # bias-only block: single factor (no SVD reference)
            offset += block[0].shape[0]


def _assert_kfoc_first_order_optimality(case):
    """Assert KFOC's weight factors are stationary points of the Frobenius residual.

    Enable grad on every weight Kronecker factor, materialize KFOC and the
    block-diagonal GGN in the standard (params) basis, and check that the
    gradient of ``||GGN - K||_F²`` w.r.t. each factor is near zero. Both
    sides are block-diagonal at the parameter-group level so the residual
    decouples per block, regardless of ``separate_weight_and_bias``.

    Args:
        case: Model, loss, parameters, data, and optional batch-size function.
    """
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    X, y = next(iter(data))

    kfoc = KFOCLinearOperator(
        model,
        loss_func,
        params,
        [(X, y)],
        check_deterministic=False,
        batch_size_fn=batch_size_fn,
    )
    _, K_op, _ = kfoc
    weight_S = [
        f.requires_grad_(True) for block in K_op if len(block) == 2 for f in block
    ]
    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        [(X, y)],
        batch_size_fn=batch_size_fn,
    )
    K = kfoc @ eye_like(kfoc)
    loss = (ggn - K).pow(2).sum()
    for g in grad(loss, weight_S):
        assert g.abs().max().item() < 1e-8


def test_kfoc_factors_match_dense_svd(
    case: tuple[
        Module,
        Module,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
        object,
    ],
):
    """KFOC's per-layer Kron product matches the top-1 SVD of ``R(G_l)`` (Linear).

    The Eckart-Young theorem on the Van Loan rearrangement is the
    closed-form Frobenius minimizer; reshape the GGN block to
    ``R(G_l) ∈ R^(d_out² × d_in²)``, take the top-1 SVD, and check that
    KFOC's Kron output matches.

    Args:
        case: Model, loss, parameters, data, and optional batch-size function.
    """
    _assert_kfoc_factors_match_dense_svd(case)


def test_kfoc_factors_match_dense_svd_cnn(
    cnn_case: tuple[
        Module,
        Module,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
        object,
    ],
):
    """Same as :func:`test_kfoc_factors_match_dense_svd`, on CNN models with Conv2d.

    Generalizes the dense-SVD oracle to ``ndim >= 2`` weights — a Conv2d
    weight ``(C_out, C_in, kH, kW)`` matricizes to ``(C_out, C_in*kH*kW)``
    for the Van Loan rearrangement.

    Args:
        cnn_case: Model, loss, parameters, data, and optional batch-size function.
    """
    _assert_kfoc_factors_match_dense_svd(cnn_case)


def test_kfoc_first_order_optimality(
    case: tuple[
        Module,
        Module,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
        object,
    ],
):
    """KFOC's weight factors are stationary points of the Frobenius residual (Linear).

    Enable grad on every Kronecker factor of a weight block, materialize
    KFOC to a dense matrix, and check that the gradient of
    ``||GGN - K||_F²`` w.r.t. each factor is near zero. ``K`` is
    block-diagonal so the residual decouples per block. Bias-only blocks
    are skipped: KFOC stores the ``t``-diagonal of the bias GGN as the
    sole factor, which is not the Frobenius minimum under weight sharing.

    Args:
        case: Model, loss, parameters, data, and optional batch-size function.
    """
    _assert_kfoc_first_order_optimality(case)


def test_kfoc_first_order_optimality_cnn(
    cnn_case: tuple[
        Module,
        Module,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
        object,
    ],
):
    """Same as :func:`test_kfoc_first_order_optimality`, on CNN models with Conv2d.

    Args:
        cnn_case: Model, loss, parameters, data, and optional batch-size function.
    """
    _assert_kfoc_first_order_optimality(cnn_case)


def test_kfoc_handles_zero_ggn():
    """Zero ``G`` (e.g. all-zero inputs) short-circuits to zero factors.

    ``svds`` otherwise fails with ARPACK error -9 ("starting vector is
    zero") on a zero operator.
    """
    manual_seed(0)
    model = Sequential(Linear(4, 2, bias=False)).double()
    loss_func = MSELoss(reduction="sum")
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    X = zeros(3, 4, dtype=float64)
    y = rand(3, 2, dtype=float64)

    kfoc = KFOCLinearOperator(
        model, loss_func, params, [(X, y)], check_deterministic=False
    )
    K = kfoc @ eye_like(kfoc)
    ggn = block_diagonal(GGNLinearOperator, model, loss_func, params, [(X, y)])
    assert allclose_report(K, ggn)
    assert K.abs().max().item() == 0.0


def test_kfoc_rejects_multi_batch():
    """Multi-batch input raises at factor computation."""
    manual_seed(0)
    model = Sequential(Linear(4, 2, bias=False))
    loss_func = MSELoss()
    params = dict(model.named_parameters())
    data = [(rand(2, 4), rand(2, 2)), (rand(3, 4), rand(3, 2))]
    with raises(ValueError, match="more than one"):
        KFOCLinearOperator(model, loss_func, params, data, check_deterministic=False)


@mark.parametrize("d_in,d_out", [(4, 1), (1, 3)], ids=["scalar_out", "scalar_in"])
def test_kfoc_handles_degenerate_svds_shapes(d_in: int, d_out: int):
    """``svds`` requires ``k < min(shape)``; the dense fallback covers scalar ends.

    Single-linear-layer scenarios where the rearranged operator has a
    trivial dimension (``d_out**2 == 1`` or ``d_in**2 == 1``) and
    ``scipy.svds`` cannot handle ``k = 1``.

    Args:
        d_in: Input feature dimension.
        d_out: Output feature dimension.
    """
    manual_seed(0)
    model = Sequential(Linear(d_in, d_out, bias=False)).double()
    loss_func = MSELoss(reduction="sum")
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    X = rand(3, d_in, dtype=float64)
    y = rand(3, d_out, dtype=float64)
    kfoc = KFOCLinearOperator(
        model,
        loss_func,
        params,
        [(X, y)],
        check_deterministic=False,
    )
    ggn = block_diagonal(GGNLinearOperator, model, loss_func, params, [(X, y)])
    assert allclose_report(kfoc @ eye_like(kfoc), ggn)
