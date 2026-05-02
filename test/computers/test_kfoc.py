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
from torch.linalg import svd as torch_svd
from torch.nn import Linear, Module, MSELoss, Sequential

from curvlinops import GGNLinearOperator, KFOCLinearOperator
from curvlinops.utils import allclose_report
from test.utils import block_diagonal, change_dtype, eye_like


def test_kfoc_factors_match_dense_svd(
    case: tuple[
        Module,
        Module,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
        object,
    ],
):
    """KFOC's per-layer Kron product matches the top-1 SVD of ``R(G_l)``.

    The Eckart-Young theorem on the Van Loan rearrangement is the
    closed-form Frobenius minimizer; reshape the GGN block to
    ``R(G_l) ∈ R^(d_out² × d_in²)``, take the top-1 SVD, and check that
    KFOC's Kron output matches.

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
    K = kfoc @ eye_like(kfoc)
    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        [(X, y)],
        batch_size_fn=batch_size_fn,
    )

    offset = 0
    for _, p in params.items():
        n = p.numel()
        if p.ndim == 2:
            d_out, d_in = p.shape
            G_l = ggn[offset : offset + n, offset : offset + n]
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
            K_l = K[offset : offset + n, offset : offset + n]
            # Individual factor signs are joint-arbitrary, but their Kron product is not.
            assert allclose_report(K_l, kron(S_1_ref, S_2_ref))
        offset += n


def test_kfoc_first_order_optimality(
    case: tuple[
        Module,
        Module,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
        object,
    ],
):
    """KFOC's factors are stationary points of the Frobenius residual.

    Set ``requires_grad=True`` on KFOC's per-layer ``(S_1, S_2)``, expand to
    ``S_1 ⊗ S_2``, compute the residual to the true GGN block, and assert
    the autograd gradients are near zero.

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
    _, K, _ = kfoc
    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        [(X, y)],
        batch_size_fn=batch_size_fn,
    )

    offset = 0
    for block in K:
        if len(block) == 2:  # weight block: factors (S_1, S_2)
            S_1, S_2 = (f.detach().requires_grad_(True) for f in block)
            n = S_1.shape[0] * S_2.shape[0]
            G_l = ggn[offset : offset + n, offset : offset + n]
            (G_l - kron(S_1, S_2)).pow(2).sum().backward()
            assert S_1.grad.abs().max().item() < 1e-8
            assert S_2.grad.abs().max().item() < 1e-8
            offset += n
        else:  # bias-only block: single factor of size d_out × d_out
            offset += block[0].shape[0]


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


@mark.parametrize(
    "d_in,d_out",
    [(4, 1), (1, 3)],
    ids=["scalar_out", "scalar_in"],
)
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
