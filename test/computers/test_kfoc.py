r"""Tests for :class:`KFOCLinearOperator` and its FX computer.

Checks three things:

1. **Factor reference**: KFOC's ``(S_1, S_2)`` match the top-1 dense SVD
   of the materialized rearranged per-layer GGN block.
2. **Rank-one recovery**: when ``G = S_1 (otimes) S_2`` exactly (single
   linear layer, no shared axes, one backward vector), KFOC recovers
   ``(S_1, S_2)`` up to a joint sign flip and scale.
3. **First-order optimality**: SVD stationarity
   ``R(G) vec(S_2) = sigma_1 vec(S_1)`` and
   ``R(G)^T vec(S_1) = sigma_1 vec(S_2)`` holds using only the
   operator's matvecs.
"""

from collections.abc import Iterable

import torch
from einops import einsum
from pytest import mark, raises
from torch import Tensor, float64, kron, manual_seed, rand
from torch import einsum as torch_einsum
from torch.linalg import svd as torch_svd
from torch.nn import Linear, Module, MSELoss, Sequential

from curvlinops import GGNLinearOperator, KFOCLinearOperator
from curvlinops.computers.kfac_make_fx import (
    make_compute_kfac_io_batch,
    make_group_gatherers,
)
from curvlinops.computers.kfoc_make_fx import (
    MakeFxKFOCComputer,
    _RearrangedGGNLinearOperator,
)
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import allclose_report, make_functional_call
from test.utils import block_diagonal, change_dtype, eye_like


def _collect_per_sample_grads(
    model: Module,
    loss_func: Module,
    params: dict[str, Tensor],
    X,
    y: Tensor,
) -> dict[tuple[str, ...], Tensor]:
    """Replay the KFOC collection path and return ``P_{v, n}`` per group.

    Args:
        model: Neural network (or callable).
        loss_func: Loss function.
        params: Named model parameters.
        X: Input batch.
        y: Target batch.

    Returns:
        Dict mapping parameter group keys to the per-sample ``vec(W)``
        gradient stack ``P`` of shape ``(V, N, d_out, d_in)``.
    """
    for p in params.values():
        p.requires_grad_(True)
    model_func = make_functional_call(model) if isinstance(model, Module) else model
    fn, mapping, io_groups, io_pnames, layer_hparams = make_compute_kfac_io_batch(
        model_func,
        loss_func,
        params,
        X,
        FisherType.TYPE2,
        intermediate_as_batch=False,
    )
    layer_inputs, layer_output_grads = fn(params, X, y)
    group_inputs, group_grads = make_group_gatherers(
        io_groups, io_pnames, layer_hparams, KFACType.EXPAND
    )
    per_sample_grads: dict[tuple[str, ...], Tensor] = {}
    for group in mapping:
        if "W" not in group:
            continue
        g = group_grads(group, layer_output_grads)
        a = group_inputs(group, layer_inputs)
        per_sample_grads[tuple(group.values())] = einsum(
            g, a, "vec batch shared out, batch shared inp -> vec batch out inp"
        )
    return per_sample_grads


def test_kfoc_factors_match_dense_svd(
    case: tuple[
        Module,
        Module,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
        object,
    ],
):
    """KFOC's ``(S_1, S_2)`` match the top-1 SVD of the dense rearrangement.

    Parametrized over the ``GGNLinearOperator`` fixture so CE/BCE/MSE, both
    reductions, 2D/3D/dict-style inputs are covered.

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

    per_sample_grads = _collect_per_sample_grads(model, loss_func, params, X, y)

    offset = 0
    for name, p in params.items():
        n = p.numel()
        if p.ndim == 2:
            d_out, d_in = p.shape
            P = per_sample_grads[(name,)]
            R = torch_einsum("vnoi,vnpj->opij", P, P).reshape(d_out**2, d_in**2)
            U, S, Vh = torch_svd(R, full_matrices=False)
            scale = S[0].sqrt()
            S_1_ref = scale * U[:, 0].reshape(d_out, d_out)
            S_2_ref = scale * Vh[0].reshape(d_in, d_in)
            K_l = K[offset : offset + n, offset : offset + n]
            # Individual factor signs are joint-arbitrary, but their Kron product is not.
            assert allclose_report(K_l, kron(S_1_ref, S_2_ref))
        offset += n


def test_kfoc_recovers_exact_rank_one_kron():
    """On a constructed ``G = S_1 (otimes) S_2`` problem, KFOC recovers the factors.

    Single linear layer + MSE with a 1D output and a single batch — the
    GGN block is exactly a rank-one Kronecker product, so KFOC's
    approximation error should be at machine precision.
    """
    manual_seed(0)
    model = Sequential(Linear(4, 1, bias=False))
    loss_func = MSELoss(reduction="sum")
    params = {n: p.double() for n, p in model.named_parameters() if p.requires_grad}
    model.double()
    X = rand(3, 4, dtype=float64)
    y = rand(3, 1, dtype=float64)

    kfoc = KFOCLinearOperator(
        model,
        loss_func,
        params,
        [(X, y)],
        check_deterministic=False,
    )
    K = kfoc @ eye_like(kfoc)

    ggn = block_diagonal(GGNLinearOperator, model, loss_func, params, [(X, y)])
    assert allclose_report(K, ggn)


def test_kfoc_first_order_optimality(
    case: tuple[
        Module,
        Module,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
        object,
    ],
):
    """Verify SVD stationarity at the returned factors via operator matvecs only.

    For each layer's ``(S_1, S_2)``:
        ``R(G) vec(S_2) ≈ ||S_2||_F^2 vec(S_1)``
        ``R(G)^T vec(S_1) ≈ ||S_1||_F^2 vec(S_2)``

    Args:
        case: Model, loss, parameters, data, and optional batch-size function.
    """
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    X, y = next(iter(data))

    computer = MakeFxKFOCComputer(
        model,
        loss_func,
        params,
        [(X, y)],
        progressbar=False,
        check_deterministic=False,
        fisher_type=FisherType.TYPE2,
        mc_samples=1,
        kfac_approx=KFACType.EXPAND,
        batch_size_fn=batch_size_fn,
    )
    input_covariances, gradient_covariances, _ = computer.compute()
    per_sample_grads = _collect_per_sample_grads(model, loss_func, params, X, y)

    for key, S_2 in input_covariances.items():
        S_1 = gradient_covariances[key]
        op = _RearrangedGGNLinearOperator(per_sample_grads[key])
        [R_S_2] = op @ [S_2]
        [Rt_S_1] = op.adjoint() @ [S_1]
        assert allclose_report(R_S_2, S_2.pow(2).sum() * S_1)
        assert allclose_report(Rt_S_1, S_1.pow(2).sum() * S_2)


def test_kfoc_handles_zero_ggn():
    """Zero ``G`` (e.g. all-zero inputs) short-circuits to zero factors.

    ``svds`` otherwise fails with ARPACK error -9 ("starting vector is
    zero") on a zero operator.
    """
    manual_seed(0)
    model = Sequential(Linear(4, 2, bias=False)).double()
    loss_func = MSELoss(reduction="sum")
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    X = torch.zeros(3, 4, dtype=float64)
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
