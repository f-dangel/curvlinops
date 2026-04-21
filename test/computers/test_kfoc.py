r"""Tests for :class:`KFOCLinearOperator` and its FX computer.

Checks three things:

1. **Factor reference**: KFOC's ``(G_star, A_star)`` match the top-1 dense SVD
   of the materialized rearranged per-layer GGN block.
2. **Rank-one recovery**: when ``B_l = G (otimes) A`` exactly (single linear
   layer, no shared axes, one backward vector), KFOC recovers ``(G, A)`` up
   to a joint sign flip and scale.
3. **First-order optimality**: SVD stationarity
   ``R(B_l) vec(A_star) = sigma_1 vec(G_star)`` and
   ``R(B_l)^T vec(G_star) = sigma_1 vec(A_star)`` holds using only the
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
    """KFOC's ``(G_star, A_star)`` match the top-1 SVD of the dense rearrangement.

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
            # Dense top-1 Kron reference: materialize R(B_l) and take top-1 SVD.
            P = per_sample_grads[(name,)]
            R = torch_einsum("vnoi,vnpj->opij", P, P).reshape(d_out**2, d_in**2)
            U, S, Vh = torch_svd(R, full_matrices=False)
            scale = S[0].sqrt()
            G_ref = scale * U[:, 0].reshape(d_out, d_out)
            A_ref = scale * Vh[0].reshape(d_in, d_in)
            K_l = K[offset : offset + n, offset : offset + n]
            # Compare the Kron product, not individual factors: sign of
            # (G, A) is joint-arbitrary but their Kron product is not.
            assert allclose_report(K_l, kron(G_ref, A_ref))
        offset += n


def test_kfoc_recovers_exact_rank_one_kron():
    """On a constructed ``B_l = G (otimes) A`` problem, KFOC recovers the factors.

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
    # The GGN block for a single linear layer with scalar output is exactly
    # 2 * sum_n a_n a_n^T (rank-one Kron in the form G (otimes) A with G=[2]).
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

    For each layer's ``(G_star, A_star)``:
        ``R(B_l) vec(A_star) ≈ ||A_star||_F^2 vec(G_star)``
        ``R(B_l)^T vec(G_star) ≈ ||G_star||_F^2 vec(A_star)``

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
            # Extract G_star, A_star from the KFOC Kronecker block K_l
            K_l = K[offset : offset + n, offset : offset + n]
            # K_l = G_star (kron) A_star → un-kron via rearrangement + rank-1 SVD
            R_K = (
                K_l
                .reshape(d_out, d_in, d_out, d_in)
                .movedim(1, 2)
                .reshape(d_out**2, d_in**2)
            )
            U, S, Vh = torch_svd(R_K, full_matrices=False)
            scale = S[0].sqrt()
            G_star = scale * U[:, 0].reshape(d_out, d_out)
            A_star = scale * Vh[0].reshape(d_in, d_in)

            P = per_sample_grads[(name,)]
            op = _RearrangedGGNLinearOperator(P)
            R_A = op._matmat([A_star.unsqueeze(-1)])[0].squeeze(-1)
            R_G = op._adjoint()._matmat([G_star.unsqueeze(-1)])[0].squeeze(-1)

            assert allclose_report(R_A, A_star.pow(2).sum() * G_star)
            assert allclose_report(R_G, G_star.pow(2).sum() * A_star)
        offset += n


def test_kfoc_factors_are_psd(
    case: tuple[
        Module,
        Module,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
        object,
    ],
):
    """KFOC applies a trace-based gauge to yield PSD factors for PSD ``B_l``.

    For the GGN blocks the ``case`` fixture produces, the factors are either
    jointly PSD or jointly NSD from the SVD; the trace convention in
    ``_top_rank_one_kron_factors`` flips to the PSD pair. This test asserts
    the expected side of the gauge.

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
    for key, A in input_covariances.items():
        G = gradient_covariances[key]
        # PSD (up to float64 eigenvalue noise); symmetrize to strip float asymmetry.
        assert torch.linalg.eigvalsh((A + A.T) / 2).min().item() > -1e-10
        assert torch.linalg.eigvalsh((G + G.T) / 2).min().item() > -1e-10


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
