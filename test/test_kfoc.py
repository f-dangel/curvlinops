"""Tests for ``KFOCLinearOperator``.

Two tests cover correctness end-to-end:

1. **Factor reference**: compare KFOC's Kronecker factors to the top-1 dense
   SVD of the explicitly-materialized Van Loan rearrangement.
2. **First-order optimality**: verify the SVD stationarity conditions hold at
   the returned factors using only the rearranged operator's matvecs.
"""

from pytest import raises
from torch import einsum, manual_seed, rand
from torch.linalg import svd as torch_svd
from torch.nn import Linear, MSELoss, Sequential
from torch.testing import assert_close

from curvlinops import KFOCLinearOperator
from curvlinops.computers.kfac_make_fx import (
    make_compute_kfac_io_batch,
    make_group_gatherers,
)
from curvlinops.computers.kfoc_make_fx import (
    MakeFxKFOCComputer,
    _RearrangedGGNLinearOperator,
)
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import make_functional_call


def _setup_problem():
    """Build a small no-bias MLP + MSE loss + single batch.

    Returns:
        Tuple ``(model, loss_fn, params, X, y)``.
    """
    manual_seed(0)
    model = Sequential(Linear(4, 3, bias=False), Linear(3, 2, bias=False))
    loss_fn = MSELoss(reduction="sum")
    params = dict(model.named_parameters())
    X = rand(5, 4)
    y = rand(5, 2)
    return model, loss_fn, params, X, y


def _collect_per_sample(model, loss_fn, params, X, y):
    """Run KFAC's IO collector and return per-layer per-sample ``(a, g)`` stacks.

    Replicates the collection logic used inside :class:`MakeFxKFOCComputer`
    so tests can form the rearranged operator without relying on the computer
    itself.

    Returns:
        List of ``(group_key, a_per_sample, g_per_sample)`` tuples (one per
        weight-bearing parameter group).
    """
    model_func = make_functional_call(model)
    for p in params.values():
        p.requires_grad_(True)

    (
        inputs_and_grad_outputs_batch,
        mapping,
        io_groups,
        io_param_names,
        layer_hparams,
    ) = make_compute_kfac_io_batch(
        model_func,
        loss_fn,
        params,
        X,
        FisherType.TYPE2,
        1,
        True,
    )

    layer_inputs, layer_output_grads = inputs_and_grad_outputs_batch(params, X, y)

    group_inputs, group_grads = make_group_gatherers(
        io_groups, io_param_names, layer_hparams, KFACType.EXPAND
    )

    out = []
    for group in mapping:
        if "W" not in group:
            continue
        x = group_inputs(group, layer_inputs)
        g = group_grads(group, layer_output_grads)
        V, N, T, _ = g.shape
        g_per = g.flatten(0, 2)
        a_flat = x.flatten(0, 1)
        a_per = (
            a_flat.unsqueeze(0).expand(V, -1, -1).reshape(V * N * T, x.shape[-1])
        ).contiguous()
        out.append((tuple(group.values()), a_per, g_per))
    return out


def _dense_rearrangement(a_per, g_per):
    """Form the Van Loan rearrangement ``sum_n vec(gg^T) vec(aa^T)^T`` densely.

    Args:
        a_per: Per-sample activations, ``(N, d_in)``.
        g_per: Per-sample output gradients, ``(N, d_out)``.

    Returns:
        Dense rearrangement matrix of shape ``(d_out^2, d_in^2)``.
    """
    _, d_in = a_per.shape
    _, d_out = g_per.shape
    gg = einsum("ni,nj->nij", g_per, g_per).reshape(-1, d_out * d_out)
    aa = einsum("ni,nj->nij", a_per, a_per).reshape(-1, d_in * d_in)
    return gg.T @ aa


def _run_kfoc(model, loss_fn, params, X, y):
    """Run ``MakeFxKFOCComputer.compute()`` and return its factor dicts.

    Returns:
        Tuple ``(input_covariances, gradient_covariances, mapping)``.
    """
    computer = MakeFxKFOCComputer(
        make_functional_call(model),
        loss_fn,
        params,
        [(X, y)],
        check_deterministic=False,
        fisher_type=FisherType.TYPE2,
        kfac_approx=KFACType.EXPAND,
        separate_weight_and_bias=True,
    )
    return computer.compute()


def test_factors_match_dense_svd():
    """KFOC factors match the top-1 dense SVD of the rearranged block."""
    model, loss_fn, params, X, y = _setup_problem()
    input_covs, gradient_covs, _ = _run_kfoc(model, loss_fn, params, X, y)

    per_sample = _collect_per_sample(model, loss_fn, params, X, y)

    for group_key, a_per, g_per in per_sample:
        R = _dense_rearrangement(a_per, g_per)
        U, S, Vt = torch_svd(R, full_matrices=False)
        sigma = S[0].item()
        d_out = g_per.shape[1]
        d_in = a_per.shape[1]
        G_ref = (sigma**0.5) * U[:, 0].reshape(d_out, d_out)
        A_ref = (sigma**0.5) * Vt[0].reshape(d_in, d_in)

        G_kfoc = gradient_covs[group_key]
        A_kfoc = input_covs[group_key]

        # SVD vectors are defined up to a joint sign flip; the product is fixed.
        assert_close(
            einsum("ij,kl->ikjl", G_kfoc, A_kfoc),
            einsum("ij,kl->ikjl", G_ref, A_ref),
            atol=1e-5,
            rtol=1e-5,
        )


def test_first_order_optimality():
    r"""At the KFOC optimum, :math:`R\,\mathrm{vec}(A^\star) = \|A^\star\|_F^2 \mathrm{vec}(G^\star)` (and adjoint)."""
    model, loss_fn, params, X, y = _setup_problem()
    input_covs, gradient_covs, _ = _run_kfoc(model, loss_fn, params, X, y)

    per_sample = _collect_per_sample(model, loss_fn, params, X, y)

    for group_key, a_per, g_per in per_sample:
        A_star = input_covs[group_key]
        G_star = gradient_covs[group_key]
        op = _RearrangedGGNLinearOperator(a_per, g_per)

        A_norm_sq = (A_star * A_star).sum().item()
        G_norm_sq = (G_star * G_star).sum().item()

        # For a PyTorchLinearOperator with block shape [(d, d)], we pass a list
        # of one tensor; _matmat adds a trailing singleton K dimension for us
        # when we go through __matmul__ with a flat tensor, so we flatten first.
        out_vec = op @ A_star.flatten()
        expected = A_norm_sq * G_star.flatten()
        assert_close(out_vec, expected, atol=1e-5, rtol=1e-5)

        out_vec_adj = op.adjoint() @ G_star.flatten()
        expected_adj = G_norm_sq * A_star.flatten()
        assert_close(out_vec_adj, expected_adj, atol=1e-5, rtol=1e-5)


def test_rejects_multi_batch():
    """Constructor raises when data does not contain exactly one batch."""
    model, loss_fn, params, X, y = _setup_problem()

    with raises(ValueError, match="got more than one"):
        KFOCLinearOperator(
            model,
            loss_fn,
            params,
            [(X, y), (X, y)],
            check_deterministic=False,
        )
