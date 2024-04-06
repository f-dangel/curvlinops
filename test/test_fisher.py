"""Contains tests for ``curvlinops/fisher.py``."""

from contextlib import suppress

from numpy import random, zeros_like
from pytest import mark, raises

from curvlinops import FisherMCLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.examples.utils import report_nonclose

MAX_REPEATS_MC_SAMPLES = [(1_000_000, 1), (10_000, 100)]
MAX_REPEATS_MC_SAMPLES_IDS = [
    f"max_repeats={n}-mc_samples={m}" for (n, m) in MAX_REPEATS_MC_SAMPLES
]
CHECK_EVERY = 1_000


@mark.montecarlo
@mark.parametrize(
    "max_repeats,mc_samples", MAX_REPEATS_MC_SAMPLES, ids=MAX_REPEATS_MC_SAMPLES_IDS
)
def test_LinearOperator_matvec_expectation(
    case, adjoint: bool, max_repeats: int, mc_samples: int
):
    F = FisherMCLinearOperator(*case, mc_samples=mc_samples)
    G_functorch = functorch_ggn(*case).detach().cpu().numpy()
    if adjoint:
        F, G_functorch = F.adjoint(), G_functorch.conj().T

    x = random.rand(F.shape[1]).astype(F.dtype)
    Gx = G_functorch @ x

    Fx = zeros_like(x)
    atol, rtol = 1e-5, 1e-1

    for m in range(max_repeats):
        Fx += F @ x
        F._seed += 1

        total_samples = (m + 1) * mc_samples
        if total_samples % CHECK_EVERY == 0:
            with suppress(ValueError):
                report_nonclose(Fx / (m + 1), Gx, rtol=rtol, atol=atol)
                print(f"Converged after {m} iterations")
                return

    report_nonclose(Fx / max_repeats, Gx, rtol=rtol, atol=atol)


@mark.montecarlo
@mark.parametrize(
    "max_repeats,mc_samples", MAX_REPEATS_MC_SAMPLES, ids=MAX_REPEATS_MC_SAMPLES_IDS
)
def test_LinearOperator_matmat_expectation(
    case, adjoint: bool, max_repeats: int, mc_samples: int, num_vecs: int = 2
):
    F = FisherMCLinearOperator(*case, mc_samples=mc_samples)
    G_functorch = functorch_ggn(*case).detach().cpu().numpy()
    if adjoint:
        F, G_functorch = F.adjoint(), G_functorch.conj().T

    X = random.rand(F.shape[1], num_vecs).astype(F.dtype)
    GX = G_functorch @ X

    FX = zeros_like(X)
    atol, rtol = 1e-5, 1e-1

    for m in range(max_repeats):
        FX += F @ X
        F._seed += 1

        total_samples = (m + 1) * mc_samples
        if total_samples % CHECK_EVERY == 0:
            with suppress(ValueError):
                report_nonclose(FX / (m + 1), GX, rtol=rtol, atol=atol)
                print(f"Converged after {m} iterations")
                return

    report_nonclose(FX, GX, rtol=rtol, atol=atol)


def test_FisherLinearOperator_dict(dict_case):
    model_func, loss_func, params, data = dict_case
    n_params = sum([p.numel() for p in params])

    with raises(ValueError):
        op = FisherMCLinearOperator(model_func, loss_func, params, data)

    batch_size_fn = lambda data: data["x"].shape[0]
    op = FisherMCLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    assert(op.shape == (n_params, n_params))
