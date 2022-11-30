"""Contains tests for ``curvlinops/fisher.py``."""

from contextlib import suppress

from numpy import random, zeros_like
from pytest import mark

from curvlinops import FisherMCLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.examples.utils import report_nonclose

MAX_REPEATS_MC_SAMPLES = [(1_000_000, 1), (10_000, 100)]
MAX_REPEATS_MC_SAMPLES_IDS = [
    f"max_repeats={n}-mc_samples={m}" for (n, m) in MAX_REPEATS_MC_SAMPLES
]
CHECK_EVERY = 1000


@mark.parametrize(
    "max_repeats,mc_samples", MAX_REPEATS_MC_SAMPLES, ids=MAX_REPEATS_MC_SAMPLES_IDS
)
def test_LinearOperator_matvec_expectation(case, max_repeats: int, mc_samples: int):
    F = FisherMCLinearOperator(*case, mc_samples=mc_samples)
    x = random.rand(F.shape[1]).astype(F.dtype)

    G_functorch = functorch_ggn(*case).detach().cpu().numpy()
    Gx = G_functorch @ x

    Fx = zeros_like(x)
    atol, rtol = 1e-5, 1e-1

    for m in range(max_repeats):
        Fx += F @ x
        F._seed += 1

        if m > 0 and m % CHECK_EVERY == 0:
            with suppress(ValueError):
                report_nonclose(Fx / (m + 1), Gx, rtol=rtol, atol=atol)
                print(f"Converged after {m} iterations")
                return

    report_nonclose(Fx / max_repeats, Gx, rtol=rtol, atol=atol)


@mark.parametrize(
    "max_repeats,mc_samples", MAX_REPEATS_MC_SAMPLES, ids=MAX_REPEATS_MC_SAMPLES_IDS
)
def test_LinearOperator_matmat_expectation(
    case, max_repeats: int, mc_samples: int, num_vecs: int = 3
):
    F = FisherMCLinearOperator(*case, mc_samples=mc_samples)
    X = random.rand(F.shape[1], num_vecs).astype(F.dtype)

    G_functorch = functorch_ggn(*case).detach().cpu().numpy()
    GX = G_functorch @ X

    FX = zeros_like(X)
    atol, rtol = 1e-5, 1e-1

    for m in range(max_repeats):
        FX += F @ X
        F._seed += 1

        if m > 0 and m % CHECK_EVERY == 0:
            with suppress(ValueError):
                report_nonclose(FX / (m + 1), GX, rtol=rtol, atol=atol)
                print(f"Converged after {m} iterations")
                return

    report_nonclose(FX, GX, rtol=rtol, atol=atol)
