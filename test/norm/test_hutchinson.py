"""Test ``curvlinops.norm.hutchinson``."""

from numpy import mean
from numpy.random import rand, seed
from pytest import mark

from curvlinops import hutchinson_squared_fro

DISTRIBUTIONS = ["rademacher", "normal"]
DISTRIBUTION_IDS = [f"distribution={distribution}" for distribution in DISTRIBUTIONS]

NUM_MATVECS = [2, 8]
NUM_MATVEC_IDS = [f"num_matvecs={num_matvecs}" for num_matvecs in NUM_MATVECS]


@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
def test_hutchinson_squared_fro(num_matvecs: int, distribution: str):
    """Test whether Hutchinson's squared Frobenius norm estimator converges.

    Args:
        num_matvecs: Number of matrix-vector products used per estimate.
        distribution: Distribution of the random vectors used for the estimation.
    """
    seed(0)
    A = rand(20, 30)
    A_squared_fro = (A**2).sum()

    max_total_matvecs = 100_000
    check_every = 100
    target_rel_error = 2e-3

    used_matvecs, converged = 0, False

    estimates = []
    while used_matvecs < max_total_matvecs and not converged:
        estimates.append(
            hutchinson_squared_fro(A, num_matvecs, distribution=distribution)
        )
        used_matvecs += num_matvecs

        if len(estimates) % check_every == 0:
            rel_error = abs(A_squared_fro - mean(estimates)) / A_squared_fro
            print(f"Relative error after {used_matvecs} matvecs: {rel_error:.5f}.")
            converged = rel_error < target_rel_error

    assert converged
