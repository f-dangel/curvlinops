"""Test ``curvlinops.diagonal.hutchinson``."""

from numpy import diag, inf, mean
from numpy.linalg import norm
from numpy.random import rand, seed
from pytest import mark

from curvlinops import hutchinson_diag

DISTRIBUTIONS = ["rademacher", "normal"]
DISTRIBUTION_IDS = [f"distribution={distribution}" for distribution in DISTRIBUTIONS]

NUM_MATVECS = [2, 8]
NUM_MATVEC_IDS = [f"num_matvecs={num_matvecs}" for num_matvecs in NUM_MATVECS]


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
def test_hutchinson_diag(num_matvecs: int, distribution: str):
    """Test whether Hutchinson's diagonal estimator converges to the true diagonal.

    Args:
        num_matvecs: Number of matrix-vector products used per estimate.
        distribution: Distribution of the random vectors used for the trace estimation.
    """
    seed(0)
    A = rand(30, 30)
    diag_A = diag(A)

    max_total_matvecs = 100_000
    check_every = 100
    target_rel_error = 2e-2

    used_matvecs, converged = 0, False

    estimates = []
    while used_matvecs < max_total_matvecs and not converged:
        estimates.append(hutchinson_diag(A, num_matvecs, distribution=distribution))
        used_matvecs += num_matvecs

        if len(estimates) % check_every == 0:
            # use the infinity norm from Section 4.4 in the XTrace paper used to
            # evaluate diagonal estimators
            rel_error = norm(diag_A - mean(estimates, axis=0), ord=inf) / norm(
                diag_A, ord=inf
            )
            print(f"Relative error after {used_matvecs} matvecs: {rel_error:.5f}.")
            converged = rel_error < target_rel_error

    assert converged
