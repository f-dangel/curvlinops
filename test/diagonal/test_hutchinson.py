"""Test ``curvlinops.diagonal.hutchinson``."""

from numpy import allclose, diag, isclose, mean
from numpy.random import rand, seed
from pytest import mark

from curvlinops import HutchinsonDiagonalEstimator

DISTRIBUTIONS = ["rademacher", "normal"]
DISTRIBUTION_IDS = [f"distribution={distribution}" for distribution in DISTRIBUTIONS]


@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
def test_HutchinsonDiagonalEstimator(distribution: str):
    """Test whether Hutchinson's diagonal estimator converges to the true diagonal.

    Args:
        distribution: Distribution of the random vectors used for the trace estimation.
    """
    seed(0)
    A = rand(10, 10)
    diag_A = diag(A)
    estimator = HutchinsonDiagonalEstimator(A)

    samples = []
    max_samples = 100_000
    chunk_size = 10_000
    atol, rtol = 1e-2, 5e-2

    while len(samples) < max_samples:
        samples.extend(
            [estimator.sample(distribution=distribution) for _ in range(chunk_size)]
        )
        diag_estimator = mean(samples, axis=0)
        if allclose(diag_A, diag_estimator, atol=atol, rtol=rtol):
            # quit once the estimator has converged
            break

        print(f"{len(samples)} samples:")
        for idx, (d1, d2) in enumerate(zip(diag_A, diag_estimator)):
            relation = "=" if isclose(d1, d2, atol=atol, rtol=rtol) else "≠"
            print(f"\tdiag(A)[{idx}]={d1:.5f}{relation}{d2:.5f}.")

    diag_estimator = mean(samples, axis=0)
    assert allclose(diag_A, diag_estimator, atol=atol, rtol=rtol)
