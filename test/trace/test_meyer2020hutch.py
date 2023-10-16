"""Test ``curvlinops.trace.meyer2020hutch."""

from numpy import isclose, mean, trace
from numpy.random import rand, seed
from pytest import mark

from curvlinops import HutchPPTraceEstimator

DISTRIBUTIONS = ["rademacher", "normal"]
DISTRIBUTION_IDS = [f"distribution={distribution}" for distribution in DISTRIBUTIONS]


@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
def test_HutchPPTraceEstimator(distribution: str):
    """Test whether Hutch++'s trace estimator converges to the true trace.

    Args:
        distribution: Distribution of the random vectors used for the trace estimation.
    """
    seed(0)
    A = rand(10, 10)
    tr_A = trace(A)

    samples = []
    max_samples = 20_000
    chunk_size = 2_000  # add that many new samples before comparing against the truth
    atol, rtol = 1e-3, 1e-2

    estimator = HutchPPTraceEstimator(A)

    while len(samples) < max_samples:
        samples.extend(
            [estimator.sample(distribution=distribution) for _ in range(chunk_size)]
        )
        tr_estimator = mean(samples)
        if not isclose(tr_A, tr_estimator, atol=atol, rtol=rtol):
            print(f"{len(samples)} samples: Tr(A)={tr_A:.5f}≠{tr_estimator:.5f}.")
        else:
            # quit once the estimator has converged
            break

    tr_estimator = mean(samples)
    assert isclose(tr_A, tr_estimator, atol=atol, rtol=rtol)
