"""Test ``curvlinops.trace``."""

from typing import Union

from numpy import isclose, mean

from curvlinops.trace.hutchinson import HutchinsonTraceEstimator
from curvlinops.trace.meyer2020hutch import HutchPPTraceEstimator

DISTRIBUTIONS = ["rademacher", "normal"]
DISTRIBUTION_IDS = [f"distribution={distribution}" for distribution in DISTRIBUTIONS]


def _test_convergence(
    estimator: Union[HutchinsonTraceEstimator, HutchPPTraceEstimator],
    tr_A: float,
    distribution: str,
    max_samples: int = 20_000,
    chunk_size: int = 2_000,
    atol: float = 1e-3,
    rtol: float = 1e-2,
):
    """Test whether a trace estimator converges to the true trace.

    Args:
        estimator: The trace estimator.
        tr_A: True trace of the linear operator.
        distribution: Distribution of the random vectors used for the trace estimation.
        max_samples: Maximum number of samples to draw before error-ing.
            Default: ``20_000``
        chunk_size: Number of samples to draw before comparing against the true trace.
            Default: ``2_000``.
        atol: Absolute toleranc e used to compare with the exact trace.
            Default: ``1e-3``.
        rtol: Relative tolerance used to compare with the exact trace.
            Default: ``1e-2``.
    """
    samples = []

    while len(samples) < max_samples:
        samples.extend(
            [estimator.sample(distribution=distribution) for _ in range(chunk_size)]
        )
        tr_estimator = mean(samples)
        if isclose(tr_A, tr_estimator, atol=atol, rtol=rtol):
            # quit once the estimator has converged
            break

        print(f"{len(samples)} samples: Tr(A)={tr_A:.5f}≠{tr_estimator:.5f}.")

    tr_estimator = mean(samples)
    assert isclose(tr_A, tr_estimator, atol=atol, rtol=rtol)
