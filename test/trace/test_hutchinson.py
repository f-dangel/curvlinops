"""Test ``curvlinops.trace.hutchinson``."""

from test.trace import DISTRIBUTION_IDS, DISTRIBUTIONS, _test_convergence

from numpy import trace
from numpy.random import rand, seed
from pytest import mark

from curvlinops import HutchinsonTraceEstimator


@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
def test_HutchinsonTraceEstimator(distribution: str):
    """Test whether Hutchinon's trace estimator converges to the true trace.

    Args:
        distribution: Distribution of the random vectors used for the trace estimation.
    """
    seed(0)
    A = rand(10, 10)
    tr_A = trace(A)
    estimator = HutchinsonTraceEstimator(A)

    _test_convergence(estimator, tr_A, distribution)
