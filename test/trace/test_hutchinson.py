"""Test ``curvlinops.trace.hutchinson``."""

from functools import partial
from test.trace import (
    DISTRIBUTION_IDS,
    DISTRIBUTIONS,
    NUM_MATVEC_IDS,
    NUM_MATVECS,
    _test_convergence,
)

from numpy import trace
from numpy.random import rand, seed
from pytest import mark

from curvlinops import hutchinson_trace


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
def test_hutchinson_trace(num_matvecs: int, distribution: str):
    """Test whether Hutchinon's trace estimator converges to the true trace.

    Args:
        num_matvecs: Number of matrix-vector products used per estimate.
        distribution: Distribution of the random vectors used for the trace estimation.
    """
    seed(0)
    A = rand(10, 10)
    tr_A = trace(A)
    estimator = partial(hutchinson_trace, distribution=distribution)
    _test_convergence(estimator, num_matvecs, A, tr_A)
