"""Test ``curvlinops.trace.meyer2020hutch."""

from functools import partial

from pytest import mark
from torch import manual_seed, rand

from curvlinops import hutchpp_trace
from test.trace import DISTRIBUTION_IDS, DISTRIBUTIONS, NUM_MATVEC_IDS, NUM_MATVECS
from test.utils import check_estimator_convergence


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
def test_hutchpp_trace(num_matvecs: int, distribution: str):
    """Test whether Hutch++'s trace estimator converges to the true trace.

    Args:
        num_matvecs: Number of matrix-vector products used per estimate.
        distribution: Distribution of the random vectors used for the trace estimation.
    """
    manual_seed(0)
    A = rand(10, 10)
    estimator = partial(
        hutchpp_trace, A=A, num_matvecs=num_matvecs, distribution=distribution
    )
    check_estimator_convergence(
        estimator, num_matvecs, A.trace(), target_rel_error=1e-3
    )
