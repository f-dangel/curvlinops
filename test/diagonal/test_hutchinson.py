"""Test ``curvlinops.diagonal.hutchinson``."""

from functools import partial

from numpy import diag
from numpy.random import rand, seed
from pytest import mark

from curvlinops import hutchinson_diag
from test.diagonal import DISTRIBUTION_IDS, DISTRIBUTIONS, NUM_MATVEC_IDS, NUM_MATVECS
from test.utils import check_estimator_convergence


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
    estimator = partial(
        hutchinson_diag, A=A, num_matvecs=num_matvecs, distribution=distribution
    )
    check_estimator_convergence(estimator, num_matvecs, diag(A), target_rel_error=2e-2)
