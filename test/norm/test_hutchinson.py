"""Test ``curvlinops.norm.hutchinson``."""

from functools import partial

from pytest import mark
from torch import manual_seed, rand

from curvlinops import hutchinson_squared_fro
from test.utils import check_estimator_convergence

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
    manual_seed(0)
    A = rand(10, 15)
    estimator = partial(
        hutchinson_squared_fro, A=A, num_matvecs=num_matvecs, distribution=distribution
    )
    check_estimator_convergence(estimator, num_matvecs, (A**2).sum())
