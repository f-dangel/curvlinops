"""Test ``curvlinops.trace``."""

from typing import Callable

from numpy import mean
from scipy.sparse.linalg import LinearOperator

DISTRIBUTIONS = ["rademacher", "normal"]
DISTRIBUTION_IDS = [f"distribution={distribution}" for distribution in DISTRIBUTIONS]

NUM_MATVECS = [3, 6]
NUM_MATVEC_IDS = [f"num_matvecs={num_matvecs}" for num_matvecs in NUM_MATVECS]


def _test_convergence(
    estimator: Callable[[LinearOperator, int], float],
    num_matvecs: int,
    A: LinearOperator,
    tr_A: float,
    max_total_matvecs: int = 100_000,
    check_every: int = 100,
    target_rel_error: float = 1e-3,
):
    """Test whether a trace estimator converges to the true trace.

    Args:
        A: Linear operator for which the trace is estimated.
        estimator: The trace estimator.
        num_matvecs: Number of matrix-vector products used per estimate.
        tr_A: True trace of the linear operator.
        max_total_matvecs: Maximum number of matrix-vector products to perform.
            Default: ``100_000``. If convergence has not been reached by then, the test
            will fail.
        check_every: Check for convergence every ``check_every`` estimates.
            Default: ``100``.
        target_rel_error: Relative error for considering the estimator converged.
            Default: ``1e-3``.
    """
    used_matvecs, converged = 0, False

    estimates = []
    while used_matvecs < max_total_matvecs and not converged:
        estimates.append(estimator(A, num_matvecs))
        used_matvecs += num_matvecs

        if len(estimates) % check_every == 0:
            rel_error = abs(tr_A - mean(estimates)) / abs(tr_A)
            print(f"Relative error after {used_matvecs} matvecs: {rel_error:.5f}.")
            converged = rel_error < target_rel_error

    assert converged
