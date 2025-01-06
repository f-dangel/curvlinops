"""Test ``curvlinops.trace.epperli2024xtrace."""

from test.trace import DISTRIBUTION_IDS, DISTRIBUTIONS

from numpy import column_stack, dot, isclose, mean, trace
from numpy.linalg import qr
from numpy.random import rand, seed
from pytest import mark
from scipy.sparse.linalg import LinearOperator

from curvlinops import xtrace
from curvlinops.sampling import random_vector

NUM_MATVECS = [4, 10]
NUM_MATVEC_IDS = [f"num_matvecs={num_matvecs}" for num_matvecs in NUM_MATVECS]


def xtrace_naive(
    A: LinearOperator, num_matvecs: int, distribution: str = "rademacher"
) -> float:
    """Naive reference implementation of XTrace.

    See Algorithm 1.2 in https://arxiv.org/pdf/2301.07825.

    Args:
        A: A square linear operator.
        num_matvecs: Total number of matrix-vector products to use. Must be even and
            less than the dimension of the linear operator.
        distribution: Distribution of the random vectors used for the trace estimation.
            Can be either ``'rademacher'`` or ``'normal'``. Default: ``'rademacher'``.

    Returns:
        The estimated trace of the linear operator.

    Raises:
        ValueError: If the linear operator is not square or if the number of matrix-
            vector products is not even or is greater than the dimension of the linear
            operator.
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square. Got shape {A.shape}.")
    dim = A.shape[1]
    if num_matvecs % 2 != 0 or num_matvecs >= dim:
        raise ValueError(
            "num_matvecs must be even and less than the dimension of A.",
            f" Got {num_matvecs}.",
        )
    sketch_dim = num_matvecs // 2

    W = column_stack([random_vector(dim, distribution) for _ in range(sketch_dim)])
    A_W = A @ W

    traces = []

    for i in range(sketch_dim):
        # compute the exact trace in the basis spanned by the sketch matrix without
        # test vector i
        not_i = [j for j in range(sketch_dim) if j != i]
        Q_i, _ = qr(A_W[:, not_i])
        A_Q_i = A @ Q_i
        tr_QT_i_A_Q_i = trace(Q_i.T @ A_Q_i)

        # apply vanilla Hutchinson in the complement, using test vector i
        w_i = W[:, i]
        A_w_i = A_W[:, i]
        A_P_w_i = A_w_i - A_Q_i @ (Q_i.T @ w_i)
        PT_A_P_w_i = A_P_w_i - Q_i @ (Q_i.T @ A_P_w_i)
        tr_w_i = dot(w_i, PT_A_P_w_i)

        traces.append(float(tr_QT_i_A_Q_i + tr_w_i))

    return mean(traces)


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
def test_xtrace(
    distribution: str,
    num_matvecs: int,
    max_total_matvecs: int = 10_000,
    check_every: int = 10,
    target_rel_error: float = 1e-3,
):
    """Test whether the XTrace estimator converges to the true trace.

    Args:
        distribution: Distribution of the random vectors used for the trace estimation.
        num_matvecs: Number of matrix-vector multiplications used by one estimator.
        max_total_matvecs: Maximum number of matrix-vector multiplications to perform.
            Default: ``1_000``. If convergence has not been reached by then, the test
            will fail.
        check_every: Check for convergence every ``check_every`` estimates.
            Default: ``10``.
        target_rel_error: Target relative error for considering the estimator converged.
            Default: ``1e-3``.
    """
    seed(0)
    A = rand(50, 50)
    tr_A = trace(A)

    used_matvecs, converged = 0, False

    estimates = []
    while used_matvecs < max_total_matvecs and not converged:
        estimates.append(xtrace(A, num_matvecs, distribution=distribution))
        used_matvecs += num_matvecs

        if len(estimates) % check_every == 0:
            rel_error = abs(tr_A - mean(estimates)) / abs(tr_A)
            print(f"Relative error after {used_matvecs} matvecs: {rel_error:.5f}.")
            converged = rel_error < target_rel_error

    assert converged


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
def test_xtrace_matches_naive(num_matvecs: int, distribution: str, num_seeds: int = 5):
    """Test whether the efficient implementation of XTrace matches the naive.

    Args:
        num_matvecs: Number of matrix-vector multiplications used by one estimator.
        distribution: Distribution of the random vectors used for the trace estimation.
        num_seeds: Number of different seeds to test the estimators with.
            Default: ``5``.
    """
    seed(0)
    A = rand(50, 50)

    # check for different seeds
    for i in range(num_seeds):
        seed(i)
        efficient = xtrace(A, num_matvecs, distribution=distribution)
        seed(i)
        naive = xtrace_naive(A, num_matvecs, distribution=distribution)
        assert isclose(efficient, naive)
