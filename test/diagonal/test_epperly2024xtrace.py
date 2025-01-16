"""Test ``curvlinops.diagonal.epperly2024xtrace``."""

from functools import partial

from numpy import allclose, column_stack, diag, mean, ndarray
from numpy.linalg import qr
from numpy.random import rand, seed
from pytest import mark
from scipy.sparse.linalg import LinearOperator

from curvlinops import xdiag
from curvlinops.sampling import random_vector
from test.diagonal import NUM_MATVEC_IDS, NUM_MATVECS
from test.utils import check_estimator_convergence


def xdiag_naive(A: LinearOperator, num_matvecs: int) -> ndarray:
    """Naive reference implementation of XDiag.

    See Section 2.4 in https://arxiv.org/pdf/2301.07825.

    Args:
        A: A square linear operator.
        num_matvecs: Total number of matrix-vector products to use. Must be even and
            less than the dimension of the linear operator.

    Returns:
        The estimated diagonal of the linear operator.

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
    num_vecs = num_matvecs // 2

    W = column_stack([random_vector(dim, "rademacher") for _ in range(num_vecs)])
    A_W = A @ W

    diagonals = []

    for i in range(num_vecs):
        # compute the exact diagonal of the projection onto the basis spanned by
        # the sketch matrix without test vector i
        not_i = [j for j in range(num_vecs) if j != i]
        Q_i, _ = qr(A_W[:, not_i])
        QT_i_A = Q_i.T @ A
        diag_Q_i_QT_i_A = diag(Q_i @ QT_i_A)

        # apply vanilla Hutchinson in the complement, using test vector i
        w_i = W[:, i]
        A_w_i = A_W[:, i]
        diag_w_i = w_i * (A_w_i - Q_i @ (Q_i.T @ A_w_i)) / w_i**2
        diagonals.append(diag_Q_i_QT_i_A + diag_w_i)

    return mean(diagonals, axis=0)


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
def test_xdiag(num_matvecs: int):
    """Test whether the XDiag estimator converges to the true diagonal.

    Args:
        num_matvecs: Number of matrix-vector multiplications used by one estimator.
    """
    seed(0)
    A = rand(30, 30)

    estimator = partial(xdiag, A=A, num_matvecs=num_matvecs)
    check_estimator_convergence(estimator, num_matvecs, diag(A), target_rel_error=3e-2)


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
def test_xdiag_matches_naive(num_matvecs: int, num_seeds: int = 5):
    """Test whether the efficient implementation of XDiag matches the naive.

    Args:
        num_matvecs: Number of matrix-vector multiplications used by one estimator.
        num_seeds: Number of different seeds to test the estimators with.
            Default: ``5``.
    """
    seed(0)
    A = rand(30, 30)

    # check for different seeds
    for i in range(num_seeds):
        seed(i)
        efficient = xdiag(A, num_matvecs)
        seed(i)
        naive = xdiag_naive(A, num_matvecs)
        assert allclose(efficient, naive)
