"""Test ``curvlinops.diagonal.epperly2024xtrace``."""

from functools import partial
from typing import Union

from pytest import mark
from torch import Tensor, column_stack, manual_seed, rand
from torch.linalg import qr

from curvlinops import xdiag
from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.sampling import random_vector
from test.diagonal import NUM_MATVEC_IDS, NUM_MATVECS
from test.utils import check_estimator_convergence


def xdiag_naive(A: Union[PyTorchLinearOperator, Tensor], num_matvecs: int) -> Tensor:
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

    W = column_stack(
        [random_vector(dim, "rademacher", A.device, A.dtype) for _ in range(num_vecs)]
    )
    A_W = A @ W

    diagonals = []

    for i in range(num_vecs):
        # compute the exact diagonal of the projection onto the basis spanned by
        # the sketch matrix without test vector i
        not_i = [j for j in range(num_vecs) if j != i]
        Q_i, _ = qr(A_W[:, not_i])
        QT_i_A = Q_i.T @ A
        diag_Q_i_QT_i_A = (Q_i @ QT_i_A).diag()

        # apply vanilla Hutchinson in the complement, using test vector i
        w_i = W[:, i]
        A_w_i = A_W[:, i]
        diag_w_i = w_i * (A_w_i - Q_i @ (Q_i.T @ A_w_i)) / w_i**2
        diagonals.append(diag_Q_i_QT_i_A + diag_w_i)

    return sum(diagonals) / len(diagonals)


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
def test_xdiag(num_matvecs: int):
    """Test whether the XDiag estimator converges to the true diagonal.

    Args:
        num_matvecs: Number of matrix-vector multiplications used by one estimator.
    """
    manual_seed(0)
    A = rand(30, 30)

    estimator = partial(xdiag, A=A, num_matvecs=num_matvecs)
    check_estimator_convergence(estimator, num_matvecs, A.diag(), target_rel_error=3e-2)


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
def test_xdiag_matches_naive(num_matvecs: int, num_seeds: int = 5):
    """Test whether the efficient implementation of XDiag matches the naive.

    Args:
        num_matvecs: Number of matrix-vector multiplications used by one estimator.
        num_seeds: Number of different seeds to test the estimators with.
            Default: ``5``.
    """
    manual_seed(0)
    A = rand(30, 30).double()

    # check for different seeds
    for i in range(num_seeds):
        manual_seed(i)
        efficient = xdiag(A, num_matvecs)
        manual_seed(i)
        naive = xdiag_naive(A, num_matvecs)
        assert efficient.allclose(naive)
