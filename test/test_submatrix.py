"""Contains tests for ``curvlinops/submatrix`` with random matrices."""

from typing import List, Tuple

from numpy import eye, ndarray, random
from pytest import fixture, mark, raises
from scipy.sparse.linalg import aslinearoperator

from curvlinops.examples.utils import report_nonclose
from curvlinops.submatrix import SubmatrixLinearOperator

SUBMATRIX_CASES = [
    # same indices
    {
        "A_fn": lambda: random.rand(300, 300),
        "row_idxs_fn": lambda: [0, 13, 42, 299],
        "col_idxs_fn": lambda: [0, 13, 42, 299],
        "seed": 0,
    },
    # different indices
    {
        "A_fn": lambda: random.rand(200, 250),
        "row_idxs_fn": lambda: [5, 67, 128],
        "col_idxs_fn": lambda: [83, 21],
        "seed": 0,
    },
]


@fixture(params=SUBMATRIX_CASES)
def submatrix_case(request) -> Tuple[ndarray, List[int], List[int]]:
    case = request.param
    random.seed(case["seed"])
    return case["A_fn"](), case["row_idxs_fn"](), case["col_idxs_fn"]()


@mark.parametrize("adjoint", [False, True], ids=["", "adjoint"])
def test_SubmatrixLinearOperator__matvec(
    submatrix_case: Tuple[ndarray, List[int], List[int]], adjoint: bool
):
    """Test the matrix-vector multiplication of a submatrix linear operator.

    Args:
        submatrix_case: A tuple with a random matrix and two index lists.
        adjoint: Whether to take the operator's adjoint before multiplying.
    """
    A, row_idxs, col_idxs = submatrix_case

    A_sub = A[row_idxs, :][:, col_idxs]
    A_sub_linop = SubmatrixLinearOperator(aslinearoperator(A), row_idxs, col_idxs)

    if adjoint:
        A_sub = A_sub.conj().T
        A_sub_linop = A_sub_linop.adjoint()

    x = random.rand(A_sub.shape[1])
    A_sub_linop_x = A_sub_linop @ x

    assert A_sub_linop_x.shape == ((len(col_idxs),) if adjoint else (len(row_idxs),))
    report_nonclose(A_sub @ x, A_sub_linop_x)


@mark.parametrize("adjoint", [False, True], ids=["", "adjoint"])
def test_SubmatrixLinearOperator__matmat(
    submatrix_case: Tuple[ndarray, List[int], List[int]],
    adjoint: bool,
    num_vecs: int = 3,
):
    """Test the matrix-matrix multiplication of a submatrix linear operator.

    Args:
        submatrix_case: A tuple with a random matrix and two index lists.
        adjoint: Whether to take the operator's adjoint before multiplying.
        num_vecs: The number of vectors to multiply. Default: ``3``.
    """
    A, row_idxs, col_idxs = submatrix_case

    A_sub = A[row_idxs, :][:, col_idxs]
    A_sub_linop = SubmatrixLinearOperator(aslinearoperator(A), row_idxs, col_idxs)

    if adjoint:
        A_sub = A_sub.conj().T
        A_sub_linop = A_sub_linop.adjoint()

    X = random.rand(A_sub.shape[1], num_vecs)
    A_sub_linop_X = A_sub_linop @ X

    assert A_sub_linop_X.shape == (
        (len(col_idxs), num_vecs) if adjoint else (len(row_idxs), num_vecs)
    )
    report_nonclose(A_sub @ X, A_sub_linop_X)


def test_SubmatrixLinearOperator_set_submatrix():
    A = aslinearoperator(eye(10))

    invalid_idxs = [
        [[0.0], [0]],  # wrong type in row_idxs
        [[0], [0.0]],  # wrong type in col_idxs
        [[2, 1, 2], [3]],  # duplicate entries in row_idxs
        [[3], [2, 1, 2]],  # duplicate entries in col_idxs
        [[10, 5], [2]],  # out-of-bounds in row_idxs
        [[6, 5], [-1]],  # out-of-bounds in col_idxs
    ]

    for row_idxs, col_idxs in invalid_idxs:
        with raises(ValueError):
            SubmatrixLinearOperator(A, row_idxs, col_idxs)
