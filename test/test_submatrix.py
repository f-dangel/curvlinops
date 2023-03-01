"""Contains tests for ``curvlinops/submatrix`` with random matrices."""

from typing import List, Tuple

from numpy import eye, ndarray, random
from pytest import fixture, raises
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


def test_SubmatrixLinearOperator__matvec(submatrix_case):
    A, row_idxs, col_idxs = submatrix_case

    A_sub = A[row_idxs, :][:, col_idxs]
    A_sub_linop = SubmatrixLinearOperator(aslinearoperator(A), row_idxs, col_idxs)

    x = random.rand(len(col_idxs))
    A_sub_linop_x = A_sub_linop @ x

    assert A_sub_linop_x.shape == (len(row_idxs),)
    report_nonclose(A_sub @ x, A_sub_linop_x)


def test_SubmatrixLinearOperator__matmat(submatrix_case, num_vecs: int = 3):
    A, row_idxs, col_idxs = submatrix_case

    A_sub = A[row_idxs, :][:, col_idxs]
    A_sub_linop = SubmatrixLinearOperator(aslinearoperator(A), row_idxs, col_idxs)

    X = random.rand(len(col_idxs), num_vecs)
    A_sub_linop_X = A_sub_linop @ X

    assert A_sub_linop_X.shape == (len(row_idxs), num_vecs)
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
