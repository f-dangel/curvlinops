"""Contains tests for ``curvlinops/submatrix`` on curvature linear operators."""

from typing import List

from pytest import mark, raises

from curvlinops import EFLinearOperator, GGNLinearOperator, HessianLinearOperator
from curvlinops.examples.functorch import (
    functorch_empirical_fisher,
    functorch_ggn,
    functorch_hessian,
)
from curvlinops.submatrix import SubmatrixLinearOperator
from test.utils import compare_consecutive_matmats, compare_matmat

CURVATURE_IN_FUNCTORCH = {
    HessianLinearOperator: functorch_hessian,
    GGNLinearOperator: functorch_ggn,
    EFLinearOperator: functorch_empirical_fisher,
}
CURVATURE_CASES = CURVATURE_IN_FUNCTORCH.keys()


def even_idxs(dim: int) -> List[int]:
    return list(range(0, dim, 2))


def odd_idxs(dim: int) -> List[int]:
    return list(range(1, dim, 2))


def every_third_idxs(dim: int):
    return list(range(0, dim, 3))


SUBMATRIX_CASES = [
    # same indices for rows and columns (square matrix)
    {
        "row_idx_fn": even_idxs,
        "col_idx_fn": even_idxs,
    },
    # different indices for rows and columns (square matrix if dim is even)
    {
        "row_idx_fn": even_idxs,
        "col_idx_fn": odd_idxs,
    },
    # different indices for rows and columns (rectangular matrix if dim>5)
    {
        "row_idx_fn": odd_idxs,
        "col_idx_fn": every_third_idxs,
    },
]
SUBMATRIX_IDS = [
    f"({case['row_idx_fn'].__name__},{case['col_idx_fn'].__name__})"
    for case in SUBMATRIX_CASES
]


def setup_submatrix_linear_operator(case, operator_case, submatrix_case):
    model_func, loss_func, params, data, batch_size_fn = case
    dim = sum(p.numel() for p in params)
    row_idxs = submatrix_case["row_idx_fn"](dim)
    col_idxs = submatrix_case["col_idx_fn"](dim)

    A = operator_case(model_func, loss_func, params, data, batch_size_fn=batch_size_fn)
    A_sub = SubmatrixLinearOperator(A, row_idxs, col_idxs)

    A_functorch = CURVATURE_IN_FUNCTORCH[operator_case](
        model_func, loss_func, params, data, "x"
    )
    A_sub_functorch = A_functorch[row_idxs, :][:, col_idxs]

    return A_sub, A_sub_functorch, row_idxs, col_idxs


@mark.parametrize("operator_case", CURVATURE_CASES)
@mark.parametrize("submatrix_case", SUBMATRIX_CASES)
def test_SubmatrixLinearOperator_on_curvatures(
    case,
    operator_case,
    submatrix_case,
    adjoint: bool,
    is_vec: bool,
):
    A_sub, A_sub_functorch, row_idxs, col_idxs = setup_submatrix_linear_operator(
        case, operator_case, submatrix_case
    )
    assert A_sub.shape == (len(row_idxs), len(col_idxs))
    compare_consecutive_matmats(A_sub, adjoint, is_vec)
    compare_matmat(A_sub, A_sub_functorch, adjoint, is_vec, atol=1e-6, rtol=1e-4)

    # try specifying the sub-matrix using invalid indices
    invalid_idxs = [
        [[0.0], [0]],  # wrong type in row_idxs
        [[0], [0.0]],  # wrong type in col_idxs
        [[2, 1, 2], [3]],  # duplicate entries in row_idxs
        [[3], [2, 1, 2]],  # duplicate entries in col_idxs
        [[1_000_000_000_000, 5], [2]],  # out-of-bounds in row_idxs
        [[6, 5], [-1]],  # out-of-bounds in col_idxs
    ]
    for row_idxs, col_idxs in invalid_idxs:
        with raises(ValueError):
            A_sub.set_submatrix(row_idxs, col_idxs)
