"""Contains tests for ``curvlinops/submatrix`` on curvature linear operators."""

from typing import List

from numpy import random
from pytest import mark

from curvlinops import EFLinearOperator, GGNLinearOperator, HessianLinearOperator
from curvlinops.examples.functorch import (
    functorch_empirical_fisher,
    functorch_ggn,
    functorch_hessian,
)
from curvlinops.examples.utils import report_nonclose
from curvlinops.submatrix import SubmatrixLinearOperator

CURVATURE_CASES = [HessianLinearOperator, GGNLinearOperator, EFLinearOperator]
CURVATURE_IN_FUNCTORCH = {
    HessianLinearOperator: functorch_hessian,
    GGNLinearOperator: functorch_ggn,
    EFLinearOperator: functorch_empirical_fisher,
}


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
    A_sub_functorch = A_functorch[row_idxs, :][:, col_idxs].detach().cpu().numpy()

    return A_sub, A_sub_functorch, row_idxs, col_idxs


@mark.parametrize("operator_case", CURVATURE_CASES)
@mark.parametrize("submatrix_case", SUBMATRIX_CASES)
def test_SubmatrixLinearOperator_on_curvatures_matvec(
    case, operator_case, submatrix_case
):
    A_sub, A_sub_functorch, row_idxs, col_idxs = setup_submatrix_linear_operator(
        case, operator_case, submatrix_case
    )
    x = random.rand(len(col_idxs)).astype(A_sub.dtype)
    A_sub_x = A_sub @ x

    assert A_sub_x.shape == (len(row_idxs),)
    report_nonclose(A_sub_x, A_sub_functorch @ x, atol=2e-7)


@mark.parametrize("operator_case", CURVATURE_CASES)
@mark.parametrize("submatrix_case", SUBMATRIX_CASES)
def test_SubmatrixLinearOperator_on_curvatures_matmat(
    case, operator_case, submatrix_case, num_vecs: int = 3
):
    A_sub, A_sub_functorch, row_idxs, col_idxs = setup_submatrix_linear_operator(
        case, operator_case, submatrix_case
    )
    X = random.rand(len(col_idxs), num_vecs).astype(A_sub.dtype)
    A_sub_X = A_sub @ X

    assert A_sub_X.shape == (len(row_idxs), num_vecs)
    report_nonclose(A_sub_X, A_sub_functorch @ X, atol=1e-6)
