"""Contains tests for ``curvlinops/ggn``."""

from typing import List

from torch.nn import BCEWithLogitsLoss

from curvlinops import GGNLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.ggn import GGNDiagonalLinearOperator
from test.utils import compare_consecutive_matmats, compare_matmat
from pytest import raises


def test_GGNLinearOperator_matvec(case, adjoint: bool, is_vec: bool):
    """Test matrix-matrix multiplication with the GGN.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = case

    G = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    G_mat = functorch_ggn(model_func, loss_func, params, data, input_key="x")

    compare_consecutive_matmats(G, adjoint, is_vec)
    compare_matmat(G, G_mat, adjoint, is_vec, atol=1e-7, rtol=1e-4)


def test_GGNDiagonalLinearOperator_matvec(case, adjoint: bool, is_vec: bool):
    """Test matrix-matrix multiplication with the GGN diagonal.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = case

    if isinstance(loss_func, BCEWithLogitsLoss):
        with raises(RuntimeError, match="BCEWithLogitsLoss does not support vmap."):
            _ = GGNDiagonalLinearOperator(
                model_func, loss_func, params, data, batch_size_fn=batch_size_fn
            )
        return

    G = GGNDiagonalLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    G_mat = (
        functorch_ggn(model_func, loss_func, params, data, input_key="x")
        .detach()
        .diag()  # extract the diagonal
        .diag()  # embed it into a matrix
    )

    compare_consecutive_matmats(G, adjoint, is_vec)
    compare_matmat(G, G_mat, adjoint, is_vec, atol=1e-7, rtol=1e-4)
