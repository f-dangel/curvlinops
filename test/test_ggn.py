"""Contains tests for ``curvlinops/ggn``."""

from pytest import raises
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

from curvlinops import GGNLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.ggn import GGNDiagonalLinearOperator
from test.utils import compare_consecutive_matmats, compare_matmat


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

    def _construct_G():
        return GGNDiagonalLinearOperator(
            model_func, loss_func, params, data, batch_size_fn=batch_size_fn
        )

    # Check that BCEWithLogitsLoss raises an error
    if isinstance(loss_func, BCEWithLogitsLoss):
        with raises(RuntimeError, match="BCEWithLogitsLoss does not support vmap."):
            _construct_G()
        return

    # Check that non-tensor inputs raise an error
    if any(not isinstance(X, Tensor) for (X, _) in data):
        with raises(
            RuntimeError, match="Only tensor-valued inputs are supported by vmap."
        ):
            _construct_G()
        return

    # Check that sequence-valued predictions are unsupported
    if model_func(data[0][0]).ndim > 2:
        with raises(RuntimeError, match="Sequence-valued predictions are unsupported."):
            _construct_G()
        return

    G = _construct_G()
    G_mat = (
        functorch_ggn(model_func, loss_func, params, data, input_key="x")
        .detach()
        .diag()  # extract the diagonal
        .diag()  # embed it into a matrix
    )

    compare_consecutive_matmats(G, adjoint, is_vec)
    compare_matmat(G, G_mat, adjoint, is_vec, atol=1e-7, rtol=1e-4)
