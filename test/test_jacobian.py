"""Contains tests for ``curvlinops/jacobian``."""

from curvlinops import JacobianLinearOperator, TransposedJacobianLinearOperator
from curvlinops.examples.functorch import functorch_jacobian
from test.utils import compare_consecutive_matmats, compare_matmat


def test_JacobianLinearOperator(case, adjoint: bool, is_vec: bool):
    """Test matrix-matrix multiplication with the Jacobian.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, _, params, data, batch_size_fn = case

    J = JacobianLinearOperator(model_func, params, data, batch_size_fn=batch_size_fn)
    J_mat = functorch_jacobian(model_func, params, data, input_key="x")

    compare_consecutive_matmats(J, adjoint, is_vec)
    compare_matmat(J, J_mat, adjoint, is_vec, rtol=1e-4, atol=1e-7)


def test_TransposedJacobianLinearOperator(case, adjoint: bool, is_vec: bool):
    """Test matrix-matrix multiplication with the transpose Jacobian.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, _, params, data, batch_size_fn = case

    JT = TransposedJacobianLinearOperator(
        model_func, params, data, batch_size_fn=batch_size_fn
    )
    JT_mat = functorch_jacobian(model_func, params, data, input_key="x").T

    compare_consecutive_matmats(JT, adjoint, is_vec)
    compare_matmat(JT, JT_mat, adjoint, is_vec, rtol=1e-4, atol=1e-7)
