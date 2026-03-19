"""Contains tests for ``curvlinops/jacobian``."""

from torch import float64

from curvlinops import JacobianLinearOperator, TransposedJacobianLinearOperator
from curvlinops.examples.functorch import functorch_jacobian
from test.utils import (
    change_dtype,
    compare_consecutive_matmats,
    compare_matmat,
    make_callable_model_func_test_case,
)


def test_JacobianLinearOperator(case):
    """Test matrix-matrix multiplication with the Jacobian.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, _, params, data, batch_size_fn = change_dtype(case, float64)

    J = JacobianLinearOperator(model_func, params, data, batch_size_fn=batch_size_fn)
    J_mat = functorch_jacobian(model_func, params, data, input_key="x").detach()

    compare_consecutive_matmats(J)
    compare_matmat(J, J_mat)


def test_TransposedJacobianLinearOperator(case):
    """Test matrix-matrix multiplication with the transpose Jacobian.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, _, params, data, batch_size_fn = change_dtype(case, float64)

    JT = TransposedJacobianLinearOperator(
        model_func, params, data, batch_size_fn=batch_size_fn
    )
    JT_mat = functorch_jacobian(model_func, params, data, input_key="x").detach().T

    compare_consecutive_matmats(JT)
    compare_matmat(JT, JT_mat)


def test_JacobianLinearOperator_callable_model_func():
    """Test Jacobian with a callable model_func and different parameter values."""
    model_fn, params_dict, data = make_callable_model_func_test_case()

    J = JacobianLinearOperator(model_fn, params_dict, data)
    J_mat = functorch_jacobian(model_fn, params_dict, data).detach()

    compare_consecutive_matmats(J)
    compare_matmat(J, J_mat)


def test_TransposedJacobianLinearOperator_callable_model_func():
    """Test transposed Jacobian with a callable model_func."""
    model_fn, params_dict, data = make_callable_model_func_test_case()

    JT = TransposedJacobianLinearOperator(model_fn, params_dict, data)
    JT_mat = functorch_jacobian(model_fn, params_dict, data).detach().T

    compare_consecutive_matmats(JT)
    compare_matmat(JT, JT_mat)
