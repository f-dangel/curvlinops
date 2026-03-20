"""Contains tests for ``curvlinops/jacobian``."""

from torch import float64

from curvlinops import JacobianLinearOperator, TransposedJacobianLinearOperator
from curvlinops.examples.functorch import functorch_jacobian
from test.utils import (
    change_dtype,
    compare_consecutive_matmats,
    compare_matmat,
    to_functional,
)


def _test_jacobian(model_func, loss_func, params, data, batch_size_fn):
    """Shared test logic for Jacobian (Module or callable)."""
    J = JacobianLinearOperator(model_func, params, data, batch_size_fn=batch_size_fn)
    J_mat = functorch_jacobian(model_func, params, data, input_key="x").detach()

    compare_consecutive_matmats(J)
    compare_matmat(J, J_mat)


def _test_transposed_jacobian(model_func, loss_func, params, data, batch_size_fn):
    """Shared test logic for transposed Jacobian (Module or callable)."""
    JT = TransposedJacobianLinearOperator(
        model_func, params, data, batch_size_fn=batch_size_fn
    )
    JT_mat = functorch_jacobian(model_func, params, data, input_key="x").detach().T

    compare_consecutive_matmats(JT)
    compare_matmat(JT, JT_mat)


def test_JacobianLinearOperator(case):
    """Test Jacobian with Module model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_jacobian(*change_dtype(case, float64))


def test_JacobianLinearOperator_functional(case):
    """Test Jacobian with callable model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_jacobian(*to_functional(*change_dtype(case, float64)))


def test_TransposedJacobianLinearOperator(case):
    """Test transposed Jacobian with Module model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_transposed_jacobian(*change_dtype(case, float64))


def test_TransposedJacobianLinearOperator_functional(case):
    """Test transposed Jacobian with callable model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_transposed_jacobian(*to_functional(*change_dtype(case, float64)))
