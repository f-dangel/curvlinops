"""Contains tests for ``curvlinops/hessian``."""

from torch import float64

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from test.utils import (
    change_dtype,
    compare_consecutive_matmats,
    compare_matmat,
    to_functional,
)


def _test_hessian(model_func, loss_func, params, data, batch_size_fn):
    """Shared test logic for Hessian (Module or callable)."""
    H = HessianLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    H_mat = functorch_hessian(
        model_func, loss_func, params, data, input_key="x"
    ).detach()

    compare_consecutive_matmats(H)
    compare_matmat(H, H_mat)


def test_HessianLinearOperator(case):
    """Test Hessian with Module model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_hessian(*change_dtype(case, float64))


def test_HessianLinearOperator_functional(case):
    """Test Hessian with callable model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_hessian(*to_functional(*change_dtype(case, float64)))
