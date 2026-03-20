"""Contains tests for ``curvlinops/gradient_moments.py``."""

from torch import float64

from curvlinops import EFLinearOperator
from curvlinops.examples.functorch import functorch_empirical_fisher
from test.utils import (
    change_dtype,
    compare_consecutive_matmats,
    compare_matmat,
    to_functional,
)


def _test_ef(model_func, loss_func, params, data, batch_size_fn):
    """Shared test logic for empirical Fisher (Module or callable)."""
    E = EFLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    E_mat = functorch_empirical_fisher(
        model_func, loss_func, params, data, input_key="x"
    ).detach()

    compare_consecutive_matmats(E)
    compare_matmat(E, E_mat)


def test_EFLinearOperator(case):
    """Test empirical Fisher with Module model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_ef(*change_dtype(case, float64))


def test_EFLinearOperator_functional(case):
    """Test empirical Fisher with callable model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_ef(*to_functional(*change_dtype(case, float64)))
