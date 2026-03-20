"""Contains tests for ``curvlinops/ggn_diagonal``."""

from torch import float64

from curvlinops.examples.functorch import functorch_ggn
from curvlinops.ggn_diagonal import GGNDiagonalLinearOperator
from test.utils import change_dtype, compare_matmat, to_functional


def _test_ggn_diagonal(model_func, loss_func, params, data, batch_size_fn):
    """Shared test logic for GGN diagonal (Module or callable)."""
    G_op = GGNDiagonalLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    G_mat = (
        functorch_ggn(model_func, loss_func, params, data, input_key="x")
        .detach()
        .diag()  # extract the diagonal
        .diag()  # embed it back into a matrix
    )
    compare_matmat(G_op, G_mat)


def test_GGNDiagonalLinearOperator_matvec(case):
    """Test GGN diagonal with Module model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_ggn_diagonal(*change_dtype(case, float64))


def test_GGNDiagonalLinearOperator_functional(case):
    """Test GGN diagonal with callable model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_ggn_diagonal(*to_functional(*change_dtype(case, float64)))
