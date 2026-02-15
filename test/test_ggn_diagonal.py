"""Contains tests for ``curvlinops/ggn_diagonal``."""

from torch import float64

from curvlinops.examples.functorch import functorch_ggn
from curvlinops.ggn_diagonal import GGNDiagonalLinearOperator
from test.utils import change_dtype, compare_matmat


def test_GGNDiagonalLinearOperator_matvec(case):
    """Verify the GGN diagonal linear operator's matrix multiplication routine.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

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
