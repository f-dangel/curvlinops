"""Contains tests for ``curvlinops/hessian``."""

from torch import float64

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from test.utils import change_dtype, compare_consecutive_matmats, compare_matmat


def test_HessianLinearOperator(case):
    """Test matrix-matrix multiplication with the Hessian.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    H = HessianLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    H_mat = functorch_hessian(
        model_func, loss_func, params, data, input_key="x"
    ).detach()

    compare_consecutive_matmats(H)
    compare_matmat(H, H_mat)
