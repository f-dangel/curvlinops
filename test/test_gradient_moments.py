"""Contains tests for ``curvlinops/gradient_moments.py``."""

from torch import float64

from curvlinops import EFLinearOperator
from curvlinops.examples.functorch import functorch_empirical_fisher
from test.utils import change_dtype, compare_consecutive_matmats, compare_matmat


def test_EFLinearOperator(case):
    """Test matrix-matrix multiplication with the (transposed) empirical Fisher.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    E = EFLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    E_mat = functorch_empirical_fisher(
        model_func, loss_func, params, data, input_key="x"
    ).detach()

    compare_consecutive_matmats(E)
    compare_matmat(E, E_mat)

    E, E_mat = E.adjoint(), E_mat.adjoint()
    compare_consecutive_matmats(E)
    compare_matmat(E, E_mat)
