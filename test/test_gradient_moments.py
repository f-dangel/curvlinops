"""Contains tests for ``curvlinops/gradient_moments.py``."""

from curvlinops import EFLinearOperator
from curvlinops.examples.functorch import functorch_empirical_fisher
from test.utils import compare_consecutive_matmats, compare_matmat


def test_EFLinearOperator(case):
    """Test matrix-matrix multiplication with the (transposed) empirical Fisher.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, batch_size_fn = case

    E = EFLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    E_mat = functorch_empirical_fisher(
        model_func, loss_func, params, data, input_key="x"
    )

    tols = {"atol": 5e-6, "rtol": 5e-4}

    compare_consecutive_matmats(E)
    compare_matmat(E, E_mat, **tols)

    E, E_mat = E.adjoint(), E_mat.adjoint()
    compare_consecutive_matmats(E)
    compare_matmat(E, E_mat, **tols)
