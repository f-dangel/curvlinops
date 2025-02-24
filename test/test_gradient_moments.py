"""Contains tests for ``curvlinops/gradient_moments.py``."""

from curvlinops import EFLinearOperator
from curvlinops.examples.functorch import functorch_empirical_fisher
from test.utils import compare_consecutive_matmats, compare_matmat


def test_EFLinearOperator(case, adjoint: bool, is_vec: bool):
    """Test matrix-matrix multiplication with the empirical Fisher.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = case

    E = EFLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    E_mat = functorch_empirical_fisher(
        model_func, loss_func, params, data, input_key="x"
    )

    compare_consecutive_matmats(E, adjoint, is_vec)
    compare_matmat(E, E_mat, adjoint, is_vec, rtol=5e-4, atol=5e-6)
