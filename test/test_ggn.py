"""Contains tests for ``curvlinops/ggn``."""

from test.utils import compare_matmat

from curvlinops import GGNLinearOperator
from curvlinops.examples.functorch import functorch_ggn


def test_GGNLinearOperator_matvec(case, adjoint: bool, is_vec: bool):
    """Test matrix-matrix multiplication with the GGN.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = case

    G = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    G_mat = functorch_ggn(model_func, loss_func, params, data, input_key="x")

    compare_matmat(G, G_mat, adjoint, is_vec, atol=1e-7)
