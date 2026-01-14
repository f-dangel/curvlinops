"""Contains tests for ``curvlinops/ggn``."""

from curvlinops import GGNLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from test.utils import compare_consecutive_matmats, compare_matmat


def test_GGNLinearOperator_matvec(case):
    """Test matrix-matrix multiplication with the GGN.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, batch_size_fn = case

    G = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    G_mat = functorch_ggn(model_func, loss_func, params, data, input_key="x")

    tols = {"atol": 1e-7, "rtol": 1e-4}

    compare_consecutive_matmats(G)
    compare_matmat(G, G_mat, **tols)

    G, G_mat = G.adjoint(), G_mat.adjoint()
    compare_consecutive_matmats(G)
    compare_matmat(G, G_mat, **tols)
