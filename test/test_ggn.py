"""Contains tests for ``curvlinops/ggn``."""

from typing import Dict

from pytest import mark, raises
from torch import Tensor

from curvlinops import GGNLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.ggn import GGNDiagonalLinearOperator
from test.utils import compare_consecutive_matmats, compare_matmat


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

    compare_consecutive_matmats(G, adjoint, is_vec)
    compare_matmat(G, G_mat, adjoint, is_vec, atol=1e-7, rtol=1e-4)


DIAGONAL_CASES = [{"mode": "exact"}, {"mode": "mc", "mc_samples": 20_000}]
DIAGONAL_IDS = [
    "_".join(f"{k}_{v}" for k, v in case.items()) for case in DIAGONAL_CASES
]


@mark.parametrize("kwargs", DIAGONAL_CASES, ids=DIAGONAL_IDS)
def test_GGNDiagonalLinearOperator_matvec(
    case, adjoint: bool, is_vec: bool, kwargs: Dict
):
    """Test matrix-matrix multiplication with the GGN diagonal.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
        kwargs: A dictionary containing additional keyword arguments for specifying how
            the GGN diagonal is approximated (either exactly or via Monte-Carlo).
    """
    model_func, loss_func, params, data, batch_size_fn = case
    print(case)
    print(loss_func.reduction)

    def _construct_G():
        return GGNDiagonalLinearOperator(
            model_func, loss_func, params, data, batch_size_fn=batch_size_fn, **kwargs
        )

    # Check that non-tensor inputs raise an error
    if any(not isinstance(X, Tensor) for (X, _) in data):
        with raises(RuntimeError, match="Only Tensor inputs are supported."):
            _construct_G()
        return

    G = _construct_G()
    G_mat = (
        functorch_ggn(model_func, loss_func, params, data, input_key="x")
        .detach()
        .diag()  # extract the diagonal
        .diag()  # embed it into a matrix
    )

    compare_consecutive_matmats(G, adjoint, is_vec)
    tols = {
        "atol": {"exact": 1e-7, "mc": 1e-4}[kwargs["mode"]],
        "rtol": {"exact": 1e-4, "mc": 2e-2}[kwargs["mode"]],
    }
    compare_matmat(G, G_mat, adjoint, is_vec, **tols)
