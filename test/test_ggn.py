"""Contains tests for ``curvlinops/ggn``."""

from typing import Dict

from pytest import mark
from torch import float64

from curvlinops import GGNLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.ggn import GGNDiagonalLinearOperator
from test.utils import change_dtype, compare_consecutive_matmats, compare_matmat


def test_GGNLinearOperator_matvec(case):
    """Test matrix-matrix multiplication with the GGN.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    G = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    G_mat = functorch_ggn(model_func, loss_func, params, data, input_key="x").detach()

    compare_consecutive_matmats(G)
    compare_matmat(G, G_mat, atol=1e-7, rtol=1e-4)


DIAGONAL_CASES = [{"mode": "exact"}, {"mode": "mc", "mc_samples": 20_000}]
DIAGONAL_IDS = [
    "_".join(f"{k}_{v}" for k, v in case.items()) for case in DIAGONAL_CASES
]


@mark.parametrize("kwargs", DIAGONAL_CASES, ids=DIAGONAL_IDS)
def test_GGNDiagonalLinearOperator_matvec(case, kwargs: Dict):
    """Test matrix-matrix multiplication with the GGN diagonal.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        kwargs: A dictionary containing additional keyword arguments for specifying how
            the GGN diagonal is approximated (either exactly or via Monte-Carlo).
    """
    model_func, loss_func, params, data, batch_size_fn = case

    G = GGNDiagonalLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn, **kwargs
    )
    G_mat = (
        functorch_ggn(model_func, loss_func, params, data, input_key="x")
        .detach()
        .diag()  # extract the diagonal
        .diag()  # embed it into a matrix
    )

    compare_consecutive_matmats(G)
    tols = {
        "atol": {"exact": 1e-7, "mc": 1e-4}[kwargs["mode"]],
        "rtol": {"exact": 1e-4, "mc": 2e-2}[kwargs["mode"]],
    }
    compare_matmat(G, G_mat, **tols)
