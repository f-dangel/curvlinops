"""Contains tests for ``curvlinops/ggn_diagonal``."""

from typing import Dict

from pytest import mark
from torch import allclose, float64, zeros_like

from curvlinops.examples.functorch import functorch_ggn
from curvlinops.ggn_diagonal import GGNDiagonalComputer, GGNDiagonalLinearOperator
from test.utils import change_dtype, compare_consecutive_matmats, compare_matmat

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
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

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


@mark.parametrize("kwargs", DIAGONAL_CASES, ids=DIAGONAL_IDS)
def test_GGNDiagonalComputer_sequential_consistency(case, kwargs: Dict):
    """Calling compute() twice produces the same diagonal.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        kwargs: Additional keyword arguments for the GGN diagonal computation mode.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    computer = GGNDiagonalComputer(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn, **kwargs
    )
    diag1 = computer.compute()
    diag2 = computer.compute()
    for d1, d2 in zip(diag1, diag2):
        assert allclose(d1, d2)


def test_GGNDiagonalComputer_mc_different_seed(case):
    """Different seeds produce different MC diagonals.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    args = (model_func, loss_func, params, data)
    kwargs = {"batch_size_fn": batch_size_fn, "mode": "mc", "mc_samples": 1}

    comp1 = GGNDiagonalComputer(*args, **kwargs, seed=0)
    diag1 = comp1.compute()

    comp2 = GGNDiagonalComputer(*args, **kwargs, seed=1)
    diag2 = comp2.compute()

    assert all(
        not allclose(d1, d2) or allclose(d1, zeros_like(d1))
        for d1, d2 in zip(diag1, diag2)
    )
