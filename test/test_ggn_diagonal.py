"""Contains tests for ``curvlinops/ggn_diagonal``."""

from typing import Dict

from pytest import mark
from torch import allclose, cat, float64, zeros_like

from curvlinops.examples.functorch import functorch_ggn
from curvlinops.ggn_diagonal import GGNDiagonalComputer
from curvlinops.utils import allclose_report
from test.utils import change_dtype

DIAGONAL_CASES = [{"mode": "exact"}, {"mode": "mc", "mc_samples": 20_000}]
DIAGONAL_IDS = [
    "_".join(f"{k}_{v}" for k, v in case.items()) for case in DIAGONAL_CASES
]


@mark.parametrize("kwargs", DIAGONAL_CASES, ids=DIAGONAL_IDS)
def test_GGNDiagonalComputer(case, kwargs: Dict):
    """Test GGN diagonal computation against functorch reference.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        kwargs: A dictionary containing additional keyword arguments for specifying how
            the GGN diagonal is approximated (either exactly or via Monte-Carlo).
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    diag = GGNDiagonalComputer(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn, **kwargs
    ).compute_ggn_diagonal()
    assert len(diag) == len(params)
    for d, p in zip(diag, params):
        assert d.shape == p.shape
    diag_flat = cat([d.flatten() for d in diag])

    diag_ref = (
        functorch_ggn(model_func, loss_func, params, data, input_key="x")
        .detach()
        .diag()
    )

    tols = {"exact": {}, "mc": {"atol": 1e-4, "rtol": 2e-2}}[kwargs["mode"]]
    assert allclose_report(diag_flat, diag_ref, **tols)


@mark.parametrize("kwargs", DIAGONAL_CASES, ids=DIAGONAL_IDS)
def test_GGNDiagonalComputer_sequential_consistency(case, kwargs: Dict):
    """Calling compute_ggn_diagonal() twice produces the same diagonal.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        kwargs: Additional keyword arguments for the GGN diagonal computation mode.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    computer = GGNDiagonalComputer(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn, **kwargs
    )
    diag1 = computer.compute_ggn_diagonal()
    diag2 = computer.compute_ggn_diagonal()
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
    diag1 = comp1.compute_ggn_diagonal()

    comp2 = GGNDiagonalComputer(*args, **kwargs, seed=1)
    diag2 = comp2.compute_ggn_diagonal()

    assert all(
        not allclose(d1, d2) or allclose(d1, zeros_like(d1))
        for d1, d2 in zip(diag1, diag2)
    )
