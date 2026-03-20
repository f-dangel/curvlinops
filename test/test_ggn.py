"""Contains tests for ``curvlinops/ggn``."""

from pytest import mark
from torch import float64

from curvlinops import GGNLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from test.test_kfac import MC_TOLS
from test.utils import (
    change_dtype,
    compare_consecutive_matmats,
    compare_matmat,
    compare_matmat_expectation,
    to_functional,
)

MAX_REPEATS_MC_SAMPLES = [(4_000, 1), (40, 100)]
MAX_REPEATS_MC_SAMPLES_IDS = [
    f"max_repeats={n}-mc_samples={m}" for (n, m) in MAX_REPEATS_MC_SAMPLES
]
CHECK_EVERY = 100


def _test_ggn(model_func, loss_func, params, data, batch_size_fn):
    """Shared test logic for GGN (Module or callable)."""
    G = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    G_mat = functorch_ggn(model_func, loss_func, params, data, input_key="x").detach()

    compare_consecutive_matmats(G)
    compare_matmat(G, G_mat)


def test_GGNLinearOperator_matvec(case):
    """Test GGN with Module model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_ggn(*change_dtype(case, float64))


def test_GGNLinearOperator_functional(case):
    """Test GGN with callable model_func.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    _test_ggn(*to_functional(*change_dtype(case, float64)))


@mark.parametrize(
    "max_repeats,mc_samples", MAX_REPEATS_MC_SAMPLES, ids=MAX_REPEATS_MC_SAMPLES_IDS
)
def test_GGNLinearOperator_mc_expectation(case, max_repeats: int, mc_samples: int):
    """Test that the MC-approximated GGN converges to the exact GGN.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        max_repeats: Number of repetitions for expectation estimation.
        mc_samples: Number of Monte Carlo samples per estimate.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    G = GGNLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        mc_samples=mc_samples,
    )
    G_mat = functorch_ggn(model_func, loss_func, params, data, input_key="x").detach()

    compare_consecutive_matmats(G)
    compare_matmat_expectation(G, G_mat, max_repeats, CHECK_EVERY, **MC_TOLS)
