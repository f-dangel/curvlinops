"""Contains tests for ``curvlinops/fisher.py``."""

from pytest import mark
from torch import float64

from curvlinops import FisherMCLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from test.test_kfac import MC_TOLS
from test.utils import (
    change_dtype,
    compare_consecutive_matmats,
    compare_matmat_expectation,
)

MAX_REPEATS_MC_SAMPLES = [(4_000, 1), (40, 100)]
MAX_REPEATS_MC_SAMPLES_IDS = [
    f"max_repeats={n}-mc_samples={m}" for (n, m) in MAX_REPEATS_MC_SAMPLES
]
CHECK_EVERY = 100


@mark.parametrize(
    "max_repeats,mc_samples", MAX_REPEATS_MC_SAMPLES, ids=MAX_REPEATS_MC_SAMPLES_IDS
)
def test_FisherMCLinearOperator_expectation(case, max_repeats: int, mc_samples: int):
    """Test matrix-matrix multiplication with the Monte-Carlo Fisher.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    F = FisherMCLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        mc_samples=mc_samples,
    )
    G_mat = functorch_ggn(model_func, loss_func, params, data, input_key="x").detach()

    compare_consecutive_matmats(F)
    compare_matmat_expectation(F, G_mat, max_repeats, CHECK_EVERY, **MC_TOLS)

    F, G_mat = F.adjoint(), G_mat.adjoint()
    compare_consecutive_matmats(F)
    compare_matmat_expectation(F, G_mat, max_repeats, CHECK_EVERY, **MC_TOLS)
