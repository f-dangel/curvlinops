"""Contains tests for ``curvlinops/fisher.py``."""

from pytest import mark

from curvlinops import FisherMCLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from test.utils import compare_consecutive_matmats, compare_matmat_expectation

MAX_REPEATS_MC_SAMPLES = [(10_000, 1), (100, 100)]
MAX_REPEATS_MC_SAMPLES_IDS = [
    f"max_repeats={n}-mc_samples={m}" for (n, m) in MAX_REPEATS_MC_SAMPLES
]
CHECK_EVERY = 100


@mark.montecarlo
@mark.parametrize(
    "max_repeats,mc_samples", MAX_REPEATS_MC_SAMPLES, ids=MAX_REPEATS_MC_SAMPLES_IDS
)
def test_FisherMCLinearOperator_expectation(
    case, adjoint: bool, is_vec: bool, max_repeats: int, mc_samples: int
):
    """Test matrix-matrix multiplication with the Monte-Carlo Fisher.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = case

    F = FisherMCLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        mc_samples=mc_samples,
    )
    G_mat = functorch_ggn(model_func, loss_func, params, data, input_key="x")

    compare_consecutive_matmats(F, adjoint, is_vec)
    compare_matmat_expectation(
        F, G_mat, adjoint, is_vec, max_repeats, CHECK_EVERY, rtol=2e-1, atol=5e-3
    )
