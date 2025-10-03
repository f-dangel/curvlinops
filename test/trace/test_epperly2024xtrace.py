"""Test ``curvlinops.trace.epperli2024xtrace."""

from functools import partial
from typing import Union

from pytest import mark
from torch import Tensor, column_stack, dot, isclose, manual_seed, rand, trace
from torch.linalg import qr

from curvlinops import xtrace
from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.sampling import random_vector
from test.trace import DISTRIBUTION_IDS, DISTRIBUTIONS
from test.utils import check_estimator_convergence

NUM_MATVECS = [6, 8]
NUM_MATVEC_IDS = [f"num_matvecs={num_matvecs}" for num_matvecs in NUM_MATVECS]


def xtrace_naive(
    A: Union[PyTorchLinearOperator, Tensor],
    num_matvecs: int,
    distribution: str = "rademacher",
) -> Tensor:
    """Naive reference implementation of XTrace.

    See Algorithm 1.2 in https://arxiv.org/pdf/2301.07825.

    Args:
        A: A square linear operator.
        num_matvecs: Total number of matrix-vector products to use. Must be even and
            less than the dimension of the linear operator.
        distribution: Distribution of the random vectors used for the trace estimation.
            Can be either ``'rademacher'`` or ``'normal'``. Default: ``'rademacher'``.

    Returns:
        The estimated trace of the linear operator.

    Raises:
        ValueError: If the linear operator is not square or if the number of matrix-
            vector products is not even or is greater than the dimension of the linear
            operator.
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square. Got shape {A.shape}.")
    dim = A.shape[1]
    if num_matvecs % 2 != 0 or num_matvecs >= dim:
        raise ValueError(
            "num_matvecs must be even and less than the dimension of A.",
            f" Got {num_matvecs}.",
        )
    num_vecs = num_matvecs // 2

    W = column_stack(
        [random_vector(dim, distribution, A.device, A.dtype) for _ in range(num_vecs)]
    )
    A_W = A @ W

    traces = []

    for i in range(num_vecs):
        # compute the exact trace in the basis spanned by the sketch matrix without
        # test vector i
        not_i = [j for j in range(num_vecs) if j != i]
        Q_i, _ = qr(A_W[:, not_i])
        A_Q_i = A @ Q_i
        tr_QT_i_A_Q_i = trace(Q_i.T @ A_Q_i)

        # apply vanilla Hutchinson in the complement, using test vector i
        w_i = W[:, i]
        A_w_i = A_W[:, i]
        A_P_w_i = A_w_i - A_Q_i @ (Q_i.T @ w_i)
        PT_A_P_w_i = A_P_w_i - Q_i @ (Q_i.T @ A_P_w_i)
        tr_w_i = dot(w_i, PT_A_P_w_i)

        traces.append(tr_QT_i_A_Q_i + tr_w_i)

    return sum(traces) / len(traces)


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
def test_xtrace(distribution: str, num_matvecs: int):
    """Test whether the XTrace estimator converges to the true trace.

    Args:
        distribution: Distribution of the random vectors used for the trace estimation.
        num_matvecs: Number of matrix-vector multiplications used by one estimator.
    """
    manual_seed(0)
    A = rand(15, 15)
    estimator = partial(xtrace, A=A, num_matvecs=num_matvecs, distribution=distribution)
    check_estimator_convergence(
        estimator,
        num_matvecs,
        A.trace(),
        # use half the target tolerance as vanilla Hutchinson
        target_rel_error=5e-4,
    )


@mark.parametrize("num_matvecs", NUM_MATVECS, ids=NUM_MATVEC_IDS)
@mark.parametrize("distribution", DISTRIBUTIONS, ids=DISTRIBUTION_IDS)
def test_xtrace_matches_naive(num_matvecs: int, distribution: str, num_seeds: int = 5):
    """Test whether the efficient implementation of XTrace matches the naive.

    Args:
        num_matvecs: Number of matrix-vector multiplications used by one estimator.
        distribution: Distribution of the random vectors used for the trace estimation.
        num_seeds: Number of different seeds to test the estimators with.
            Default: ``5``.
    """
    manual_seed(0)
    A = rand(50, 50)

    # check for different seeds
    for i in range(num_seeds):
        manual_seed(i)
        efficient = xtrace(A, num_matvecs, distribution=distribution)
        manual_seed(i)
        naive = xtrace_naive(A, num_matvecs, distribution=distribution)
        assert isclose(efficient, naive)
