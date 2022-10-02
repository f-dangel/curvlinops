"""Contains tests for ``curvlinops/ggn``."""

from test.cases import CASES, initialize_case

from numpy import random
from pytest import mark

from curvlinops import GGNLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.examples.utils import report_nonclose


@mark.parametrize("case", CASES)
def test_GGNLinearOperator_matvec(case):
    model_func, loss_func, params, data = initialize_case(case)

    GGN = GGNLinearOperator(model_func, loss_func, params, data)
    GGN_functorch = (
        functorch_ggn(model_func, loss_func, params, data).detach().cpu().numpy()
    )

    x = random.rand(GGN.shape[1])
    report_nonclose(GGN @ x, GGN_functorch @ x)


@mark.parametrize("case", CASES)
def test_GGNLinearOperator_matmat(case, num_vecs: int = 3):
    model_func, loss_func, params, data = initialize_case(case)

    GGN = GGNLinearOperator(model_func, loss_func, params, data)
    GGN_functorch = (
        functorch_ggn(model_func, loss_func, params, data).detach().cpu().numpy()
    )

    X = random.rand(GGN.shape[1], num_vecs)
    report_nonclose(GGN @ X, GGN_functorch @ X)