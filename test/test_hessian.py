"""Contains tests for ``curvlinops/hessian``."""

from test.cases import CASES, initialize_case

from numpy import random
from pytest import mark

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from curvlinops.examples.utils import report_nonclose


@mark.parametrize("case", CASES)
def test_HessianLinearOperator_matvec(case):
    model_func, loss_func, params, data = initialize_case(case)

    H = HessianLinearOperator(model_func, loss_func, params, data)
    H_functorch = (
        functorch_hessian(model_func, loss_func, params, data).detach().cpu().numpy()
    )

    x = random.rand(H.shape[1])
    report_nonclose(H @ x, H_functorch @ x)


@mark.parametrize("case", CASES)
def test_HessianLinearOperator_matmat(case, num_vecs: int = 3):
    model_func, loss_func, params, data = initialize_case(case)

    H = HessianLinearOperator(model_func, loss_func, params, data)
    H_functorch = (
        functorch_hessian(model_func, loss_func, params, data).detach().cpu().numpy()
    )

    X = random.rand(H.shape[1], num_vecs)
    report_nonclose(H @ X, H_functorch @ X)
