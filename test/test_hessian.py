"""Contains tests for ``curvlinops/hessian``."""

from numpy import random

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from curvlinops.examples.utils import report_nonclose


def test_HessianLinearOperator_matvec(case):
    H = HessianLinearOperator(*case)
    H_functorch = functorch_hessian(*case).detach().cpu().numpy()

    x = random.rand(H.shape[1])
    report_nonclose(H @ x, H_functorch @ x)


def test_HessianLinearOperator_matmat(case, num_vecs: int = 3):
    H = HessianLinearOperator(*case)
    H_functorch = functorch_hessian(*case).detach().cpu().numpy()

    X = random.rand(H.shape[1], num_vecs)
    report_nonclose(H @ X, H_functorch @ X, atol=1e-6, rtol=5e-4)
