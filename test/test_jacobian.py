"""Contains tests for ``curvlinops/jacobian``."""

from numpy import random

from curvlinops import JacobianLinearOperator
from curvlinops.examples.functorch import functorch_jacobian
from curvlinops.examples.utils import report_nonclose


def test_JacobianLinearOperator_matvec(case):
    model_func, _, params, data = case

    op = JacobianLinearOperator(model_func, params, data)
    op_functorch = functorch_jacobian(model_func, params, data).detach().cpu().numpy()

    x = random.rand(op.shape[1])
    report_nonclose(op @ x, op_functorch @ x)


def test_JacobianLinearOperator_matmat(case, num_vecs: int = 3):
    model_func, _, params, data = case

    op = JacobianLinearOperator(model_func, params, data)
    op_functorch = functorch_jacobian(model_func, params, data).detach().cpu().numpy()

    X = random.rand(op.shape[1], num_vecs)
    report_nonclose(op @ X, op_functorch @ X)
