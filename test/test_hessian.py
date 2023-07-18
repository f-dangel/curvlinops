"""Contains tests for ``curvlinops/hessian``."""

from numpy import random

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from curvlinops.examples.utils import report_nonclose


def test_HessianLinearOperator_matvec(case, adjoint: bool):
    op = HessianLinearOperator(*case)
    op_functorch = functorch_hessian(*case).detach().cpu().numpy()
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    x = random.rand(op.shape[1])
    report_nonclose(op @ x, op_functorch @ x)


def test_HessianLinearOperator_matmat(case, adjoint: bool, num_vecs: int = 3):
    op = HessianLinearOperator(*case)
    op_functorch = functorch_hessian(*case).detach().cpu().numpy()
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    X = random.rand(op.shape[1], num_vecs)
    report_nonclose(op @ X, op_functorch @ X, atol=1e-6, rtol=5e-4)
