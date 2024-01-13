"""Contains tests for ``curvlinops/jacobian``."""

from numpy import random

from curvlinops import JacobianLinearOperator, TransposedJacobianLinearOperator
from curvlinops.examples.functorch import functorch_jacobian
from curvlinops.examples.utils import report_nonclose


def test_JacobianLinearOperator_matvec(case, adjoint: bool):
    model_func, _, params, data = case

    op = JacobianLinearOperator(model_func, params, data)
    op_functorch = functorch_jacobian(model_func, params, data).detach().cpu().numpy()
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    x = random.rand(op.shape[1])
    report_nonclose(op @ x, op_functorch @ x)


def test_JacobianLinearOperator_matmat(case, adjoint: bool, num_vecs: int = 3):
    model_func, _, params, data = case

    op = JacobianLinearOperator(model_func, params, data)
    op_functorch = functorch_jacobian(model_func, params, data).detach().cpu().numpy()
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    X = random.rand(op.shape[1], num_vecs)
    report_nonclose(op @ X, op_functorch @ X)


def test_TransposedJacobianLinearOperator_matvec(case, adjoint: bool):
    model_func, _, params, data = case

    op = TransposedJacobianLinearOperator(model_func, params, data)
    op_functorch = functorch_jacobian(model_func, params, data).detach().cpu().numpy().T
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    x = random.rand(op.shape[1])
    report_nonclose(op @ x, op_functorch @ x)


def test_TransposedJacobianLinearOperator_matmat(
    case, adjoint: bool, num_vecs: int = 3
):
    model_func, _, params, data = case

    op = TransposedJacobianLinearOperator(model_func, params, data)
    op_functorch = functorch_jacobian(model_func, params, data).detach().cpu().numpy().T
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    X = random.rand(op.shape[1], num_vecs)
    report_nonclose(op @ X, op_functorch @ X)
