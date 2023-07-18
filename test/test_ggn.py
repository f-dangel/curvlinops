"""Contains tests for ``curvlinops/ggn``."""

from numpy import random

from curvlinops import GGNLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.examples.utils import report_nonclose


def test_GGNLinearOperator_matvec(case, adjoint: bool):
    model_func, loss_func, params, data = case

    op = GGNLinearOperator(model_func, loss_func, params, data)
    op_functorch = (
        functorch_ggn(model_func, loss_func, params, data).detach().cpu().numpy()
    )
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    x = random.rand(op.shape[1])
    report_nonclose(op @ x, op_functorch @ x)


def test_GGNLinearOperator_matmat(case, adjoint: bool, num_vecs: int = 3):
    model_func, loss_func, params, data = case

    op = GGNLinearOperator(model_func, loss_func, params, data)
    op_functorch = (
        functorch_ggn(model_func, loss_func, params, data).detach().cpu().numpy()
    )
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    X = random.rand(op.shape[1], num_vecs)
    report_nonclose(op @ X, op_functorch @ X)
