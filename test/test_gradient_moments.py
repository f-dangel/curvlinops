"""Contains tests for ``curvlinops/gradient_moments.py``."""

from numpy import random

from curvlinops import EFLinearOperator
from curvlinops.examples.functorch import functorch_empirical_fisher
from curvlinops.examples.utils import report_nonclose


def test_EFLinearOperator_matvec(case, adjoint: bool):
    op = EFLinearOperator(*case)
    op_functorch = functorch_empirical_fisher(*case).detach().cpu().numpy()
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    x = random.rand(op.shape[1]).astype(op.dtype)
    report_nonclose(op @ x, op_functorch @ x)


def test_EFLinearOperator_matmat(case, adjoint: bool, num_vecs: int = 3):
    op = EFLinearOperator(*case)
    op_functorch = functorch_empirical_fisher(*case).detach().cpu().numpy()
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    X = random.rand(op.shape[1], num_vecs).astype(op.dtype)
    report_nonclose(op @ X, op_functorch @ X, atol=1e-7, rtol=1e-4)
