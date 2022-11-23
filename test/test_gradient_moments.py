"""Contains tests for ``curvlinops/gradient_moments.py``."""

from numpy import random

from curvlinops import EFLinearOperator
from curvlinops.examples.functorch import functorch_empirical_fisher
from curvlinops.examples.utils import report_nonclose


def test_EFLinearOperator_matvec(case):
    EF = EFLinearOperator(*case)
    EF_functorch = functorch_empirical_fisher(*case).detach().cpu().numpy()

    x = random.rand(EF.shape[1]).astype(EF.dtype)
    report_nonclose(EF @ x, EF_functorch @ x)


def test_EFLinearOperator_matmat(case, num_vecs: int = 3):
    EF = EFLinearOperator(*case)
    EF_functorch = functorch_empirical_fisher(*case).detach().cpu().numpy()

    X = random.rand(EF.shape[1], num_vecs).astype(EF.dtype)
    report_nonclose(EF @ X, EF_functorch @ X)
