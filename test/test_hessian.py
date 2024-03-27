"""Contains tests for ``curvlinops/hessian``."""

from numpy import random
from pytest import mark
from torch import block_diag

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from curvlinops.examples.utils import report_nonclose
from curvlinops.utils import split_list


def test_HessianLinearOperator_matvec(case, adjoint: bool):
    op = HessianLinearOperator(*case)
    op_functorch = functorch_hessian(*case).detach().cpu().numpy()
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    x = random.rand(op.shape[1])
    report_nonclose(op @ x, op_functorch @ x, atol=1e-7)


def test_HessianLinearOperator_matmat(case, adjoint: bool, num_vecs: int = 3):
    op = HessianLinearOperator(*case)
    op_functorch = functorch_hessian(*case).detach().cpu().numpy()
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    X = random.rand(op.shape[1], num_vecs)
    report_nonclose(op @ X, op_functorch @ X, atol=1e-6, rtol=5e-4)


BLOCKING_FNS = {
    "per-parameter": lambda params: [1 for _ in range(len(params))],
    "two-blocks": lambda params: (
        [1] if len(params) == 1 else [len(params) // 2, len(params) - len(params) // 2]
    ),
}


@mark.parametrize("blocking", BLOCKING_FNS.keys(), ids=BLOCKING_FNS.keys())
def test_blocked_HessianLinearOperator_matmat(
    case, adjoint: bool, blocking: str, num_vecs: int = 2
):
    """Test matrix-matrix multiplication with the block-diagonal Hessian.

    Args:
        case: Tuple of model, loss function, parameters, and data.
        adjoint: Whether to test the adjoint operator.
        blocking: Blocking scheme.
        num_vecs: Number of vectors to multiply with. Default is ``2``.
    """
    model, loss_func, params, data = case
    block_sizes = BLOCKING_FNS[blocking](params)

    op = HessianLinearOperator(model, loss_func, params, data, block_sizes=block_sizes)

    # compute the blocks with functorch and build the block diagonal matrix
    op_functorch = [
        functorch_hessian(model, loss_func, params_block, data).detach()
        for params_block in split_list(params, block_sizes)
    ]
    op_functorch = block_diag(*op_functorch).cpu().numpy()

    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    X = random.rand(op.shape[1], num_vecs)
    report_nonclose(op @ X, op_functorch @ X, atol=1e-6, rtol=5e-4)
