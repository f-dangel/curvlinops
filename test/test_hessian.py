"""Contains tests for ``curvlinops/hessian``."""

from collections.abc import MutableMapping
from test.utils import compare_matmat

from pytest import mark, raises
from torch import block_diag

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from curvlinops.utils import split_list


@mark.parametrize("is_vec", [True, False], ids=["matvec", "matmat"])
def test_HessianLinearOperator(case, adjoint: bool, is_vec: bool):
    """Test matrix-matrix multiplication with the Hessian.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = case

    # Test when X is dict-like but batch_size_fn = None (default)
    if isinstance(data[0][0], MutableMapping):
        with raises(ValueError):
            _ = HessianLinearOperator(model_func, loss_func, params, data)

    H = HessianLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    H_mat = functorch_hessian(model_func, loss_func, params, data, input_key="x")

    compare_matmat(H, H_mat, adjoint, is_vec, rtol=1e-4, atol=1e-6)


BLOCKING_FNS = {
    "per-parameter": lambda params: [1 for _ in range(len(params))],
    "two-blocks": lambda params: (
        [1] if len(params) == 1 else [len(params) // 2, len(params) - len(params) // 2]
    ),
}


@mark.parametrize("blocking", BLOCKING_FNS.keys(), ids=BLOCKING_FNS.keys())
@mark.parametrize("is_vec", [True, False], ids=["matvec", "matmat"])
def test_blocked_HessianLinearOperator(
    case, adjoint: bool, blocking: str, is_vec: bool
):
    """Test matrix-matrix multiplication with the block-diagonal Hessian.

    Args:
        case: Tuple of model, loss function, parameters, data and batch size getter.
        adjoint: Whether to test the adjoint operator.
        blocking: Blocking scheme.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = case
    block_sizes = BLOCKING_FNS[blocking](params)

    H = HessianLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        block_sizes=block_sizes,
    )

    # compute the blocks with functorch and build the block diagonal matrix
    H_mat = [
        functorch_hessian(model_func, loss_func, params_block, data, input_key="x")
        for params_block in split_list(params, block_sizes)
    ]
    H_mat = block_diag(*H_mat)

    compare_matmat(H, H_mat, adjoint, is_vec, rtol=1e-4, atol=1e-6)
