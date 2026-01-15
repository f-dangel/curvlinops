"""Contains tests for ``curvlinops/hessian``."""

from typing import Callable, List, Optional

from torch import block_diag, float64
from torch.nn import Parameter

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from curvlinops.utils import split_list
from test.utils import change_dtype, compare_consecutive_matmats, compare_matmat


def test_HessianLinearOperator(
    case,
    block_sizes_fn: Callable[[List[Parameter]], Optional[List[int]]],
):
    """Test matrix-matrix multiplication with the Hessian.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        block_sizes_fn: The function that generates the block sizes used to define
            block diagonal approximations from the parameters.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    block_sizes = block_sizes_fn(params)

    H = HessianLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        block_sizes=block_sizes,
    )

    # compute the blocks with functorch and build the block diagonal matrix
    H_blocks = [
        functorch_hessian(
            model_func, loss_func, params_block, data, input_key="x"
        ).detach()
        for params_block in split_list(
            params, [len(params)] if block_sizes is None else block_sizes
        )
    ]
    H_mat = block_diag(*H_blocks)

    compare_consecutive_matmats(H)
    compare_matmat(H, H_mat)

    H, H_mat = H.adjoint(), H_mat.adjoint()
    compare_consecutive_matmats(H)
    compare_matmat(H, H_mat)
