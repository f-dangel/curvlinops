"""Contains tests for ``curvlinops/hessian``."""

from torch import float64, manual_seed, rand, rand_like
from torch.func import functional_call
from torch.nn import Linear, MSELoss, Sequential

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from test.utils import change_dtype, compare_consecutive_matmats, compare_matmat


def test_HessianLinearOperator(case):
    """Test matrix-matrix multiplication with the Hessian.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    H = HessianLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )

    H_mat = functorch_hessian(
        model_func, loss_func, params, data, input_key="x"
    ).detach()

    compare_consecutive_matmats(H)
    compare_matmat(H, H_mat)


def test_HessianLinearOperator_callable_model_func():
    """Test Hessian with a callable model_func and different parameter values."""
    manual_seed(0)
    model = Sequential(Linear(4, 3), Linear(3, 2)).to(dtype=float64)
    loss_func = MSELoss()
    data = [(rand(5, 4, dtype=float64), rand(5, 2, dtype=float64))]

    params_dict = {n: rand_like(p) for n, p in model.named_parameters()}

    def model_fn(params_dict, X):
        return functional_call(model, params_dict, (X,))

    H = HessianLinearOperator(model_fn, loss_func, params_dict, data)
    H_mat = functorch_hessian(model_fn, loss_func, params_dict, data).detach()

    compare_consecutive_matmats(H)
    compare_matmat(H, H_mat)
