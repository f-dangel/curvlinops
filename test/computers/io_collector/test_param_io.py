"""Tests for with_param_io: collecting parameter in- and output relationships."""

from typing import Any

from pytest import raises
from torch import Tensor, arange, manual_seed, rand, zeros, zeros_like
from torch.func import functional_call
from torch.nn import Conv2d, Linear
from torch.nn.functional import conv2d, linear

from curvlinops.computers.io_collector import with_param_io
from curvlinops.computers.io_collector._base import NOT_A_PARAM
from curvlinops.computers.io_collector.conv import CONV_STR
from curvlinops.computers.io_collector.linear import LINEAR_STR
from curvlinops.utils import allclose_report


def compare_io(
    info: tuple[tuple[str | Tensor | None | dict[str, Any], ...], ...],
    info_true: tuple[tuple[str | Tensor | None | dict[str, Any], ...], ...],
) -> None:
    """Compare two layer info tuple collections for equality.

    Args:
        info: Info tuples from with_param_io output.
        info_true: Expected info tuples (ground truth).
    """
    assert len(info) == len(info_true), (
        f"Length mismatch: {len(info)} vs {len(info_true)}"
    )
    for layer_info, layer_info_true in zip(info, info_true):
        assert len(layer_info) == len(layer_info_true), (
            f"Layer info length mismatch: {len(layer_info)} vs {len(layer_info_true)}"
        )
        for item, item_true in zip(layer_info, layer_info_true):
            if isinstance(item, Tensor) and isinstance(item_true, Tensor):
                assert allclose_report(item, item_true)
            elif isinstance(item, dict) and isinstance(item_true, dict):
                assert item == item_true, f"Dict mismatch: {item} vs {item_true}"
            else:
                assert item == item_true, f"Value mismatch: {item} vs {item_true}"


def _verify_io(
    f,
    x: Tensor,
    params: dict[str, Tensor],
    io_true: tuple[tuple[str | Tensor | None | dict[str, Any], ...], ...],
) -> None:
    """Verify that with_param_io produces correct outputs and IO information.

    Args:
        f: Function to test, with signature f(x, params) -> output.
        x: Input tensor to the function.
        params: Dictionary mapping parameter names to parameter tensors.
        io_true: Expected tuple of layer information tuples.
    """
    y_true = f(x, params)

    dummy_x = zeros_like(x)
    dummy_params = {n: zeros_like(p) for n, p in params.items()}
    f_with_io = with_param_io(f, dummy_x, dummy_params)

    out = f_with_io(x, params)
    y, io = out[0], out[1:]

    assert allclose_report(y, y_true)
    compare_io(io, io_true)


def test_fully_connected():
    """Test with_param_io on a simple linear layer."""
    manual_seed(0)
    N, D_in, D_out = 2, 3, 4

    # 1) Both weight and bias as free parameters
    def f(x: Tensor, params: dict) -> Tensor:
        return linear(x, params["weight"], bias=params["bias"])

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in), "bias": rand(D_out)}
    io_true = ((LINEAR_STR, f(x, params), x, "weight", "bias", {}),)
    _verify_io(f, x, params, io_true)

    # 2) Only weight as free parameter (frozen bias)
    def f(x: Tensor, params: dict) -> Tensor:
        bias = arange(D_out, dtype=x.dtype)
        return linear(x, params["weight"], bias=bias)

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in)}
    io_true = ((LINEAR_STR, f(x, params), x, "weight", NOT_A_PARAM, {}),)
    _verify_io(f, x, params, io_true)

    # 3) Only bias as free parameter (frozen weight)
    def f(x: Tensor, params: dict) -> Tensor:
        weight = arange(D_out * D_in, dtype=x.dtype).reshape(D_out, D_in)
        return linear(x, weight, params["bias"])

    x, params = rand(N, D_in), {"bias": rand(D_out)}
    io_true = ((LINEAR_STR, f(x, params), x, NOT_A_PARAM, "bias", {}),)
    _verify_io(f, x, params, io_true)

    # 4) Without bias
    def f(x: Tensor, params: dict) -> Tensor:
        return linear(x, params["weight"])

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in)}
    io_true = ((LINEAR_STR, f(x, params), x, "weight", None, {}),)
    _verify_io(f, x, params, io_true)

    # 5) Use torch.nn
    fc = Linear(D_in, D_out, bias=True)

    def f(x: Tensor, params: dict) -> Tensor:
        return functional_call(fc, params, x)

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in), "bias": rand(D_out)}
    io_true = ((LINEAR_STR, f(x, params), x, "weight", "bias", {}),)
    _verify_io(f, x, params, io_true)


CONV2D_DEFAULT_PARAMS = {
    "stride": [1, 1],
    "padding": [0, 0],
    "dilation": [1, 1],
    "transposed": False,
    "output_padding": [0, 0],
    "groups": 1,
}


def test_convolution():
    """Test that convolution patterns are detected correctly."""
    manual_seed(0)
    N, C_out, C_in, K1, K2, I1, I2 = 2, 3, 4, 5, 6, 10, 11

    # 1) Standard convolution with weight and bias
    def f(x: Tensor, params: dict) -> Tensor:
        return conv2d(x, params["weight"], bias=params["bias"])

    x = rand(N, C_in, I1, I2)
    params = {"weight": rand(C_out, C_in, K1, K2), "bias": rand(C_out)}
    io_true = ((CONV_STR, f(x, params), x, "weight", "bias", CONV2D_DEFAULT_PARAMS),)
    _verify_io(f, x, params, io_true)

    # 2) Non-standard convolution with weight and bias
    def f(x: Tensor, params: dict) -> Tensor:
        stride = 2
        # NOTE Supply padding as keyword arg and bias and strides
        # as positional to attempt to confuse fx
        return conv2d(x, params["weight"], params["bias"], stride, padding=1)

    x = rand(N, C_in, I1, I2)
    params = {"weight": rand(C_out, C_in, K1, K2), "bias": rand(C_out)}
    hyperparams_true = {
        **CONV2D_DEFAULT_PARAMS,
        **{"stride": [2, 2], "padding": [1, 1]},
    }
    io_true = ((CONV_STR, f(x, params), x, "weight", "bias", hyperparams_true),)
    _verify_io(f, x, params, io_true)

    # 3) Standard convolution with weight only
    def f(x: Tensor, params: dict) -> Tensor:
        return conv2d(x, params["weight"])

    x = rand(N, C_in, I1, I2)
    params = {"weight": rand(C_out, C_in, K1, K2)}
    io_true = ((CONV_STR, f(x, params), x, "weight", None, CONV2D_DEFAULT_PARAMS),)
    _verify_io(f, x, params, io_true)

    # 4) Use torch.nn nn
    conv = Conv2d(C_in, C_out, (K1, K2), stride=(2, 1), bias=False)

    def f(x: Tensor, params: dict) -> Tensor:
        return functional_call(conv, params, x)

    hyperparams_true = {**CONV2D_DEFAULT_PARAMS, **{"stride": [2, 1]}}
    x, params = (rand(N, C_in, I1, I2), {"weight": rand(C_out, C_in, K1, K2)})
    io_true = ((CONV_STR, f(x, params), x, "weight", None, hyperparams_true),)
    _verify_io(f, x, params, io_true)


def test_unsupported_patterns():
    """Test that unsupported patterns raise ValueError."""
    manual_seed(0)
    N, D_in, D_out = 2, 3, 4

    # NOTE Although mathematically equivalent, these patterns are not implemented
    # and can therefore not be detected.
    # Unsupported: Using .T (calls aten.permute.default)
    # Unsupported: Using + (calls aten.add.Tensor)
    def f(x: Tensor, params: dict) -> Tensor:
        return x @ params["weight"].T + params["bias"]

    x_dummy = zeros(N, D_in)
    params_dummy = {"weight": zeros(D_out, D_in), "bias": zeros(D_out)}

    with raises(ValueError, match="Some parameters are used in unsupported patterns."):
        _ = with_param_io(f, x_dummy, params_dummy)


def test_multiple_parameter_usages():
    """Test multiple usages of the same parameter (recurrent structure)."""
    manual_seed(0)
    N, D = 2, 3

    def f(x: Tensor, params: dict) -> Tensor:
        xW = linear(x, params["weight"], bias=params["bias"])
        return linear(xW, params["weight"])

    x, params = rand(N, D), {"weight": rand(D, D), "bias": rand(D)}
    xW = linear(x, params["weight"], bias=params["bias"])
    io_true = (
        (LINEAR_STR, xW, x, "weight", "bias", {}),
        (LINEAR_STR, f(x, params), xW, "weight", None, {}),
    )
    _verify_io(f, x, params, io_true)


def test_undetected_parameter_paths():
    """Test case where not all parameter usage paths are detected."""
    manual_seed(0)
    N, D_in, D_out = 2, 3, 4

    def f(x: Tensor, params: dict) -> Tensor:
        W = params["weight"]
        return linear(x, W, bias=W.sum(1))

    x_dummy = zeros(N, D_in)
    params_dummy = {"weight": zeros(D_out, D_in)}

    with raises(ValueError, match="Some parameters are used in unsupported patterns."):
        _ = with_param_io(f, x_dummy, params_dummy)


def test_supports_multiple_batch_sizes():
    """Test with_param_io supports batch sizes other than the one used for tracing."""
    manual_seed(0)
    N1, N2, D_in, D_out = 2, 3, 4, 5

    def f(x: Tensor, params: dict) -> Tensor:
        return linear(x, params["weight"], bias=params["bias"])

    x1, x2 = rand(N1, D_in), rand(N2, D_in)
    params = {"weight": rand(D_out, D_in), "bias": rand(D_out)}

    # Use x1 for tracing
    f_with_io = with_param_io(f, x1, params)

    # Check IO for batch sizes N1 and N2
    for x in [x1, x2]:
        io_true = ((LINEAR_STR, f(x, params), x, "weight", "bias", {}),)
        io = f_with_io(x, params)[1:]
        compare_io(io, io_true)
