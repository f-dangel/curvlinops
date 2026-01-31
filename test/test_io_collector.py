"""Tests collecting parameter in- and output relationships."""

from typing import Tuple, Union

from torch import Tensor, manual_seed, rand, zeros_like, arange, zeros
from torch.nn.functional import linear
from torch.nn import Linear
from torch.func import functional_call
from curvlinops.utils import allclose_report
from pytest import raises

from curvlinops.io_collector import with_param_io
from curvlinops.io_patterns import NOT_A_PARAM


def compare_io(
    info: Tuple[Tuple[Union[str, Tensor, None], ...], ...],
    info_true: Tuple[Tuple[Union[str, Tensor, None], ...], ...],
) -> None:
    """Compare two layer info tuple collections for equality.

    Args:
        info: Info tuples from with_param_io output.
        info_true: Expected info tuples (ground truth).

    Raises:
        AssertionError: If the tuples don't match.
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
            else:
                assert item == item_true, f"Value mismatch: {item} vs {item_true}"


def _verify_io(f, x, params, io_true):
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
    io_true = (("Linear(y=x@W^T+b)", x, f(x, params), "weight", "bias"),)
    _verify_io(f, x, params, io_true)

    # 2) Only weight as free parameter (frozen bias)
    def f(x: Tensor, params: dict) -> Tensor:
        bias = arange(D_out, dtype=x.dtype)
        return linear(x, params["weight"], bias=bias)

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in)}
    io_true = (("Linear(y=x@W^T+b)", x, f(x, params), "weight", NOT_A_PARAM),)
    _verify_io(f, x, params, io_true)

    # 3) Fully-connected layer with only bias as free parameter (frozen weight)
    def f(x: Tensor, params: dict) -> Tensor:
        weight = arange(D_out * D_in, dtype=x.dtype).reshape(D_out, D_in)
        return linear(x, weight, params["bias"])

    x, params = rand(N, D_in), {"bias": rand(D_out)}
    io_true = (("Linear(y=x@W^T+b)", x, f(x, params), NOT_A_PARAM, "bias"),)
    _verify_io(f, x, params, io_true)

    # 4) Fully-connected layer without bias
    def f(x: Tensor, params: dict) -> Tensor:
        return linear(x, params["weight"])

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in)}
    io_true = (("Linear(y=x@W^T+b)", x, f(x, params), "weight", None),)
    _verify_io(f, x, params, io_true)

    # 5) Use torch.nn
    fc = Linear(D_out, D_in, bias=True)

    def f(x: Tensor, params: dict) -> Tensor:
        return functional_call(fc, params, x)

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in), "bias": rand(D_out)}
    io_true = (("Linear(y=x@W^T+b)", x, f(x, params), "weight", "bias"),)
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

    with raises(ValueError, match="is used in an unsupported pattern"):
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
        ("Linear(y=x@W^T+b)", x, xW, "weight", "bias"),
        ("Linear(y=x@W^T+b)", xW, f(x, params), "weight", None),
    )
    _verify_io(f, x, params, io_true)


def test_undetected_parameter_paths():
    """Test case where not all parameter usage paths are detected."""
    manual_seed(0)
    N, D_in, D_out = 2, 3, 4

    def f(x: Tensor, params: dict) -> Tensor:
        W = params["weight"]
        return linear(x, W, bias=W.sum(1))

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in)}

    # TODO This example is breaking the path detection.
    # NOTE Maybe I have to introduce a W and WT node
    # NOTE I need to implement a function that returns all paths.
    with raises(ValueError, match="is used in an unsupported pattern"):
        x_dummy = zeros_like(x)
        params_dummy = {n: zeros_like(p) for n, p in params.items()}
        _ = with_param_io(f, x_dummy, params_dummy)
