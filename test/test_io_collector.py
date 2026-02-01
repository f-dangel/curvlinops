"""Tests collecting parameter in- and output relationships."""

from typing import Callable, Dict, Tuple, Union

from pytest import raises
from torch import Tensor, arange, manual_seed, rand, zeros, zeros_like
from torch.func import functional_call
from torch.nn import Linear
from torch.nn.functional import linear

from curvlinops.io_collector import with_kfac_io, with_param_io
from curvlinops.io_patterns import NOT_A_PARAM
from curvlinops.utils import allclose_report


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


def _verify_io(
    f: Callable[[Tensor, Dict[str, Tensor]], Tensor],
    x: Tensor,
    params: Dict[str, Tensor],
    io_true: Tuple[Tuple[Union[str, Tensor, None], ...], ...],
) -> None:
    """Verify that with_param_io produces correct outputs and IO information.

    Tests that the traced function with IO collection produces the same output
    as the original function and captures the expected layer IO relationships.

    Args:
        f: Function to test, with signature f(x, params) -> output.
        x: Input tensor to the function.
        params: Dictionary mapping parameter names to parameter tensors.
        io_true: Expected tuple of layer information tuples.
            Each layer info tuple contains:
            (layer_type, output_node, input_node, weight_name, bias_name)

    Raises:
        AssertionError: If the function output doesn't match the expected output
            or if the collected IO information doesn't match expectations.
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
    io_true = (("Linear(y=x@W^T+b)", f(x, params), x, "weight", "bias"),)
    _verify_io(f, x, params, io_true)

    # 2) Only weight as free parameter (frozen bias)
    def f(x: Tensor, params: dict) -> Tensor:
        bias = arange(D_out, dtype=x.dtype)
        return linear(x, params["weight"], bias=bias)

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in)}
    io_true = (("Linear(y=x@W^T+b)", f(x, params), x, "weight", NOT_A_PARAM),)
    _verify_io(f, x, params, io_true)

    # 3) Fully-connected layer with only bias as free parameter (frozen weight)
    def f(x: Tensor, params: dict) -> Tensor:
        weight = arange(D_out * D_in, dtype=x.dtype).reshape(D_out, D_in)
        return linear(x, weight, params["bias"])

    x, params = rand(N, D_in), {"bias": rand(D_out)}
    io_true = (("Linear(y=x@W^T+b)", f(x, params), x, NOT_A_PARAM, "bias"),)
    _verify_io(f, x, params, io_true)

    # 4) Fully-connected layer without bias
    def f(x: Tensor, params: dict) -> Tensor:
        return linear(x, params["weight"])

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in)}
    io_true = (("Linear(y=x@W^T+b)", f(x, params), x, "weight", None),)
    _verify_io(f, x, params, io_true)

    # 5) Use torch.nn
    fc = Linear(D_out, D_in, bias=True)

    def f(x: Tensor, params: dict) -> Tensor:
        return functional_call(fc, params, x)

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in), "bias": rand(D_out)}
    io_true = (("Linear(y=x@W^T+b)", f(x, params), x, "weight", "bias"),)
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
        ("Linear(y=x@W^T+b)", xW, x, "weight", "bias"),
        ("Linear(y=x@W^T+b)", f(x, params), xW, "weight", None),
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
        io_true = (("Linear(y=x@W^T+b)", f(x, params), x, "weight", "bias"),)
        io = f_with_io(x, params)[1:]
        compare_io(io, io_true)


def test_with_kfac_io_multiple_parameter_usages():
    manual_seed(0)
    N, D = 2, 3

    def f(x: Tensor, params: dict) -> Tensor:
        xW = linear(x, params["weight"], bias=params["bias"])
        return linear(xW, params["weight"])

    x, params = rand(N, D), {"weight": rand(D, D), "bias": rand(D)}

    with raises(ValueError, match="Parameters used multiple times"):
        with_kfac_io(f, x, params, "empirical")
