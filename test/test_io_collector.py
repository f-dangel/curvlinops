"""Tests collecting parameter in- and output relationships."""

from typing import Tuple, Union

from torch import Tensor, manual_seed, rand, zeros_like, arange
from torch.nn.functional import linear
from curvlinops.utils import allclose_report

from curvlinops.io_collector import with_param_io


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
    io_true = (("Linear(y=x@W^T+b)", x, f(x, params), "weight", "__not_a_param"),)
    _verify_io(f, x, params, io_true)

    # 3) Fully-connected layer with only bias as free parameter (frozen weight)
    def f(x: Tensor, params: dict) -> Tensor:
        weight = arange(D_out * D_in, dtype=x.dtype).reshape(D_out, D_in)
        return linear(x, weight, params["bias"])

    x, params = rand(N, D_in), {"bias": rand(D_out)}
    io_true = (("Linear(y=x@W^T+b)", x, f(x, params), "__not_a_param", "bias"),)
    _verify_io(f, x, params, io_true)

    # Fully-connected layer without bias
    def f(x: Tensor, params: dict) -> Tensor:
        return linear(x, params["weight"])

    x, params = rand(N, D_in), {"weight": rand(D_out, D_in)}
    io_true = (("Linear(y=x@W^T+b)", x, f(x, params), "weight", None),)
    _verify_io(f, x, params, io_true)
