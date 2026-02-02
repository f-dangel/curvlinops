"""Tests collecting parameter in- and output relationships."""

from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union

from pytest import mark, raises
from torch import (
    Tensor,
    arange,
    flatten,
    manual_seed,
    rand,
    randn_like,
    zeros,
    zeros_like,
)
from torch.func import functional_call
from torch.nn import (
    AdaptiveAvgPool2d,
    Conv2d,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from torch.nn.functional import adaptive_avg_pool2d, conv2d, linear, max_pool2d, relu

from curvlinops.io_collector import with_kfac_io, with_param_io
from curvlinops.io_patterns._base import NOT_A_PARAM
from curvlinops.io_patterns.conv import CONV_STR
from curvlinops.io_patterns.linear import LINEAR_STR
from curvlinops.kfac import FisherType
from curvlinops.utils import allclose_report


def compare_io(
    info: Tuple[Tuple[Union[str, Tensor, None, Dict[str, Any]], ...], ...],
    info_true: Tuple[Tuple[Union[str, Tensor, None, Dict[str, Any]], ...], ...],
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
            elif isinstance(item, dict) and isinstance(item_true, dict):
                assert item == item_true, f"Dict mismatch: {item} vs {item_true}"
            else:
                assert item == item_true, f"Value mismatch: {item} vs {item_true}"


def _verify_io(
    f: Callable[[Tensor, Dict[str, Tensor]], Tensor],
    x: Tensor,
    params: Dict[str, Tensor],
    io_true: Tuple[Tuple[Union[str, Tensor, None, Dict[str, Any]], ...], ...],
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
            (layer_type, output_node, input_node, weight_name, bias_name, hyperparams)

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
    fc = Linear(D_out, D_in, bias=True)

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


def test_with_kfac_io_multiple_parameter_usages():
    manual_seed(0)
    N, D = 2, 3

    def f(x: Tensor, params: dict) -> Tensor:
        xW = linear(x, params["weight"], bias=params["bias"])
        return linear(xW, params["weight"])

    x, params = rand(N, D), {"weight": rand(D, D), "bias": rand(D)}

    with raises(ValueError, match="Parameters used multiple times"):
        with_kfac_io(f, x, params, "empirical")


def _verify_kfac_io(
    f: Callable[[Tensor, Dict[str, Tensor]], Tensor],
    x: Tensor,
    params: Dict[str, Tensor],
    fisher_type: str,
    kfac_io_true: Tuple[
        Tensor,
        Dict[str, Tensor],
        Dict[str, Tensor],
        Dict[str, Dict[str, str]],
        Dict[str, Dict[str, Any]],
    ],
) -> None:
    y_true, inputs_true, outputs_true, layers_true, hyperparams_true = kfac_io_true

    dummy_x = zeros_like(x)
    dummy_params = {n: zeros_like(p) for n, p in params.items()}
    f_and_kfac_io = with_kfac_io(f, dummy_x, dummy_params, fisher_type)

    y, inputs, outputs, layers, hyperparams = f_and_kfac_io(x, params)

    # Compare function value
    assert allclose_report(y, y_true)

    # Compare layer inputs
    assert set(inputs.keys()) == set(inputs_true.keys())
    for key in inputs_true:
        assert allclose_report(inputs[key], inputs_true[key])

    # Compare layer outputs
    assert set(outputs.keys()) == set(outputs_true.keys())
    for key in outputs_true:
        assert allclose_report(outputs[key], outputs_true[key])

    # Compare layer dicts
    assert layers == layers_true

    # Compare hyperparameters
    assert hyperparams == hyperparams_true


def test_with_kfac_io_fully_connected():
    """Test with_kfac_io on a simple linear layer.

    Verifies correctness of returned KFAC IO information, as well as its dependency
    on the Fisher type (forward-only KFAC and bias-only layers do not require storing
    inputs).
    """
    manual_seed(0)
    N, D_in, D_out = 2, 3, 4

    # 1) Both weight and bias as free parameters
    def f(x: Tensor, params: dict) -> Tensor:
        return linear(x, params["w"], bias=params["b"])

    x, params = rand(N, D_in), {"w": rand(D_out, D_in), "b": rand(D_out)}

    for fisher_type in FisherType:
        kfac_io_true = (
            f(x, params),
            {"Linear0": x},
            # Forward-only KFAC does not require storing outputs
            {} if fisher_type == FisherType.FORWARD_ONLY else {"Linear0": f(x, params)},
            {"Linear0": {"weight": "w", "bias": "b"}},
            {"Linear0": {}},  # Linear layers have empty hyperparams
        )
        _verify_kfac_io(f, x, params, fisher_type, kfac_io_true)

    # 2) Only weight as free parameter (frozen bias)
    def f(x: Tensor, params: dict) -> Tensor:
        bias = arange(D_out, dtype=x.dtype)
        return linear(x, params["w"], bias=bias)

    x, params = rand(N, D_in), {"w": rand(D_out, D_in)}

    for fisher_type in FisherType:
        kfac_io_true = (
            f(x, params),
            {"Linear0": x},
            # Forward-only KFAC does not require storing outputs
            {} if fisher_type == FisherType.FORWARD_ONLY else {"Linear0": f(x, params)},
            {"Linear0": {"weight": "w"}},
            {"Linear0": {}},  # Linear layers have empty hyperparams
        )
        _verify_kfac_io(f, x, params, fisher_type, kfac_io_true)

    # 3) Only bias as free parameter (frozen weight)
    def f(x: Tensor, params: dict) -> Tensor:
        weight = arange(D_out * D_in, dtype=x.dtype).reshape(D_out, D_in)
        return linear(x, weight, params["b"])

    x, params = rand(N, D_in), {"b": rand(D_out)}

    for fisher_type in FisherType:
        kfac_io_true = (
            f(x, params),
            {},  # No need to store inputs for biases
            # Forward-only KFAC does not require storing outputs
            {} if fisher_type == FisherType.FORWARD_ONLY else {"Linear0": f(x, params)},
            {"Linear0": {"bias": "b"}},
            {"Linear0": {}},  # Linear layers have empty hyperparams
        )
        _verify_kfac_io(f, x, params, fisher_type, kfac_io_true)

    # 4) Fully-connected layer without bias
    def f(x: Tensor, params: dict) -> Tensor:
        return linear(x, params["w"])

    x, params = rand(N, D_in), {"w": rand(D_out, D_in)}

    for fisher_type in FisherType:
        kfac_io_true = (
            f(x, params),
            {"Linear0": x},
            # Forward-only KFAC does not require storing outputs
            {} if fisher_type == FisherType.FORWARD_ONLY else {"Linear0": f(x, params)},
            {"Linear0": {"weight": "w"}},
            {"Linear0": {}},  # Linear layers have empty hyperparams
        )
        _verify_kfac_io(f, x, params, fisher_type, kfac_io_true)


@mark.parametrize("inplace", [False, True], ids=["out-of-place", "in-place"])
def test_kfac_io_mlp(inplace: bool):
    """Test correctness of KFAC IO on a three-layer MLP.

    Test with in- or out-of-place activations to make sure that the IO
    collector replaces in-place activations correctly (otherwise we
    would wrongly collect the modified IO).

    Args:
        inplace: Whether to use in-place activations.
    """
    manual_seed(0)
    D_in, D_hidden, D_out, N = 5, 4, 3, 2

    mlp = Sequential(
        OrderedDict(
            [
                # We want KFAC w.r.t. weight & bias
                ("0", Linear(D_in, D_hidden)),
                ("1", ReLU(inplace=inplace)),
                # Has no bias, we only want KFAC w.r.t. weight
                ("2", Linear(D_hidden, D_hidden, bias=False)),
                ("3", ReLU(inplace=inplace)),
                # We will freeze the weight, only want KFAC w.r.t. bias
                ("4", Linear(D_hidden, D_out)),
            ]
        )
    )
    frozen_weight_4 = randn_like(mlp.get_submodule("4").weight)

    def f(x, params):
        assert "4.weight" not in params
        return functional_call(mlp, {**params, "4.weight": frozen_weight_4}, x)

    x, params = (
        rand(N, D_in),
        {n: randn_like(p) for n, p in mlp.named_parameters() if n != "4.weight"},
    )

    # Manually compute in and outputs of linear layers
    lin0_in = x
    lin0_out = linear(x, params["0.weight"], bias=params["0.bias"])

    lin1_in = relu(lin0_out)
    lin1_out = linear(lin1_in, params["2.weight"])

    lin2_in = relu(lin1_out)
    lin2_out = linear(lin2_in, frozen_weight_4, bias=params["4.bias"])

    # Verify KFAC IOs for different Fisher types
    for fisher_type in FisherType:
        kfac_io_true = (
            f(x, params),
            # No lin2_in here because this layer only has a free bias
            {"Linear0": lin0_in, "Linear1": lin1_in},
            # Forward-only KFAC does not require storing outputs
            {}
            if fisher_type == FisherType.FORWARD_ONLY
            else {"Linear0": lin0_out, "Linear1": lin1_out, "Linear2": lin2_out},
            {
                "Linear0": {"weight": "0.weight", "bias": "0.bias"},
                "Linear1": {"weight": "2.weight"},
                "Linear2": {"bias": "4.bias"},
            },
            {
                "Linear0": {},  # Linear layers have empty hyperparams
                "Linear1": {},  # Linear layers have empty hyperparams
                "Linear2": {},  # Linear layers have empty hyperparams
            },
        )
        _verify_kfac_io(f, x, params, fisher_type, kfac_io_true)


@mark.parametrize("inplace", [False, True], ids=["out-of-place", "in-place"])
def test_kfac_io_cnn(inplace: bool):
    """Test correctness of KFAC IO on a CNN with convolutions, pooling, and linear.

    Test with in- or out-of-place activations to make sure that the IO
    collector replaces in-place activations correctly (otherwise we
    would wrongly collect the modified IO).

    Args:
        inplace: Whether to use in-place activations.
    """
    manual_seed(0)
    N, C_in, H, W = 2, 3, 8, 8
    C_hidden, C_out = 4, 6
    Linear_in, Linear_hidden, Linear_out = C_out * 2 * 2, 10, 5  # After 2x2x2 pooling

    cnn = Sequential(
        OrderedDict(
            [
                # Conv layer with weight & bias
                ("0", Conv2d(C_in, C_hidden, kernel_size=3, padding=1)),
                ("1", ReLU(inplace=inplace)),
                ("2", MaxPool2d(kernel_size=2, stride=2)),
                # Conv layer with weight only (no bias)
                ("3", Conv2d(C_hidden, C_out, kernel_size=4, stride=2, bias=False)),
                ("4", ReLU(inplace=inplace)),
                ("5", AdaptiveAvgPool2d((2, 2))),
                ("6", Flatten()),
                # Linear layer with bias only (frozen weight)
                ("7", Linear(Linear_in, Linear_hidden)),
                ("8", ReLU(inplace=inplace)),
                # Linear layer with weight & bias
                ("9", Linear(Linear_hidden, Linear_out)),
            ]
        )
    )

    # Freeze the weight of the first linear layer (layer 7)
    frozen_weight_7 = randn_like(cnn.get_submodule("7").weight)

    def f(x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        assert "7.weight" not in params
        return functional_call(cnn, {**params, "7.weight": frozen_weight_7}, x)

    x, params = (
        rand(N, C_in, H, W),
        {n: randn_like(p) for n, p in cnn.named_parameters() if n != "7.weight"},
    )

    # Manually compute inputs and outputs of layers with parameters
    # Conv layer 0 (weight + bias)
    conv0_in = x
    conv0_out = conv2d(x, params["0.weight"], bias=params["0.bias"], padding=1)
    conv0_relu = relu(conv0_out, inplace=False)
    conv0_pool = max_pool2d(conv0_relu, kernel_size=2, stride=2)

    # Conv layer 3 (weight only, no bias)
    conv1_in = conv0_pool
    conv1_out = conv2d(conv0_pool, params["3.weight"], stride=2, padding=0)
    conv1_relu = relu(conv1_out, inplace=False)
    conv1_pool = adaptive_avg_pool2d(conv1_relu, (2, 2))

    # Linear layer 7 (bias only, frozen weight)
    linear0_in = flatten(conv1_pool, start_dim=1)
    linear0_out = linear(linear0_in, frozen_weight_7, bias=params["7.bias"])
    linear0_relu = relu(linear0_out, inplace=False)

    # Linear layer 9 (weight + bias)
    linear1_in = linear0_relu
    linear1_out = linear(linear0_relu, params["9.weight"], bias=params["9.bias"])

    # Define expected hyperparameters for convolution layers
    conv_hyperparams = {**CONV2D_DEFAULT_PARAMS, **{"padding": [1, 1]}}

    # Verify KFAC IOs for different Fisher types
    for fisher_type in FisherType:
        kfac_io_true = (
            f(x, params),
            # Inputs are stored for layers that have weights (not bias-only layers)
            {"Conv0": conv0_in, "Conv1": conv1_in, "Linear1": linear1_in},
            # Forward-only KFAC does not require storing outputs
            {}
            if fisher_type == FisherType.FORWARD_ONLY
            else {
                "Conv0": conv0_out,
                "Conv1": conv1_out,
                "Linear0": linear0_out,
                "Linear1": linear1_out,
            },
            {
                "Conv0": {"weight": "0.weight", "bias": "0.bias"},
                "Conv1": {"weight": "3.weight"},
                "Linear0": {"bias": "7.bias"},
                "Linear1": {"weight": "9.weight", "bias": "9.bias"},
            },
            {
                "Conv0": {**CONV2D_DEFAULT_PARAMS, **{"padding": [1, 1]}},
                "Conv1": {**CONV2D_DEFAULT_PARAMS, **{"stride": [2, 2]}},
                "Linear0": {},  # Linear layers have empty hyperparams
                "Linear1": {},  # Linear layers have empty hyperparams
            },
        )
        _verify_kfac_io(f, x, params, fisher_type, kfac_io_true)
