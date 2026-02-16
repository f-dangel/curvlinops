"""Tests for with_kfac_io: collecting KFAC-specific layer inputs and outputs."""

from collections import OrderedDict
from typing import Any

from pytest import mark, raises
from torch import Tensor, arange, flatten, manual_seed, rand, randn_like, zeros_like
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
from torch.nn.functional import (
    adaptive_avg_pool2d,
    conv2d,
    linear,
    max_pool2d,
    relu,
)

from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac import FisherType
from curvlinops.utils import allclose_report
from test.computers.io_collector.test_param_io import CONV2D_DEFAULT_PARAMS


def _verify_kfac_io(
    f,
    x: Tensor,
    params: dict[str, Tensor],
    fisher_type: FisherType,
    kfac_io_true: tuple[
        Tensor,
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, dict[str, str]],
        dict[str, dict[str, Any]],
    ],
) -> None:
    """Verify that with_kfac_io produces correct outputs and IO information.

    Args:
        f: Function to test, with signature f(x, params) -> output.
        x: Input tensor to the function.
        params: Dictionary mapping parameter names to parameter tensors.
        fisher_type: Type of Fisher information computation.
        kfac_io_true: Expected 5-tuple of KFAC IO information.
    """
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


def test_with_kfac_io_multiple_parameter_usages():
    """Test that KFAC rejects multiple usages of the same parameter."""
    manual_seed(0)
    N, D = 2, 3

    def f(x: Tensor, params: dict) -> Tensor:
        xW = linear(x, params["weight"], bias=params["bias"])
        return linear(xW, params["weight"])

    x, params = rand(N, D), {"weight": rand(D, D), "bias": rand(D)}

    with raises(ValueError, match="Parameters used multiple times"):
        with_kfac_io(f, x, params, "empirical")


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
        OrderedDict([
            # We want KFAC w.r.t. weight & bias
            ("0", Linear(D_in, D_hidden)),
            ("1", ReLU(inplace=inplace)),
            # Has no bias, we only want KFAC w.r.t. weight
            ("2", Linear(D_hidden, D_hidden, bias=False)),
            ("3", ReLU(inplace=inplace)),
            # We will freeze the weight, only want KFAC w.r.t. bias
            ("4", Linear(D_hidden, D_out)),
        ])
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
        OrderedDict([
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
        ])
    )

    # Freeze the weight of the first linear layer (layer 7)
    frozen_weight_7 = randn_like(cnn.get_submodule("7").weight)

    def f(x: Tensor, params: dict[str, Tensor]) -> Tensor:
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
