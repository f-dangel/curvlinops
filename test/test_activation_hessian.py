"""Contains tests for ``curvlinops.activation_hessian``."""

from test.cases import DEVICES, DEVICES_IDS

from pytest import mark, raises
from torch import allclose, device, manual_seed, rand
from torch.nn import Linear, ReLU, Sequential, Sigmoid

from curvlinops.activation_hessian import store_activation


@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_store_activation(dev: device):
    """Test context that stores the input/output of a layer.

    Args:
        dev: Device on which to run the tests.
    """
    manual_seed(0)
    layers = [
        Linear(10, 8),
        ReLU(),
        Linear(8, 6),
        Sigmoid(),
        Linear(6, 4),
    ]
    layers = [l.to(dev) for l in layers]
    model = Sequential(*layers).to(dev)
    X = rand(5, 10, device=dev)

    # compare with manual forward pass
    activation = ("3", "input", 0)  # input to the sigmoid layer
    activation_storage = []
    with store_activation(model, *activation, activation_storage):
        model(X)
    act = activation_storage.pop()
    assert act.shape == (5, 6)
    assert not activation_storage
    truth = layers[2](layers[1](layers[0](X)))
    assert allclose(act, truth)
    # make sure the hooks were removed when computing ``truth``
    assert not activation_storage

    # check failure scenarios
    # layer name does not exist
    invalid_activation = ("foo", "input", 0)
    with raises(ValueError):
        store_activation(model, *invalid_activation, [])

    # io type does not exist
    invalid_activation = ("3", "foo", 0)
    with raises(ValueError):
        store_activation(model, *invalid_activation, [])

    # destination not empty
    invalid_activation = ("3", "foo", 0)
    with raises(ValueError):
        store_activation(model, *invalid_activation, [42])
