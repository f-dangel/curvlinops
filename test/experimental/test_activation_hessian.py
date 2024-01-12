"""Contains tests for ``curvlinops.activation_hessian``."""

from test.cases import DEVICES, DEVICES_IDS
from test.utils import classification_targets

from numpy import eye as numpy_eye
from pytest import mark, raises
from torch import (
    allclose,
    block_diag,
    device,
    einsum,
    eye,
    from_numpy,
    manual_seed,
    rand,
)
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential, Sigmoid

from curvlinops.experimental.activation_hessian import (
    ActivationHessianLinearOperator,
    store_activation,
)


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
    layers = [layer.to(dev) for layer in layers]
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


@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_ActivationHessianLinearOperator(dev: device):
    """Check the Hessian w.r.t. an activation.

    Verifies the Hessian of a function ``l(id(X), y)`` w.r.t. ``X`` where ``id`` is the
    identity.

    Args:
        dev: Device on which to run the tests.
    """
    manual_seed(0)
    batch_size, num_classes = 2, 10

    # model does nothing to the input but needs parameters so the linear
    # operator can infer the device
    model = Linear(num_classes, num_classes, bias=False).to(dev)
    model.weight.data = eye(num_classes)

    loss_func = CrossEntropyLoss(reduction="sum")
    X = rand(batch_size, num_classes, requires_grad=True, device=dev)
    y = classification_targets((batch_size,), num_classes).to(dev)
    data = [(X, y)]
    activation = ("", "input", 0)

    # compute the Hessian matrix representation
    H_linop = ActivationHessianLinearOperator(model, loss_func, activation, data)
    H_mat = from_numpy(H_linop @ numpy_eye(H_linop.shape[1])).to(dev, X.dtype)

    # we know that the Hessian of softmax CE loss is ``diag(p(x)) - p(x) p(x)ᵀ``
    # where ``p(x)`` is the softmax probability on a single datum ``x``. On a batch,
    # the Hessian is the block diagonal stack of these per-sample Hessians
    p = X.softmax(dim=1).detach()
    blocks = []
    for n in range(batch_size):
        p_n = p[n]
        H_n = p_n.diag() - einsum("i,j->ij", p_n, p_n)
        blocks.append(H_n)
    truth = block_diag(*blocks)

    assert allclose(H_mat, truth)
