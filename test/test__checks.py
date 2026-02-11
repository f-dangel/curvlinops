"""Tests for ``curvlinops._checks``."""

from collections import UserDict

from pytest import mark, raises, warns
from torch import Tensor, manual_seed, rand, vmap
from torch.nn import Linear, ReLU, Sequential

from curvlinops._checks import _register_userdict_as_pytree, _userdict_pytree_registered
from curvlinops.utils import allclose_report


# Must run before any test that triggers _register_userdict_as_pytree, because
# PyTorch's pytree registration is global and irreversible within a process.
@mark.order(0)
def test_register_userdict_as_pytree():
    """Test that registering UserDict as a pytree enables vmap over UserDicts."""
    manual_seed(0)

    net = Sequential(Linear(4, 3), ReLU(), Linear(3, 2))

    def model(data: UserDict) -> Tensor:
        return net(data["x"])

    X = UserDict({"x": rand(5, 4)})

    # 1) Before registration, the flag must be False
    assert not _userdict_pytree_registered

    # 2) vmap must crash on UserDict input before registration
    with raises(ValueError, match="We cannot vmap over non-Tensor arguments"):
        vmap(model)(X)

    # 3) Register UserDict as pytree node
    with warns(
        UserWarning,
        match="UserDict PyTree registration relies on PyTorch's private "
        + "`torch.utils._pytree` module, which may change in future versions.",
    ):
        _register_userdict_as_pytree()

    vmapped_result = vmap(model)(X)

    # 4) Compare to manual per-sample evaluation
    batched_result = net(X["x"])
    assert allclose_report(vmapped_result, batched_result)
