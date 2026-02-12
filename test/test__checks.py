"""Tests for ``curvlinops._checks``."""

from collections import UserDict

from pytest import mark, raises, warns
from torch import Tensor, manual_seed, rand
from torch.func import vmap
from torch.nn import Linear, ReLU, Sequential

from curvlinops._checks import (
    _check_supports_batched_and_unbatched_inputs,
    _register_userdict_as_pytree,
    _userdict_pytree_registered,
)
from curvlinops.utils import allclose_report


# Must run before any test that triggers _register_userdict_as_pytree, because
# PyTorch's pytree registration is global and irreversible within a process.
@mark.order(0)
def test_register_userdict_as_pytree():
    """Test that registering UserDict as a pytree enables vmap over UserDicts."""
    manual_seed(0)

    net = Sequential(Linear(4, 3), ReLU(), Linear(3, 2))
    X = UserDict({"x": rand(5, 4)})

    def model(data: UserDict) -> Tensor:
        return net(data["x"])

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


def test_check_supports_batched_and_unbatched_inputs_detects_violation():
    """Test that a function using the batch dimension is detected as incompatible."""
    manual_seed(0)
    X = rand(5, 4)

    # This function normalizes by the batch mean, so f(X) != vmap(f)(X)
    def batch_dependent(x: Tensor) -> Tensor:
        """Handle unbatched and batched data differently.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        # Mean is over all axes. This includes the batch axis in the batched case, but
        # does not in the un-batched case. Hence this function operates differently
        # on batched and unbatched inputs.
        return x - x.mean()

    with raises(
        ValueError, match="Function does not support batched and un-batched inputs"
    ):
        _check_supports_batched_and_unbatched_inputs(X, batch_dependent)
