"""Helpers to verify determinism of the empirical risk, gradient, and model."""

from __future__ import annotations

from collections import UserDict
from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING, Any
from warnings import warn

from torch import Tensor
from torch.func import vmap

from curvlinops.utils import allclose_report

if TYPE_CHECKING:
    from curvlinops._torch_base import PyTorchLinearOperator

# Track whether UserDict has been registered as a PyTree node
_userdict_pytree_registered = False


def _check_matmul_compatible_shape(
    left: PyTorchLinearOperator, right: PyTorchLinearOperator
):
    """Check that two linear operators can be chained (left @ right).

    Args:
        left: The left operator.
        right: The right operator.

    Raises:
        ValueError: If the left operator's input shape doesn't match the right
            operator's output shape.
    """
    if left._in_shape != right._out_shape:
        raise ValueError(
            f"Shape mismatch: input shape {left._in_shape} does not match"
            f" output shape {right._out_shape}."
        )


def _check_same_shape(old: Tensor, new: Tensor):
    """Check that two tensors have the same shape.

    Args:
        old: The original tensor.
        new: The new tensor.

    Raises:
        ValueError: If shapes don't match.
    """
    if old.shape != new.shape:
        raise ValueError(f"Shape mismatch: expected {old.shape}, got {new.shape}.")


def _check_same_tensor_list_shape(
    old: PyTorchLinearOperator, new: PyTorchLinearOperator
):
    """Check that two linear operators have the same tensor list shapes.

    Compares ``_in_shape`` and ``_out_shape`` (the structured list-of-tuples
    shapes used by ``PyTorchLinearOperator``).

    Args:
        old: The original linear operator.
        new: The new linear operator.

    Raises:
        ValueError: If ``_in_shape`` or ``_out_shape`` don't match.
    """
    if old._in_shape != new._in_shape or old._out_shape != new._out_shape:
        raise ValueError(
            f"Shape mismatch: expected in_shape={old._in_shape}, "
            f"out_shape={old._out_shape}, got in_shape={new._in_shape}, "
            f"out_shape={new._out_shape}."
        )


def _check_same_device(
    old: Tensor | PyTorchLinearOperator, new: Tensor | PyTorchLinearOperator
):
    """Check that two objects live on the same device.

    Args:
        old: The original tensor or linear operator.
        new: The new tensor or linear operator.

    Raises:
        ValueError: If devices don't match.
    """
    if old.device != new.device:
        raise ValueError(f"Device mismatch: expected {old.device}, got {new.device}.")


def _check_same_dtype(
    old: Tensor | PyTorchLinearOperator, new: Tensor | PyTorchLinearOperator
):
    """Check that two objects have the same dtype.

    Args:
        old: The original tensor or linear operator.
        new: The new tensor or linear operator.

    Raises:
        ValueError: If dtypes don't match.
    """
    if old.dtype != new.dtype:
        raise ValueError(f"Dtype mismatch: expected {old.dtype}, got {new.dtype}.")


def _register_userdict_as_pytree():
    """Register UserDict as a PyTree node for torch.vmap compatibility.

    This allows torch.vmap to accept UserDict arguments directly by defining
    how to flatten and unflatten UserDict instances.

    Warning:
        This function relies on PyTorch's private ``torch.utils._pytree`` module,
        which may change in future PyTorch versions without notice.
    """
    global _userdict_pytree_registered  # noqa: PLW0603
    if _userdict_pytree_registered:
        return

    warn(
        "UserDict PyTree registration relies on PyTorch's private "
        "`torch.utils._pytree` module, which may change in future versions.",
        UserWarning,
        stacklevel=3,
    )

    from torch.utils._pytree import register_pytree_node

    def userdict_flatten(ud: UserDict) -> tuple[list[Any], tuple[str, ...]]:
        """Flatten a UserDict into a list of values and a tuple of keys.

        Args:
            ud: The UserDict to flatten.

        Returns:
            A tuple of (list of values, tuple of keys).
        """
        keys = tuple(ud.data.keys())
        values = [ud.data[k] for k in keys]
        return values, keys

    def userdict_unflatten(values: list[Any], keys: tuple[str, ...]) -> UserDict:
        """Unflatten a list of values and keys back into a UserDict.

        Args:
            values: The values to unflatten.
            keys: The keys corresponding to the values.

        Returns:
            A UserDict with the given keys and values.
        """
        return UserDict(dict(zip(keys, values)))

    register_pytree_node(UserDict, userdict_flatten, userdict_unflatten)
    _userdict_pytree_registered = True


def _check_supports_batched_and_unbatched_inputs(
    X: Tensor | MutableMapping,
    f: Callable[[Tensor | MutableMapping], Tensor],
    batch_axis: int = 0,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """Verify that a function f works correctly on batched and unbatched inputs.

    This function checks that ``f(X) = vmap(f)(X)``, where ``X`` is a batched
    tensor and ``vmap`` applies ``f`` to each individual sample in the batch.

    Args:
        X: Batched input tensor where one axis represents the batch dimension.
        f: Function to test. Should accept a tensor and return a tensor.
        batch_axis: Which axis represents the batch dimension. Default: ``0``.
        rtol: Relative tolerance for comparison. Default: ``1e-5``.
        atol: Absolute tolerance for comparison. Default: ``1e-8``.

    Raises:
        ValueError: If the results of batched and unbatched evaluation differ.
    """
    batched_result = f(X)

    if isinstance(X, UserDict):
        _register_userdict_as_pytree()

    vmapped_f = vmap(f, in_dims=batch_axis, out_dims=batch_axis)
    vmapped_result = vmapped_f(X)

    if not allclose_report(batched_result, vmapped_result, rtol=rtol, atol=atol):
        raise ValueError("Function does not support batched and un-batched inputs.")
