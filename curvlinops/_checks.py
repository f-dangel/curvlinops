"""Helpers to verify determinism of the empirical risk, gradient, and model."""

from collections import UserDict
from typing import Any, Callable, List, MutableMapping, Tuple, Union
from warnings import warn

from torch import Tensor, allclose, rand, vmap

from curvlinops.utils import allclose_report

# Track whether UserDict has been registered as a PyTree node
_userdict_pytree_registered = False


def _check_deterministic_matvec(
    linop, rtol: float = 1e-5, atol: float = 1e-8
):
    """Probe whether a linear operator's matrix-vector product is deterministic.

    Performs two sequential matrix-vector products and compares them.

    Args:
        linop: The linear operator to check.
        rtol: Relative tolerance for comparison. Defaults to ``1e-5``.
        atol: Absolute tolerance for comparison. Defaults to ``1e-8``.

    Raises:
        RuntimeError: If the two matrix-vector products yield different results.
    """
    v = rand(linop.shape[1], device=linop.device, dtype=linop.dtype)
    Av1 = linop @ v
    Av2 = linop @ v
    if not allclose_report(Av1, Av2, rtol=rtol, atol=atol):
        raise RuntimeError("Check for deterministic matvec failed.")


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

    def userdict_flatten(ud: UserDict) -> Tuple[List[Any], Tuple[str, ...]]:
        """Flatten a UserDict into a list of values and a tuple of keys.

        Args:
            ud: The UserDict to flatten.

        Returns:
            A tuple of (list of values, tuple of keys).
        """
        keys = tuple(ud.data.keys())
        values = tuple(ud.data[k] for k in keys)
        return list(values), keys

    def userdict_unflatten(values: List[Any], keys: Tuple[str, ...]) -> UserDict:
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
    X: Union[Tensor, MutableMapping],
    f: Callable[[Tensor], Tensor],
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

    if not allclose(batched_result, vmapped_result, rtol=rtol, atol=atol):
        raise ValueError("Function does not support batched and un-batched inputs.")
