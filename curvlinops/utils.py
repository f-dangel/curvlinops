"""General utility functions."""

from collections import abc
from typing import List, Mapping, Sequence, Tuple, Union

from numpy import cumsum, ndarray
from torch import Tensor, as_tensor


def split_list(x: Union[List, Tuple], sizes: List[int]) -> List[List]:
    """Split a list into multiple lists of specified size.

    Args:
        x: List or tuple to be split.
        sizes: Sizes of the resulting lists.

    Returns:
        List of lists. Each sub-list has the size specified in ``sizes``.

    Raises:
        ValueError: If the sum of ``sizes`` does not match the input list's length.
    """
    if len(x) != sum(sizes):
        raise ValueError(
            f"List to be split has length {len(x)}, but requested sub-list with a total"
            + f" of {sum(sizes)} entries."
        )
    boundaries = cumsum([0] + sizes)
    return [list(x[boundaries[i] : boundaries[i + 1]]) for i in range(len(sizes))]


def allclose_report(
    tensor1: Union[Tensor, ndarray],
    tensor2: Union[Tensor, ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Same as ``allclose``, but prints entries that differ.

    Args:
        tensor1: First tensor for comparison.
        tensor2: Second tensor for comparison.
        rtol: Relative tolerance. Default is ``1e-5``.
        atol: Absolute tolerance. Default is ``1e-8``.

    Returns:
        ``True`` if the tensors are close, ``False`` otherwise.
    """
    tensor1, tensor2 = as_tensor(tensor1), as_tensor(tensor2)

    close = tensor1.allclose(tensor2, rtol=rtol, atol=atol)
    if not close:
        # print non-close values
        nonclose_idx = tensor1.isclose(tensor2, rtol=rtol, atol=atol).logical_not_()
        nonclose_entries = 0
        for idx, t1, t2 in zip(
            nonclose_idx.argwhere(),
            tensor1[nonclose_idx].flatten(),
            tensor2[nonclose_idx].flatten(),
        ):
            print(f"at index {idx.tolist()}: {t1:.5e} ≠ {t2:.5e}, ratio: {t1 / t2:.5e}")
            nonclose_entries += 1

        # print largest and smallest absolute entries
        amax1, amax2 = tensor1.abs().max().item(), tensor2.abs().max().item()
        print(f"Abs max: {amax1:.5e} vs. {amax2:.5e}.")
        amin1, amin2 = tensor1.abs().min().item(), tensor2.abs().min().item()
        print(f"Abs min: {amin1:.5e} vs. {amin2:.5e}.")

        # print number of nonclose values and tolerances
        print(f"Non-close entries: {nonclose_entries} / {tensor1.numel()}.")
        print(f"rtol = {rtol}, atol = {atol}.")

    return close


def assert_is_square(A) -> int:
    """Assert that a matrix or linear operator is square.

    Args:
        A: Matrix or linear operator to be checked. Must have a ``.shape`` attribute.

    Returns:
        The dimension of the square matrix.

    Raises:
        ValueError: If the matrix is not square.
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Operator must be square. Got shape {A.shape}.")
    (dim,) = set(A.shape)
    return dim


def assert_matvecs_subseed_dim(A, num_matvecs: int):
    """Assert that the number of matrix-vector products is smaller than the dimension.

    Args:
        A: Matrix or linear operator to be checked. Must have a ``.shape`` attribute.
        num_matvecs: Number of matrix-vector products.

    Raises:
        ValueError: If the number of matrix-vector products is greater than the dimension.
    """
    if any(num_matvecs >= d for d in A.shape):
        raise ValueError(
            f"num_matvecs ({num_matvecs}) must be less than A's size ({A.shape})."
        )


def assert_divisible_by(num: int, divisor: int, name: str):
    """Assert that a number is divisible by another number.

    Args:
        num: Number to be checked.
        divisor: Divisor.
        name: Name of the number.

    Raises:
        ValueError: If the number is not divisible by the divisor.
    """
    if num % divisor != 0:
        raise ValueError(f"{name} ({num}) must be divisible by {divisor}.")


StateDictType = Union[
    Mapping[str, "StateDictType"], Sequence["StateDictType"], Tensor, float, int, str
]


def do_statedicts_match(statedict1: StateDictType, statedict2: StateDictType) -> bool:
    """Compare two state dictionaries for equality. Each statedict can be a nested
    dictionary from string keys to Tensors, floats, ints or another statedict.

    Args:
        statedict1: First state dictionary to compare.
        statedict2: Second state dictionary to compare.

    Returns:
        bool: True if the state dictionaries match exactly, False otherwise.

    Note:
        Performs deep comparison of nested dictionaries and tensors. For tensors,
        checks element-wise equality.
    """
    if type(statedict1) is not type(statedict2):
        return False
    elif isinstance(statedict1, abc.Mapping):
        return _do_mappings_match(statedict1, statedict2)
    elif isinstance(statedict1, abc.Sequence):
        return _do_sequences_match(statedict1, statedict2)
    elif isinstance(statedict1, Tensor):
        return (statedict1 == statedict2).all()
    else:
        return statedict1 == statedict2


def _do_sequences_match(
    statedict1: Sequence[StateDictType], statedict2: Sequence[StateDictType]
) -> bool:
    if len(statedict1) != len(statedict2):
        return False
    for i in range(len(statedict1)):
        if not do_statedicts_match(statedict1[i], statedict2[i]):
            return False
    return True


def _do_mappings_match(
    statedict1: Mapping[str, StateDictType], statedict2: Mapping[str, StateDictType]
) -> bool:
    if len(statedict1) != len(statedict2):
        return False
    for key in statedict1.keys():
        if key not in statedict2:
            return False
        if not do_statedicts_match(statedict1[key], statedict2[key]):
            return False
    return True
