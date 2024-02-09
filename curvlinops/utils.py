"""General utility functions."""

from typing import List

from numpy import cumsum


def split_list(x: List, sizes: List[int]) -> List[List]:
    """Split a list into multiple lists of specified size.

    Args:
        x: List to be split.
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
    return [x[boundaries[i] : boundaries[i + 1]] for i in range(len(sizes))]
