"""Utility functions for the examples in the documentation."""

from numpy import allclose, isclose, ndarray


def report_nonclose(
    array1: ndarray,
    array2: ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
):
    """Compare two numpy arrays, raise exception if nonclose values and print them.

    Args:
        array1: First array.
        array2: Second array.
        rtol: Relative tolerance (see ``numpy.allclose``). Default: ``1e-5``.
        atol: Absolute tolerance (see ``numpy.allclose``). Default: ``1e-8``.
        equal_nan: Whether comparing two NaNs should be considered as ``True``
            (see ``numpy.allclose``). Default: ``False``.

    Raises:
        ValueError: If the two arrays don't match in shape or have nonclose values.
    """
    if array1.shape != array2.shape:
        raise ValueError(
            f"Arrays shapes don't match: {array1.shape} vs. {array2.shape}."
        )

    if allclose(array1, array2, rtol=rtol, atol=atol, equal_nan=equal_nan):
        print("Compared arrays match.")
    else:
        for a1, a2 in zip(array1.flatten(), array2.flatten()):
            if not isclose(a1, a2, atol=atol, rtol=rtol, equal_nan=equal_nan):
                print(f"{a1} ≠ {a2} (ratio {a1 / a2:.5f})")
        print(f"Max: {array1.max():.5f}, {array2.max():.5f}")
        print(f"Min: {array1.min():.5f}, {array2.min():.5f}")
        raise ValueError("Compared arrays don't match.")
