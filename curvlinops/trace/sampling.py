"""Sampling methods for random vectors."""

from numpy import ndarray
from numpy.random import binomial, randn


def rademacher(dim: int) -> ndarray:
    """Draw a vector with i.i.d. Rademacher elements.

    Args:
        dim: Dimension of the vector.

    Returns:
        Vector with i.i.d. Rademacher elements and specified dimension.
    """
    num_trials, success_prob = 1, 0.5
    return binomial(num_trials, success_prob, size=dim).astype(float) * 2 - 1


def normal(dim: int) -> ndarray:
    """Drawa vector with i.i.d. standard normal elements.

    Args:
        dim: Dimension of the vector.

    Returns:
        Vector with i.i.d. standard normal elements and specified dimension.
    """
    return randn(dim)
