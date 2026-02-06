"""Sampling methods for random vectors."""

from torch import Tensor, device, dtype, empty, randn


def rademacher(dim: int, device: device, dtype: dtype) -> Tensor:
    """Draw a vector with i.i.d. Rademacher elements.

    Args:
        dim: Dimension of the vector.
        device: Device on which the vector is allocated.
        dtype: Data type of the vector.

    Returns:
        Vector with i.i.d. Rademacher elements and specified dimension.
    """
    p_success = 0.5
    return empty(dim, device=device, dtype=dtype).bernoulli_(p_success).mul_(2).sub_(1)


def normal(dim: int, device: device, dtype: dtype) -> Tensor:
    """Draw a vector with i.i.d. standard normal elements.

    Args:
        dim: Dimension of the vector.
        device: Device on which the vector is allocated.
        dtype: Data type of the vector.

    Returns:
        Vector with i.i.d. standard normal elements and specified dimension.
    """
    return randn(dim, device=device, dtype=dtype)


def random_vector(dim: int, distribution: str, device: device, dtype: dtype) -> Tensor:
    """Draw a vector with i.i.d. elements.

    Args:
        dim: Dimension of the vector.
        distribution: Distribution of the vector's elements. Either ``'rademacher'`` or
            ``'normal'``.
        device: Device on which the vector is allocated.
        dtype: Data type of the vector.

    Returns:
        Vector with i.i.d. elements and specified dimension.

    Raises:
        ValueError: If the distribution is unknown.
    """
    if distribution == "rademacher":
        return rademacher(dim, device, dtype)
    elif distribution == "normal":
        return normal(dim, device, dtype)
    else:
        raise ValueError(f"Unknown distribution {distribution:!r}.")
