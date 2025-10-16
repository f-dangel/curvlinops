"""Hutchinson-style matrix norm estimation."""

from typing import Union

from torch import Tensor, column_stack

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.sampling import random_vector


def hutchinson_squared_fro(
    A: Union[Tensor, PyTorchLinearOperator],
    num_matvecs: int,
    distribution: str = "rademacher",
) -> Tensor:
    r"""Estimate the squared Frobenius norm of a matrix using Hutchinson's method.

    Let :math:`\mathbf{A} \in \mathbb{R}^{M \times N}` be some matrix. It's Frobenius
    norm :math:`\lVert\mathbf{A}\rVert_\text{F}` is defined via:

    .. math::
        \lVert\mathbf{A}\rVert_\text{F}^2
        =
        \sum_{m=1}^M \sum_{n=1}^N \mathbf{A}_{n,m}^2
        =
        \text{Tr}(\mathbf{A}^\top \mathbf{A}).

    Due to the last equality, we can use Hutchinson-style trace estimation to estimate
    the squared Frobenius norm.

    Args:
        A: A matrix whose squared Frobenius norm is estimated.
        num_matvecs: Total number of matrix-vector products to use. Must be smaller
            than the minimum dimension of the matrix.
        distribution: Distribution of the random vectors used for the trace estimation.
            Can be either ``'rademacher'`` or ``'normal'``. Default: ``'rademacher'``.

    Returns:
        The estimated squared Frobenius norm of the matrix.

    Raises:
        ValueError: If the matrix is not two-dimensional or if the number of matrix-
            vector products is greater than the minimum dimension of the matrix
            (because then you can evaluate the true squared Frobenius norm directly
            atthe same cost).

    Example:
        >>> from torch.linalg import matrix_norm
        >>> from torch import rand, manual_seed
        >>> _ = manual_seed(0) # make deterministic
        >>> A = rand(40, 40)
        >>> fro2_A = matrix_norm(A).item()**2 # reference: exact squared Frobenius norm
        >>> # one- and multi-sample approximations
        >>> fro2_A_low_prec = hutchinson_squared_fro(A, num_matvecs=1).item()
        >>> fro2_A_high_prec = hutchinson_squared_fro(A, num_matvecs=30).item()
        >>> assert abs(fro2_A - fro2_A_low_prec) > abs(fro2_A - fro2_A_high_prec)
        >>> round(fro2_A, 1), round(fro2_A_low_prec, 1), round(fro2_A_high_prec, 1)
        (530.9, 156.7, 628.9)
    """
    if len(A.shape) != 2:
        raise ValueError(f"A must be a matrix. Got shape {A.shape}.")
    dim = min(A.shape)
    if num_matvecs >= dim:
        raise ValueError(
            f"num_matvecs ({num_matvecs}) must be less than the minimum dimension of A."
        )
    # Instead of AT @ A, use A @ AT if the matrix is wider than tall
    if A.shape[1] > A.shape[0]:
        A = A.T

    G = column_stack(
        [
            random_vector(dim, distribution, A.device, A.dtype)
            for _ in range(num_matvecs)
        ]
    )
    AG = A @ G
    return (AG**2 / num_matvecs).sum()
