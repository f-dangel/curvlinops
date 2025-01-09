"""Hutchinson-style matrix diagonal estimation."""

from numpy import column_stack, einsum, ndarray
from scipy.sparse.linalg import LinearOperator

from curvlinops.sampling import random_vector


def hutchinson_diag(
    A: LinearOperator, num_matvecs: int, distribution: str = "rademacher"
) -> ndarray:
    r"""Estimate a linear operator's diagonal using Hutchinson's method.

    For details, see

    - Bekas, C., Kokiopoulou, E., & Saad, Y. (2007). An estimator for the diagonal
    of a matrix. Applied Numerical Mathematics.

    Let :math:`\mathbf{A}` be a square linear operator. We can approximate its diagonal
    :math:`\mathrm{diag}(\mathbf{A})` by drawing random vectors :math:`N`
    :math:`\mathbf{v}_n \sim \mathbf{v}` from a distribution :math:`\mathbf{v}` that
    satisfies :math:`\mathbb{E}[\mathbf{v} \mathbf{v}^\top] = \mathbf{I}`, and compute
    the estimator

    .. math::
        \mathbf{a}
        := \frac{1}{N} \sum_{n=1}^N \mathbf{v}_n \odot \mathbf{A} \mathbf{v}_n
        \approx \mathrm{diag}(\mathbf{A})\,.

    This estimator is unbiased,

    .. math::
        \mathbb{E}[a_i]
        = \sum_j \mathbb{E}[v_i A_{i,j} v_j]
        = \sum_j A_{i,j} \mathbb{E}[v_i  v_j]
        = \sum_j A_{i,j} \delta_{i, j}
        = A_{i,i}\,.

    Args:
        A: A square linear operator whose diagonal is estimated.
        num_matvecs: Total number of matrix-vector products to use. Must be smaller
            than the dimension of the linear operator.
        distribution: Distribution of the random vectors used for the diagonal
            estimation. Can be either ``'rademacher'`` or ``'normal'``.
            Default: ``'rademacher'``.

    Returns:
        The estimated diagonal of the linear operator.

    Raises:
        ValueError: If the linear operator is not square or if the number of matrix-
            vector products is greater than the dimension of the linear operator
            (because then you can evaluate the true diagonal directly at the same cost).

    Example:
        >>> from numpy import diag
        >>> from numpy.random import rand, seed
        >>> from numpy.linalg import norm
        >>> seed(0) # make deterministic
        >>> A = rand(40, 40)
        >>> diag_A = diag(A) # exact diagonal as reference
        >>> # one- and multi-sample approximations
        >>> diag_A_low_precision = hutchinson_diag(A, num_matvecs=1)
        >>> diag_A_high_precision = hutchinson_diag(A, num_matvecs=30)
        >>> # compute residual norms
        >>> error_low_precision = norm(diag_A - diag_A_low_precision) / norm(diag_A)
        >>> error_high_precision = norm(diag_A - diag_A_high_precision) / norm(diag_A)
        >>> assert error_low_precision > error_high_precision
        >>> round(error_low_precision, 4), round(error_high_precision, 4)
        (4.616, 1.2441)
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square. Got shape {A.shape}.")
    dim = A.shape[1]
    if num_matvecs >= dim:
        raise ValueError(
            f"num_matvecs ({num_matvecs}) must be less than A's size ({dim})."
        )
    G = column_stack([random_vector(dim, distribution) for _ in range(num_matvecs)])
    AG = A @ G
    return einsum("ij,ij->i", G, AG) / num_matvecs
