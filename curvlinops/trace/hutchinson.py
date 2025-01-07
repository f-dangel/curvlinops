"""Vanilla Hutchinson trace estimation."""

from numpy import column_stack, einsum
from scipy.sparse.linalg import LinearOperator

from curvlinops.sampling import random_vector


def hutchinson_trace(
    A: LinearOperator, num_matvecs: int, distribution: str = "rademacher"
) -> float:
    r"""Estimate a linear operator's trace using the Girard-Hutchinson method.

    For details, see

    - Girard, D. A. (1989). A fast 'monte-carlo cross-validation' procedure for
      large least squares problems with noisy data. Numerische Mathematik.
    - Hutchinson, M. (1989). A stochastic estimator of the trace of the influence
      matrix for laplacian smoothing splines. Communication in Statistics---Simulation
      and Computation.

    Let :math:`\mathbf{A}` be a square linear operator. We can approximate its trace
    :math:`\mathrm{Tr}(\mathbf{A})` by drawing :math:`N` random vectors
    :math:`\mathbf{v}_n \sim \mathbf{v}` from a distribution that satisfies
    :math:`\mathbb{E}[\mathbf{v} \mathbf{v}^\top] = \mathbf{I}` and compute

    .. math::
        := \frac{1}{N} \mathbf{v}_n^\top \mathbf{A} \mathbf{v}_n
        \approx \mathrm{Tr}(\mathbf{A})\,.

    This estimator is unbiased,

    .. math::
        \mathbb{E}[a]
        = \mathrm{Tr}(\mathbb{E}[\mathbf{v}^\top\mathbf{A} \mathbf{v}])
        = \mathrm{Tr}(\mathbf{A} \mathbb{E}[\mathbf{v} \mathbf{v}^\top])
        = \mathrm{Tr}(\mathbf{A} \mathbf{I})
        = \mathrm{Tr}(\mathbf{A})\,.

    Args:
        A: A square linear operator whose trace is estimated.
        num_matvecs: Total number of matrix-vector products to use. Must be smaller
            than the dimension of the linear operator.
        distribution: Distribution of the random vectors used for the trace estimation.
            Can be either ``'rademacher'`` or ``'normal'``. Default: ``'rademacher'``.

    Returns:
        The estimated trace of the linear operator.

    Raises:
        ValueError: If the linear operator is not square or if the number of matrix-
            vector products is greater than the dimension of the linear operator
            (because then you can evaluate the true trace directly at the same cost).

    Example:
        >>> from numpy import trace, mean
        >>> from numpy.random import rand, seed
        >>> seed(0) # make deterministic
        >>> A = rand(50, 50)
        >>> tr_A = trace(A) # exact trace as reference
        >>> # one- and multi-sample approximations
        >>> tr_A_low_precision = hutchinson_trace(A, num_matvecs=1)
        >>> tr_A_high_precision = hutchinson_trace(A, num_matvecs=40)
        >>> # compute the relative errors
        >>> rel_error_low_precision = abs(tr_A - tr_A_low_precision) / abs(tr_A)
        >>> rel_error_high_precision = abs(tr_A - tr_A_high_precision) / abs(tr_A)
        >>> assert rel_error_low_precision > rel_error_high_precision
        >>> round(tr_A, 4), round(tr_A_low_precision, 4), round(tr_A_high_precision, 4)
        (25.7342, 59.7307, 20.033)
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square. Got shape {A.shape}.")
    dim = A.shape[1]
    if num_matvecs >= dim:
        raise ValueError(
            f"num_matvecs ({num_matvecs}) must be less than A's size ({dim})."
        )
    G = column_stack([random_vector(dim, distribution) for _ in range(num_matvecs)])

    return einsum("ij,ij", G, A @ G) / num_matvecs
