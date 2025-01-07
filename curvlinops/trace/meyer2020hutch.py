"""Implementation of Hutch++ trace estimation from Meyer et al."""

from numpy import column_stack, einsum
from numpy.linalg import qr
from scipy.sparse.linalg import LinearOperator

from curvlinops.sampling import random_vector


def hutchpp_trace(
    A: LinearOperator, num_matvecs: int, distribution: str = "rademacher"
) -> float:
    r"""Estimate a linear operator's trace using the Hutch++ method.

    In contrast to vanilla Hutchinson, Hutch++ has lower variance, but requires more
    memory. The method is presented in

    - Meyer, R. A., Musco, C., Musco, C., & Woodruff, D. P. (2020). Hutch++:
      optimal stochastic trace estimation.

    Let :math:`\mathbf{A}` be a square linear operator whose trace we want to
    approximate. First, using one third of the available matrix-vector products,
    we compute an orthonormal basis :math:`\mathbf{Q}` of a sub-space spanned by
    :math:`\mathbf{A} \mathbf{S}` where :math:`\mathbf{S}` is a tall random matrix
    with i.i.d. elements. Then, using one third of the available matrix-vector
    products, we compute the trace in the sub-space. Finally, we apply Hutchinson's
    estimator in the remaining space spanned by
    :math:`\mathbf{I} - \mathbf{Q} \mathbf{Q}^\top`. Let :math:`3N` denote the
    total number of matrix-vector products. We can draw :math:`2N` random vectors
    :math:`\mathbf{v}_n \sim \mathbf{v}` from a distribution which satisfies
    :math:`\mathbb{E}[\mathbf{v} \mathbf{v}^\top] = \mathbf{I}`, compute
    :math:`\mathbf{Q}` from the first :math:`N` vectors, and use the remaining
    to compute the estimator

    .. math::
        a
        := \mathrm{Tr}(\mathbf{Q}^\top \mathbf{A} \mathbf{Q})
        + \sum_{n = N+1}^{2N} \mathbf{v}_n^\top
          (\mathbf{I} - \mathbf{Q} \mathbf{Q}^\top)^\top
          \mathbf{A} (\mathbf{I} - \mathbf{Q} \mathbf{Q}^\top) \mathbf{v}_n
        \approx \mathrm{Tr}(\mathbf{A})\,.

    This estimator is unbiased, :math:`\mathbb{E}[a] = \mathrm{Tr}(\mathbf{A})`, as the
    first term is the exact trace in the space spanned by :math:`\mathbf{Q}`, and the
    second part is Hutchinson's unbiased estimator in the complementary space.

    Args:
        A: A square linear operator whose trace is estimated.
        num_matvecs: Total number of matrix-vector products to use. Must be smaller
            than the dimension of the linear operator, and divisible by 3.
        distribution: Distribution of the random vectors used for the trace estimation.
            Can be either ``'rademacher'`` or ``'normal'``. Default: ``'rademacher'``.

    Returns:
        The estimated trace of the linear operator.

    Raises:
        ValueError: If the linear operator is not square or if the number of matrix-
            vector products is greater than the dimension of the linear operator
            (because then you can evaluate the true trace directly at the same cost)
            or not divisible by 3.

    Example:
        >>> from numpy import trace, mean
        >>> from numpy.random import rand, seed
        >>> seed(0) # make deterministic
        >>> A = rand(50, 50)
        >>> tr_A = trace(A) # exact trace as reference
        >>> # one- and multi-sample approximations
        >>> tr_A_low_precision = hutchpp_trace(A, num_matvecs=3)
        >>> tr_A_high_precision = hutchpp_trace(A, num_matvecs=30)
        >>> # compute the relative errors
        >>> rel_error_low_precision = abs(tr_A - tr_A_low_precision) / abs(tr_A)
        >>> rel_error_high_precision = abs(tr_A - tr_A_high_precision) / abs(tr_A)
        >>> assert rel_error_low_precision > rel_error_high_precision
        >>> round(tr_A, 4), round(tr_A_low_precision, 4), round(tr_A_high_precision, 4)
        (25.7342, 50.3488, 26.052)
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square. Got shape {A.shape}.")
    dim = A.shape[1]
    if num_matvecs >= dim or num_matvecs % 3 != 0:
        raise ValueError(
            f"num_matvecs ({num_matvecs}) must be less than A's size ({dim})"
            " and divisible by 3."
        )

    N = num_matvecs // 3

    # compute the orthogonal basis for the subspace spanned by AS, and evaluate the
    # exact trace using 2/3 of the available matrix-vector products
    AS = A @ column_stack([random_vector(dim, distribution) for _ in range(N)])
    Q, _ = qr(AS)
    tr_QT_A_Q = einsum("ji,ji", Q, A @ Q)

    # compute the trace in the complementary space using the remaining 1/3 of the
    # matrix-vector products
    G = column_stack([random_vector(dim, distribution) for _ in range(N)])

    # project out subspace
    A_proj_G = A @ (G - Q @ (Q.T @ G))
    A_proj_G -= Q @ (Q.T @ A_proj_G)
    # compute trace with vanilla Hutchinson
    tr_A_proj = einsum("ij,ij", G, A_proj_G) / N

    return tr_QT_A_Q + tr_A_proj
