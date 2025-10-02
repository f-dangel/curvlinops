"""Hutchinson-style matrix diagonal estimation."""

from typing import Union

from torch import Tensor, column_stack, einsum

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.sampling import random_vector
from curvlinops.utils import assert_is_square, assert_matvecs_subseed_dim


def hutchinson_diag(
    A: Union[PyTorchLinearOperator, Tensor],
    num_matvecs: int,
    distribution: str = "rademacher",
) -> Tensor:
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
            than the dimension of the linear operator (because otherwise one can
            evaluate the true diagonal directly at the same cost).
        distribution: Distribution of the random vectors used for the diagonal
            estimation. Can be either ``'rademacher'`` or ``'normal'``.
            Default: ``'rademacher'``.

    Returns:
        The estimated diagonal of the linear operator.

    Example:
        >>> from torch import manual_seed, rand
        >>> from torch.linalg import vector_norm
        >>> _ = manual_seed(0) # make deterministic
        >>> A = rand(40, 40)
        >>> diag_A = A.diag() # exact diagonal as reference
        >>> # one- and multi-sample approximations
        >>> diag_A_low_precision = hutchinson_diag(A, num_matvecs=1)
        >>> diag_A_high_precision = hutchinson_diag(A, num_matvecs=30)
        >>> # compute residual norms
        >>> error_low_precision = (vector_norm(diag_A - diag_A_low_precision) / vector_norm(diag_A)).item()
        >>> error_high_precision = (vector_norm(diag_A - diag_A_high_precision) / vector_norm(diag_A)).item()
        >>> assert error_low_precision > error_high_precision
        >>> round(error_low_precision, 4), round(error_high_precision, 4)
        (3.2648, 0.9253)
    """
    dim = assert_is_square(A)
    assert_matvecs_subseed_dim(A, num_matvecs)
    G = column_stack(
        [
            random_vector(dim, distribution, A.device, A.dtype)
            for _ in range(num_matvecs)
        ]
    )

    return einsum("ij,ij->i", G, A @ G) / num_matvecs
