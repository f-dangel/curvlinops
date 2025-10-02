"""Vanilla Hutchinson trace estimation."""

from typing import Union

from torch import Tensor, column_stack, einsum

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.sampling import random_vector
from curvlinops.utils import assert_is_square, assert_matvecs_subseed_dim


def hutchinson_trace(
    A: Union[Tensor, PyTorchLinearOperator],
    num_matvecs: int,
    distribution: str = "rademacher",
) -> Tensor:
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
        a := \frac{1}{N} \sum_{n=1}^N \mathbf{v}_n^\top \mathbf{A} \mathbf{v}_n
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
            than the dimension of the linear operator (because otherwise one can
            evaluate the true trace directly at the same cost).
        distribution: Distribution of the random vectors used for the trace estimation.
            Can be either ``'rademacher'`` or ``'normal'``. Default: ``'rademacher'``.

    Returns:
        The estimated trace of the linear operator.

    Example:
        >>> from torch import manual_seed, rand
        >>> _ = manual_seed(0) # make deterministic
        >>> A = rand(50, 50)
        >>> tr_A = A.trace().item() # exact trace as reference
        >>> # one- and multi-sample approximations
        >>> tr_A_low_precision = hutchinson_trace(A, num_matvecs=1).item()
        >>> tr_A_high_precision = hutchinson_trace(A, num_matvecs=40).item()
        >>> # compute the relative errors
        >>> rel_error_low_precision = abs(tr_A - tr_A_low_precision) / abs(tr_A)
        >>> rel_error_high_precision = abs(tr_A - tr_A_high_precision) / abs(tr_A)
        >>> assert rel_error_low_precision > rel_error_high_precision
        >>> round(tr_A, 4), round(tr_A_low_precision, 4), round(tr_A_high_precision, 4)
        (23.7836, -10.0279, 20.8427)
    """
    dim = assert_is_square(A)
    assert_matvecs_subseed_dim(A, num_matvecs)
    G = column_stack(
        [
            random_vector(dim, distribution, A.device, A.dtype)
            for _ in range(num_matvecs)
        ]
    )

    return einsum("ij,ij", G, A @ G) / num_matvecs
