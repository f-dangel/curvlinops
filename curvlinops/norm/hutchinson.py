"""Hutchinson-style matrix norm estimation."""

from numpy import dot
from scipy.sparse.linalg import LinearOperator

from curvlinops.sampling import random_vector


class HutchinsonSquaredFrobeniusNormEstimator:
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

    Example:
        >>> from numpy import mean, round
        >>> from numpy.linalg import norm
        >>> from numpy.random import rand, seed
        >>> seed(0) # make deterministic
        >>> A = rand(5, 5)
        >>> fro2_A = norm(A, ord='fro')**2 # exact squared Frobenius norm as reference
        >>> estimator = HutchinsonSquaredFrobeniusNormEstimator(A)
        >>> # one- and multi-sample approximations
        >>> fro2_A_low_prec = estimator.sample()
        >>> fro2_A_high_prec = mean([estimator.sample() for _ in range(1_000)])
        >>> assert abs(fro2_A - fro2_A_low_prec) > abs(fro2_A - fro2_A_high_prec)
        >>> round(fro2_A, 4), round(fro2_A_low_prec, 4), round(fro2_A_high_prec, 4)
        (10.7192, 8.3257, 10.6406)
    """

    def __init__(self, A: LinearOperator):
        """Store the linear operator whose squared Frobenius norm will be estimated.

        Args:
            A: Linear operator whose squared Frobenius norm will be estimated.
        """
        self._A = A

    def sample(self, distribution: str = "rademacher") -> float:
        """Draw a sample from the squared Frobenius norm estimator.

        Multiple samples can be combined into a more accurate squared Frobenius norm
        estimation via averaging.

        Args:
            distribution: Distribution of the vector along which the linear operator
                will be evaluated. Either ``'rademacher'`` or ``'normal'``.
                Default is ``'rademacher'``.

        Returns:
            Sample from the squared Frobenius norm estimator.
        """
        dim = self._A.shape[1]
        v = random_vector(dim, distribution)
        Av = self._A @ v
        return dot(Av, Av)
