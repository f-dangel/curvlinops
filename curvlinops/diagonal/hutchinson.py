"""Hutchinson-style matrix diagonal estimation."""


from numpy import ndarray
from scipy.sparse.linalg import LinearOperator

from curvlinops.sampling import random_vector


class HutchinsonDiagonalEstimator:
    """Class to perform diagonal estimation with Hutchinson's method.

    For details, see

    - Martens, J., Sutskever, I., & Swersky, K. (2012). Estimating the hessian by
      back-propagating curvature. International Conference on Machine Learning (ICML).

    Example:
        >>> from numpy import diag, mean, round
        >>> from numpy.random import rand, seed
        >>> from numpy.linalg import norm
        >>> seed(0) # make deterministic
        >>> A = rand(10, 10)
        >>> diag_A = diag(A) # exact diagonal as reference
        >>> estimator = HutchinsonDiagonalEstimator(A)
        >>> # one- and multi-sample approximations
        >>> diag_A_low_precision = estimator.sample()
        >>> samples = [estimator.sample() for _ in range(1_000)]
        >>> diag_A_high_precision = mean(samples, axis=0)
        >>> # compute residual norms
        >>> error_low_precision = norm(diag_A - diag_A_low_precision)
        >>> error_high_precision = norm(diag_A - diag_A_high_precision)
        >>> assert error_low_precision > error_high_precision
        >>> round(error_low_precision, 4), round(error_high_precision, 4)
        (5.7268, 0.1525)
    """

    def __init__(self, A: LinearOperator):
        """Store the linear operator whose diagonal will be estimated.

        Args:
            A: Linear square-shaped operator whose diagonal will be estimated.

        Raises:
            ValueError: If the operator is not square.
        """
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square. Got shape {A.shape}.")
        self._A = A

    def sample(self, distribution: str = "rademacher") -> ndarray:
        """Draw a sample from the trace estimator.

        Multiple samples can be combined into a more accurate trace estimation via
        averaging.

        Args:
            distribution: Distribution of the vector along which the linear operator
                will be evaluated. Either ``'rademacher'`` or ``'normal'``.
                Default is ``'rademacher'``.

        Returns:
            Sample from the diagonal estimator.
        """
        dim = self._A.shape[1]
        v = random_vector(dim, distribution)
        Av = self._A @ v
        return v * Av
