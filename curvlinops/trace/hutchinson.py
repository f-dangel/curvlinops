"""Vanilla Hutchinson trace estimation."""

from typing import Callable, Dict

from numpy import dot, ndarray
from scipy.sparse.linalg import LinearOperator

from curvlinops.trace.sampling import normal, rademacher


class HutchinsonTraceEstimator:
    """Class to perform trace estimation with Hutchinson's method.

    For details, see

    - Hutchinson, M. (1989). A stochastic estimator of the trace of the influence
      matrix for laplacian smoothing splines. Communication in Statistics---Simulation
      and Computation.

    Example:
        >>> from numpy import trace, mean, round
        >>> from numpy.random import rand, seed
        >>> seed(0) # make deterministic
        >>> A = rand(10, 10)
        >>> tr_A = trace(A) # exact trace as reference
        >>> estimator = HutchinsonTraceEstimator(A)
        >>> # one- and multi-sample approximations
        >>> tr_A_low_precision = estimator.sample()
        >>> tr_A_high_precision = mean([estimator.sample() for _ in range(1_000)])
        >>> assert abs(tr_A - tr_A_low_precision) > abs(tr_A - tr_A_high_precision)
        >>> round(tr_A, 4), round(tr_A_low_precision, 4), round(tr_A_high_precision, 4)
        (4.4575, 6.6796, 4.3886)

    Attributes:
        SUPPORTED_DISTRIBUTIONS: Dictionary mapping supported distributions to their
            sampling functions.
    """

    SUPPORTED_DISTRIBUTIONS: Dict[str, Callable[[int], ndarray]] = {
        "rademacher": rademacher,
        "normal": normal,
    }

    def __init__(self, A: LinearOperator):
        """Store the linear operator whose trace will be estimated.

        Args:
            A: Linear square-shaped operator whose trace will be estimated.

        Raises:
            ValueError: If the operator is not square.
        """
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square. Got shape {A.shape}.")
        self._A = A

    def sample(self, distribution: str = "rademacher") -> float:
        """Draw a sample from the trace estimator.

        Multiple samples can be combined into a more accurate trace estimation via
        averaging.

        Args:
            distribution: Distribution of the vector along which the linear operator
                will be evaluated. Either ``'rademacher'`` or ``'normal'``.
                Default is ``'rademacher'``.

        Returns:
            Sample from the trace estimator.

        Raises:
            ValueError: If the distribution is not supported.
        """
        dim = self._A.shape[1]

        if distribution not in self.SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution {distribution:!r}. "
                f"Supported distributions are {list(self.SUPPORTED_DISTRIBUTIONS)}."
            )

        v = self.SUPPORTED_DISTRIBUTIONS[distribution](dim)
        Av = self._A @ v

        return dot(v, Av)
