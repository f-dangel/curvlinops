"""Implementation of Hutch++ trace estimation from Meyer et al."""

from typing import Callable, Dict, Optional, Union

from numpy import column_stack, dot, ndarray
from numpy.linalg import qr
from scipy.sparse.linalg import LinearOperator

from curvlinops.trace.sampling import normal, rademacher


class HutchPPTraceEstimator:
    """Class to perform trace estimation with the Huch++ method.

    In contrast to vanilla Hutchinson, Hutch++ has lower variance, but requires more
    memory.

    For details, see

    - Meyer, R. A., Musco, C., Musco, C., & Woodruff, D. P. (2020). Hutch++:
      optimal stochastic trace estimation.

    Example:
        >>> from numpy import trace, mean, round
        >>> from numpy.random import rand, seed
        >>> seed(0) # make deterministic
        >>> A = rand(10, 10)
        >>> tr_A = trace(A) # exact trace as reference
        >>> estimator = HutchPPTraceEstimator(A)
        >>> # one- and multi-sample approximations
        >>> tr_A_low_precision = estimator.sample()
        >>> tr_A_high_precision = mean([estimator.sample() for _ in range(333)])
        >>> # assert abs(tr_A - tr_A_low_precision) > abs(tr_A - tr_A_high_precision)
        >>> round(tr_A, 4), round(tr_A_low_precision, 4), round(tr_A_high_precision, 4)
        (4.4575, 7.1653, 4.4588)

    Attributes:
        SUPPORTED_DISTRIBUTIONS: Dictionary mapping supported distributions to their
            sampling functions.
    """

    SUPPORTED_DISTRIBUTIONS: Dict[str, Callable[[int], ndarray]] = {
        "rademacher": rademacher,
        "normal": normal,
    }

    def __init__(
        self,
        A: LinearOperator,
        basis_dim: Optional[int] = None,
        basis_distribution: str = "rademacher",
    ):
        """Store the linear operator whose trace will be estimated.

        Args:
            A: Linear square-shaped operator whose trace will be estimated.
            basis_dim: Dimension of the subspace used for exact trace estimation.
                Can be at most the linear operator's dimension. By default, its
                size will be 1% of the matrix's dimension, but at most ``10``.
                This assumes that we are working with very large matrices and we can
                only afford storing a small number of columns at a time.
            basis_distribution: Distribution of the vectors used to construct the
                subspace. Either ``'rademacher'` or ``'normal'``. Default is
                ``'rademacher'``.

        Raises:
            ValueError: If the operator is not square, the basis dimension is too
                large, or the sampling distribution is not supported.

        Note:
            If you are planning to perform a fair (i.e. same computation budget)
            comparison with vanilla Hutchinson, ``basis_dim`` should be ``s / 3``
            where ``s`` is the number of samples used by vanilla Hutchinson. If
            ``s / 3`` requires storing a too large matrix, you can pick
            ``basis_dim = s1`` and draw ``s2`` samples from Hutch++ such that
            ``2 * s1 + s2 = s``.
        """
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square. Got shape {A.shape}.")
        self._A = A

        dim = A.shape[1]
        basis_dim = basis_dim if basis_dim is not None else min(10, max(dim // 100, 1))
        if basis_dim > self._A.shape[1]:
            raise ValueError(
                f"Basis dimension must be at most {self._A.shape[1]}. Got {basis_dim}."
            )
        self._basis_dim = basis_dim

        if basis_distribution not in self.SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution {basis_distribution:!r}. "
                f"Supported distributions are {list(self.SUPPORTED_DISTRIBUTIONS)}."
            )
        self._basis_distribution = basis_distribution

        # When drawing the first sample, the basis and its subspace trace will be
        # computed and stored in the following buffers for further samples
        self._Q: Union[ndarray, None] = None
        self._tr_QT_A_Q: Union[float, None] = None

    def sample(self, distribution: str = "rademacher") -> float:
        """Draw a sample from the trace estimator.

        Multiple samples can be combined into a more accurate trace estimation via
        averaging.

        Note:
            Calling this function for the first time will also compute the sub-space and
            its trace. Future calls will be faster as the latter are cached internally.

        Args:
            distribution: Distribution of the vector along which the linear operator
                will be evaluated. Either ``'rademacher'`` or ``'normal'``.
                Default is ``'rademacher'``.

        Returns:
            Sample from the trace estimator.

        Raises:
            ValueError: If the distribution is not supported.
        """
        self.maybe_compute_and_cache_subspace()

        if distribution not in self.SUPPORTED_DISTRIBUTIONS:
            raise ValueError(
                f"Unsupported distribution {distribution:!r}. "
                f"Supported distributions are {list(self.SUPPORTED_DISTRIBUTIONS)}."
            )

        dim = self._A.shape[1]
        v = self.SUPPORTED_DISTRIBUTIONS[distribution](dim)
        # project out subspace
        v -= self._Q @ (self._Q.T @ v)

        Av = self._A @ v

        return self._tr_QT_A_Q + dot(v, Av)

    def maybe_compute_and_cache_subspace(self):
        """Compute and cache the subspace and its trace if not already done."""
        if self._Q is not None and self._tr_QT_A_Q is not None:
            return

        dim = self._A.shape[1]
        S = column_stack(
            [
                self.SUPPORTED_DISTRIBUTIONS[self._basis_distribution](dim)
                for _ in range(self._basis_dim)
            ]
        )
        self._Q, _ = qr(S)

        self._tr_QT_A_Q = 0.0
        for i in range(self._basis_dim):
            v = self._Q[:, i]
            Av = self._A @ v
            self._tr_QT_A_Q += dot(v, Av)
