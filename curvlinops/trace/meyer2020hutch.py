"""Implementation of Hutch++ trace estimation from Meyer et al."""

from typing import Optional, Union

from numpy import column_stack, dot, ndarray
from numpy.linalg import qr
from scipy.sparse.linalg import LinearOperator

from curvlinops.sampling import random_vector


class HutchPPTraceEstimator:
    r"""Class to perform trace estimation with the Huch++ method.

    In contrast to vanilla Hutchinson, Hutch++ has lower variance, but requires more
    memory.

    For details, see

    - Meyer, R. A., Musco, C., Musco, C., & Woodruff, D. P. (2020). Hutch++:
      optimal stochastic trace estimation.

    Let :math:`\mathbf{A}` be a square linear operator whose trace we want to
    approximate. First, we compute an orthonormal basis :math:`\mathbf{Q}` of a
    sub-space spanned by :math:`\mathbf{A} \mathbf{S}` where :math:`\mathbf{S}` is a
    tall random matrix with i.i.d. elements. Then, we compute the trace in the sub-space
    and apply Hutchinson's estimator in the remaining space spanned by
    :math:`\mathbf{I} - \mathbf{Q} \mathbf{Q}^\top`: We can draw a random vector
    :math:`\mathbf{v}` which satisfies
    :math:`\mathbb{E}[\mathbf{v} \mathbf{v}^\top] = \mathbf{I}` and sample from the
    estimator

    .. math::
        a
        := \mathrm{Tr}(\mathbf{Q}^\top \mathbf{A} \mathbf{Q})
        + \mathbf{v}^\top (\mathbf{I} - \mathbf{Q} \mathbf{Q}^\top)^\top
          \mathbf{A} (\mathbf{I} - \mathbf{Q} \mathbf{Q}^\top) \mathbf{v}
        \approx \mathrm{Tr}(\mathbf{A})\,.

    This estimator is unbiased, :math:`\mathbb{E}[a] = \mathrm{Tr}(\mathbf{A})`, as the
    first term is constant and the second part is Hutchinson's estimator in a sub-space.

    Example:
        >>> from numpy import trace, mean, round
        >>> from numpy.random import rand, seed
        >>> seed(0) # make deterministic
        >>> A = rand(10, 10)
        >>> tr_A = trace(A) # exact trace as reference
        >>> estimator = HutchPPTraceEstimator(A)
        >>> # one- and multi-sample approximations
        >>> tr_A_low_precision = estimator.sample()
        >>> tr_A_high_precision = mean([estimator.sample() for _ in range(998)])
        >>> # assert abs(tr_A - tr_A_low_precision) > abs(tr_A - tr_A_high_precision)
        >>> round(tr_A, 4), round(tr_A_low_precision, 4), round(tr_A_high_precision, 4)
        (4.4575, 2.4085, 4.5791)
    """

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
            ValueError: If the operator is not square or the basis dimension is too
                large.

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
        """
        self.maybe_compute_and_cache_subspace()

        dim = self._A.shape[1]
        v = random_vector(dim, distribution)
        # project out subspace
        v -= self._Q @ (self._Q.T @ v)
        Av = self._A @ v
        return self._tr_QT_A_Q + dot(v, Av)

    def maybe_compute_and_cache_subspace(self):
        """Compute and cache the subspace and its trace if not already done."""
        if self._Q is not None and self._tr_QT_A_Q is not None:
            return

        dim = self._A.shape[1]
        AS = column_stack(
            [
                self._A @ random_vector(dim, self._basis_distribution)
                for _ in range(self._basis_dim)
            ]
        )
        self._Q, _ = qr(AS)

        self._tr_QT_A_Q = 0.0
        for i in range(self._basis_dim):
            v = self._Q[:, i]
            Av = self._A @ v
            self._tr_QT_A_Q += dot(v, Av)
