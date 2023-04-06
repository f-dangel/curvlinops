"""Implements linear operator inverses."""

from numpy import allclose, column_stack, ndarray
from scipy.sparse.linalg import LinearOperator, cg


class _InverseLinearOperator(LinearOperator):
    """Base class for (approximate) inverses of linear operators."""

    def _matmat(self, X: ndarray) -> ndarray:
        """Matrix-matrix multiplication.

        Args:
            X: Matrix for multiplication.

        Returns:
            Matrix-multiplication result ``A⁻¹@ X``.
        """
        return column_stack([self @ col for col in X.T])


class CGInverseLinearOperator(_InverseLinearOperator):
    """Class for inverse linear operators via conjugate gradients."""

    def __init__(self, A: LinearOperator):
        """Store the linear operator whose inverse should be represented.

        Args:
            A: Linear operator whose inverse is formed. Must be symmetric and
                positive-definite

        """
        super().__init__(A.dtype, A.shape)
        self._A = A

        # CG hyperparameters
        self.set_cg_hyperparameters()

    def set_cg_hyperparameters(
        self, x0=None, tol=1e-05, maxiter=None, M=None, callback=None, atol=None
    ):
        """Store hyperparameters for CG.

        They will be used to approximate the inverse matrix-vector products.

        For more detail, see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html

        # noqa: DAR101
        """
        self._cg_hyperparameters = {
            "x0": x0,
            "tol": tol,
            "maxiter": maxiter,
            "M": M,
            "callback": callback,
            "atol": atol,
        }

    def _matvec(self, x: ndarray) -> ndarray:
        """Multiply x by the inverse of A.

        Args:
             x: Vector for multiplication.

        Returns:
             Result of inverse matrix-vector multiplication, ``A⁻¹ @ x``.
        """
        result, _ = cg(self._A, x, **self._cg_hyperparameters)
        return result


class NeumannInverseLinearOperator(_InverseLinearOperator):
    """Class for inverse linear operators via truncated Neumann series.

    See https://en.wikipedia.org/w/index.php?title=Neumann_series&oldid=1131424698#Approximate_matrix_inversion.

    Motivated by (referred to as lorraine2020optimizing in the following)

    - Lorraine, J., Vicol, P., & Duvenaud, D. (2020). Optimizing millions of
      hyperparameters by implicit differentiation. In International Conference on
      Artificial Intelligence and Statistics (AISTATS).

    .. warning::
        The Neumann series can be a very poor approximation of the inverse.
        Use :py:class:`curvlinops.CGInverLinearOperator` for better accuracy.

    .. warning::
        The Neumann series can be numerically unstable, leading to ``NaN``s.
    """

    def __init__(self, A: LinearOperator):
        """Store the linear operator whose inverse should be represented.

        Args:
            A: Linear operator whose inverse is formed.
        """
        super().__init__(A.dtype, A.shape)
        self._A = A

        # Neumann hyperparameters
        self.set_neumann_hyperparameters()

    def set_neumann_hyperparameters(
        self, num_terms: int = 10, scale: float = 1.0, check_nan: bool = True
    ):  # e-3):
        r"""Store hyperparameters for the truncated Neumann series.

        The Neumann series for an invertible linear operator :math:`\mathbf{A}` is

        .. math::
            \mathbf{A}^{-1}
            =
            \sum_{k=0}^{\infty}
            \left(\mathbf{I} - \mathbf{A} \right)^k\,.

        We truncate the series at ``num_terms`` (:math:`K`) and re-rescale the matrix
        by a number ``scale`` (:math:`\alpha`), which helps convergence of IVPs
        according to lorraine2020optimizing:

        .. math::
            \mathbf{A}^{-1}
            =
            \alpha (\alpha \mathbf{A})^{-1}
            =
            \alpha \sum_{k=0}^{\infty}
            \left(\mathbf{I} - \alpha \mathbf{A} \right)^{k}
            \approx
            \alpha \sum_{k=0}^{K}
            \left(\mathbf{I} - \alpha \mathbf{A} \right)^{k}\,.

        Args:
            num_terms: Number of terms in the truncated Neumann series. Default: ``20``.
            scale: Scale applied to the matrix in the Neumann iteration. Helps
                convergence of IVPs according to lorraine2020optimizing.
                Default: ``1e-3``.
            check_nan: Whether to check for NaNs while applying the truncated Neumann
                series. Default: ``True``.
        """
        self._num_terms = num_terms
        self._scale = scale
        self._check_nan = check_nan

    def _matvec(self, x: ndarray) -> ndarray:
        """Multiply x by the inverse of A.

        Args:
             x: Vector for multiplication.

        Returns:
             Result of inverse matrix-vector multiplication, ``A⁻¹ @ x``.
        """
        result, v = x.copy(), x.copy()

        for idx in range(self._num_terms):
            v = v - self._scale * (self._A @ v)
            result = result + v

            if self._check_nan and not allclose(result, result):
                raise ValueError(f"Detected NaNs after application of {idx}-th term.")

        return self._scale * result
