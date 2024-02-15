"""Implements linear operator inverses."""

from typing import Dict, Optional, Tuple

from einops import einsum, rearrange
from numpy import allclose, column_stack, ndarray
from scipy.sparse.linalg import LinearOperator, cg
from torch import Tensor, cat, cholesky_inverse, eye
from torch.linalg import cholesky

from curvlinops.kfac import KFACLinearOperator


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

    # noqa: B950

    See https://en.wikipedia.org/w/index.php?title=Neumann_series&oldid=1131424698#Approximate_matrix_inversion.

    Motivated by (referred to as lorraine2020optimizing in the following)

    - Lorraine, J., Vicol, P., & Duvenaud, D. (2020). Optimizing millions of
      hyperparameters by implicit differentiation. In International Conference on
      Artificial Intelligence and Statistics (AISTATS).

    .. warning::
        The Neumann series can be non-convergent. In this case, the iterations
        will become numerically unstable, leading to ``NaN``s.

    .. warning::
        The Neumann series can converge slowly.
        Use :py:class:`curvlinops.CGInverLinearOperator` for better accuracy.
    """

    DEFAULT_NUM_TERMS = 100
    DEFAULT_SCALE = 1.0
    DEFAULT_CHECK_NAN = True

    def __init__(
        self,
        A: LinearOperator,
        num_terms: int = DEFAULT_NUM_TERMS,
        scale: float = DEFAULT_SCALE,
        check_nan: bool = DEFAULT_CHECK_NAN,
    ):
        r"""Store the linear operator whose inverse should be represented.

        The Neumann series for an invertible linear operator :math:`\mathbf{A}` is

        .. math::
            \mathbf{A}^{-1}
            =
            \sum_{k=0}^{\infty}
            \left(\mathbf{I} - \mathbf{A} \right)^k\,.

        and is convergent if all eigenvalues satisfy
        :math:`0 < \lambda(\mathbf{A}) < 2`.

        By re-rescaling the matrix by ``scale`` (:math:`\alpha`), we have:

        .. math::
            \mathbf{A}^{-1}
            =
            \alpha (\alpha \mathbf{A})^{-1}
            =
            \alpha \sum_{k=0}^{\infty}
            \left(\mathbf{I} - \alpha \mathbf{A} \right)^k\,,

        which and is convergent if :math:`0 < \lambda(\mathbf{A}) < \frac{2}{\alpha}`.

        Additionally, we truncate the series at ``num_terms`` (:math:`K`):

        .. math::
            \mathbf{A}^{-1}
            \approx
            \alpha \sum_{k=0}^{K}
            \left(\mathbf{I} - \alpha \mathbf{A} \right)^k\,.

        Args:
            A: Linear operator whose inverse is formed.
            num_terms: Number of terms in the truncated Neumann series.
                Default: ``100``.
            scale: Scale applied to the matrix in the Neumann iteration. Crucial
                for convergence of Neumann series (details above). Default: ``1.0``.
            check_nan: Whether to check for NaNs while applying the truncated Neumann
                series. Default: ``True``.
        """
        super().__init__(A.dtype, A.shape)
        self._A = A
        self.set_neumann_hyperparameters(
            num_terms=num_terms, scale=scale, check_nan=check_nan
        )

    def set_neumann_hyperparameters(
        self,
        num_terms: int = DEFAULT_NUM_TERMS,
        scale: float = DEFAULT_SCALE,
        check_nan: bool = DEFAULT_CHECK_NAN,
    ):
        """Store hyperparameters for the truncated Neumann series.

        Args:
            num_terms: Number of terms in the truncated Neumann series.
                Default: ``100``.
            scale: Scale applied to the matrix in the Neumann iteration. Crucial
                for convergence of Neumann series. Default: ``1.0``.
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

        Raises:
            ValueError: If ``NaN`` check is turned on and ``NaN``s are detected.
        """
        result, v = x.copy(), x.copy()

        for idx in range(self._num_terms):
            v = v - self._scale * (self._A @ v)
            result = result + v

            if self._check_nan and not allclose(result, result):
                raise ValueError(
                    f"Detected NaNs after application of {idx}-th term."
                    + " This is probably because the Neumann series is non-convergent."
                    + " Try decreasing `scale` and read the comment on convergence."
                )

        return self._scale * result


class KFACInverseLinearOperator(_InverseLinearOperator):
    """Class to invert instances of the ``KFACLinearOperator``."""

    def __init__(
        self,
        A: KFACLinearOperator,
        damping: Tuple[float, float] = (0.0, 0.0),
        cache: bool = True,
    ):
        """Store the linear operator whose inverse should be represented.

        Args:
            A: ``KFACLinearOperator`` whose inverse is formed.
            damping: Damping values for all input and gradient covariances.
                Default: ``(0., 0.)``.
            cache: Whether to cache the inverses of the Kronecker factors.
                Default: ``True``.

        Raises:
            ValueError: If the linear operator is not a ``KFACLinearOperator``.
        """
        if not isinstance(A, KFACLinearOperator):
            raise ValueError(
                "The input `A` must be an instance of `KFACLinearOperator`."
            )
        super().__init__(A.dtype, A.shape)
        self._A = A
        self._damping = damping
        self._cache = cache
        self._inverse_input_covariances: Dict[str, Tensor] = {}
        self._inverse_gradient_covariances: Dict[str, Tensor] = {}

    def _compute_or_get_cached_inverse(
        self, name: str
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Invert the Kronecker factors of the KFACLinearOperator or retrieve them.

        Args:
            name: Name of the layer for which to invert Kronecker factors.

        Returns:
            Tuple of inverses of the input and gradient covariance Kronecker factors.
        """
        if name in self._inverse_input_covariances:
            aaT_inv = self._inverse_input_covariances.get(name)
            ggT_inv = self._inverse_gradient_covariances.get(name)
            return aaT_inv, ggT_inv
        aaT = self._A._input_covariances.get(name)
        ggT = self._A._gradient_covariances.get(name)
        aaT_inv = (
            None
            if aaT is None
            else cholesky_inverse(
                cholesky(aaT + self._damping[0] * eye(aaT.shape[0], device=aaT.device))
            )
        )
        ggT_inv = (
            None
            if ggT is None
            else cholesky_inverse(
                cholesky(ggT + self._damping[1] * eye(ggT.shape[0], device=ggT.device))
            )
        )
        if self._cache:
            self._inverse_input_covariances[name] = aaT_inv
            self._inverse_gradient_covariances[name] = ggT_inv
        return aaT_inv, ggT_inv

    def _matmat(self, M: ndarray) -> ndarray:
        """Multiply a matrix ``M`` x by the inverse of KFAC.

        Args:
             M: Matrix for multiplication.

        Returns:
             Result of inverse matrix-matrixmultiplication, ``KFAC⁻¹ @ M``.
        """
        if not self._A._input_covariances and not self._A._gradient_covariances:
            self._A._compute_kfac()

        M_torch = self._A._preprocess(M)

        for mod_name, param_pos in self._A._mapping.items():
            # retrieve the inverses of the Kronecker factors from cache or invert them
            aaT_inv, ggT_inv = self._compute_or_get_cached_inverse(mod_name)

            # bias and weights are treated jointly
            if (
                not self._A._separate_weight_and_bias
                and "weight" in param_pos.keys()
                and "bias" in param_pos.keys()
            ):
                w_pos, b_pos = param_pos["weight"], param_pos["bias"]
                M_w = rearrange(M_torch[w_pos], "m c_out ... -> m c_out (...)")
                M_joint = cat([M_w, M_torch[b_pos].unsqueeze(2)], dim=2)
                M_joint = einsum(ggT_inv, M_joint, aaT_inv, "i j,m j k, k l -> m i l")

                w_cols = M_w.shape[2]
                M_torch[w_pos], M_torch[b_pos] = M_joint.split([w_cols, 1], dim=2)

            # for weights we need to multiply from the right with aaT
            # for weights and biases we need to multiply from the left with ggT
            else:
                for p_name, pos in param_pos.items():
                    if p_name == "weight":
                        M_w = rearrange(M_torch[pos], "m c_out ... -> m c_out (...)")
                        M_torch[pos] = einsum(M_w, aaT_inv, "m i j, j k -> m i k")

                    M_torch[pos] = einsum(
                        ggT_inv, M_torch[pos], "i j, m j ... -> m i ..."
                    )

        return self._A._postprocess(M_torch)
