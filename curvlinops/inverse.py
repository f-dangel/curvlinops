"""Implements linear operator inverses."""

from math import sqrt
from typing import Dict, List, Optional, Tuple, Union

from einops import einsum, rearrange
from numpy import allclose, column_stack, ndarray
from scipy.sparse.linalg import LinearOperator, cg
from torch import Tensor, cat, cholesky_inverse, eye, outer
from torch.linalg import cholesky, eigh

from curvlinops.kfac import KFACLinearOperator

KFAC_INV_TYPE = Union[Optional[Tensor], Tuple[Optional[Tensor], Optional[Tensor]]]


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
        damping: Union[float, Tuple[float, float]] = 0.0,
        use_heuristic_damping: bool = False,
        min_damping: float = 1e-8,
        use_exact_damping: bool = False,
        cache: bool = True,
    ):
        """Store the linear operator whose inverse should be represented.

        Args:
            A: ``KFACLinearOperator`` whose inverse is formed.
            damping: Damping value(s) for all input and gradient covariances.
                Default: ``0.``.
            use_heuristic_damping: Whether to use a heuristic damping strategy.
                Default: ``False``.
            min_damping: Minimum damping value. Default: ``1e-8``. Only used if
                ``use_heuristic_damping`` is ``True``.
            use_exact_damping: Whether to use exact damping.
                Default: ``False``.
            cache: Whether to cache the inverses of the Kronecker factors.
                Default: ``True``.

        Raises:
            ValueError: If the linear operator is not a ``KFACLinearOperator``.
            ValueError: If both heuristic and exact damping are selected.
            ValueError: If heuristic or exact damping is used and the damping value is a
                tuple.
        """
        if not isinstance(A, KFACLinearOperator):
            raise ValueError(
                "The input `A` must be an instance of `KFACLinearOperator`."
            )
        super().__init__(A.dtype, A.shape)
        self._A = A
        if use_heuristic_damping and use_exact_damping:
            raise ValueError("Either use heuristic damping or exact damping, not both.")
        if (use_heuristic_damping or use_exact_damping) and isinstance(damping, tuple):
            raise ValueError(
                "Heuristic and exact damping require a single damping value."
            )

        self._damping = damping
        self._use_heuristic_damping = use_heuristic_damping
        self._min_damping = min_damping
        self._use_exact_damping = use_exact_damping
        self._cache = cache
        self._inverse_input_covariances: Dict[str, KFAC_INV_TYPE] = {}
        self._inverse_gradient_covariances: Dict[str, KFAC_INV_TYPE] = {}

    def _compute_damping(
        self, aaT: Optional[Tensor], ggT: Optional[Tensor]
    ) -> Tuple[float, float]:
        """Compute the damping values for the input and gradient covariances.

        Args:
            aaT: Input covariance matrix.
            ggT: Gradient covariance matrix.

        Returns:
            Damping values for the input and gradient covariances.
        """
        if self._use_heuristic_damping and aaT is not None and ggT is not None:
            aaT_eig_mean = aaT.trace() / len(aaT)
            ggT_eig_mean = ggT.trace() / len(ggT)
            if aaT_eig_mean >= 0.0 and ggT_eig_mean > 0.0:
                sqrt_eig_mean_ratio = (aaT_eig_mean / ggT_eig_mean).sqrt()
                sqrt_damping = sqrt(self._damping)
                damping_aaT = max(sqrt_damping * sqrt_eig_mean_ratio, self._min_damping)
                damping_ggT = max(sqrt_damping / sqrt_eig_mean_ratio, self._min_damping)
                return damping_aaT, damping_ggT

        if isinstance(self._damping, tuple):
            return self._damping

        return self._damping, self._damping

    def _compute_inverse_factors(
        self, aaT: Optional[Tensor], ggT: Optional[Tensor]
    ) -> Tuple[KFAC_INV_TYPE, KFAC_INV_TYPE]:
        """Compute the inverses of the Kronecker factors for a given layer.

        Args:
            aaT: Input covariance matrix.
            ggT: Gradient covariance matrix.

        Returns:
            Tuple of inverses (or eigendecompositions) of the input and gradient
            covariance Kronecker factors.
        """
        if self._use_exact_damping:
            # Compute eigendecomposition to perform damped preconditioning in
            # Kronecker-factored eigenbasis (KFE).
            aaT_eigvals, aaT_eigvecs = (None, None) if aaT is None else eigh(aaT)
            ggT_eigvals, ggT_eigvecs = (None, None) if ggT is None else eigh(ggT)
            return (aaT_eigvecs, aaT_eigvals), (ggT_eigvecs, ggT_eigvals)
        else:
            damping_aaT, damping_ggT = self._compute_damping(aaT, ggT)
            aaT_inv = (
                None
                if aaT is None
                else cholesky_inverse(
                    cholesky(
                        aaT.add(eye(aaT.shape[0], device=aaT.device), alpha=damping_aaT)
                    )
                )
            )
            ggT_inv = (
                None
                if ggT is None
                else cholesky_inverse(
                    cholesky(
                        ggT.add(eye(ggT.shape[0], device=ggT.device), alpha=damping_ggT)
                    )
                )
            )
            return aaT_inv, ggT_inv

    def _compute_or_get_cached_inverse(
        self, name: str
    ) -> Tuple[KFAC_INV_TYPE, KFAC_INV_TYPE]:
        """Invert the Kronecker factors of the KFACLinearOperator or retrieve them.

        Args:
            name: Name of the layer for which to invert Kronecker factors.

        Returns:
            Tuple of inverses (or eigendecompositions) of the input and gradient
            covariance Kronecker factors.
        """
        if name in self._inverse_input_covariances:
            aaT_inv = self._inverse_input_covariances.get(name)
            ggT_inv = self._inverse_gradient_covariances.get(name)
            return aaT_inv, ggT_inv

        aaT = self._A._input_covariances.get(name)
        ggT = self._A._gradient_covariances.get(name)
        aaT_inv, ggT_inv = self._compute_inverse_factors(aaT, ggT)

        if self._cache:
            self._inverse_input_covariances[name] = aaT_inv
            self._inverse_gradient_covariances[name] = ggT_inv

        return aaT_inv, ggT_inv

    def _left_and_right_multiply(
        self, M_joint: Tensor, aaT_inv: KFAC_INV_TYPE, ggT_inv: KFAC_INV_TYPE
    ) -> Tensor:
        """Left and right multiply matrix with inverse Kronecker factors.

        Args:
            M_joint: Matrix for multiplication.
            aaT_inv: Inverse of the input covariance Kronecker factor.
            ggT_inv: Inverse of the gradient covariance Kronecker factor.

        Returns:
            Matrix-multiplication result ``KFAC⁻¹ @ M_joint``.
        """
        if self._use_exact_damping:
            # Perform damped preconditioning in KFE, e.g. see equation (21) in
            # https://arxiv.org/abs/2308.03296.
            aaT_eigvecs, aaT_eigvals = aaT_inv
            ggT_eigvecs, ggT_eigvals = ggT_inv
            M_joint = einsum(
                ggT_eigvecs, M_joint, aaT_eigvecs, "i j, m i k, k l -> m j l"
            )
            M_joint.div_(outer(ggT_eigvals, aaT_eigvals).add_(self._damping))
            M_joint = einsum(
                ggT_eigvecs, M_joint, aaT_eigvecs, "i j, m j k, l k -> m i l"
            )
        else:
            M_joint = einsum(ggT_inv, M_joint, aaT_inv, "i j, m j k, k l -> m i l")
        return M_joint

    def _separate_left_and_right_multiply(
        self,
        M_torch: Tensor,
        param_pos: Dict[str, int],
        aaT_inv: KFAC_INV_TYPE,
        ggT_inv: KFAC_INV_TYPE,
    ) -> Tensor:
        """Multiply matrix with inverse Kronecker factors for separated weight and bias.

        Args:
            M_torch: Matrix for multiplication.
            param_pos: Dictionary with positions of the weight and bias parameters.
            aaT_inv: Inverse of the input covariance Kronecker factor.
            ggT_inv: Inverse of the gradient covariance Kronecker factor.

        Returns:
            Matrix-multiplication result ``KFAC⁻¹ @ M_torch``.
        """
        if self._use_exact_damping:
            # Perform damped preconditioning in KFE, e.g. see equation (21) in
            # https://arxiv.org/abs/2308.03296.
            aaT_eigvecs, aaT_eigvals = aaT_inv
            ggT_eigvecs, ggT_eigvals = ggT_inv

        for p_name, pos in param_pos.items():
            # for weights we need to multiply from the right with aaT
            # for weights and biases we need to multiply from the left with ggT
            if p_name == "weight":
                M_w = rearrange(M_torch[pos], "m c_out ... -> m c_out (...)")
                aaT_fac = aaT_eigvecs if self._use_exact_damping else aaT_inv
                M_torch[pos] = einsum(M_w, aaT_fac, "m i j, j k -> m i k")

            ggT_fac = ggT_eigvecs if self._use_exact_damping else ggT_inv
            dims = (
                "m i ... -> m j ..."
                if self._use_exact_damping
                else " m j ... -> m i ..."
            )
            M_torch[pos] = einsum(ggT_fac, M_torch[pos], f"i j, {dims}")

            if self._use_exact_damping:
                if p_name == "weight":
                    M_torch[pos].div_(
                        outer(ggT_eigvals, aaT_eigvals).add_(self._damping)
                    )
                    M_torch[pos] = einsum(
                        M_torch[pos], aaT_eigvecs, "m i j, k j -> m i k"
                    )
                else:
                    M_torch[pos].div_(ggT_eigvals.add_(self._damping))
                M_torch[pos] = einsum(
                    ggT_eigvecs, M_torch[pos], "i j, m j ... -> m i ..."
                )

        return M_torch

    def torch_matmat(
        self, M_torch: Union[Tensor, List[Tensor]]
    ) -> Union[Tensor, List[Tensor]]:
        """Apply the inverse of KFAC to a matrix (multiple vectors) in PyTorch.

        This allows for matrix-matrix products with the inverse KFAC approximation in
        PyTorch without converting tensors to numpy arrays, which avoids unnecessary
        device transfers when working with GPUs and flattening/concatenating.

        Args:
            M_torch: Matrix for multiplication. If list of tensors, each entry has the
                same shape as a parameter with an additional leading dimension of size
                ``K`` for the columns, i.e. ``[(K,) + p1.shape), (K,) + p2.shape, ...]``.
                If tensor, has shape ``[D, K]`` with some ``K``.

        Returns:
            Matrix-multiplication result ``KFAC⁻¹ @ M``. Return type is the same as the
            type of the input. If list of tensors, each entry has the same shape as a
            parameter with an additional leading dimension of size ``K`` for the columns,
            i.e. ``[(K,) + p1.shape, (K,) + p2.shape, ...]``. If tensor, has shape
            ``[D, K]`` with some ``K``.
        """
        return_tensor, M_torch = self._A._check_input_type_and_preprocess(M_torch)
        if not self._A._input_covariances and not self._A._gradient_covariances:
            self._A._compute_kfac()

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
                M_joint = self._left_and_right_multiply(M_joint, aaT_inv, ggT_inv)
                w_cols = M_w.shape[2]
                M_torch[w_pos], M_torch[b_pos] = M_joint.split([w_cols, 1], dim=2)
            else:
                M_torch = self._separate_left_and_right_multiply(
                    M_torch, param_pos, aaT_inv, ggT_inv
                )

        if return_tensor:
            M_torch = cat([rearrange(M, "k ... -> (...) k") for M in M_torch])

        return M_torch

    def torch_matvec(
        self, v_torch: Union[Tensor, List[Tensor]]
    ) -> Union[Tensor, List[Tensor]]:
        """Apply the inverse of KFAC to a vector in PyTorch.

        This allows for matrix-vector products with the inverse KFAC approximation in
        PyTorch without converting tensors to numpy arrays, which avoids unnecessary
        device transfers when working with GPUs and flattening/concatenating.

        Args:
            v_torch: Vector for multiplication. If list of tensors, each entry has the
                same shape as a parameter, i.e. ``[p1.shape, p2.shape, ...]``.
                If tensor, has shape ``[D]``.

        Returns:
            Matrix-multiplication result ``KFAC⁻¹ @ v``. Return type is the same as the
            type of the input. If list of tensors, each entry has the same shape as a
            parameter, i.e. ``[p1.shape, p2.shape, ...]``. If tensor, has shape ``[D]``.

        Raises:
            ValueError: If the input tensor has the wrong data type.
        """
        if isinstance(v_torch, list):
            v_torch = [v_torch_i.unsqueeze(0) for v_torch_i in v_torch]
            result = self.torch_matmat(v_torch)
            return [res.squeeze(0) for res in result]
        elif isinstance(v_torch, Tensor):
            return self.torch_matmat(v_torch.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(
                f"Invalid input type: {type(v_torch)}. Expected list of tensors or tensor."
            )

    def _matmat(self, M: ndarray) -> ndarray:
        """Apply the inverse of KFAC to a matrix (multiple vectors).

        Args:
            M: Matrix for multiplication. Has shape ``[D, K]`` with some ``K``.

        Returns:
            Matrix-multiplication result ``KFAC⁻¹ @ M``. Has shape ``[D, K]``.
        """
        M_torch = self._A._preprocess(M)
        M_torch = self.torch_matmat(M_torch)
        return self._A._postprocess(M_torch)
