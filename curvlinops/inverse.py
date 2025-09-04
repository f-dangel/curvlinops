"""Implements linear operator inverses."""

from __future__ import annotations

from math import sqrt
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
from warnings import warn

from einops import rearrange
from numpy import allclose, column_stack, ndarray
from scipy.sparse.linalg import LinearOperator, cg, lsmr
from torch import (
    Tensor,
    cat,
    cholesky_inverse,
    device,
    dtype,
    eye,
    float64,
    from_numpy,
    outer,
)
from torch.linalg import cholesky, eigh

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.ekfac import EKFACLinearOperator
from curvlinops.kfac import FactorType, KFACLinearOperator

KFACInvType = TypeVar(
    "KFACInvType", Optional[Tensor], Tuple[Optional[Tensor], Optional[Tensor]]
)


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


class InversePyTorchLinearOperator(PyTorchLinearOperator):
    """Base class for inverses of PyTorch linear operators."""

    def __init__(self, A: PyTorchLinearOperator):
        """Store the linear operator whose inverse should be represented.

        Args:
            A: PyTorch linear operator whose inverse is formed.
        """
        super().__init__(A._in_shape, A._out_shape)
        self._A = A

    def _infer_dtype(self) -> dtype:
        """Determine the linear operator's data type.

        Returns:
            The linear operator's dtype.
        """
        return self._A._infer_dtype()

    def _infer_device(self) -> device:
        """Determine the device the linear operators is defined on.

        Returns:
            The linear operator's device.
        """
        return self._A._infer_device()


class CGInverseLinearOperator(InversePyTorchLinearOperator):
    """Class for inverse linear operators via conjugate gradients.

    Note:
        Internally, this operator uses SciPy's CPU implementation of CG as PyTorch
        currently does not offer a CG interface that purely relies on matrix-vector
        products.
    """

    def __init__(self, A: PyTorchLinearOperator, **cg_hyperparameters):
        """Store the linear operator whose inverse should be represented.

        Args:
            A: PyTorch linear operator whose inverse is formed. Must represent a
                symmetric and positive-definite matrix.
            cg_hyperparameters: Keyword arguments for SciPy's CG implementation.
                For details, see
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html.
        """
        super().__init__(A)
        self._A_scipy = A.to_scipy()
        self._cg_hyperparameters = cg_hyperparameters

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Multiply X by the inverse of A.

        Args:
             X: Matrix for multiplication.

        Returns:
             Result of inverse matrix-vector multiplication, ``A⁻¹ @ X``.
        """
        # flatten and convert to numpy
        X_np = (
            cat([x.flatten(end_dim=-2) for x in X])
            .cpu()
            .numpy()
            .astype(self._A_scipy.dtype)
        )
        _, num_vecs = X_np.shape

        # apply CG to each vector in SciPy
        Ainv_X = [cg(self._A_scipy, x, **self._cg_hyperparameters)[0] for x in X_np.T]
        Ainv_X = column_stack(Ainv_X)

        # convert to PyTorch and unflatten
        dev, dt = self._infer_device(), self._infer_dtype()
        Ainv_X = from_numpy(Ainv_X).to(dev, dt)
        Ainv_X = [
            r.reshape(*s, num_vecs)
            for r, s in zip(Ainv_X.split(self._out_shape_flat), self._out_shape)
        ]
        return Ainv_X

    def _adjoint(self) -> CGInverseLinearOperator:
        """Return the linear operator's adjoint: (A^-1)* = (A*)^-1.

        Returns:
            A linear operator representing the adjoint.
        """
        return CGInverseLinearOperator(self._A._adjoint(), **self._cg_hyperparameters)


class LSMRInverseLinearOperator(_InverseLinearOperator):
    """Class for inverse linear operators via LSMR.

    See https://arxiv.org/abs/1006.0758 for details on the LSMR algorithm.
    """

    def __init__(self, A: LinearOperator):
        """Store the linear operator whose inverse should be represented.

        Args:
            A: Linear operator whose inverse is formed.
        """
        super().__init__(A.dtype, A.shape)
        self._A = A

        # LSMR hyperparameters
        self.set_lsmr_hyperparameters()

    def set_lsmr_hyperparameters(
        self,
        damp: float = 0.0,
        atol: Optional[float] = 1e-6,
        btol: Optional[float] = 1e-6,
        conlim: Optional[float] = 1e8,
        maxiter: Optional[int] = None,
        show: Optional[bool] = False,
        x0: Optional[ndarray] = None,
    ):
        """Store hyperparameters for LSMR.

        They will be used to approximate the inverse matrix-vector products.

        For more detail, see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html.

        # noqa: DAR101
        """
        self._lsmr_hyperparameters = {
            "damp": damp,
            "atol": atol,
            "btol": btol,
            "conlim": conlim,
            "maxiter": maxiter,
            "show": show,
            "x0": x0,
        }

    def matvec_with_info(
        self, x: ndarray
    ) -> Tuple[ndarray, int, int, float, float, float, float, float]:
        """Multiply x by the inverse of A and return additional information.

        Args:
             x: Vector for multiplication.

        Returns:
            Result of inverse matrix-vector multiplication, ``A⁻¹ @ x`` with additional
            information; see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html
            for details (same return values).
        """
        return lsmr(self._A, x, **self._lsmr_hyperparameters)

    def _matvec(self, x: ndarray) -> ndarray:
        """Multiply x by the inverse of A.

        Args:
             x: Vector for multiplication.

        Returns:
             Result of inverse matrix-vector multiplication, ``A⁻¹ @ x``.
        """
        return self.matvec_with_info(x)[0]


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


class KFACInverseLinearOperator(PyTorchLinearOperator):
    """Class to invert instances of the ``KFACLinearOperator``.

    Attributes:
        SELF_ADJOINT: Whether the operator is self-adjoint. ``True`` for KFAC-inverse.
    """

    SELF_ADJOINT: bool = True

    def __init__(
        self,
        A: KFACLinearOperator,
        damping: Union[float, Tuple[float, float]] = 0.0,
        use_heuristic_damping: bool = False,
        min_damping: float = 1e-8,
        use_exact_damping: bool = False,
        cache: bool = True,
        retry_double_precision: bool = True,
    ):
        r"""Store the linear operator whose inverse should be represented.

        Args:
            A: ``KFACLinearOperator`` whose inverse is formed.
            damping: Damping value(s) for all input and gradient covariances. If tuple,
                the first value is used for the input covariances and the second value
                for the gradient covariances. Note that if heuristic or exact damping is
                used the damping cannot be a tuple. Default: ``0.``.
            use_heuristic_damping: Whether to use a heuristic damping strategy by
                `Martens and Grosse, 2015 <https://arxiv.org/abs/1503.05671>`_
                (Section 6.3). For input covariances
                :math:`A \in \mathbb{R}^{n \times n}` and gradient covariances
                :math:`B \in \mathbb{R}^{m \times m}`, we define
                :math:`\pi := \sqrt{\frac{m\; \text{tr}(A)}{n\; \text{tr}(B)}}` and set the
                damping for the input covariances :math:`A` to
                :math:`\max(\pi\; \sqrt{\text{damping}}, \text{min_damping})` and for the
                gradient covariances :math:`B` to
                :math:`\max(\frac{1}{\pi}\; \sqrt{\text{damping}}, \text{min_damping})`.
                Default: ``False``.
            min_damping: Minimum damping value. Only used if ``use_heuristic_damping``
                is ``True``. Default: ``1e-8``.
            use_exact_damping: Whether to use exact damping, i.e. to invert
                :math:`(A \otimes B) + \text{damping}\; \mathbf{I}`. This is implemented
                via eigendecompositions of the Kronecker factors, e.g. see equation (21)
                in `Grosse et al., 2023 <https://arxiv.org/abs/2308.03296>`_.
                Note that the eigendecomposition synchronizes the device with the CPU.
                Default: ``False``.
            cache: Whether to cache the inverses of the Kronecker factors.
                Default: ``True``.
            retry_double_precision: Whether to retry Cholesky decomposition used for
                inversion in double precision. Default: ``True``.

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
        super().__init__(
            [tuple(s) for s in A._in_shape], [tuple(s) for s in A._out_shape]
        )
        self._A = A
        self._infer_device = A._infer_device
        self._infer_dtype = A._infer_dtype
        if use_heuristic_damping and use_exact_damping:
            raise ValueError("Either use heuristic damping or exact damping, not both.")
        if (use_heuristic_damping or use_exact_damping) and isinstance(damping, tuple):
            raise ValueError(
                "Heuristic and exact damping require a single damping value."
            )
        if isinstance(self._A, EKFACLinearOperator) and not use_exact_damping:
            raise ValueError("Only exact damping is supported for EKFAC.")

        self._damping = damping
        self._use_heuristic_damping = use_heuristic_damping
        self._min_damping = min_damping
        self._use_exact_damping = use_exact_damping
        self._cache = cache
        self._retry_double_precision = retry_double_precision
        self._inverse_input_covariances: Dict[str, FactorType] = {}
        self._inverse_gradient_covariances: Dict[str, FactorType] = {}

    def _compute_damping(
        self, aaT: Optional[Tensor], ggT: Optional[Tensor]
    ) -> Tuple[float, float]:
        """Compute the damping values for the input and gradient covariances.

        Args:
            aaT: Input covariance matrix. ``None`` for biases.
            ggT: Gradient covariance matrix.

        Returns:
            Damping values for the input and gradient covariances.
        """
        if self._use_heuristic_damping and aaT is not None and ggT is not None:
            # Martens and Grosse, 2015 (https://arxiv.org/abs/1503.05671) (Section 6.3)
            aaT_eig_mean = aaT.trace() / aaT.shape[0]
            ggT_eig_mean = ggT.trace() / ggT.shape[0]
            if aaT_eig_mean >= 0.0 and ggT_eig_mean > 0.0:
                sqrt_eig_mean_ratio = (aaT_eig_mean / ggT_eig_mean).sqrt()
                sqrt_damping = sqrt(self._damping)
                damping_aaT = max(sqrt_damping * sqrt_eig_mean_ratio, self._min_damping)
                damping_ggT = max(sqrt_damping / sqrt_eig_mean_ratio, self._min_damping)
                return damping_aaT, damping_ggT

        if isinstance(self._damping, tuple):
            return self._damping

        return self._damping, self._damping

    def _damped_cholesky(self, M: Tensor, damping: float) -> Tensor:
        """Compute the Cholesky decomposition of a matrix with damping.

        Args:
            M: Matrix for Cholesky decomposition.
            damping: Damping value.

        Returns:
            Cholesky decomposition of the matrix with damping.
        """
        return cholesky(
            M.add(eye(M.shape[0], dtype=M.dtype, device=M.device), alpha=damping)
        )

    def _compute_inv_damped_eigenvalues(
        self, aaT_eigenvalues: Tensor, ggT_eigenvalues: Tensor, name: str
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Compute the inverses of the damped eigenvalues for a given layer.

        Args:
            aaT_eigenvalues: Eigenvalues of the input covariance matrix.
            ggT_eigenvalues: Eigenvalues of the gradient covariance matrix.
            name: Name of the layer for which to damp and invert eigenvalues.

        Returns:
            Inverses of the damped eigenvalues.
        """
        param_pos = self._A._mapping[name]
        if (
            not self._A._separate_weight_and_bias
            and "weight" in param_pos
            and "bias" in param_pos
        ):
            inv_damped_eigenvalues = (
                outer(ggT_eigenvalues, aaT_eigenvalues).add_(self._damping).pow_(-1)
            )
        else:
            inv_damped_eigenvalues: Dict[str, Tensor] = {}
            for p_name, pos in param_pos.items():
                inv_damped_eigenvalues[pos] = (
                    outer(ggT_eigenvalues, aaT_eigenvalues)
                    if p_name == "weight"
                    else ggT_eigenvalues.clone()
                )
                inv_damped_eigenvalues[pos].add_(self._damping).pow_(-1)
        return inv_damped_eigenvalues

    def _compute_factors_eigendecomposition(
        self, aaT: Optional[Tensor], ggT: Optional[Tensor]
    ) -> Tuple[FactorType, FactorType]:
        """Compute the eigendecompositions of the Kronecker factors for a given layer.

        Used to perform damped preconditioning in Kronecker-factored eigenbasis (KFE).

        Args:
            aaT: Input covariance matrix. ``None`` for biases.
            ggT: Gradient covariance matrix.

        Returns:
            Tuple of eigenvalues and eigenvectors of the input and gradient covariance
            Kronecker factors. Can be ``None`` if the input or gradient covariance is
            ``None`` (e.g. the input covariances for biases).
        """
        aaT_eigenvalues, aaT_eigenvectors = (None, None) if aaT is None else eigh(aaT)
        ggT_eigenvalues, ggT_eigenvectors = (None, None) if ggT is None else eigh(ggT)
        return (aaT_eigenvalues, aaT_eigenvectors), (ggT_eigenvalues, ggT_eigenvectors)

    def _compute_inverse_factors(
        self, aaT: Optional[Tensor], ggT: Optional[Tensor]
    ) -> Tuple[FactorType, FactorType]:
        """Compute the inverses of the Kronecker factors for a given layer.

        Args:
            aaT: Input covariance matrix. ``None`` for biases.
            ggT: Gradient covariance matrix.

        Returns:
            Tuple of inverses (or eigendecompositions) of the input and gradient
            covariance Kronecker factors and optionally eigenvalues. Can be ``None`` if
            the input or gradient covariance is ``None`` (e.g. the input covariances for
            biases).

        Raises:
            RuntimeError: If a Cholesky decomposition (and optionally the retry in
                double precision) fails.
        """
        inverse_factors = []
        for factor, damping in zip((aaT, ggT), self._compute_damping(aaT, ggT)):
            # Compute inverse of factor matrix via Cholesky decomposition
            try:
                factor_chol = (
                    None if factor is None else self._damped_cholesky(factor, damping)
                )
            except RuntimeError as error:
                if self._retry_double_precision and factor.dtype != float64:
                    warn(
                        f"Failed to compute Cholesky decomposition in {factor.dtype} "
                        f"precision with error {error}. "
                        "Retrying in double precision...",
                        stacklevel=2,
                    )
                    # Retry in double precision
                    original_type = factor.dtype
                    factor_chol = self._damped_cholesky(factor.to(float64), damping)
                    factor_chol = factor_chol.to(original_type)
                else:
                    raise error
            factor_inv = None if factor_chol is None else cholesky_inverse(factor_chol)
            inverse_factors.append(factor_inv)
        return tuple(inverse_factors)

    def _compute_or_get_cached_inverse(
        self, name: str
    ) -> Tuple[FactorType, FactorType]:
        """Invert the Kronecker factors of the KFACLinearOperator or retrieve them.

        Args:
            name: Name of the layer for which to invert Kronecker factors.

        Returns:
            Tuple of inverses (or eigendecompositions) of the input and gradient
            covariance Kronecker factors and optionally eigenvalues. Can be ``None`` if
            the input or gradient covariance is ``None`` (e.g. the input covariances for
            biases).
        """
        if isinstance(self._A, EKFACLinearOperator):
            aaT_eigenvectors = self._A._input_covariances_eigenvectors.get(name)
            ggT_eigenvectors = self._A._gradient_covariances_eigenvectors.get(name)
            eigenvalues = self._A._corrected_eigenvalues[name]
            if isinstance(eigenvalues, dict):
                inv_damped_eigenvalues = {
                    k: v.add(self._damping).pow_(-1) for k, v in eigenvalues.items()
                }
            elif isinstance(eigenvalues, Tensor):
                inv_damped_eigenvalues = eigenvalues.add(self._damping).pow_(-1)
            return aaT_eigenvectors, ggT_eigenvectors, inv_damped_eigenvalues

        if name in self._inverse_input_covariances:
            aaT_inv = self._inverse_input_covariances.get(name)
            ggT_inv = self._inverse_gradient_covariances.get(name)
            if self._use_exact_damping:
                aaT_eigenvectors, aaT_eigenvalues = aaT_inv
                ggT_eigenvectors, ggT_eigenvalues = ggT_inv
                inv_damped_eigenvalues = self._compute_inv_damped_eigenvalues(
                    aaT_eigenvalues, ggT_eigenvalues, name
                )
                return aaT_eigenvectors, ggT_eigenvectors, inv_damped_eigenvalues
            return aaT_inv, ggT_inv, None

        aaT = self._A._input_covariances.get(name)
        ggT = self._A._gradient_covariances.get(name)
        if self._use_exact_damping:
            (aaT_eigenvalues, aaT_eigenvectors), (ggT_eigenvalues, ggT_eigenvectors) = (
                self._compute_factors_eigendecomposition(aaT, ggT)
            )
            aaT_inv = (aaT_eigenvectors, aaT_eigenvalues)
            ggT_inv = (ggT_eigenvectors, ggT_eigenvalues)
            inv_damped_eigenvalues = self._compute_inv_damped_eigenvalues(
                aaT_eigenvalues, ggT_eigenvalues, name
            )
        else:
            aaT_inv, ggT_inv = self._compute_inverse_factors(aaT, ggT)
            inv_damped_eigenvalues = None

        if self._cache:
            self._inverse_input_covariances[name] = aaT_inv
            self._inverse_gradient_covariances[name] = ggT_inv

        return (
            aaT_inv[0] if self._use_exact_damping else aaT_inv,
            ggT_inv[0] if self._use_exact_damping else ggT_inv,
            inv_damped_eigenvalues,
        )

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication with the KFAC inverse.

        Args:
            X: Matrix for multiplication. Is a list of tensors where each entry has the
                same shape as a parameter with an additional trailing dimension of size
                ``K`` for the columns, i.e. ``[(*p1.shape, K), (*p2.shape, K), ...]``.

        Returns:
            Matrix-multiplication result ``KFAC⁻¹ @ X``. Has same shape as ``X``.
        """
        # Maybe compute (E)KFAC if not already done.
        if isinstance(self._A, EKFACLinearOperator):
            self._A._maybe_compute_ekfac()
        elif not (self._A._input_covariances or self._A._gradient_covariances):
            self._A.compute_kronecker_factors()

        KX: List[Tensor | None] = [None] * len(X)

        for mod_name, param_pos in self._A._mapping.items():
            # retrieve the inverses of the Kronecker factors from cache or invert them.
            # aaT_inv/ggT_inv are the eigenvectors of aaT/ggT if exact damping is used.
            aaT_inv, ggT_inv, inv_damped_eigenvalues = (
                self._compute_or_get_cached_inverse(mod_name)
            )
            # cache the weight shape to ensure correct shapes are returned
            if "weight" in param_pos:
                weight_shape = X[param_pos["weight"]].shape

            # bias and weights are treated jointly
            if (
                not self._A._separate_weight_and_bias
                and "weight" in param_pos.keys()
                and "bias" in param_pos.keys()
            ):
                w_pos, b_pos = param_pos["weight"], param_pos["bias"]
                X_w = rearrange(X[w_pos], "c_out ... m -> c_out (...) m")
                X_joint = cat([X_w, X[b_pos].unsqueeze(1)], dim=1)
                X_joint = self._A._left_and_right_multiply(
                    X_joint, aaT_inv, ggT_inv, inv_damped_eigenvalues
                )
                w_cols = X_w.shape[1]
                KX[w_pos], KX[b_pos] = X_joint.split([w_cols, 1], dim=1)
                KX[b_pos].squeeze_(1)
            else:
                self._A._separate_left_and_right_multiply(
                    KX, X, param_pos, aaT_inv, ggT_inv, inv_damped_eigenvalues
                )

            # restore original shapes
            if "weight" in param_pos:
                KX[param_pos["weight"]] = KX[param_pos["weight"]].view(weight_shape)

        return KX

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the inverse KFAC linear operator.

        Returns:
            State dictionary.
        """
        return {
            "A": self._A.state_dict(),
            # Attributes
            "damping": self._damping,
            "use_heuristic_damping": self._use_heuristic_damping,
            "min_damping": self._min_damping,
            "use_exact_damping": self._use_exact_damping,
            "cache": self._cache,
            "retry_double_precision": self._retry_double_precision,
            # Inverse Kronecker factors (if computed and cached)
            "inverse_input_covariances": self._inverse_input_covariances,
            "inverse_gradient_covariances": self._inverse_gradient_covariances,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the state of the inverse KFAC linear operator.

        Args:
            state_dict: State dictionary.
        """
        self._A.load_state_dict(state_dict["A"])

        # Set attributes
        self._damping = state_dict["damping"]
        self._use_heuristic_damping = state_dict["use_heuristic_damping"]
        self._min_damping = state_dict["min_damping"]
        self._use_exact_damping = state_dict["use_exact_damping"]
        self._cache = state_dict["cache"]
        self._retry_double_precision = state_dict["retry_double_precision"]

        # Set inverse Kronecker factors (if computed and cached)
        self._inverse_input_covariances = state_dict["inverse_input_covariances"]
        self._inverse_gradient_covariances = state_dict["inverse_gradient_covariances"]

    @classmethod
    def from_state_dict(
        cls, state_dict: Dict[str, Any], A: KFACLinearOperator
    ) -> "KFACInverseLinearOperator":
        """Load an inverse KFAC linear operator from a state dictionary.

        Args:
            state_dict: State dictionary.
            A: ``KFACLinearOperator`` whose inverse is formed.

        Returns:
            Linear operator of inverse KFAC approximation.
        """
        inv_kfac = cls(
            A,
            damping=state_dict["damping"],
            use_heuristic_damping=state_dict["use_heuristic_damping"],
            min_damping=state_dict["min_damping"],
            use_exact_damping=state_dict["use_exact_damping"],
            cache=state_dict["cache"],
            retry_double_precision=state_dict["retry_double_precision"],
        )
        inv_kfac.load_state_dict(state_dict)
        return inv_kfac
