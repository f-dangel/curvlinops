"""Implements linear operator inverses."""

from __future__ import annotations

from math import sqrt
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
from warnings import warn

from einops import rearrange
from linear_operator.utils.linear_cg import linear_cg
from numpy import column_stack
from scipy.sparse.linalg import lsmr
from torch import (
    Tensor,
    as_tensor,
    cat,
    cholesky_inverse,
    device,
    dtype,
    eye,
    float64,
    isnan,
    outer,
)
from torch.linalg import cholesky, eigh

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.ekfac import EKFACLinearOperator
from curvlinops.kfac import FactorType, KFACLinearOperator

KFACInvType = TypeVar(
    "KFACInvType", Optional[Tensor], Tuple[Optional[Tensor], Optional[Tensor]]
)


class _InversePyTorchLinearOperator(PyTorchLinearOperator):
    """Base class for inverses of PyTorch linear operators."""

    def __init__(self, A: PyTorchLinearOperator):
        """Store the linear operator whose inverse should be represented.

        Args:
            A: PyTorch linear operator whose inverse is formed.

        Raises:
            ValueError: If the passed linear operator is not quadratic.
        """
        if A._in_shape != A._out_shape:
            raise ValueError(
                "Input linear operator must be square to form an inverse."
                + f"Got {A._in_shape} != {A._out_shape}."
            )
        super().__init__(A._in_shape, A._out_shape)
        self._A = A

    @property
    def dtype(self) -> dtype:
        """Determine the linear operator's data type.

        Returns:
            The linear operator's dtype.
        """
        return self._A.dtype

    @property
    def device(self) -> device:
        """Determine the device the linear operators is defined on.

        Returns:
            The linear operator's device.
        """
        return self._A.device


class CGInverseLinearOperator(_InversePyTorchLinearOperator):
    """Class for inverse linear operators via conjugate gradients.

    Note:
        Internally, this operator uses GPyTorch's implementation of CG.
    """

    def __init__(self, A: PyTorchLinearOperator, **cg_hyperparameters):
        """Store the linear operator whose inverse should be represented.

        Args:
            A: PyTorch linear operator whose inverse is formed. Must represent a
                symmetric and positive-definite matrix.
            cg_hyperparameters: Keyword arguments for GPyTorch's CG implementation.
                For details, see the documentation of the ``linear_cg`` function in
                https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/utils/linear_cg.py.
        """
        super().__init__(A)
        self._cg_hyperparameters = cg_hyperparameters

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Multiply X by the inverse of A.

        Args:
             X: Matrix for multiplication.

        Returns:
             Result of inverse matrix-vector multiplication, ``A⁻¹ @ X``.
        """
        X_flat = cat([x.flatten(end_dim=-2) for x in X])
        _, num_vecs = X_flat.shape

        # batched CG for all vectors in parallel
        Ainv_X = linear_cg(self._A.__matmul__, X_flat, **self._cg_hyperparameters)

        return [
            r.reshape(*s, num_vecs)
            for r, s in zip(Ainv_X.split(self._out_shape_flat), self._out_shape)
        ]

    def _adjoint(self) -> CGInverseLinearOperator:
        """Return the linear operator's adjoint: (A^-1)* = (A*)^-1.

        Returns:
            A linear operator representing the adjoint.
        """
        return CGInverseLinearOperator(self._A._adjoint(), **self._cg_hyperparameters)


class LSMRInverseLinearOperator(_InversePyTorchLinearOperator):
    """Class for inverse PyTorch linear operators via LSMR.

    See https://arxiv.org/abs/1006.0758 for details on the LSMR algorithm.

    Note:
        Internally, this operator uses SciPy's CPU implementation of LSMR as PyTorch
        currently does not offer an LSMR interface that purely relies on matrix-vector
        products.
    """

    def __init__(self, A: PyTorchLinearOperator, **lsmr_hyperparameters):
        """Store the linear operator whose inverse should be represented.

        Args:
            A: Linear operator whose inverse is formed.
            lsmr_hyperparameters: The hyper-parameters that will be passed to the
                LSMR implementation in SciPy. For more detail, see
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html.
        """
        super().__init__(A)
        self._A_scipy = A.to_scipy()
        self._lsmr_hyperparameters = lsmr_hyperparameters

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Multiply the inverse of A onto a matrix X in list format.

        Args:
             X: Matrix for multiplication in list format.

        Returns:
             Result of inverse matrix-matrix multiplication, ``A⁻¹ @ X`` in list format.
        """
        # flatten and convert to numpy
        X_np = (
            cat([x.flatten(end_dim=-2) for x in X])
            .cpu()
            .numpy()
            .astype(self._A_scipy.dtype)
        )
        _, num_vecs = X_np.shape

        # apply LSMR to each vector in SciPy (returns solution and info)
        Ainv_X = [lsmr(self._A_scipy, x, **self._lsmr_hyperparameters) for x in X_np.T]
        self._lsmr_info = [result[1:] for result in Ainv_X]
        Ainv_X = column_stack([result[0] for result in Ainv_X])

        # convert to PyTorch and unflatten
        Ainv_X = as_tensor(Ainv_X, device=self.device, dtype=self.dtype)
        Ainv_X = [
            r.reshape(*s, num_vecs)
            for r, s in zip(Ainv_X.split(self._out_shape_flat), self._out_shape)
        ]
        return Ainv_X

    def _adjoint(self) -> LSMRInverseLinearOperator:
        """Return the linear operator's adjoint: (A^-1)* = (A*)^-1.

        Returns:
            A linear operator representing the adjoint.
        """
        return LSMRInverseLinearOperator(
            self._A._adjoint(), **self._lsmr_hyperparameters
        )


class NeumannInverseLinearOperator(_InversePyTorchLinearOperator):
    """Class for inverse linear operators via truncated Neumann series.

    # noqa: B950

    See https://en.wikipedia.org/w/index.php?title=Neumann_series&oldid=1131424698#Approximate_matrix_inversion.

    Motivated by

    - Lorraine, J., Vicol, P., & Duvenaud, D. (2020). Optimizing millions of
      hyperparameters by implicit differentiation. In International Conference on
      Artificial Intelligence and Statistics (AISTATS).

    .. warning::
        The Neumann series can be non-convergent. In this case, the iterations
        will become numerically unstable, leading to ``NaN`` values.

    .. warning::
        The Neumann series can converge slowly.
        Use :py:class:`curvlinops.CGInverLinearOperator` for better accuracy.
    """

    def __init__(
        self,
        A: PyTorchLinearOperator,
        num_terms: int = 100,
        scale: float = 1.0,
        check_nan: bool = True,
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
        super().__init__(A)
        self._num_terms = num_terms
        self._scale = scale
        self._check_nan = check_nan

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Multiply the inverse of A onto a matrix in list format.

        Args:
             X: Matrix for multiplication in list format.

        Returns:
             Result of inverse matrix-vector multiplication, ``A⁻¹ @ x``.

        Raises:
            ValueError: If ``NaN`` check is turned on and ``NaN`` values are detected.
        """
        result_list, v_list = [x.clone() for x in X], [x.clone() for x in X]

        for idx in range(self._num_terms):
            v_list = [
                v.sub_(Av, alpha=self._scale)
                for v, Av in zip(v_list, self._A._matmat(v_list))
            ]
            result_list = [result.add_(v) for result, v in zip(result_list, v_list)]

            if self._check_nan and any(isnan(result).any() for result in result_list):
                raise ValueError(
                    f"Detected NaNs after application of {idx}-th term."
                    + " This is probably because the Neumann series is non-convergent."
                    + " Try decreasing `scale` and read the comment on convergence."
                )

        return [result.mul_(self._scale) for result in result_list]

    def _adjoint(self) -> NeumannInverseLinearOperator:
        """Return the linear operator's adjoint: (A^-1)* = (A*)^-1.

        Returns:
            A linear operator representing the adjoint.
        """
        return NeumannInverseLinearOperator(
            self._A._adjoint(),
            num_terms=self._num_terms,
            scale=self._scale,
            check_nan=self._check_nan,
        )


class KFACInverseLinearOperator(_InversePyTorchLinearOperator):
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
        super().__init__(A)
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
            (
                (aaT_eigenvalues, aaT_eigenvectors),
                (ggT_eigenvalues, ggT_eigenvectors),
            ) = self._compute_factors_eigendecomposition(aaT, ggT)
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
            (
                aaT_inv,
                ggT_inv,
                inv_damped_eigenvalues,
            ) = self._compute_or_get_cached_inverse(mod_name)
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
