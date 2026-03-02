"""Implements linear operator inverses."""

from __future__ import annotations

from linear_operator.utils.linear_cg import linear_cg
from numpy import column_stack
from scipy.sparse.linalg import lsmr
from torch import Tensor, as_tensor, cat, device, dtype, isnan

from curvlinops._checks import _check_same_tensor_list_shape
from curvlinops._torch_base import PyTorchLinearOperator


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

    def _matmat(self, X: list[Tensor]) -> list[Tensor]:
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

    def _matmat(self, X: list[Tensor]) -> list[Tensor]:
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
    - Wang, A., Nguyen, E., Yang, R., Bae, J., McIlraith, S. A., & Grosse,
      R. B. (2025). Better Training Data Attribution via Better Inverse
      Hessian-Vector Products. In Advances in Neural Information Processing
      Systems (NeurIPS 2025).

    .. warning::
        The Neumann series can be non-convergent. In this case, the iterations
        will become numerically unstable, leading to ``NaN`` values.

    .. warning::
        The Neumann series can converge slowly.
        Use :py:class:`curvlinops.CGInverseLinearOperator` for better accuracy.
    """

    def __init__(
        self,
        A: PyTorchLinearOperator,
        num_terms: int = 100,
        scale: float = 1.0,
        check_nan: bool = True,
        preconditioner: None | PyTorchLinearOperator = None,
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
            preconditioner: Optional preconditioner :math:`\mathbf{P}` used in the
                preconditioned Neumann/Richardson iteration
                :math:`\mathbf{A}^{-1} \approx \alpha \sum_{k=0}^{K}
                (\mathbf{I} - \alpha \mathbf{P}\mathbf{A})^k \mathbf{P}`,
                where :math:`\alpha` is given by ``scale``. This preconditioned
                formulation is inspired by Wang et al. (NeurIPS 2025). ``preconditioner``
                should have the same in-/out-shapes as ``A``. Default: ``None``.

        Raises:
            ValueError: If ``preconditioner`` is provided with incompatible shapes.
        """
        super().__init__(A)
        self._num_terms = num_terms
        self._scale = scale
        self._check_nan = check_nan
        self._preconditioner = preconditioner
        if preconditioner is not None:
            _check_same_tensor_list_shape(A, preconditioner)

    def _matmat(self, X: list[Tensor]) -> list[Tensor]:
        """Multiply the inverse of A onto a matrix in list format.

        Args:
             X: Matrix for multiplication in list format.

        Returns:
             Result of inverse matrix-matrix multiplication, ``A⁻¹ @ X``.

        Raises:
            ValueError: If ``NaN`` check is turned on and ``NaN`` values are detected.
        """
        preconditioned = self._preconditioner is not None
        if not preconditioned:
            rhs_list = X
            apply_iteration_operator = self._A._matmat
        else:
            rhs_list = self._preconditioner @ X

            # Use the public matmul interface for preconditioned updates so
            # tensor/list pre-/post-processing stays consistent across all
            # preconditioner operator types.
            def apply_iteration_operator(v_list: list[Tensor]) -> list[Tensor]:
                return self._preconditioner @ (self._A @ v_list)

        result_list = [x.clone() for x in rhs_list]
        v_list = [x.clone() for x in rhs_list]

        for idx in range(self._num_terms):
            A_v_list = apply_iteration_operator(v_list)
            v_list = [
                v.sub_(A_v, alpha=self._scale) for v, A_v in zip(v_list, A_v_list)
            ]
            result_list = [result.add_(v) for result, v in zip(result_list, v_list)]

            if self._check_nan and any(isnan(v).any() for v in v_list):
                raise ValueError(
                    f"Detected NaNs after application of {idx}-th term."
                    + " This is probably because the Neumann series is non-convergent."
                    + " Try using a better preconditioner or fewer terms."
                )

        return [result.mul_(self._scale) for result in result_list]

    def _adjoint(self) -> NeumannInverseLinearOperator:
        """Return the linear operator's adjoint: (A^-1)* = (A*)^-1.

        Returns:
            A linear operator representing the adjoint.
        """
        preconditioner = (
            None if self._preconditioner is None else self._preconditioner._adjoint()
        )
        return NeumannInverseLinearOperator(
            self._A._adjoint(),
            num_terms=self._num_terms,
            scale=self._scale,
            check_nan=self._check_nan,
            preconditioner=preconditioner,
        )
