"""Contains functionality for examples in the documentation."""

from __future__ import annotations

from typing import List, Tuple

from torch import Tensor, device, dtype, einsum, tensor

from curvlinops._torch_base import PyTorchLinearOperator


class TensorLinearOperator(PyTorchLinearOperator):
    """Linear operator wrapping a single tensor as a linear operator."""

    def __init__(self, A: Tensor):
        """Initialize linear operator from a 2D tensor.

        Args:
            A: A 2D tensor representing the matrix.

        Raises:
            ValueError: If ``A`` is not a 2D tensor.
        """
        if A.ndim != 2:
            raise ValueError(f"Input tensor must be 2D. Got {A.ndim}D.")
        super().__init__([(A.shape[1],)], [(A.shape[0],)])
        self._A = A
        self.SELF_ADJOINT = A.shape == A.T.shape and A.allclose(A.T)

    @property
    def device(self) -> device:
        """Infer the linear operator's device.

        Returns:
            The linear operator's device.
        """
        return self._A.device

    @property
    def dtype(self) -> dtype:
        """Infer the linear operator's data type.

        Returns:
            The linear operator's data type.
        """
        return self._A.dtype

    def _adjoint(self) -> TensorLinearOperator:
        """Return a linear operator representing the adjoint.

        Returns:
            The adjoint linear operator.
        """
        return TensorLinearOperator(self._A.conj().T)

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Multiply the linear operator onto a matrix in list format.

        Args:
            M: Matrix for multiplication in list format.

        Returns:
            Result of the matrix-matrix multiplication in list format.
        """
        (M0,) = M
        return [self._A @ M0]

    @property
    def trace(self) -> Tensor:
        """Trace of the matrix.

        Returns:
            Trace of the underlying tensor matrix.
        """
        return self._A.trace()

    @property
    def det(self) -> Tensor:
        """Determinant of the matrix.

        Returns:
            Determinant of the underlying tensor matrix.
        """
        return self._A.det()

    @property
    def logdet(self) -> Tensor:
        """Log determinant of the matrix.

        Returns:
            Log determinant of the underlying tensor matrix.
        """
        return self._A.logdet()

    @property
    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the matrix.

        Returns:
            Frobenius norm of the underlying tensor matrix.
        """
        return self._A.norm(p="fro")


class OuterProductLinearOperator(PyTorchLinearOperator):
    """Linear operator for low-rank matrices of the form ``∑ᵢ cᵢ aᵢ aᵢᵀ``.

    ``cᵢ`` is the coefficient for the vector ``aᵢ``.
    """

    SELF_ADJOINT = True

    def __init__(self, c: Tensor, A: Tensor):
        """Store coefficients and vectors for low-rank representation.

        Args:
            c: Coefficients ``cᵢ``. Has shape ``[K]`` where ``K`` is the rank.
            A: Matrix of shape ``[D, K]``, where ``D`` is the linear operators
                dimension, that stores the low-rank vectors columnwise, i.e. ``aᵢ``
                is stored in ``A[:,i]``.
        """
        shape = [(A.shape[0],)]
        super().__init__(shape, shape)
        self._A = A
        self._c = c

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Apply the linear operator to a matrix in list format.

        Args:
            M: The matrix to multiply onto in list format.

        Returns:
            The result of the multiplication in list format.
        """
        (M0,) = M
        # Compute ∑ᵢ cᵢ aᵢ aᵢᵀ @ X
        return [einsum("ik,k,jk,jl->il", self._A, self._c, self._A, M0)]

    def _adjoint(self) -> OuterProductLinearOperator:
        """Return the linear operator representing the adjoint.

        An outer product is self-adjoint.

        Returns:
            Self.
        """
        return self

    @property
    def dtype(self) -> dtype:
        """Return the data type of the linear operator.

        Returns:
            The data type of the linear operator.
        """
        return self._A.dtype

    @property
    def device(self) -> device:
        """Return the linear operator's device.

        Returns:
            The device on which the linear operator is defined.
        """
        return self._A.device

    @property
    def trace(self) -> Tensor:
        """Trace of the low-rank matrix ∑ᵢ cᵢ aᵢ aᵢᵀ.

        For outer product matrices, tr(∑ᵢ cᵢ aᵢ aᵢᵀ) = ∑ᵢ cᵢ (aᵢᵀ aᵢ).

        Returns:
            Trace of the low-rank matrix.
        """
        return (self._c * (self._A**2).sum(dim=0)).sum()

    @property
    def det(self) -> Tensor:
        """Determinant of the low-rank matrix ∑ᵢ cᵢ aᵢ aᵢᵀ.

        Note: Computing the determinant of a low-rank matrix requires
        constructing the full matrix, which may be computationally expensive
        for large dimensions.

        Returns:
            Determinant of the low-rank matrix.
        """
        full_matrix = einsum("ik,k,jk->ij", self._A, self._c, self._A)
        return full_matrix.det()

    @property
    def logdet(self) -> Tensor:
        """Log determinant of the low-rank matrix ∑ᵢ cᵢ aᵢ aᵢᵀ.

        Note: Computing the log determinant of a low-rank matrix requires
        constructing the full matrix, which may be computationally expensive
        for large dimensions.

        Returns:
            Log determinant of the low-rank matrix.
        """
        full_matrix = einsum("ik,k,jk->ij", self._A, self._c, self._A)
        return full_matrix.logdet()

    @property
    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the low-rank matrix ∑ᵢ cᵢ aᵢ aᵢᵀ.

        For outer product matrices, ||∑ᵢ cᵢ aᵢ aᵢᵀ||_F² = ∑ᵢⱼ cᵢ cⱼ (aᵢᵀ aⱼ)².

        Returns:
            Frobenius norm of the low-rank matrix.
        """
        inner_products = self._A.T @ self._A
        weighted = self._c.unsqueeze(0) * inner_products * self._c.unsqueeze(1)
        frobenius_norm_squared = (inner_products * weighted).sum()
        return frobenius_norm_squared.sqrt()


class IdentityLinearOperator(PyTorchLinearOperator):
    """Linear operator representing the identity matrix."""

    SELF_ADJOINT = True

    def __init__(self, shape: List[Tuple[int, ...]], device: device, dtype: dtype):
        """Store the linear operator's input and output space dimensions.

        Args:
            shape: A list of shapes specifying the identity's input and output space.
            device: The device on which the identity operator is defined.
            dtype: The data type of the identity operator.
        """
        super().__init__(shape, shape)
        self._device = device
        self._dtype = dtype

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Apply the linear operator to a matrix in list format.

        Args:
            M: The matrix to multiply onto in list format.

        Returns:
            The result of the matrix multiplication in list format.
        """
        return M

    @property
    def dtype(self) -> dtype:
        """Return the data type of the linear operator.

        Returns:
            The data type of the linear operator.
        """
        return self._dtype

    @property
    def device(self) -> device:
        """Return the linear operator's device.

        Returns:
            The device on which the linear operator is defined.
        """
        return self._device

    @property
    def trace(self) -> Tensor:
        """Trace of the identity matrix.

        For an identity matrix, tr(I) = dimension.

        Returns:
            Trace of the identity matrix.
        """
        total_dim = sum(self._in_shape_flat)
        return tensor(float(total_dim), device=self.device, dtype=self.dtype)

    @property
    def det(self) -> Tensor:
        """Determinant of the identity matrix.

        For an identity matrix, det(I) = 1.

        Returns:
            Determinant of the identity matrix.
        """
        return tensor(1.0, device=self.device, dtype=self.dtype)

    @property
    def logdet(self) -> Tensor:
        """Log determinant of the identity matrix.

        For an identity matrix, logdet(I) = log(1) = 0.

        Returns:
            Log determinant of the identity matrix.
        """
        return tensor(0.0, device=self.device, dtype=self.dtype)

    @property
    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the identity matrix.

        For an identity matrix, ||I||_F = sqrt(dimension).

        Returns:
            Frobenius norm of the identity matrix.
        """

        total_dim = sum(self._in_shape_flat)
        return tensor(float(total_dim), device=self.device, dtype=self.dtype).sqrt()
