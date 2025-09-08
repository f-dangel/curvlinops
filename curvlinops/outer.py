"""Utility linear operators."""

from __future__ import annotations

from typing import List, Tuple

from numpy import einsum, einsum_path, ndarray, ones
from scipy.sparse.linalg import LinearOperator
from torch import Tensor, device, dtype

from curvlinops._torch_base import PyTorchLinearOperator


class OuterProductLinearOperator(LinearOperator):
    """Linear operator for low-rank matrices of the form ``∑ᵢ cᵢ aᵢ aᵢᵀ``.

    ``cᵢ`` is the coefficient for the vector ``aᵢ``.
    """

    def __init__(self, c: ndarray, A: ndarray):
        """Store coefficients and vectors for low-rank representation.

        Args:
            c: Coefficients ``cᵢ``. Has shape ``[K]`` where ``K`` is the rank.
            A: Matrix of shape ``[D, K]``, where ``D`` is the linear operators
                dimension, that stores the low-rank vectors columnwise, i.e. ``aᵢ``
                is stored in ``A[:,i]``.
        """
        super().__init__(A.dtype, (A.shape[0], A.shape[0]))
        self._A = A
        self._c = c

        # optimize einsum
        self._equation = "ij,j,kj,k->i"
        self._operands = (self._A, self._c, self._A)
        placeholder = ones(self.shape[0])
        self._path = einsum_path(
            self._equation, *self._operands, placeholder, optimize="optimal"
        )[0]

    def _matvec(self, x: ndarray) -> ndarray:
        """Apply the linear operator to a vector.

        Args:
            x: Vector.

        Returns:
            Result of linear operator applied to the vector.
        """
        return einsum(self._equation, *self._operands, x, optimize=self._path)

    def _adjoint(self) -> OuterProductLinearOperator:
        """Return the linear operator representing the adjoint.

        An outer product is self-adjoint.

        Returns:
            Self.
        """
        return self


class Projector(OuterProductLinearOperator):
    """Linear operator for the projector onto the orthonormal basis ``{ aᵢ }``."""

    def __init__(self, A: ndarray):
        """Store orthonormal basis.

        Args:
            A: Matrix of shape ``[D, K]``, where ``D`` is the linear operators
                dimension, that stores the K orthonormal basis vectors columnwise,
                i.e. ``aᵢ`` is stored in ``A[:,i]``.
        """
        super().__init__(ones(A.shape[1]), A)


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

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Apply the linear operator to a matrix in list format.

        Args:
            X: The matrix to multiply onto in list format.

        Returns:
            The result of the matrix multiplication in list format.
        """
        return X

    @property
    def dtype(self) -> dtype:
        """Return the linear operator's device.

        Returns:
            The device on which the linear operator is defined.
        """
        return self._dtype

    @property
    def device(self) -> device:
        """Return the data type of the linear operator.

        Returns:
            The data type of the linear operator.
        """
        return self._device
