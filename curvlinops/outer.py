"""Utility linear operators."""

from __future__ import annotations

from numpy import einsum, einsum_path, ndarray, ones
from scipy.sparse.linalg import LinearOperator


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
