"""Implements slices of linear operators."""

from __future__ import annotations

from typing import List

from numpy import column_stack, ndarray, zeros
from scipy.sparse.linalg import LinearOperator


class SubmatrixLinearOperator(LinearOperator):
    """Class for sub-matrices of linear operators."""

    def __init__(self, A: LinearOperator, row_idxs: List[int], col_idxs: List[int]):
        """Store the linear operator and indices of its sub-matrix.

        Represents the sub-matrix ``A[row_idxs, :][col_idxs, :]``.

        Args:
            A: A linear operator.
            row_idxs: The sub-matrix's row indices.
            col_idxs: The sub-matrix's column indices.
        """
        self._A = A
        self.set_submatrix(row_idxs, col_idxs)

    def set_submatrix(self, row_idxs: List[int], col_idxs: List[int]):
        """Define the sub-matrix.

        Internally sets the linear operator's shape.

        Args:
            row_idxs: The sub-matrix's row indices.
            col_idxs: The sub-matrix's column indices.

        Raises:
            ValueError: If the index lists contain duplicate values, non-integers,
                or out-of-bounds indices.
        """
        shape = []

        for ax_idx, idxs in enumerate([row_idxs, col_idxs]):
            if any(not isinstance(i, int) for i in idxs):
                raise ValueError("Index lists must contain integers.")
            if len(idxs) != len(set(idxs)):
                raise ValueError("Index lists cannot contain duplicates.")
            if any(i < 0 or i >= self._A.shape[ax_idx] for i in idxs):
                raise ValueError("Index lists contain out-of-bounds indices.")
            shape.append(len(idxs))

        super().__init__(self._A.dtype, shape)
        self._row_idxs = row_idxs
        self._col_idxs = col_idxs

    def _matvec(self, x: ndarray) -> ndarray:
        """Multiply x by the sub-matrix of A.

        Args:
             x: Vector for multiplication. Has shape ``[len(col_idxs)]``.

        Returns:
             Result of the (sub-matrix)-vector-multiplication,
             ``A[row_idxs, :][:, col_idxs] @ x``. Has shape ``[len(row_idxs)]``.
        """
        v = zeros((self._A.shape[1],), dtype=self._A.dtype)
        v[self._col_idxs] = x
        Av = self._A @ v

        return Av[self._row_idxs]

    def _matmat(self, X: ndarray) -> ndarray:
        """Multiply each column of X by the sub-matrix of A.

        Args:
            X: Matrix for multiplication. Has shape ``[len(col_idxs), N]`` with
                abitrary ``N``.

        Returns:
            Result of the (sub-matrix)-matrix-multiplication,
            ``A[row_idxs, :][:, col_idxs] @ x``. Has shape ``[len(row_idxs), N]``.
        """
        return column_stack([self @ col for col in X.T])

    def _adjoint(self) -> SubmatrixLinearOperator:
        """Return the adjoint of the sub-matrix.

        For that, we need to take the adjoint operator, and swap row and column indices.

        Returns:
            The linear operator for the adjoint sub-matrix.
        """
        return type(self)(self._A.adjoint(), self._col_idxs, self._row_idxs)
