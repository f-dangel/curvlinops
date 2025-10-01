"""Implements slices of linear operators."""

from __future__ import annotations

from typing import List

from torch import Tensor, device, dtype, zeros

from curvlinops._torch_base import PyTorchLinearOperator


class SubmatrixLinearOperator(PyTorchLinearOperator):
    """Class for sub-matrices of linear operators."""

    def __init__(
        self, A: PyTorchLinearOperator, row_idxs: List[int], col_idxs: List[int]
    ):
        """Store the linear operator and indices of its sub-matrix.

        Represents the sub-matrix ``A[row_idxs, :][col_idxs, :]``.

        Args:
            A: A linear operator.
            row_idxs: The sub-matrix's row indices.
            col_idxs: The sub-matrix's column indices.
        """
        self._A = A
        self.set_submatrix(row_idxs, col_idxs)

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

        in_shape, out_shape = [(shape[1],)], [(shape[0],)]
        super().__init__(in_shape, out_shape)
        self._row_idxs = row_idxs
        self._col_idxs = col_idxs

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication.

        Args:
            X: A list that contains a single tensor, which is the input tensor.

        Returns:
            A list that contains a single tensor, which is the output tensor.
        """
        (M,) = X
        V = zeros(self._A.shape[1], M.shape[-1], dtype=self.dtype, device=self.device)
        V[self._col_idxs] = M
        AV = self._A @ V
        return [AV[self._row_idxs]]

    def _adjoint(self) -> SubmatrixLinearOperator:
        """Return the adjoint of the sub-matrix.

        For that, we need to take the adjoint operator, and swap row and column indices.

        Returns:
            The linear operator for the adjoint sub-matrix.
        """
        return type(self)(self._A.adjoint(), self._col_idxs, self._row_idxs)
