"""Implements a linear operator for block-diagonal matrices."""

from __future__ import annotations

from collections.abc import Iterator

from torch import Tensor, device, dtype, stack

from curvlinops._checks import (
    _check_same_device,
    _check_same_dtype,
    _check_same_tensor_list_shape,
)
from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.kronecker import ensure_all_square
from curvlinops.utils import _infer_device, _infer_dtype, split_list


class BlockDiagonalLinearOperator(PyTorchLinearOperator):
    """Linear operator for block-diagonal matrices with PyTorch linear operator blocks.

    The blocks are arranged diagonally:
    [B1  0  0 ]
    [ 0 B2  0 ]
    [ 0  0 B3 ]

    where each Bi is itself a PyTorch linear operator.
    """

    def __init__(self, blocks: list[PyTorchLinearOperator]):
        """Initialize the block-diagonal linear operator.

        Args:
            blocks: List of PyTorch linear operators forming the diagonal blocks.

        Raises:
            ValueError: If no blocks are provided.
        """
        if not blocks:
            raise ValueError("At least one block must be provided.")

        self._blocks = blocks

        # Construct input and output shapes from blocks
        in_shape = [
            tuple(s) for s in sum([B._in_shape for B in self._blocks], start=[])
        ]
        out_shape = [
            tuple(s) for s in sum([B._out_shape for B in self._blocks], start=[])
        ]
        super().__init__(in_shape, out_shape)

        # Block diagonal is self-adjoint if all blocks are self-adjoint
        self.SELF_ADJOINT = all(B.SELF_ADJOINT for B in blocks)

    def __iter__(self) -> Iterator[PyTorchLinearOperator]:
        """Iterate over the diagonal blocks.

        Returns:
            Iterator over the block linear operators.
        """
        return iter(self._blocks)

    def __len__(self) -> int:
        """Return the number of diagonal blocks.

        Returns:
            The number of blocks.
        """
        return len(self._blocks)

    def __getitem__(self, index: int) -> PyTorchLinearOperator:
        """Get a block by index.

        Args:
            index: Index of the block.

        Returns:
            The block at the given index.
        """
        return self._blocks[index]

    def __setitem__(self, index: int, value: PyTorchLinearOperator):
        """Replace a block by index.

        The replacement must have the same shape, device, and dtype as
        the block it replaces.

        Args:
            index: Index of the block to replace.
            value: The new block.
        """
        old = self._blocks[index]
        _check_same_tensor_list_shape(old, value)
        _check_same_device(old, value)
        _check_same_dtype(old, value)
        self._blocks[index] = value

    def _matmat(self, X: list[Tensor]) -> list[Tensor]:
        """Matrix-matrix multiplication with block-diagonal structure.

        Args:
            X: Input matrix in tensor list format.

        Returns:
            Result of block-diagonal matrix multiplication.
        """
        # split tensor list into per-block lists
        X_blocks = split_list(X, [len(B._in_shape) for B in self._blocks])
        # Multiply per-block and concatenate the resulting lists into the result
        return sum([B @ X_B for B, X_B in zip(self._blocks, X_blocks)], start=[])

    def _adjoint(self) -> BlockDiagonalLinearOperator:
        """Return the adjoint of the block-diagonal linear operator.

        The adjoint of a block-diagonal matrix is block-diagonal with adjoint blocks.

        Returns:
            Block-diagonal linear operator with adjoint blocks.
        """
        adjoint_blocks = [block.adjoint() for block in self._blocks]
        return BlockDiagonalLinearOperator(adjoint_blocks)

    @property
    def device(self) -> device:
        """Get the device of the linear operator.

        Returns:
            Device of the blocks.
        """
        return _infer_device(self._blocks)

    @property
    def dtype(self) -> dtype:
        """Get the dtype of the linear operator.

        Returns:
            Data type of the blocks.
        """
        return _infer_dtype(self._blocks)

    def trace(self) -> Tensor:
        """Trace of the block-diagonal matrix.

        For a block-diagonal matrix, tr(block_diag(B1, B2, ..., Bk)) = ∑ tr(Bi).
        Only works if all blocks are square.

        Returns:
            Trace of the block-diagonal matrix.
        """
        ensure_all_square(*self._blocks)
        return stack([B.trace() for B in self._blocks]).sum()

    def det(self) -> Tensor:
        """Compute the determinant of the block-diagonal matrix.

        For a block-diagonal matrix, det(block_diag(B1, B2, ..., Bk)) = ∏ det(Bi).
        Only works if all blocks are square.

        Returns:
            Determinant of the block-diagonal matrix.
        """
        ensure_all_square(*self._blocks)
        return stack([B.det() for B in self._blocks]).prod()

    def logdet(self) -> Tensor:
        """Log determinant of the block-diagonal matrix.

        For a block-diagonal matrix, logdet(block_diag(B1, B2, ..., Bk)) = ∑ logdet(Bi).
        More numerically stable than det property. Only works if all blocks are square.

        Returns:
            Log determinant of the block-diagonal matrix.
        """
        ensure_all_square(*self._blocks)
        return stack([B.logdet() for B in self._blocks]).sum()

    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the block-diagonal matrix.

        For a block-diagonal matrix,
        ||block_diag(B1, B2, ..., Bk)||_F = sqrt(∑ ||Bi||_F^2).
        Works for any rectangular blocks.

        Returns:
            Frobenius norm of the block-diagonal matrix.
        """
        return stack([B.frobenius_norm() ** 2 for B in self._blocks]).sum().sqrt()
