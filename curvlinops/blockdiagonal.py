"""Implements a linear operator for block-diagonal matrices."""

from typing import List

from torch import Tensor, device, dtype, stack

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.kronecker import ensure_all_square


class BlockDiagonalLinearOperator(PyTorchLinearOperator):
    """Linear operator for block-diagonal matrices with PyTorch linear operator blocks.

    The blocks are arranged diagonally:
    [B1  0  0 ]
    [ 0 B2  0 ]
    [ 0  0 B3 ]

    where each Bi is itself a PyTorch linear operator.
    """

    def __init__(self, blocks: List[PyTorchLinearOperator]):
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

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication with block-diagonal structure.

        Args:
            X: Input matrix in tensor list format.

        Returns:
            Result of block-diagonal matrix multiplication.
        """
        # split tensor list into per-block lists
        X_blocks = []
        start = 0
        for B in self._blocks:
            num_tensors = len(B._in_shape)
            X_blocks.append(X[start : start + num_tensors])
            start += num_tensors

        # Multiply per-block and concatenate the resulting lists into the result
        return sum([B @ X_B for B, X_B in zip(self._blocks, X_blocks)], start=[])

    def _adjoint(self) -> "BlockDiagonalLinearOperator":
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

        Raises:
            RuntimeError: If blocks have inconsistent devices.
        """
        devices = {block.device for block in self._blocks}
        if len(devices) > 1:
            raise RuntimeError(f"Blocks have inconsistent devices: {devices}")
        return devices.pop()

    @property
    def dtype(self) -> dtype:
        """Get the dtype of the linear operator.

        Returns:
            Data type of the blocks.

        Raises:
            RuntimeError: If blocks have inconsistent dtypes.
        """
        dtypes = {block.dtype for block in self._blocks}
        if len(dtypes) > 1:
            raise RuntimeError(f"Blocks have inconsistent dtypes: {dtypes}")
        return dtypes.pop()

    # NOTE Keep in mind that some properties only work if blocks are all square.

    @property
    def trace(self) -> Tensor:
        """Trace of the block-diagonal matrix.

        For a block-diagonal matrix, tr(block_diag(B1, B2, ..., Bk)) = ∑ tr(Bi).
        Only works if all blocks are square.

        Returns:
            Trace of the block-diagonal matrix.

        Raises:
            RuntimeError: If any block is not square.
        """
        ensure_all_square(*self._blocks)
        return stack([B.trace for B in self._blocks]).sum()

    @property
    def det(self) -> Tensor:
        """Determinant of the block-diagonal matrix.

        For a block-diagonal matrix, det(block_diag(B1, B2, ..., Bk)) = ∏ det(Bi).
        Only works if all blocks are square.

        Returns:
            Determinant of the block-diagonal matrix.

        Raises:
            RuntimeError: If any block is not square.
        """
        ensure_all_square(*self._blocks)
        return stack([B.det for B in self._blocks]).prod()

    @property
    def logdet(self) -> Tensor:
        """Log determinant of the block-diagonal matrix.

        For a block-diagonal matrix, logdet(block_diag(B1, B2, ..., Bk)) = ∑ logdet(Bi).
        More numerically stable than det property. Only works if all blocks are square.

        Returns:
            Log determinant of the block-diagonal matrix.

        Raises:
            RuntimeError: If any block is not square.
        """
        ensure_all_square(*self._blocks)
        return stack([B.logdet for B in self._blocks]).sum()

    @property
    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the block-diagonal matrix.

        For a block-diagonal matrix,
        ||block_diag(B1, B2, ..., Bk)||_F = sqrt(∑ ||Bi||_F^2).
        Works for any rectangular blocks.

        Returns:
            Frobenius norm of the block-diagonal matrix.
        """
        return stack([B.frobenius_norm**2 for B in self._blocks]).sum().sqrt()
