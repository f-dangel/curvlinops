"""Linear operator implementation for diagonal matrices."""

from __future__ import annotations

from typing import List, Union

from torch import Tensor, device, dtype

from curvlinops._torch_base import PyTorchLinearOperator, _SumPyTorchLinearOperator
from curvlinops.utils import _infer_device, _infer_dtype


class DiagonalLinearOperator(PyTorchLinearOperator):
    """Linear operator representing a diagonal matrix."""

    def __init__(self, diagonal: List[Tensor]):
        """Initialize diagonal linear operator.

        The diagonal entries are specified in tensor list format, where each tensor
        corresponds to the diagonal block for that parameter. Both the input and output
        shapes are inferred from the shapes of the diagonal tensors. The operator is
        automatically marked as self-adjoint if all diagonal entries are real-valued
        (i.e. equal to their complex conjugate).

        Args:
            diagonal: List of tensors representing the diagonal entries. Each tensor's
                shape defines both the input and output shape for that block of the
                operator.
        """
        shapes = [tuple(d.shape) for d in diagonal]
        super().__init__(shapes, shapes)
        self._diagonal = diagonal
        self.SELF_ADJOINT = all(
            not d.is_complex() or d.conj().allclose(d) for d in diagonal
        )

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication with the diagonal matrix.

        Multiplies each column of ``X`` element-wise by the diagonal entries.

        Args:
            X: A list of tensors representing the matrix to multiply onto. Each
                tensor has shape ``[*Ni, K]``, where ``Ni`` is the shape of the
                corresponding diagonal tensor and ``K`` is the number of columns.

        Returns:
            A list of tensors with the same shapes as ``X``, where each tensor is the
            element-wise product of the diagonal and the corresponding input tensor.
        """
        return [d.unsqueeze(-1) * x for d, x in zip(self._diagonal, X)]

    def adjoint(self) -> DiagonalLinearOperator:
        """Return the adjoint (conjugate transpose) of the diagonal operator.

        For a diagonal matrix, the adjoint is obtained by taking the complex
        conjugate of each diagonal entry.

        Returns:
            A new ``DiagonalLinearOperator`` whose diagonal entries are the complex
            conjugates of this operator's diagonal entries.
        """
        return DiagonalLinearOperator([d.conj() for d in self._diagonal])

    @property
    def device(self) -> device:
        """Infer the linear operator's device.

        Returns:
            The linear operator's device.
        """
        return _infer_device(self._diagonal)

    @property
    def dtype(self) -> dtype:
        """Infer the linear operator's data type.

        Returns:
            The linear operator's data type.
        """
        return _infer_dtype(self._diagonal)

    def inverse(self, damping: float) -> DiagonalLinearOperator:
        """Return the inverse of the damped diagonal operator ``(D + damping * I)^{-1}``.

        Computes the element-wise inverse ``1 / (d + damping)`` for each diagonal
        entry ``d``.

        Args:
            damping: Non-negative scalar added to each diagonal entry before inverting.
                A positive value ensures the operator is invertible even if the original
                diagonal contains zeros.

        Returns:
            A new ``DiagonalLinearOperator`` representing the inverse of the damped
            operator.
        """
        return DiagonalLinearOperator([1.0 / (d + damping) for d in self._diagonal])

    def __add__(
        self, other: PyTorchLinearOperator
    ) -> Union[_SumPyTorchLinearOperator, DiagonalLinearOperator]:
        """Add another linear operator to this diagonal operator.

        If ``other`` is also a ``DiagonalLinearOperator`` with matching input shapes,
        the addition is performed element-wise on the diagonals, returning a new
        ``DiagonalLinearOperator``. Otherwise, falls back to the generic sum operator
        from the parent class.

        Args:
            other: Another ``PyTorchLinearOperator`` to add.

        Returns:
            A ``DiagonalLinearOperator`` if both operands are diagonal with matching
            shapes, otherwise a ``_SumPyTorchLinearOperator``.
        """
        if (
            isinstance(other, DiagonalLinearOperator)
            and self._in_shape == other._in_shape
        ):
            return DiagonalLinearOperator([
                d1 + d2 for d1, d2 in zip(self._diagonal, other._diagonal)
            ])
        else:
            return super().__add__(other)

    def __matmul__(
        self, other: Union[List[Tensor], Tensor, PyTorchLinearOperator]
    ) -> Union[List[Tensor], Tensor, PyTorchLinearOperator]:
        """Multiply this operator with a vector, matrix, or another operator.

        If ``other`` is a ``DiagonalLinearOperator`` with matching shapes, the
        product is computed element-wise on the diagonals, returning a new
        ``DiagonalLinearOperator``. Otherwise, falls back to the parent class.

        Args:
            other: A vector/matrix (as tensor or tensor list) or another linear
                operator.

        Returns:
            A ``DiagonalLinearOperator`` if both operands are diagonal with matching
            shapes, otherwise the result from the parent class.
        """
        if (
            isinstance(other, DiagonalLinearOperator)
            and self._in_shape == other._in_shape
        ):
            return DiagonalLinearOperator([
                d1 * d2 for d1, d2 in zip(self._diagonal, other._diagonal)
            ])
        return super().__matmul__(other)

    def __mul__(self, scalar: int | float) -> DiagonalLinearOperator:
        """Multiply the diagonal operator by a scalar.

        Scales each diagonal entry by the given scalar, returning a new
        ``DiagonalLinearOperator`` rather than a generic scaled operator.

        Args:
            scalar: The scalar factor to multiply with.

        Returns:
            A new ``DiagonalLinearOperator`` with scaled diagonal entries.
        """
        return DiagonalLinearOperator([d * scalar for d in self._diagonal])
