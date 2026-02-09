"""Linear operator implementation for diagonal matrices."""

from __future__ import annotations

from typing import List, Union

from torch import Tensor, device, dtype

from curvlinops._torch_base import PyTorchLinearOperator, _SumPyTorchLinearOperator


class DiagonalLinearOperator(PyTorchLinearOperator):
    """Linear operator representing a diagonal matrix."""

    def __init__(self, diagonal: List[Tensor]):
        """Initialize diagonal linear operator.

        Args:
        """
        shapes = [tuple(d.shape) for d in diagonal]
        super().__init__(shapes, shapes)
        self._diagonal = diagonal
        self.SELF_ADJOINT = all(d.conj().allclose(d) for d in diagonal)

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication with diagonal matrix."""
        return [d.unsqueeze(-1) * x for d, x in zip(self._diagonal, X)]

    def adjoint(self) -> DiagonalLinearOperator:
        return DiagonalLinearOperator([d.conj() for d in self._diagonal])

    @property
    def device(self) -> device:
        """Infer the linear operator's device.

        Returns:
            The linear operator's device.
        """
        devices = {d.device for d in self._diagonal}
        if len(devices) != 1:
            raise ValueError(f"Inconsistent devices detected: {devices}.")
        return devices.pop()

    @property
    def dtype(self) -> dtype:
        """Infer the linear operator's data type.

        Returns:
            The linear operator's data type.
        """
        dtypes = {d.dtype for d in self._diagonal}
        if len(dtypes) != 1:
            raise ValueError(f"Inconsistent dtypes detected: {dtypes}.")
        return dtypes.pop()

    def inverse(self, damping: float) -> DiagonalLinearOperator:
        """Return the inverse of the damped linear operator."""
        return DiagonalLinearOperator([1.0 / (d + damping) for d in self._diagonal])

    def __add__(
        self, other: PyTorchLinearOperator
    ) -> Union[_SumPyTorchLinearOperator, DiagonalLinearOperator]:
        if isinstance(other, DiagonalLinearOperator):
            if self._in_shape != other._in_shape:
                raise ValueError(
                    f"Diagonal shapes differ: {self._in_shape} vs {other._in_shape}."
                )
            return DiagonalLinearOperator([
                d1 + d2 for d1, d2 in zip(self._diagonal, other._diagonal)
            ])
        return super().__add__(other)

    def __mul__(self, scalar: int | float) -> DiagonalLinearOperator:
        return DiagonalLinearOperator([d * scalar for d in self._diagonal])
