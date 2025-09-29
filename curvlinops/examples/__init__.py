"""Contains functionality for examples in the documentation."""

from __future__ import annotations

from typing import List

from torch import Tensor, device, dtype

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

    def _infer_device(self) -> device:
        return self._A.device

    def _infer_dtype(self) -> dtype:
        return self._A.dtype

    def _adjoint(self) -> TensorLinearOperator:
        return TensorLinearOperator(self._A.conj().T)

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        return [self._A @ X[0]]
