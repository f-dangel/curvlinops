"""Utility linear operators."""

from __future__ import annotations

from typing import List, Tuple

from torch import Tensor, device, dtype, einsum

from curvlinops._torch_base import PyTorchLinearOperator


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

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Apply the linear operator to a matrix in list format.

        Args:
            X: The matrix to multiply onto in list format.

        Returns:
            The result of the multiplication in list format.
        """
        # Compute ∑ᵢ cᵢ aᵢ aᵢᵀ @ X
        return [einsum("ik,k,jk,jl->il", self._A, self._c, self._A, X[0])]

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
