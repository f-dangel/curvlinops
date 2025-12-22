"""PyTorch linear operator implementation of Kronecker product S_1 ⊗ S_2 ⊗ ... ."""

from math import prod
from typing import List

from einops import einsum
from torch import Tensor, device, dtype, stack
from torch.linalg import matrix_norm

from curvlinops._torch_base import PyTorchLinearOperator


class KroneckerProductLinearOperator(PyTorchLinearOperator):
    """Linear operator representing a Kronecker product S_1 ⊗ S_2 ⊗ ... ⊗ S_k.

    The Kronecker product of matrices S_1 (m_1 × n_1), S_2 (m_2 × n_2), ...,
    S_k (m_k × n_k) results in a matrix of shape (∏m_i) × (∏n_i).
    """

    def __init__(self, *factors: Tensor):
        """Initialize Kronecker product linear operator.

        Args:
            *factors: Variable number of 2D tensors representing the Kronecker
                factors S_1, S_2, ..., S_k. Each factor should have shape (m_i, n_i).

        Raises:
            ValueError: If any factor is not a 2D tensor.
            ValueError: If no factors are provided.
        """
        if len(factors) == 0:
            raise ValueError("At least one factor must be provided.")

        for i, factor in enumerate(factors):
            if factor.ndim != 2:
                raise ValueError(
                    f"Factor {i} must be a 2D tensor, got shape {factor.shape}."
                )

        # Store the Kronecker factors and compute the product's dimensions
        self._factors = list(factors)
        D_in = prod(S.shape[1] for S in self._factors)
        D_out = prod(S.shape[0] for S in self._factors)
        in_shapes, out_shapes = [(D_in,)], [(D_out,)]

        # Assemble the einsum equation for matrix multiplication
        num_dims = len(self._factors)
        S_strs = [f"out{i} in{i}" for i in range(num_dims)]
        x_str = " ".join([f"in{i}" for i in range(num_dims)]) + " k"
        result_str = " ".join([f"out{i}" for i in range(num_dims)]) + " k"
        # For two Kronecker factors: 'in0 in1, out0 in0, out1 in1 -> out0 out1'
        equation = f"{x_str}, {','.join(S_strs)} -> {result_str}"
        self._einsum_equation = equation

        super().__init__(in_shapes, out_shapes)

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Apply Kronecker product to matrix in tensor list format.

        Args:
            X: List with a single tensor of shape (n_1 * n_2 * ... * n_k , K) where
                K is the number of columns.

        Returns:
            List with a single tensor of shape (m_1 * m_2 * ... * m_k, K).
        """
        (x,) = X
        x = x.reshape(*[S.shape[1] for S in self._factors], x.shape[-1])
        return [einsum(x, *self._factors, self._einsum_equation).flatten(end_dim=-2)]

    def _adjoint(self) -> "KroneckerProductLinearOperator":
        """Return the adjoint of the Kronecker product.

        The adjoint of S_1 ⊗ S_2 ⊗ ... ⊗ S_k is S_1^H ⊗ S_2^H ⊗ ... ⊗ S_k^H.

        Returns:
            New KroneckerProductLinearOperator representing the adjoint.
        """
        adjoint_factors = [factor.T for factor in self._factors]
        return KroneckerProductLinearOperator(*adjoint_factors)

    @property
    def device(self) -> device:
        """Return the device of the Kronecker factors.

        Returns:
            Device of the factors.

        Raises:
            RuntimeError: If factors are on different devices.
        """
        devices = {factor.device for factor in self._factors}
        if len(devices) != 1:
            raise RuntimeError(f"Factors are on different devices: {devices}")
        return devices.pop()

    @property
    def dtype(self) -> dtype:
        """Return the data type of the Kronecker factors.

        Returns:
            Data type of the factors.

        Raises:
            RuntimeError: If factors have different data types.
        """
        dtypes = {factor.dtype for factor in self._factors}
        if len(dtypes) != 1:
            raise RuntimeError(f"Factors have different dtypes: {dtypes}")
        return dtypes.pop()

    @staticmethod
    def ensure_all_square(*factors: Tensor):
        """Check that all provided tensor factors are square matrices.

        Args:
            *factors: Variable number of 2D tensors to check for squareness.

        Raises:
            RuntimeError: If any factor is not square.
        """
        for i, factor in enumerate(factors):
            if factor.shape[0] != factor.shape[1]:
                raise RuntimeError(f"Factor {i} is not square: {factor.shape}.")

    @property
    def trace(self) -> Tensor:
        """Trace of the Kronecker product.

        For square matrices, tr(S_1 ⊗ S_2 ⊗ ... ⊗ S_k) = ∏ tr(S_i).

        Returns:
            Trace of the Kronecker product.

        Raises:
            RuntimeError: If any factor is not square.
        """
        self.ensure_all_square(*self._factors)
        return stack([S.trace() for S in self._factors]).prod()

    @property
    def det(self) -> Tensor:
        """Determinant of the Kronecker product.

        For square matrices S_1 (n_1×n_1), S_2 (n_2×n_2), ..., S_k (n_k×n_k):
        det(S_1 ⊗ S_2 ⊗ ... ⊗ S_k) = ∏_i det(S_i)^(∏_{j≠i} n_j)

        Returns:
            Determinant of the Kronecker product.

        """
        self.ensure_all_square(*self._factors)
        dim = prod(S.shape[0] for S in self._factors)
        return stack([S.det() ** (dim // S.shape[0]) for S in self._factors]).prod()

    @property
    def logdet(self) -> Tensor:
        """Log determinant of the Kronecker product.

        More numerically stable than det property.
        For square matrices S_1 (n_1×n_1), S_2 (n_2×n_2), ..., S_k (n_k×n_k):
        logdet(S_1 ⊗ S_2 ⊗ ... ⊗ S_k) = ∑_i (∏_{j≠i} n_j) * logdet(S_i)

        Returns:
            Log determinant of the Kronecker product.
        """
        self.ensure_all_square(*self._factors)
        dim = prod(S.shape[0] for S in self._factors)
        return stack([(dim // S.shape[0]) * S.logdet() for S in self._factors]).sum()

    @property
    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the Kronecker product.

        ||S_1 ⊗ S_2 ⊗ ... ⊗ S_k||_F = ∏_i ||S_i||_F

        Returns:
            Frobenius norm of the Kronecker product.
        """
        return stack([matrix_norm(S) for S in self._factors]).prod()
