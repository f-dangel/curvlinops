"""PyTorch linear operator implementation of Kronecker product S_1 ⊗ S_2 ⊗ ... ."""

from math import prod, sqrt
from typing import List, Tuple, Union
from warnings import warn

from einops import einsum
from torch import (
    Tensor,
    cholesky_inverse,
    device,
    diagonal_scatter,
    dtype,
    float64,
    kron,
    stack,
)
from torch.linalg import cholesky, eigh, matrix_norm

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.eigh import EighDecomposedLinearOperator


def ensure_all_square(*factors: Union[Tensor, PyTorchLinearOperator]):
    """Check that all provided tensors/linear operators are square.

    Args:
        *factors: Variable number of tensors/operators to check.

    Raises:
        RuntimeError: If any factor is not square.
    """
    for i, factor in enumerate(factors):
        if len(factor.shape) != 2 or factor.shape[0] != factor.shape[1]:
            raise RuntimeError(f"Factor {i} is not square: {factor.shape}.")


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

    @property
    def trace(self) -> Tensor:
        """Trace of the Kronecker product.

        For square matrices, tr(S_1 ⊗ S_2 ⊗ ... ⊗ S_k) = ∏ tr(S_i).

        Returns:
            Trace of the Kronecker product.

        Raises:
            RuntimeError: If any factor is not square.
        """
        ensure_all_square(*self._factors)
        return stack([S.trace() for S in self._factors]).prod()

    @property
    def det(self) -> Tensor:
        """Determinant of the Kronecker product.

        For square matrices S_1 (n_1×n_1), S_2 (n_2×n_2), ..., S_k (n_k×n_k):
        det(S_1 ⊗ S_2 ⊗ ... ⊗ S_k) = ∏_i det(S_i)^(∏_{j≠i} n_j)

        Returns:
            Determinant of the Kronecker product.

        """
        ensure_all_square(*self._factors)
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
        ensure_all_square(*self._factors)
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

    def inverse(
        self,
        damping: Union[float, Tuple[float, ...]] = 0.0,
        use_heuristic_damping: bool = False,
        min_damping: float = 1e-8,
        use_exact_damping: bool = False,
        retry_double_precision: bool = True,
    ):
        ensure_all_square(*self._factors)

        if use_heuristic_damping and use_exact_damping:
            raise ValueError("Either use heuristic damping or exact damping, not both.")
        if (use_heuristic_damping or use_exact_damping) and isinstance(damping, tuple):
            raise ValueError(
                "Heuristic and exact damping require a single damping value."
            )
        if isinstance(damping, tuple) and len(damping) != len(self._factors):
            raise ValueError(
                f"Damping tuple length {len(damping)} does not match number of factors {len(self._factors)}."
            )

        if use_heuristic_damping and len(self._factors) != 2:
            raise ValueError(
                f"Heuristic damping only implemented for two factors. Got {len(self._factors)}"
            )

        if use_exact_damping:
            # NOTE We assume all Kronecker factors are symmetric
            eigvals, eigvecs = zip(*[eigh(S) for S in self._factors])
            eigvals_expanded = eigvals[0]
            for eigval in eigvals[1]:
                eigvals_expanded = kron(eigvals_expanded, eigval)
            return EighDecomposedLinearOperator(
                eigvals_expanded, KroneckerProductLinearOperator(*eigvecs)
            ).inverse(damping=damping)

        else:
            # Martens and Grosse, 2015 (https://arxiv.org/abs/1503.05671) (Section 6.3)
            if use_heuristic_damping:
                S1, S2 = self._factors
                mean_eig1, mean_eig2 = S1.diag().mean(), S2.diag().mean()
                if any(mean_eig < 0 for mean_eig in [mean_eig1, mean_eig2]):
                    raise RuntimeError("Negative mean eigenvalue detected")

                sqrt_eig_mean_ratio = (mean_eig2 / mean_eig1).sqrt()
                sqrt_damping = sqrt(damping)
                damping1 = max(sqrt_damping / sqrt_eig_mean_ratio, min_damping)
                damping2 = max(sqrt_damping * sqrt_eig_mean_ratio, min_damping)
                damping = (damping1, damping2)

            else:
                damping = (
                    damping
                    if isinstance(damping, tuple)
                    else tuple(len(self._factors) * [damping])
                )

            factors_inv = [
                self._damped_cholesky_inverse(S_i, damping_i, retry_double_precision)
                for S_i, damping_i in zip(self._factors, damping)
            ]
            return KroneckerProductLinearOperator(*factors_inv)

    @staticmethod
    def _damped_cholesky_inverse(
        A: Tensor, damping: float, retry_double_precision: bool
    ) -> Tensor:
        def _damped_cholesky(A: Tensor, damping: float) -> Tensor:
            A_damped = diagonal_scatter(A, A.diag() + damping)
            return cholesky(A_damped)

        try:
            L = _damped_cholesky(A, damping)
        except RuntimeError as error:
            if not retry_double_precision or A.dtype == float64:
                raise error

            warn(
                f"Failed to compute Cholesky decomposition in {A.dtype} "
                f"precision with error {error}. Retrying in double precision...",
                stacklevel=2,
            )
            # Retry in double precision
            original_dt = A.dtype
            L = _damped_cholesky(A.to(float64), damping).to(original_dt)

        return cholesky_inverse(L)
