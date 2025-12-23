"""Linear operator for eigen-decompositions."""

from typing import List, Union

from torch import Tensor, device, dtype

from curvlinops._torch_base import PyTorchLinearOperator


class EighDecomposedLinearOperator(PyTorchLinearOperator):
    """Linear operator representing Q diag(λ) Q^T with Q orthogonal.

    This represents a symmetric matrix through its eigendecomposition, where Q contains
    the eigenvectors as columns and λ contains the corresponding eigenvalues.
    """

    SELF_ADJOINT: bool = True

    def __init__(
        self, eigenvalues: Tensor, eigenvectors: Union[Tensor, PyTorchLinearOperator]
    ):
        """Initialize eigendecomposition linear operator.

        Args:
            eigenvalues: 1D tensor of shape (n,) containing eigenvalues.
            eigenvectors: 2D tensor of shape (n, n) containing eigenvectors as columns.

        Raises:
            ValueError: If eigenvalues is not 1D or eigenvectors is not 2D.
            ValueError: If eigenvalues and eigenvectors have incompatible shapes.
        """
        if eigenvalues.ndim != 1:
            raise ValueError(f"Eigenvalues must be 1D, got shape {eigenvalues.shape}.")

        if len(eigenvectors.shape) != 2:
            raise ValueError(
                f"Eigenvectors must be 2D, got shape {eigenvectors.shape}."
            )

        if eigenvectors.shape[0] != eigenvectors.shape[1]:
            raise ValueError(
                f"Eigenvectors must be square, got shape {eigenvectors.shape}."
            )

        if eigenvalues.shape[0] != eigenvectors.shape[0]:
            raise ValueError(
                f"Incompatible shapes: eigenvalues {eigenvalues.shape}, "
                f"eigenvectors {eigenvectors.shape}."
            )

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors

        n = eigenvalues.shape[0]
        in_shapes, out_shapes = [(n,)], [(n,)]

        super().__init__(in_shapes, out_shapes)

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Apply eigendecomposition operator to matrix.

        Computes Q diag(λ) Q^T @ X efficiently as Q @ (λ * (Q^T @ X)).

        Args:
            X: List with single tensor of shape (n, k).

        Returns:
            List with single tensor of shape (n, k).
        """
        (x,) = X
        result = self._eigenvectors @ (
            self._eigenvalues.unsqueeze(1) * (self._eigenvectors.adjoint() @ x)
        )
        return [result]

    @property
    def device(self) -> device:
        """Return the device of the eigendecomposition.

        Returns:
            Device of the eigenvalues and eigenvectors.

        Raises:
            RuntimeError: If eigenvalues and eigenvectors are on different devices.
        """
        if self._eigenvalues.device != self._eigenvectors.device:
            raise RuntimeError(
                f"Eigenvalues and eigenvectors on different devices: "
                f"{self._eigenvalues.device} vs {self._eigenvectors.device}"
            )
        return self._eigenvalues.device

    @property
    def dtype(self) -> dtype:
        """Return the data type of the eigendecomposition.

        Returns:
            Data type of the eigenvalues and eigenvectors.

        Raises:
            RuntimeError: If eigenvalues and eigenvectors have different dtypes.
        """
        if self._eigenvalues.dtype != self._eigenvectors.dtype:
            raise RuntimeError(
                f"Eigenvalues and eigenvectors have different dtypes: "
                f"{self._eigenvalues.dtype} vs {self._eigenvectors.dtype}"
            )
        return self._eigenvalues.dtype

    @property
    def trace(self) -> Tensor:
        """Trace of the eigendecomposition operator.

        For Q diag(λ) Q^T, the trace is sum(λ).

        Returns:
            Trace as sum of eigenvalues.
        """
        return self._eigenvalues.sum()

    @property
    def det(self) -> Tensor:
        """Determinant of the eigendecomposition operator.

        For Q diag(λ) Q^T, the determinant is prod(λ).

        Returns:
            Determinant as product of eigenvalues.
        """
        return self._eigenvalues.prod()

    @property
    def logdet(self) -> Tensor:
        """Log determinant of the eigendecomposition operator.

        For Q diag(λ) Q^T, the log determinant is sum(log(λ)).

        Returns:
            Log determinant as sum of log eigenvalues.
        """
        return self._eigenvalues.log().sum()

    @property
    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the eigendecomposition operator.

        For Q diag(λ) Q^T, ||A||_F = sqrt(sum(λ²)).

        Returns:
            Frobenius norm.
        """
        return self._eigenvalues.norm(p="fro")
