"""Test linear operator representing an eigen-decomposed symmetric matrix."""

from torch import diag, float64, isclose, manual_seed, rand, randn
from torch.linalg import matrix_norm, qr

from curvlinops.eigh import EighDecomposedLinearOperator
from test.utils import compare_matmat


def test_EighDecomposedLinearOperator(adjoint: bool, is_vec: bool):
    """Test matrix multiplication and properties of an eigen-decomposed matrix."""
    manual_seed(0)
    # Use double precision to improve stability of det/logdet
    kwargs = {"dtype": float64}
    n = 6

    # Create orthogonal matrix with QR decomposition
    Q = randn(n, n, **kwargs)
    Q, _ = qr(Q)  # Make orthogonal
    eigvals = rand(n, **kwargs) + 1e-3  # Ensure positive eigenvalues

    # Create eigendecomposition operator
    op = EighDecomposedLinearOperator(eigvals, Q)
    # Create equivalent matrix: Q @ diag(λ) @ Q^T
    mat = Q @ diag(eigvals) @ Q.T

    compare_matmat(op, mat, adjoint, is_vec)

    # Test mathematical properties with higher precision
    assert isclose(mat.trace(), op.trace)
    assert isclose(mat.det(), op.det)
    assert isclose(mat.logdet(), op.logdet)
    assert isclose(matrix_norm(mat), op.frobenius_norm)
