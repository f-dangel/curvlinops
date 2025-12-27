"""Test linear operator representing an eigen-decomposed symmetric matrix."""

from torch import float64, isclose, manual_seed, rand, randn
from torch.linalg import inv, matrix_norm, qr

from curvlinops.eigh import EighDecomposedLinearOperator
from test.utils import compare_matmat, eye_like


def test_EighDecomposedLinearOperator(adjoint: bool, is_vec: bool):
    """Test matrix multiplication and properties of an eigen-decomposed matrix."""
    manual_seed(0)
    # Use double precision to improve stability of det/logdet
    kwargs = {"dtype": float64}
    n = 6

    # Create orthogonal matrix with QR decomposition
    Q, _ = qr(randn(n, n, **kwargs))
    eigvals = rand(n, **kwargs) + 1e-3  # Ensure positive eigenvalues

    # Create eigendecomposition operator and its matrix representation
    op = EighDecomposedLinearOperator(eigvals, Q)
    mat = Q @ eigvals.diag() @ Q.T

    # Test matrix multiplication
    compare_matmat(op, mat, adjoint, is_vec)

    # Test mathematical properties
    assert isclose(mat.trace(), op.trace())
    assert isclose(mat.det(), op.det())
    assert isclose(mat.logdet(), op.logdet())
    assert isclose(matrix_norm(mat), op.frobenius_norm())

    # Test inversion
    damping = 1e-4
    op_inv = op.inverse(damping=damping)
    mat_inv = inv(mat + damping * eye_like(mat))
    compare_matmat(op_inv, mat_inv, adjoint, is_vec)
