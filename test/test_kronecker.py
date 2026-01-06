"""Test linear operator of a Kronecker product."""

from pytest import raises
from torch import float64, isclose, kron, manual_seed, rand
from torch.linalg import matrix_norm

from curvlinops.kronecker import KroneckerProductLinearOperator
from test.utils import compare_matmat


def test_KroneckerProductLinearOperator(adjoint: bool, is_vec: bool):
    """Test matrix multiplication with rectangular Kronecker factors."""
    manual_seed(0)
    # Create rectangular factors: 3x2, 4x3, 2x5
    S1, S2, S3 = rand(3, 2), rand(4, 3), rand(2, 5)

    # Create and test Kronecker product operator's @ operation
    op = KroneckerProductLinearOperator(S1, S2, S3)
    mat = kron(kron(S1, S2), S3)
    compare_matmat(op, mat, adjoint, is_vec)

    # Test mathematical properties
    for property in ["trace", "det", "logdet"]:
        with raises(RuntimeError):
            getattr(op, property)
    assert isclose(matrix_norm(mat), op.frobenius_norm)


def test_KroneckerProductLinearOperator_square(adjoint: bool, is_vec: bool):
    """Test matrix multiplication with square Kronecker factors."""
    manual_seed(0)
    # Use double-precision because we are testing logdet below
    kwargs = {"dtype": float64}
    S1, S2, S3 = rand(2, 2, **kwargs), rand(4, 4, **kwargs), rand(3, 3, **kwargs)
    S1, S2, S3 = S1 @ S1.T, S2 @ S2.T, S3 @ S3.T  # make PSD

    # Create and test Kronecker product operator's @ operation
    op = KroneckerProductLinearOperator(S1, S2, S3)
    mat = kron(kron(S1, S2), S3)
    compare_matmat(op, mat, adjoint, is_vec)

    # Test mathematical properties
    assert isclose(mat.trace(), op.trace)
    assert isclose(mat.det(), op.det)
    assert isclose(mat.logdet(), op.logdet)
    assert isclose(matrix_norm(mat), op.frobenius_norm)
