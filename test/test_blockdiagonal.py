"""Test linear operator of a block-diagonal matrix."""

from pytest import raises
from torch import block_diag, float64, isclose, manual_seed, rand
from torch.linalg import matrix_norm

from curvlinops.blockdiagonal import BlockDiagonalLinearOperator
from curvlinops.examples import TensorLinearOperator
from test.utils import compare_matmat


def test_BlockDiagonalLinearOperator():
    """Test matrix multiplication with rectangular blocks."""
    manual_seed(0)
    # Create rectangular blocks: 3x2, 4x3, 2x5
    B1, B2, B3 = rand(3, 2), rand(4, 3), rand(2, 5)

    # Create and test block-diagonal product operator's @ operation
    op = BlockDiagonalLinearOperator([TensorLinearOperator(B) for B in [B1, B2, B3]])
    mat = block_diag(B1, B2, B3)
    compare_matmat(op, mat)

    # Test mathematical properties
    for property in ["trace", "det", "logdet"]:
        with raises(RuntimeError):
            getattr(op, property)()
    assert isclose(matrix_norm(mat), op.frobenius_norm())


def test_BlockDiagonalLinearOperator_square():
    """Test matrix multiplication with square blocks."""
    # Use double-precision because we are testing logdet below
    kwargs = {"dtype": float64}
    B1, B2, B3 = rand(2, 2, **kwargs), rand(4, 4, **kwargs), rand(3, 3, **kwargs)
    B1, B2, B3 = B1 @ B1.T, B2 @ B2.T, B3 @ B3.T  # make PSD

    # Create and test block diagonal operator's @ operation
    op = BlockDiagonalLinearOperator([TensorLinearOperator(B) for B in [B1, B2, B3]])
    mat = block_diag(B1, B2, B3)
    compare_matmat(op, mat)

    # Test mathematical properties
    assert isclose(mat.trace(), op.trace())
    assert isclose(mat.det(), op.det())
    assert isclose(mat.logdet(), op.logdet())
    assert isclose(matrix_norm(mat), op.frobenius_norm())
