"""Test ``DiagonalLinearOperator``."""

from typing import List, Tuple

from pytest import mark
from torch import cat, manual_seed, rand

from curvlinops.diag import DiagonalLinearOperator
from test.utils import compare_matmat

SHAPES = [[(3, 2), (5,)]]


@mark.parametrize("shape", SHAPES)
def test_matvec(shape: List[Tuple[int, ...]]):
    """Test matrix-vector multiplication against dense diagonal matrix."""
    manual_seed(0)
    diag = [rand(*s) for s in shape]
    op = DiagonalLinearOperator(diag)
    mat = cat([d.flatten() for d in diag]).diag()
    compare_matmat(op, mat)


@mark.parametrize("shape", SHAPES)
def test_add_and_sub_closure(shape: List[Tuple[int, ...]]):
    """Test that addition/subtraction yields a DiagonalLinearOperator."""
    manual_seed(0)
    diag1, diag2 = [rand(*s) for s in shape], [rand(*s) for s in shape]
    op1, op2 = DiagonalLinearOperator(diag1), DiagonalLinearOperator(diag2)

    # Test closure and correctness of addition
    op_add = op1 + op2
    mat_add = cat([d1.flatten() + d2.flatten() for d1, d2 in zip(diag1, diag2)]).diag()
    assert isinstance(op_add, DiagonalLinearOperator)
    compare_matmat(op_add, mat_add)

    # Test closure and correctness of subtraction
    op_sub = op1 - op2
    mat_sub = cat([d1.flatten() - d2.flatten() for d1, d2 in zip(diag1, diag2)]).diag()
    assert isinstance(op_sub, DiagonalLinearOperator)
    compare_matmat(op_sub, mat_sub)


@mark.parametrize("shape", SHAPES)
def test_mul_and_div_closure(shape: List[Tuple[int, ...]]):
    """Test that scalar multiplication/division yields a DiagonalLinearOperator."""
    manual_seed(0)
    diag = [rand(*s) for s in shape]
    op = DiagonalLinearOperator(diag)
    scalar = 0.1

    # Test closure and correctness of left-multiplication onto scalar
    op_mul = op * scalar
    mat_mul = cat([scalar * d.flatten() for d in diag]).diag()
    assert isinstance(op_mul, DiagonalLinearOperator)
    compare_matmat(op_mul, mat_mul)

    # Test closure and correctness of right-multiplication onto scalar
    op_rmul = scalar * op
    assert isinstance(op_rmul, DiagonalLinearOperator)
    compare_matmat(op_rmul, mat_mul)

    # Test closure and correctness of division by scalar
    op_div = op / scalar
    mat_div = cat([d.flatten() / scalar for d in diag]).diag()
    assert isinstance(op_div, DiagonalLinearOperator)
    compare_matmat(op_div, mat_div)


@mark.parametrize("shape", SHAPES)
def test_matmul_closure(shape: List[Tuple[int, ...]]):
    """Test that matmul of two diagonal operators yields a DiagonalLinearOperator."""
    manual_seed(0)
    diag1, diag2 = [rand(*s) for s in shape], [rand(*s) for s in shape]
    op1, op2 = DiagonalLinearOperator(diag1), DiagonalLinearOperator(diag2)

    op_matmul = op1 @ op2
    mat_matmul = cat([(d1 * d2).flatten() for d1, d2 in zip(diag1, diag2)]).diag()
    assert isinstance(op_matmul, DiagonalLinearOperator)
    compare_matmat(op_matmul, mat_matmul)


@mark.parametrize("shape", SHAPES)
def test_inverse_closure(shape: List[Tuple[int, ...]]):
    """Test the damped inverse of a diagonal operator is a diagonal operator."""
    manual_seed(0)
    diag = [rand(*s) for s in shape]
    op = DiagonalLinearOperator(diag)
    damping = 1.0

    op_inv = op.inverse(damping)
    mat_inv = (cat([d.flatten() for d in diag]) + damping).reciprocal().diag()
    assert isinstance(op_inv, DiagonalLinearOperator)
    compare_matmat(op_inv, mat_inv)
