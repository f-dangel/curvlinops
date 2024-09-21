"""Tests the linear operator interface in PyTorch."""

from typing import List

from pytest import raises
from torch import Tensor, zeros

from curvlinops._torch_base import PyTorchLinearOperator


def test_input_formatting():
    """Test format checks of the input to a matrix multiplication."""
    in_shape = [(2, 3), (4, 5)]

    L = PyTorchLinearOperator(in_shape, in_shape)
    assert L._in_shape_flat == L._out_shape_flat == [6, 20]
    assert L.shape == (26, 26)

    # try multiplying with invalid vectors/matrices
    with raises(ValueError):
        _ = L @ zeros(25)  # too few numbers

    with raises(ValueError):
        _ = L @ [zeros(2, 3), zeros(4, 4)]  # wrong shape in second tensor

    with raises(ValueError):
        _ = L @ [zeros(2, 3, 6), zeros(4, 5, 7)]  # ambiguous number of vectors


class IdentityLinearOperator(PyTorchLinearOperator):
    """Linear operator in PyTorch representing the identity matrix."""

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        return X


def test_output_formatting():
    """Test format checks of the output of a matrix multiplication."""
    in_shape = [(2, 3), (4, 5)]
    out_shape = [(2, 3), (4, 6)]  # NOTE that this will trigger an error

    I = IdentityLinearOperator(in_shape, out_shape)
    assert I._in_shape_flat == [6, 20]
    assert I._out_shape_flat == [6, 24]
    assert I.shape == (30, 26)

    # using valid input vectors/matrices will trigger errors because we
    # initialized the identity with different input/output spaces
    with raises(ValueError):
        _ = I @ [zeros(2, 3), zeros(4, 5)]  # valid vector in list format

    with raises(ValueError):
        _ = I @ [zeros(2, 3, 6), zeros(4, 5, 6)]  # valid matrix in list format

    with raises(ValueError):
        _ = I @ zeros(26)  # valid vector in tensor format

    with raises(ValueError):
        _ = I @ zeros(26, 6)  # valid matrix in tensor format


def test_preserve_input_format():
    """Test whether the input format is preserved by matrix multiplication."""
    in_shape = out_shape = [(2, 3), (4, 5)]
    I = IdentityLinearOperator(in_shape, out_shape)
    assert I._in_shape_flat == I._out_shape_flat == [6, 20]

    X = [zeros(2, 3), zeros(4, 5)]  # vector in tensor list format
    IX = I @ X
    assert len(IX) == len(X) and all(Ix.allclose(x) for Ix, x in zip(IX, X))

    X = [zeros(2, 3, 6), zeros(4, 5, 6)]  # matrix in tensor list format
    IX = I @ X
    assert len(IX) == len(X) and all(Ix.allclose(x) for Ix, x in zip(IX, X))

    X = zeros(26)  # vector in tensor format
    IX = I @ X
    assert IX.allclose(X)

    X = zeros(26, 6)  # matrix in tensor format
    IX = I @ X
    assert IX.allclose(X)
