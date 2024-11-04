"""Tests the linear operator interface in PyTorch."""

from typing import List

from pytest import raises
from torch import Tensor, zeros

from curvlinops._torch_base import CurvatureLinearOperator, PyTorchLinearOperator
from curvlinops.examples.functorch import functorch_gradient_and_loss
from curvlinops.utils import allclose_report


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

    Id = IdentityLinearOperator(in_shape, out_shape)
    assert Id._in_shape_flat == [6, 20]
    assert Id._out_shape_flat == [6, 24]
    assert Id.shape == (30, 26)

    # using valid input vectors/matrices will trigger errors because we
    # initialized the identity with different input/output spaces
    with raises(ValueError):
        _ = Id @ [zeros(2, 3), zeros(4, 5)]  # valid vector in list format

    with raises(ValueError):
        _ = Id @ [zeros(2, 3, 6), zeros(4, 5, 6)]  # valid matrix in list format

    with raises(ValueError):
        _ = Id @ zeros(26)  # valid vector in tensor format

    with raises(ValueError):
        _ = Id @ zeros(26, 6)  # valid matrix in tensor format


def test_preserve_input_format():
    """Test whether the input format is preserved by matrix multiplication."""
    in_shape = out_shape = [(2, 3), (4, 5)]
    Id = IdentityLinearOperator(in_shape, out_shape)
    assert Id._in_shape_flat == Id._out_shape_flat == [6, 20]

    X = [zeros(2, 3), zeros(4, 5)]  # vector in tensor list format
    IdX = Id @ X
    assert len(IdX) == len(X) and all(Idx.allclose(x) for Idx, x in zip(IdX, X))

    X = [zeros(2, 3, 6), zeros(4, 5, 6)]  # matrix in tensor list format
    IdX = Id @ X
    assert len(IdX) == len(X) and all(Idx.allclose(x) for Idx, x in zip(IdX, X))

    X = zeros(26)  # vector in tensor format
    IdX = Id @ X
    assert IdX.allclose(X)

    X = zeros(26, 6)  # matrix in tensor format
    IdX = Id @ X
    assert IdX.allclose(X)


def test_gradient_and_loss(case):
    """Test the gradient and loss computation over a data loader."""
    model, loss_func, params, data, batch_size_fn = case

    linop = CurvatureLinearOperator(
        model,
        loss_func,
        params,
        data,
        # turn off because this would trigger the un-implemented `matmat`
        check_deterministic=False,
        batch_size_fn=batch_size_fn,
    )
    gradient, loss = linop.gradient_and_loss()

    gradient_functorch, loss_functorch = functorch_gradient_and_loss(
        model, loss_func, params, data, input_key="x"
    )

    assert allclose_report(loss, loss_functorch)
    assert len(gradient) == len(gradient_functorch)
    for g, g_functorch in zip(gradient, gradient_functorch):
        assert allclose_report(g, g_functorch)
