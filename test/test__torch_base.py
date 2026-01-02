"""Tests the linear operator interface in PyTorch."""

from __future__ import annotations

from typing import Iterable, Iterator, List, MutableMapping, Tuple, Union

from pytest import raises
from torch import Tensor, linspace, manual_seed, rand, rand_like, randperm, zeros

from curvlinops._torch_base import CurvatureLinearOperator, PyTorchLinearOperator
from curvlinops.examples import TensorLinearOperator
from curvlinops.examples.functorch import functorch_gradient_and_loss
from curvlinops.utils import allclose_report
from test.utils import compare_matmat


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


def test_MutableMapping_no_batch_size_fn(case):
    """Trigger error with ``MutableMapping`` data + unspecified batch size getter.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, _ = case

    if isinstance(data[0][0], MutableMapping):
        with raises(ValueError):
            _ = CurvatureLinearOperator(model_func, loss_func, params, data)


def test_check_deterministic(non_deterministic_case):
    """Test that non-deterministic behavior is recognized."""
    model_func, loss_func, params, data, batch_size_fn = non_deterministic_case

    with raises(RuntimeError):
        CurvatureLinearOperator(
            model_func,
            loss_func,
            params,
            data,
            batch_size_fn=batch_size_fn,
            check_deterministic=True,
        )


class FixedBatchesIdentityLinearOperator(CurvatureLinearOperator):
    """Linear identity operator which demands deterministic batches."""

    FIXED_DATA_ORDER: bool = True

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        return X


class PermutedBatchLoader:
    """Randomly shuffle data points in a batch before returning it."""

    def __init__(self, data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]]):
        self.data = data

    def __iter__(self) -> Iterator[Tuple[Union[Tensor, MutableMapping], Tensor]]:
        for X, y in self.data:
            permutation = randperm(y.shape[0])

            if isinstance(X, MutableMapping):
                for key, value in X.items():
                    if isinstance(value, Tensor):
                        X[key] = X[key][permutation]
            else:
                X = X[permutation]

            yield X, y[permutation]


def test_check_deterministic_batch(case):
    """Test that non-deterministic batches are recognized.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model_func, loss_func, params, data, batch_size_fn = case

    data = PermutedBatchLoader(data)
    with raises(RuntimeError):
        _ = FixedBatchesIdentityLinearOperator(
            model_func, loss_func, params, data, batch_size_fn=batch_size_fn
        )


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


def test_SumPyTorchLinearOperator(adjoint: bool, is_vec: bool):
    """Test adding two PyTorch linear operators.

    Args:
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    manual_seed(0)
    A = linspace(1, 10, steps=20).reshape(5, 4)
    B = rand_like(A)

    # test addition
    A_plus_B = A + B
    A_linop, B_linop = TensorLinearOperator(A), TensorLinearOperator(B)
    A_plus_B_linop = A_linop + B_linop
    compare_matmat(A_plus_B_linop, A_plus_B, adjoint, is_vec)

    # test subtraction
    A_minus_B = A - B
    A_linop, B_linop = TensorLinearOperator(A), TensorLinearOperator(B)
    A_minus_B_linop = A_linop - B_linop
    compare_matmat(A_minus_B_linop, A_minus_B, adjoint, is_vec)


def test_ScalePyTorchLinearOperator(adjoint: bool, is_vec: bool):
    """Test scaling a PyTorch linear operator with a scalar.

    Args:
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    A = linspace(1, 10, steps=20).reshape(5, 4)
    scalar = 0.1

    A_scaled = scalar * A
    A_linop = TensorLinearOperator(A)

    # scale from the right
    compare_matmat(A_linop * scalar, A_scaled, adjoint, is_vec)
    # scale from the left
    compare_matmat(scalar * A_linop, A_scaled, adjoint, is_vec)


def test_ChainPyTorchLinearOperator(adjoint: bool, is_vec: bool):
    """Test chaining two PyTorch linear operators.

    Args:
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    manual_seed(0)
    A = linspace(1, 10, steps=20).reshape(5, 4)
    B = rand(4, 3)
    C = rand(3, 2)

    ABC = A @ B @ C
    ABC_linop = (
        TensorLinearOperator(A) @ TensorLinearOperator(B) @ TensorLinearOperator(C)
    )
    compare_matmat(ABC_linop, ABC, adjoint, is_vec)
