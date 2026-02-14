"""Tests the linear operator interface in PyTorch."""

from __future__ import annotations

from typing import Iterable, Iterator, List, MutableMapping, Tuple, Union

from pytest import raises
from torch import Size, Tensor, linspace, manual_seed, rand, rand_like, randperm, zeros

from curvlinops._torch_base import CurvatureLinearOperator, PyTorchLinearOperator
from curvlinops.examples import TensorLinearOperator
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


def test_empty_shapes_exception():
    """Test that empty input and output shapes raise exceptions."""
    in_and_out_shapes = [
        [[], []],
        [[(1, 2, 3)], []],
        [[], [(4, 5, 6)]],
    ]
    for in_shape, out_shape in in_and_out_shapes:
        with raises(ValueError, match="must be non-empty."):
            _ = PyTorchLinearOperator(in_shape, out_shape)


class MockLinearOperator(PyTorchLinearOperator):
    """Dummy linear operator in PyTorch. Implements the zero matrix."""

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        ((dev, dt, num_vecs),) = {(x.device, x.dtype, x.shape[-1]) for x in X}
        return [zeros(*x.shape[:-1], num_vecs, device=dev, dtype=dt) for x in X]


def test_output_formatting():
    """Test format checks of the output of a matrix multiplication."""
    shape = [(2, 3), (4, 5)]

    Id = MockLinearOperator(shape, shape)
    assert Id._in_shape_flat == Id._out_shape_flat == [6, 20]
    assert Id.shape == (26, 26)

    # Verify multiplying onto invalid vectors raises exceptions during input processing
    invalid_type = [1.0, 2.0, 3.0]
    with raises(ValueError, match="Input must be tensor or list of tensors."):
        _ = Id @ invalid_type

    for invalid_list in [
        [zeros(2, 3), zeros(4, 6)],
        [zeros(2, 3, 6), zeros(4, 5, 7)],
    ]:
        with raises(ValueError, match="Input list must contain tensors with shapes"):
            _ = Id @ invalid_list

    for invalid_tensor in [zeros(25), zeros(25, 6)]:
        with raises(ValueError, match="Input tensor must have shape"):
            _ = Id @ invalid_tensor

    too_few_entries = [zeros(2, 3)]
    with raises(ValueError, match="Input list must have 2 tensors. Got 1."):
        _ = Id @ too_few_entries


def test__check_output_and_postprocess_exceptions():
    """Trigger the exceptions in output checking and post-processing."""
    expected_shapes = [Size([1, 2]), Size([3, 4])]
    list_format, is_vec, num_vecs, free_dim = False, True, 1, "trailing"

    for free_dim in ["trailing", "leading"]:
        other_args = (list_format, is_vec, num_vecs, expected_shapes, free_dim)

        # NOTE Shape does not really matter as long as we have one tensor
        too_few_entries = [zeros(1, 2)]
        with raises(ValueError, match="Output tensor list must have 2 tensors. Got 1."):
            _ = PyTorchLinearOperator._check_output_and_postprocess(
                too_few_entries, *other_args
            )

        # NOTE Shape does not really matter as long as we have two tensors
        invalid_shapes = [zeros(1, 2, num_vecs), zeros(3, 4, num_vecs + 1)]
        with raises(ValueError, match="Output tensors must have shapes"):
            _ = PyTorchLinearOperator._check_output_and_postprocess(
                invalid_shapes, *other_args
            )


def test_preserve_input_format():
    """Test whether the input format is preserved by matrix multiplication."""
    in_shape = out_shape = [(2, 3), (4, 5)]
    A = MockLinearOperator(in_shape, out_shape)
    assert A._in_shape_flat == A._out_shape_flat == [6, 20]

    X = [zeros(2, 3), zeros(4, 5)]  # vector in tensor list format
    AX = A @ X
    assert len(AX) == len(X) and all(Ax.allclose(x) for Ax, x in zip(AX, X))

    X = [zeros(2, 3, 6), zeros(4, 5, 6)]  # matrix in tensor list format
    AX = A @ X
    assert len(AX) == len(X) and all(Ax.allclose(x) for Ax, x in zip(AX, X))

    X = zeros(26)  # vector in tensor format
    AX = A @ X
    assert AX.allclose(X)

    X = zeros(26, 6)  # matrix in tensor format
    AX = A @ X
    assert AX.allclose(X)


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


class FixedBatchesMockLinearOperator(CurvatureLinearOperator):
    """Mock linear operator which demands deterministic batch order."""

    FIXED_DATA_ORDER: bool = True

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        ((dev, dt, num_vecs),) = {(x.device, x.dtype, x.shape[-1]) for x in X}
        return [zeros(*x.shape[:-1], num_vecs, device=dev, dtype=dt) for x in X]


class PermutedBatchLoader:
    """Randomly shuffle data points in a batch before returning it."""

    def __init__(self, data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]]):
        """Store data used for permutation.

        Args:
            data: Iterable of input/target batches.
        """
        self.data = data

    def __iter__(self) -> Iterator[Tuple[Union[Tensor, MutableMapping], Tensor]]:
        """Iterate over permuted batches.

        Yields:
            Permuted batches of inputs and targets.
        """
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
        _ = FixedBatchesMockLinearOperator(
            model_func, loss_func, params, data, batch_size_fn=batch_size_fn
        )


def test_SumPyTorchLinearOperator():
    """Test adding and subtracting two PyTorch linear operators."""
    manual_seed(0)
    A = linspace(1, 10, steps=20).reshape(5, 4)
    B = rand_like(A)
    A_linop, B_linop = TensorLinearOperator(A), TensorLinearOperator(B)

    # test addition and subtraction
    compare_matmat(A_linop + B_linop, A + B)
    compare_matmat(A_linop - B_linop, A - B)


def test_ScalePyTorchLinearOperator():
    """Test scaling a PyTorch linear operator with a scalar."""
    A = linspace(1, 10, steps=20).reshape(5, 4)
    A_linop = TensorLinearOperator(A)
    scalar = 0.1

    # test scaling from the left and right
    compare_matmat(scalar * A_linop, scalar * A)
    compare_matmat(A_linop * scalar, A * scalar)

    # test division (internally relies on scaling with the inverse)
    compare_matmat(A_linop / scalar, A / scalar)


def test_ChainPyTorchLinearOperator():
    """Test chaining two PyTorch linear operators."""
    manual_seed(0)
    A, B, C = linspace(1, 10, steps=20).reshape(5, 4), rand(4, 3), rand(3, 2)
    A_linop, B_linop, C_linop = [TensorLinearOperator(T) for T in [A, B, C]]

    compare_matmat(A_linop @ B_linop @ C_linop, A @ B @ C)
