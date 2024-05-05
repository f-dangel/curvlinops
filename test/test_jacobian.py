"""Contains tests for ``curvlinops/jacobian``."""

from collections.abc import MutableMapping

from numpy import random
from pytest import raises

from curvlinops import JacobianLinearOperator, TransposedJacobianLinearOperator
from curvlinops.examples.functorch import functorch_jacobian
from curvlinops.examples.utils import report_nonclose


def test_JacobianLinearOperator_matvec(case, adjoint: bool):
    model_func, _, params, data, batch_size_fn = case

    # Test when X is dict-like but batch_size_fn = None (default)
    if isinstance(data[0][0], MutableMapping):
        with raises(AttributeError):
            op = JacobianLinearOperator(model_func, params, data)

    op = JacobianLinearOperator(model_func, params, data, batch_size_fn=batch_size_fn)
    op_functorch = (
        functorch_jacobian(model_func, params, data, input_key="x")
        .detach()
        .cpu()
        .numpy()
    )
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    x = random.rand(op.shape[1])
    report_nonclose(op @ x, op_functorch @ x, rtol=1e-4)


def test_JacobianLinearOperator_matmat(case, adjoint: bool, num_vecs: int = 3):
    model_func, _, params, data, batch_size_fn = case

    op = JacobianLinearOperator(model_func, params, data, batch_size_fn=batch_size_fn)
    op_functorch = (
        functorch_jacobian(model_func, params, data, input_key="x")
        .detach()
        .cpu()
        .numpy()
    )
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    X = random.rand(op.shape[1], num_vecs)
    report_nonclose(op @ X, op_functorch @ X, rtol=1e-4)


def test_TransposedJacobianLinearOperator_matvec(case, adjoint: bool):
    model_func, _, params, data, batch_size_fn = case

    # Test when X is dict-like but batch_size_fn = None (default)
    if isinstance(data[0][0], MutableMapping):
        with raises(AttributeError):
            op = TransposedJacobianLinearOperator(model_func, params, data)

    op = TransposedJacobianLinearOperator(
        model_func, params, data, batch_size_fn=batch_size_fn
    )
    op_functorch = (
        functorch_jacobian(model_func, params, data, input_key="x")
        .detach()
        .cpu()
        .numpy()
        .T
    )
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    x = random.rand(op.shape[1])
    report_nonclose(op @ x, op_functorch @ x, rtol=1e-4)


def test_TransposedJacobianLinearOperator_matmat(
    case, adjoint: bool, num_vecs: int = 3
):
    model_func, _, params, data, batch_size_fn = case

    op = TransposedJacobianLinearOperator(
        model_func, params, data, batch_size_fn=batch_size_fn
    )
    op_functorch = (
        functorch_jacobian(model_func, params, data, input_key="x")
        .detach()
        .cpu()
        .numpy()
        .T
    )
    if adjoint:
        op, op_functorch = op.adjoint(), op_functorch.conj().T

    X = random.rand(op.shape[1], num_vecs)
    report_nonclose(op @ X, op_functorch @ X, rtol=1e-4)
