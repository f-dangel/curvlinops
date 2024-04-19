"""Tests for ``curvlinops/_base``."""

from pytest import raises

from curvlinops._base import _LinearOperator


def test_check_deterministic(non_deterministic_case):
    """Test that non-deterministic behavior is recognized."""
    model_func, loss_func, params, data, batch_size_fn = non_deterministic_case

    with raises(RuntimeError):
        _LinearOperator(
            model_func,
            loss_func,
            params,
            data,
            batch_size_fn=batch_size_fn,
            check_deterministic=True,
        )
