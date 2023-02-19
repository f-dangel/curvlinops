"""Tests for ``curvlinops/_base``."""

from pytest import raises

from curvlinops._base import _LinearOperator


def test_check_deterministic(non_deterministic_case):
    """Test that non-deterministic behavior is recognized."""
    model_func, loss_func, params, data = non_deterministic_case

    with raises(RuntimeError):
        _LinearOperator(model_func, loss_func, params, data, check_deterministic=True)
