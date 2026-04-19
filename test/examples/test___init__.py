"""Tests for ``curvlinops.examples``."""

from collections.abc import Callable, Iterable, MutableMapping
from math import sqrt

import pytest
from torch import Tensor, float64
from torch import tensor as torch_tensor

from curvlinops.examples import (
    GradientNormDamping,
    gradient_and_loss,
    gradient_l2_norm,
    gradient_norm_damping,
)
from curvlinops.examples.functorch import functorch_gradient_and_loss
from curvlinops.utils import allclose_report
from test.utils import change_dtype


def test_gradient_and_loss(
    case: tuple[
        Callable[[Tensor], Tensor],
        Callable[[Tensor, Tensor], Tensor],
        dict[str, Tensor],
        Iterable[tuple[Tensor | MutableMapping, Tensor]],
        Callable[[MutableMapping], int] | None,
    ],
):
    """Test the standalone ``gradient_and_loss`` against ``functorch``.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    gradient, loss = gradient_and_loss(
        model, loss_func, params, data, batch_size_fn=batch_size_fn
    )

    gradient_functorch, loss_functorch = functorch_gradient_and_loss(
        model, loss_func, params, data, input_key="x"
    )

    assert allclose_report(loss, loss_functorch)
    assert len(gradient) == len(gradient_functorch)
    for g, g_functorch in zip(gradient, gradient_functorch):
        assert allclose_report(g, g_functorch)


def test_gradient_l2_norm():
    """Test the Euclidean norm helper for gradient tensor collections."""
    gradient = [
        torch_tensor([[3.0, 4.0]]),
        None,
        torch_tensor([12.0]),
    ]

    assert allclose_report(gradient_l2_norm(gradient), torch_tensor(13.0))


def test_gradient_l2_norm_raises_on_missing_gradient():
    """The norm helper should reject collections without gradient tensors."""
    with pytest.raises(ValueError, match="Expected at least one gradient tensor."):
        gradient_l2_norm([None, None])


def test_gradient_norm_damping():
    """Test the explicit gradient-norm damping helper."""
    gradient = [torch_tensor([3.0, 4.0])]

    assert gradient_norm_damping(gradient, damping_scale=0.8, min_damping=2.5) == 2.5
    assert gradient_norm_damping(
        gradient, damping_scale=1.8, min_damping=0.5
    ) == pytest.approx(sqrt(1.8 * 5.0))


def test_gradient_norm_damping_ignores_none_entries():
    """Gradient-norm damping should use the norm over the non-``None`` entries."""
    gradient = [
        torch_tensor([[3.0, 4.0]]),
        None,
        torch_tensor([12.0]),
    ]

    assert gradient_norm_damping(
        gradient, damping_scale=1.3, min_damping=0.0
    ) == pytest.approx(sqrt(1.3 * 13.0))


def test_gradient_norm_damping_on_full_dataset_gradient(
    case: tuple[
        Callable[[Tensor], Tensor],
        Callable[[Tensor, Tensor], Tensor],
        dict[str, Tensor],
        Iterable[tuple[Tensor | MutableMapping, Tensor]],
        Callable[[MutableMapping], int] | None,
    ],
):
    """Gradient-norm damping should match the rule on a real autograd gradient."""
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    gradient, _ = gradient_and_loss(
        model, loss_func, params, data, batch_size_fn=batch_size_fn
    )

    damping_scale = 0.7
    min_damping = 1e-4
    expected = max(min_damping, sqrt(damping_scale * gradient_l2_norm(gradient).item()))

    assert gradient_norm_damping(gradient, damping_scale, min_damping) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("damping_scale,min_damping", [(-1.0, 0.0), (1.0, -1.0)])
def test_gradient_norm_damping_raises_on_negative_inputs(
    damping_scale: float, min_damping: float
):
    """Gradient-norm damping should reject negative hyperparameters."""
    with pytest.raises(ValueError):
        gradient_norm_damping([torch_tensor([1.0])], damping_scale, min_damping)


def test_gradient_norm_damping_raises_on_missing_gradient():
    """Gradient-norm damping should reject collections without gradient tensors."""
    with pytest.raises(ValueError, match="Expected at least one gradient tensor."):
        gradient_norm_damping([None, None], damping_scale=1.0)


def test_gradient_norm_damping_policy():
    """The callable policy should evaluate the gradient-norm rule."""
    policy = GradientNormDamping(damping_scale=1.8, min_damping=0.5)
    gradient = [torch_tensor([3.0, 4.0])]

    assert policy(gradient) == pytest.approx(sqrt(1.8 * 5.0))


def test_gradient_norm_damping_policy_per_block():
    """The callable policy should produce one damping value per block."""
    policy = GradientNormDamping(damping_scale=0.8, min_damping=0.5)
    block_gradients = (
        [torch_tensor([3.0, 4.0])],
        [torch_tensor([0.0]), torch_tensor([0.0])],
    )

    assert policy.per_block(block_gradients) == pytest.approx((2.0, 0.5))


@pytest.mark.parametrize("damping_scale,min_damping", [(-1.0, 0.0), (1.0, -1.0)])
def test_gradient_norm_damping_policy_raises_on_negative_inputs(
    damping_scale: float, min_damping: float
):
    """The callable policy should reject negative hyperparameters."""
    with pytest.raises(ValueError):
        GradientNormDamping(damping_scale=damping_scale, min_damping=min_damping)
