"""Tests for ``curvlinops.examples``."""

from collections.abc import Callable, Iterable, MutableMapping

from torch import Tensor

from curvlinops.examples import gradient_and_loss
from curvlinops.examples.functorch import functorch_gradient_and_loss
from curvlinops.utils import allclose_report


def test_gradient_and_loss(
    case: tuple[
        Callable[[Tensor], Tensor],
        Callable[[Tensor, Tensor], Tensor],
        list[Tensor],
        Iterable[tuple[Tensor | MutableMapping, Tensor]],
        Callable[[MutableMapping], int] | None,
    ],
):
    """Test the standalone ``gradient_and_loss`` against ``functorch``.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
    """
    model, loss_func, params, data, batch_size_fn = case

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
