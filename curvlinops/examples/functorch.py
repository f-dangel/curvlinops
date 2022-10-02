"""Contains functorch functionality for the examples."""

from math import sqrt
from typing import Iterable, List, Tuple

from functorch import grad, hessian, jvp, make_functional
from torch import Tensor, cat
from torch.nn import Module


def blocks_to_matrix(blocks: Tuple[Tuple[Tensor]]) -> Tensor:
    """Convert a block representation into a matrix.

    Assumes the diagonal blocks to be quadratic to automatically detect their dimension.

    Args:
        blocks: Nested tuple of tensors that contains the ``(i, j)``th matrix
            block at index ``[i, j]``.

    Returns:
        Two-dimensional matrix with concatenated and flattened blocks.
    """
    num_blocks = len(blocks)
    row_blocks = []

    for idx in range(num_blocks):
        block_num_rows = int(sqrt(blocks[idx][idx].numel()))
        col_blocks = [b.reshape(block_num_rows, -1) for b in blocks[idx]]
        row_blocks.append(cat(col_blocks, dim=1))

    return cat(row_blocks)


def functorch_hessian(
    model_func: Module,
    loss_func: Module,
    params: List[Tensor],
    data: Iterable[Tuple[Tensor, Tensor]],
) -> Tensor:
    """Compute the Hessian with functorch."""
    # convert modules to functions
    model_fn, _ = make_functional(model_func)
    loss_fn, loss_fn_params = make_functional(loss_func)

    # concatenate batches
    X, y = list(zip(*list(data)))
    X, y = cat(X), cat(y)

    def loss(X: Tensor, y: Tensor, params: Tuple[Tensor]) -> Tensor:
        """Compute the loss given a mini-batch and the neural network parameters."""
        output = model_fn(params, X)
        return loss_fn(loss_fn_params, output, y)

    params_argnum = 2
    hessian_fn = hessian(loss, argnums=params_argnum)

    return blocks_to_matrix(hessian_fn(X, y, params))


def functorch_ggn(
    model_func: Module,
    loss_func: Module,
    params: List[Tensor],
    data: Iterable[Tuple[Tensor, Tensor]],
) -> Tensor:
    """Compute the GGN with functorch."""
    # convert modules to functions
    model_fn, _ = make_functional(model_func)
    loss_fn, loss_fn_params = make_functional(loss_func)

    # concatenate batches
    X, y = list(zip(*list(data)))
    X, y = cat(X), cat(y)

    def linearized_model(
        anchor: Tuple[Tensor], params: Tuple[Tensor], X: Tensor
    ) -> Tensor:
        """Evaluate the model at params, using its linearization around anchor."""

        def model_fn_params_only(params: Tuple[Tensor]) -> Tensor:
            return model_fn(params, X)

        diff = tuple(p - a for p, a in zip(params, anchor))
        model_at_anchor, jvp_diff = jvp(model_fn_params_only, (anchor,), (diff,))

        return model_at_anchor + jvp_diff

    def linearized_loss(
        X: Tensor, y: Tensor, anchor: Tuple[Tensor], params: Tuple[Tensor]
    ) -> Tensor:
        """Compute the loss given a mini-batch under a linearized NN around anchor.

        Returns:
            f(X, θ₀) + (J_θ₀ f(X, θ₀)) @ (θ - θ₀) with f the neural network, θ₀ the anchor
            point of the linearization, and θ the evaluation point.
        """
        output = linearized_model(anchor, params, X)
        return loss_fn(loss_fn_params, output, y)

    params_argnum = 3
    ggn_fn = hessian(linearized_loss, argnums=params_argnum)

    anchor = tuple(p.clone() for p in params)

    return blocks_to_matrix(ggn_fn(X, y, anchor, params))


def functorch_gradient(
    model_func: Module,
    loss_func: Module,
    params: List[Tensor],
    data: Iterable[Tuple[Tensor, Tensor]],
) -> Tuple[Tensor]:
    """Compute the gradient with functorch."""
    # convert modules to functions
    model_fn, _ = make_functional(model_func)
    loss_fn, loss_fn_params = make_functional(loss_func)

    # concatenate batches
    X, y = list(zip(*list(data)))
    X, y = cat(X), cat(y)

    def loss(X: Tensor, y: Tensor, params: Tuple[Tensor]) -> Tensor:
        """Compute the loss given a mini-batch and the neural network parameters."""
        output = model_fn(params, X)
        return loss_fn(loss_fn_params, output, y)

    params_argnum = 2
    grad_fn = grad(loss, argnums=params_argnum)

    return grad_fn(X, y, params)
