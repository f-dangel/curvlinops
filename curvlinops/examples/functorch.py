"""Contains functorch functionality for the examples."""

from math import sqrt
from typing import Iterable, List, Tuple

from functorch import grad, hessian, jvp, make_functional, vmap
from torch import Tensor, cat, einsum
from torch.func import jacrev
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
    """Compute the Hessian with functorch.

    Args:
        model_func: A function that maps the mini-batch input X to predictions.
            Could be a PyTorch module representing a neural network.
        loss_func: Loss function criterion. Maps predictions and mini-batch labels
            to a scalar value.
        params: List of differentiable parameters used by the prediction function.
        data: Source from which mini-batches can be drawn, for instance a list of
            mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``.

    Returns:
        Square matrix containing the Hessian.
    """
    # convert modules to functions
    model_fn, _ = make_functional(model_func)
    loss_fn, loss_fn_params = make_functional(loss_func)

    X, y = _concatenate_batches(data)

    def loss(X: Tensor, y: Tensor, params: Tuple[Tensor]) -> Tensor:
        """Compute the loss given a mini-batch and the neural network parameters.

        # noqa: DAR101
        # noqa: DAR201
        """
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
    """Compute the GGN with functorch.

    The GGN is the Hessian when the model is replaced by its linearization.

    Args:
        model_func: A function that maps the mini-batch input X to predictions.
            Could be a PyTorch module representing a neural network.
        loss_func: Loss function criterion. Maps predictions and mini-batch labels
            to a scalar value.
        params: List of differentiable parameters used by the prediction function.
        data: Source from which mini-batches can be drawn, for instance a list of
            mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``.

    Returns:
        Square matrix containing the GGN.
    """
    # convert modules to functions
    model_fn, _ = make_functional(model_func)
    loss_fn, loss_fn_params = make_functional(loss_func)

    X, y = _concatenate_batches(data)

    def linearized_model(
        anchor: Tuple[Tensor], params: Tuple[Tensor], X: Tensor
    ) -> Tensor:
        """Evaluate the model at params, using its linearization around anchor.

        # noqa: DAR101
        # noqa: DAR201
        """

        def model_fn_params_only(params: Tuple[Tensor]) -> Tensor:
            return model_fn(params, X)

        diff = tuple(p - a for p, a in zip(params, anchor))
        model_at_anchor, jvp_diff = jvp(model_fn_params_only, (anchor,), (diff,))

        return model_at_anchor + jvp_diff

    def linearized_loss(
        X: Tensor, y: Tensor, anchor: Tuple[Tensor], params: Tuple[Tensor]
    ) -> Tensor:
        """Compute the loss given a mini-batch under a linearized NN around anchor.

        # noqa: DAR101

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
    """Compute the gradient with functorch.

    Args:
        model_func: A function that maps the mini-batch input X to predictions.
            Could be a PyTorch module representing a neural network.
        loss_func: Loss function criterion. Maps predictions and mini-batch labels
            to a scalar value.
        params: List of differentiable parameters used by the prediction function.
        data: Source from which mini-batches can be drawn, for instance a list of
            mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``.

    Returns:
        Gradient in same format as the parameters.
    """
    # convert modules to functions
    model_fn, _ = make_functional(model_func)
    loss_fn, loss_fn_params = make_functional(loss_func)

    X, y = _concatenate_batches(data)

    def loss(X: Tensor, y: Tensor, params: Tuple[Tensor]) -> Tensor:
        """Compute the loss given a mini-batch and the neural network parameters.

        # noqa: DAR101
        # noqa: DAR201
        """
        output = model_fn(params, X)
        return loss_fn(loss_fn_params, output, y)

    params_argnum = 2
    grad_fn = grad(loss, argnums=params_argnum)

    return grad_fn(X, y, params)


def functorch_empirical_fisher(
    model_func: Module,
    loss_func: Module,
    params: List[Tensor],
    data: Iterable[Tuple[Tensor, Tensor]],
) -> Tensor:
    """Compute the empirical Fisher with functorch.

    Args:
        model_func: A function that maps the mini-batch input X to predictions.
            Could be a PyTorch module representing a neural network.
        loss_func: Loss function criterion. Maps predictions and mini-batch labels
            to a scalar value.
        params: List of differentiable parameters used by the prediction function.
        data: Source from which mini-batches can be drawn, for instance a list of
            mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``.

    Returns:
        Square matrix containing the empirical Fisher.

    Raises:
        ValueError: If the loss function's reduction cannot be determined.
    """
    # convert modules to functions
    model_fn, _ = make_functional(model_func)
    loss_fn, loss_fn_params = make_functional(loss_func)

    X, y = _concatenate_batches(data)

    # compute batched gradients
    def loss_n(X_n: Tensor, y_n: Tensor, params: List[Tensor]) -> Tensor:
        """Compute the gradient for a single sample.

        # noqa: DAR101
        # noqa: DAR201
        """
        output = model_fn(params, X_n)
        return loss_fn(loss_fn_params, output, y_n)

    params_argnum = 2
    batch_grad_fn = vmap(grad(loss_n, argnums=params_argnum))

    N = X.shape[0]
    params_replicated = [p.unsqueeze(0).expand(N, *(p.dim() * [-1])) for p in params]

    batch_grad = batch_grad_fn(X, y, params_replicated)
    batch_grad = cat([bg.flatten(start_dim=1) for bg in batch_grad], dim=1)

    if loss_func.reduction == "sum":
        normalization = 1
    elif loss_func.reduction == "mean":
        normalization = N
    else:
        raise ValueError("Cannot detect reduction method from loss function.")

    return 1 / normalization * einsum("ni,nj->ij", batch_grad, batch_grad)


def functorch_jacobian(
    model_func: Module,
    params: List[Tensor],
    data: Iterable[Tuple[Tensor, Tensor]],
) -> Tensor:
    """Compute the Jacobian with functorch.

    Args:
        model_func: A function that maps the mini-batch input X to predictions.
            Could be a PyTorch module representing a neural network.
        params: List of differentiable parameters used by the prediction function.
        data: Source from which mini-batches can be drawn, for instance a list of
            mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``.

    Returns:
        Matrix containing the Jacobian. Has shape ``[N * C, D]`` where ``D`` is the
        total number of parameters, ``N`` the total number of data points, and ``C``
        the model's output space dimension.
    """
    model_fn, _ = make_functional(model_func)
    X, _ = _concatenate_batches(data)

    def model_fn_params_only(params: Tuple[Tensor]) -> Tensor:
        return model_fn(params, X)

    # concatenate over flattened parameters and flattened outputs
    jac = jacrev(model_fn_params_only)(params)
    jac = [j.flatten(start_dim=-p.dim()) for j, p in zip(jac, params)]
    jac = cat(jac, dim=-1).flatten(end_dim=-2)

    return jac


def _concatenate_batches(
    data: Iterable[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, Tensor]:
    """Concatenate all batches in the dataset along the batch dimension.

    Args:
        data: A dataloader or iterable of batches.

    Returns:
        Concatenated model inputs.
        Concatenated targets.
    """
    X, y = list(zip(*list(data)))
    return cat(X), cat(y)
