"""Contains functorch functionality for the examples."""

from math import sqrt
from typing import Dict, Iterable, List, Tuple

from torch import Tensor, cat, einsum
from torch.func import functional_call, grad, hessian, jacrev, jvp, vmap
from torch.nn import Module


def blocks_to_matrix(blocks: Dict[str, Dict[str, Tensor]]) -> Tensor:
    """Convert a block representation into a matrix.

    Assumes the diagonal blocks to be quadratic to automatically detect their dimension.

    Args:
        blocks: Nested dictionaries with keys denoting parameter names, and blocks
            ``(i, j)`` denoting the matrix block w.r.t parameters ``[i, j]``.

    Returns:
        Two-dimensional matrix with concatenated and flattened blocks.
    """
    row_blocks = []

    param_names = list(blocks.keys())
    for name, param_block in blocks.items():
        num_rows = int(sqrt(param_block[name].numel()))
        col_blocks = [param_block[n].reshape(num_rows, -1) for n in param_names]
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
    X, y = _concatenate_batches(data)
    params_dict = _make_params_dict(model_func, params)

    def loss(X: Tensor, y: Tensor, params_dict: Dict[str, Tensor]) -> Tensor:
        """Compute the loss given a mini-batch and the neural network parameters.

        # noqa: DAR101
        # noqa: DAR201
        """
        output = functional_call(model_func, params_dict, X)
        return functional_call(loss_func, {}, (output, y))

    params_argnum = 2
    hessian_fn = hessian(loss, argnums=params_argnum)

    return blocks_to_matrix(hessian_fn(X, y, params_dict))


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
    X, y = _concatenate_batches(data)
    params_dict = _make_params_dict(model_func, params)

    def linearized_model(
        anchor_dict: Dict[str, Tensor], params_dict: Dict[str, Tensor], X: Tensor
    ) -> Tensor:
        """Evaluate the model at params, using its linearization around anchor.

        # noqa: DAR101
        # noqa: DAR201
        """

        def model_fn_params_only(params_dict: Dict[str, Tensor]) -> Tensor:
            return functional_call(model_func, params_dict, X)

        diff_dict = {n: params_dict[n] - anchor_dict[n] for n in params_dict}
        model_at_anchor, jvp_diff = jvp(
            model_fn_params_only, (anchor_dict,), (diff_dict,)
        )

        return model_at_anchor + jvp_diff

    def linearized_loss(
        X: Tensor,
        y: Tensor,
        anchor_dict: Dict[str, Tensor],
        params_dict: Dict[str, Tensor],
    ) -> Tensor:
        """Compute the loss given a mini-batch under a linearized NN around anchor.

        # noqa: DAR101

        Returns:
            f(X, θ₀) + (J_θ₀ f(X, θ₀)) @ (θ - θ₀) with f the neural network, θ₀ the anchor
            point of the linearization, and θ the evaluation point.
        """
        output = linearized_model(anchor_dict, params_dict, X)
        return functional_call(loss_func, {}, (output, y))

    params_argnum = 3
    ggn_fn = hessian(linearized_loss, argnums=params_argnum)

    anchor_dict = {n: p.clone() for n, p in params_dict.items()}

    return blocks_to_matrix(ggn_fn(X, y, anchor_dict, params_dict))


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
    X, y = _concatenate_batches(data)
    params_dict = _make_params_dict(model_func, params)

    def loss(X: Tensor, y: Tensor, params_dict: Dict[str, Tensor]) -> Tensor:
        """Compute the loss given a mini-batch and the neural network parameters.

        # noqa: DAR101
        # noqa: DAR201
        """
        output = functional_call(model_func, params_dict, X)
        return functional_call(loss_func, {}, (output, y))

    params_argnum = 2
    grad_fn = grad(loss, argnums=params_argnum)

    return tuple(grad_fn(X, y, params_dict).values())


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
    X, y = _concatenate_batches(data)
    params_dict = _make_params_dict(model_func, params)

    # compute batched gradients
    def loss_n(X_n: Tensor, y_n: Tensor, params_dict: Dict[str, Tensor]) -> Tensor:
        """Compute the gradient for a single sample.

        # noqa: DAR101
        # noqa: DAR201
        """
        output = functional_call(model_func, params_dict, X_n)
        return functional_call(loss_func, {}, (output, y_n))

    params_argnum = 2
    batch_grad_fn = vmap(grad(loss_n, argnums=params_argnum))

    N = X.shape[0]
    params_replicated_dict = {
        name: p.unsqueeze(0).expand(N, *(p.dim() * [-1]))
        for name, p in params_dict.items()
    }

    batch_grad = batch_grad_fn(X, y, params_replicated_dict)
    batch_grad = cat([bg.flatten(start_dim=1) for bg in batch_grad.values()], dim=1)

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
    X, _ = _concatenate_batches(data)
    params_dict = _make_params_dict(model_func, params)

    def model_fn_params_only(params_dict: Dict[str, Tensor]) -> Tensor:
        return functional_call(model_func, params_dict, X)

    # concatenate over flattened parameters and flattened outputs
    jac = jacrev(model_fn_params_only)(params_dict)
    jac = [
        j.flatten(start_dim=-p.dim())
        for j, p in zip(jac.values(), params_dict.values())
    ]
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


def _make_params_dict(model_func: Module, params: List[Tensor]) -> Dict[str, Tensor]:
    """Create a named dictionary for the parameter list.

    Required for ``functorch``'s ``functional_call`` API.

    Args:
        model_func: A PyTorch module representing a neural network.
        params: List of differentiable parameters used by the prediction function.

    Returns:
        Dictionary mapping parameter names to parameter tensors.
    """
    name_dict = {p.data_ptr(): name for name, p in model_func.named_parameters()}
    return {name_dict[p.data_ptr()]: p for p in params}
