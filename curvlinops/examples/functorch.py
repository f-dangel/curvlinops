"""Contains functorch functionality for the examples."""

from collections.abc import MutableMapping
from math import sqrt
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, cat, stack
from torch.func import functional_call, grad, hessian, jacrev, jvp
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss


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
    data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
    input_key: Optional[str] = None,
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
        input_key: Key to obtain the input tensor when ``X`` is a dict-like object.

    Returns:
        Square matrix containing the Hessian.
    """
    (dev,) = {p.device for p in params}
    X, y = _concatenate_batches(data, input_key, device=dev)
    params_dict = _make_params_dict(model_func, params)

    def loss(
        X: Union[Tensor, MutableMapping], y: Tensor, params_dict: Dict[str, Tensor]
    ) -> Tensor:
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
    data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
    input_key: Optional[str] = None,
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
        input_key: Key to obtain the input tensor when ``X`` is a dict-like object.

    Returns:
        Square matrix containing the GGN.
    """
    (dev,) = {p.device for p in params}
    X, y = _concatenate_batches(data, input_key, device=dev)
    params_dict = _make_params_dict(model_func, params)

    def linearized_model(
        anchor_dict: Dict[str, Tensor],
        params_dict: Dict[str, Tensor],
        X: Union[Tensor, MutableMapping],
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
        X: Union[Tensor, MutableMapping],
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
    data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
    input_key: Optional[str] = None,
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
        input_key: Key to obtain the input tensor when ``X`` is a dict-like object.

    Returns:
        Gradient in same format as the parameters.
    """
    (dev,) = {p.device for p in params}
    X, y = _concatenate_batches(data, input_key, device=dev)
    params_dict = _make_params_dict(model_func, params)

    def loss(
        X: Union[Tensor, MutableMapping], y: Tensor, params_dict: Dict[str, Tensor]
    ) -> Tensor:
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
    data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
    input_key: Optional[str] = None,
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
        input_key: Key to obtain the input tensor when ``X`` is a dict-like object.

    Returns:
        Square matrix containing the empirical Fisher.
    """
    (dev,) = {p.device for p in params}
    X, y = _concatenate_batches(data, input_key, device=dev)
    params_dict = _make_params_dict(model_func, params)

    def losses(
        X: Union[Tensor, MutableMapping], y: Tensor, params_dict: Dict[str, Tensor]
    ) -> Tensor:
        """Compute all elementary losses without reduction.

        An elementary loss results from a scalar entry of `y`.

        Args:
            X: Mini-batch input.
            y: Mini-batch labels.
            params_dict: Dictionary of parameters.

        Returns:
            1d tensor containing all elementary losses.
        """
        output = functional_call(model_func, params_dict, X)

        flatten_output = {
            MSELoss: "batch ... d_out -> (batch ... d_out)",
            BCEWithLogitsLoss: "batch ... d_out -> (batch ... d_out)",
            CrossEntropyLoss: "batch c ... -> (batch ...) c",
        }[loss_func.__class__]
        flatten_y = "... -> (...)"
        output_flat, y_flat = rearrange(output, flatten_output), rearrange(y, flatten_y)

        return stack(
            [
                functional_call(loss_func, {}, (o_el, y_el))
                for o_el, y_el in zip(output_flat, y_flat)
            ]
        )

    params_argnum = 2
    jac = jacrev(losses, argnums=params_argnum)(X, y, params_dict)
    jac = cat([j.flatten(start_dim=1) for j in jac.values()], dim=1)

    # the losses over which the expectation is taken
    num_losses = {
        CrossEntropyLoss: y.numel(),
        MSELoss: y.shape[:-1].numel(),
        BCEWithLogitsLoss: y.shape[:-1].numel(),
    }[loss_func.__class__]
    num_params = sum(p.numel() for p in params)

    # the losses which model the same random variable
    grouped_losses = y.numel() // num_losses
    jac = jac.reshape(num_losses, grouped_losses, num_params).sum(1)
    if (
        isinstance(loss_func, (MSELoss, BCEWithLogitsLoss))
        and loss_func.reduction == "mean"
    ):
        jac /= sqrt(grouped_losses)

    normalization = {"sum": 1, "mean": num_losses}[loss_func.reduction]
    return jac.T @ jac / normalization


def functorch_jacobian(
    model_func: Module,
    params: List[Tensor],
    data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
    input_key: Optional[str] = None,
) -> Tensor:
    """Compute the Jacobian with functorch.

    Args:
        model_func: A function that maps the mini-batch input X to predictions.
            Could be a PyTorch module representing a neural network.
        params: List of differentiable parameters used by the prediction function.
        data: Source from which mini-batches can be drawn, for instance a list of
            mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``.
        input_key: Key to obtain the input tensor when ``X`` is a dict-like object.

    Returns:
        Matrix containing the Jacobian. Has shape ``[N * C, D]`` where ``D`` is the
        total number of parameters, ``N`` the total number of data points, and ``C``
        the model's output space dimension.
    """
    (dev,) = {p.device for p in params}
    X, _ = _concatenate_batches(data, input_key, device=dev)
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
    data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
    input_key: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """Concatenate all batches in the dataset along the batch dimension.

    Args:
        data: A dataloader or iterable of batches.
        input_key: Key to obtain the input tensor when ``X`` is a dict-like object.
        device: The device the data should live in.

    Returns:
        Concatenated model inputs.
        Concatenated targets.

    Raises:
        ValueError: If ``X`` in ``data`` is a dict-like object and ``input_key`` is
            not provided.
    """
    X, y = list(zip(*list(data)))
    device = y[0].device if device is None else device
    y = cat(y).to(device)

    if isinstance(X[0], MutableMapping) and input_key is None:
        raise ValueError("input_key must be provided for dict-like X!")

    if isinstance(X[0], Tensor):
        return cat(X).to(device), y
    else:
        X = {input_key: cat([d[input_key] for d in X]).to(device)}
        return X, y


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
