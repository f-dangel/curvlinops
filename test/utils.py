"""Utility functions to test `curvlinops`."""

from itertools import product
from typing import Iterable, List, Tuple

from numpy import eye, ndarray
from torch import Tensor, cat, cuda, device, from_numpy, rand, randint
from torch.nn import Module, Parameter

from curvlinops import GGNLinearOperator


def get_available_devices():
    """Return CPU and, if present, GPU device.

    Returns:
        [device]: Available devices for `torch`.
    """
    devices = [device("cpu")]

    if cuda.is_available():
        devices.append(device("cuda"))

    return devices


def classification_targets(size, num_classes):
    """Create random targets for classes 0, ..., `num_classes - 1`."""
    return randint(size=size, low=0, high=num_classes)


def regression_targets(size):
    """Create random targets for regression."""
    return rand(*size)


def ggn_block_diagonal(
    model: Module,
    loss_func: Module,
    params: List[Parameter],
    data: Iterable[Tuple[Tensor, Tensor]],
    separate_weight_and_bias: bool = True,
) -> ndarray:
    """Compute the block-diagonal GGN.

    Args:
        model: The neural network.
        loss_func: The loss function.
        params: The parameters w.r.t. which the GGN block-diagonals will be computed.
        data: A data loader.
        separate_weight_and_bias: Whether to treat weight and bias of a layer as
            separate blocks in the block-diagonal GGN. Default: ``True``.

    Returns:
        The block-diagonal GGN.
    """
    # compute the full GGN then zero out the off-diagonal blocks
    ggn = GGNLinearOperator(model, loss_func, params, data)
    ggn = from_numpy(ggn @ eye(ggn.shape[1]))
    sizes = [p.numel() for p in params]
    # ggn_blocks[i, j] corresponds to the block of (params[i], params[j])
    ggn_blocks = [list(block.split(sizes, dim=1)) for block in ggn.split(sizes, dim=0)]

    # find out which blocks to keep
    num_params = len(params)
    keep = [(i, i) for i in range(num_params)]
    param_ids = [p.data_ptr() for p in params]

    # keep blocks corresponding to jointly-treated weights and biases
    if not separate_weight_and_bias:
        # find all layers with weight and bias
        has_weight_and_bias = [
            mod
            for mod in model.modules()
            if hasattr(mod, "weight") and hasattr(mod, "bias") and mod.bias is not None
        ]
        # only keep those whose parameters are included
        has_weight_and_bias = [
            mod
            for mod in has_weight_and_bias
            if mod.weight.data_ptr() in param_ids and mod.bias.data_ptr() in param_ids
        ]
        for mod in has_weight_and_bias:
            w_pos = param_ids.index(mod.weight.data_ptr())
            b_pos = param_ids.index(mod.bias.data_ptr())
            keep.extend([(w_pos, b_pos), (b_pos, w_pos)])

    for i, j in product(range(num_params), range(num_params)):
        if (i, j) not in keep:
            ggn_blocks[i][j].zero_()

    # concatenate all blocks
    return cat([cat(row_blocks, dim=1) for row_blocks in ggn_blocks], dim=0).numpy()
