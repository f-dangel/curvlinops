"""Shared math for KFAC/EKFAC Kronecker factor computation.

Pure functions for preparing layer inputs/outputs and computing loss corrections,
shared between the hook-based and FX-based backends.
"""

from einops import reduce
from torch import Tensor, cat

from curvlinops.kfac_utils import (
    KFACType,
    extract_averaged_patches,
    extract_patches,
)


def conv2d_input_to_kfac_format(
    x: Tensor,
    kfac_approx: str,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...] | str,
    dilation: tuple[int, ...],
    groups: int,
) -> Tensor:
    """Convert a Conv2d layer's input to KFAC format ``[batch, ..., d_in]``.

    Extracts patches (expand) or averaged patches (reduce) from the spatial
    input, producing a tensor with the feature dimension last.

    Args:
        x: Conv2d layer input ``[batch, C_in, H, W]``.
        kfac_approx: KFAC approximation type (``KFACType.EXPAND`` or
            ``KFACType.REDUCE``).
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        groups: Convolution groups.

    Returns:
        Tensor with shape ``[batch, ..., d_in]`` where ``...`` are
        weight-sharing dimensions (spatial patches for expand, empty for reduce).
    """
    patch_extractor_fn = {
        KFACType.EXPAND: extract_patches,
        KFACType.REDUCE: extract_averaged_patches,
    }[kfac_approx]
    return patch_extractor_fn(x, kernel_size, stride, padding, dilation, groups)


def conv2d_grad_to_kfac_format(g: Tensor, num_leading_dims: int = 1) -> Tensor:
    """Move Conv2d channel dimension to last position.

    Converts from ``[..., batch, C_out, H, W]`` to ``[..., batch, H, W, C_out]``
    so the feature dimension is last, matching KFAC's expected layout.

    Args:
        g: Conv2d gradient with channel dim at position ``num_leading_dims``.
        num_leading_dims: Number of leading dims to preserve (1 for hooks,
            2 for FX batched grads).

    Returns:
        Gradient with channel dim moved to last position.
    """
    # [leading..., C_out, spatial...] -> [leading..., spatial..., C_out]
    return g.movedim(num_leading_dims, -1)


def prepare_layer_input(
    x: Tensor,
    kfac_approx: str,
    append_ones_for_bias: bool = False,
) -> tuple[Tensor, float]:
    """Prepare a layer's input for KFAC input covariance computation.

    Expects ``x`` in shape ``[batch, ..., d_in]`` where ``...`` are optional
    weight-sharing dimensions. Callers must convert layer-specific formats
    (e.g. Conv2d spatial inputs) to this shape before calling.

    Handles KFAC-expand/reduce rearrangement and optional ones-column for
    joint weight+bias treatment.

    Args:
        x: Layer input ``[batch, ..., d_in]``. For fully-connected layers
            without weight sharing, shape is simply ``[batch, d_in]``.
        kfac_approx: KFAC approximation type (``KFACType.EXPAND`` or
            ``KFACType.REDUCE``).
        append_ones_for_bias: Whether to append a ones column for joint
            weight+bias treatment.

    Returns:
        ``(x_prepared, scale)`` where ``x_prepared`` has shape ``[B, d_in]``
        and ``scale`` is the weight-sharing normalization factor.
    """
    if kfac_approx == KFACType.EXPAND:
        scale = x.shape[1:-1].numel()
        x = x.flatten(end_dim=-2)  # "batch ... d_in -> (batch ...) d_in"
    else:
        scale = 1.0
        x = reduce(x, "batch ... d_in -> batch d_in", "mean")

    if append_ones_for_bias:
        x = cat([x, x.new_ones(x.shape[0], 1)], dim=1)

    return x, scale


def prepare_grad_output(
    g: Tensor,
    kfac_approx: str,
    num_leading_dims: int = 1,
) -> Tensor:
    """Prepare a layer's output gradient for KFAC gradient covariance computation.

    Expects ``g`` in shape ``[..., batch, ..., d_out]`` where the first ``...``
    are leading dimensions and the second ``...`` are weight-sharing dimensions,
    with the feature dimension last. Callers must convert layer-specific formats
    (e.g. Conv2d channel-first layout) before calling.

    Args:
        g: Output gradient. Hooks: ``[batch, ..., d_out]``.
            FX: ``[v, batch, ..., d_out]``.
        kfac_approx: KFAC approximation type (``KFACType.EXPAND`` or
            ``KFACType.REDUCE``).
        num_leading_dims: Number of leading dims to preserve (1 for hooks,
            2 for FX batched grads).

    Returns:
        Prepared gradient with spatial dims handled.
    """
    if kfac_approx == KFACType.EXPAND:
        # Flatten spatial dims into the last leading dim
        # hooks: [batch, s1, s2, d] -> [(batch s1 s2), d]
        # FX:    [v, batch, s1, s2, d] -> [v, (batch s1 s2), d]
        leading = g.shape[: num_leading_dims - 1]
        d_out = g.shape[-1]
        g = g.reshape(*leading, -1, d_out)
    else:
        # Sum over spatial dims
        spatial_dims = tuple(range(num_leading_dims, g.ndim - 1))
        if spatial_dims:
            g = g.sum(dim=spatial_dims)

    return g


def compute_loss_correction(
    batch_size: int,
    num_per_example_loss_terms: int,
    loss_reduction: str,
    n_data: int | None = None,
) -> float:
    """Compute the loss correction factor for gradient covariances.

    For ``"sum"`` reduction, returns ``1.0`` (no correction needed).
    For ``"mean"`` reduction, returns
    ``num_loss_terms² / (num_per_example_loss_terms * n_data)`` when ``n_data``
    is given, or ``num_loss_terms² / num_per_example_loss_terms`` otherwise.

    Args:
        batch_size: Batch size.
        num_per_example_loss_terms: Number of loss terms per example.
        loss_reduction: ``"sum"`` or ``"mean"``.
        n_data: Total dataset size. When given, the ``1/N_data`` normalization
            is included in the correction. When ``None``, the caller handles
            ``N_data`` normalization separately.

    Returns:
        Scalar correction factor.
    """
    num_loss_terms = batch_size * num_per_example_loss_terms
    denominator = num_per_example_loss_terms
    if n_data is not None:
        denominator *= n_data
    return {
        "sum": 1.0,
        "mean": num_loss_terms**2 / denominator,
    }[loss_reduction]
