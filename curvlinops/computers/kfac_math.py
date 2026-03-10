"""Shared math for KFAC/EKFAC Kronecker factor computation.

Pure functions for preparing layer inputs/outputs and computing loss corrections,
shared between the hook-based and FX-based backends.

Weight sharing format
---------------------
KFAC treats every supported layer as a linear map applied identically across
zero or more *weight-sharing* positions (spatial locations for Conv2d, sequence
positions for Linear with matrix or higher-dimensional features). The
**weight sharing format** normalises inputs and gradients to::

    [batch, *sharing, features]

where ``*sharing`` are the weight-sharing dimensions and ``features`` is
``d_in`` (for inputs) or ``d_out`` (for gradients). In this layout every layer
looks like ``output[b, s] = W @ input[b, s] + bias``, so the downstream KFAC
math (covariance computation, expand/reduce) is uniform.

Examples of the conversion:

* **Linear** (vector input): ``[batch, d_in]`` — already in format (no sharing dims).
* **Linear** (matrix input): ``[batch, seq, d_in]`` — already in format.
* **Conv2d input**: ``[batch, C_in, H, W]`` → patch extraction →
  ``[batch, O1*O2, C_in*K1*K2]``.
* **Conv2d gradient**: ``[batch, C_out, H, W]`` → channel-last →
  ``[batch, H, W, C_out]``.
"""

from typing import Any

from einops import reduce
from torch import Tensor, cat

from curvlinops.kfac_utils import (
    KFACType,
    extract_averaged_patches,
    extract_patches,
)


def input_to_weight_sharing_format(
    x: Tensor,
    kfac_approx: str,
    layer_hyperparams: dict[str, Any] | None = None,
    append_ones_for_bias: bool = False,
) -> tuple[Tensor, float]:
    """Convert a layer's input to weight sharing format and prepare for covariance.

    Converts the input to ``[batch, *sharing, d_in]``, then collapses the
    sharing dimensions (flatten for expand, mean for reduce) and optionally
    appends a ones column for joint weight+bias treatment.

    For Conv2d layers, pass the convolution hyperparameters (``kernel_size``,
    ``stride``, ``padding``, ``dilation``, ``groups``) to trigger patch
    extraction. For Linear layers, pass ``None`` or an empty dict (the input
    is already in weight sharing format).

    Args:
        x: Layer input. Linear: ``[batch, (*sharing,) d_in]``.
            Conv2d: ``[batch, C_in, H, W]``.
        kfac_approx: KFAC approximation type (``KFACType.EXPAND`` or
            ``KFACType.REDUCE``).
        layer_hyperparams: Convolution hyperparameters (``kernel_size``,
            ``stride``, ``padding``, ``dilation``, ``groups``). Empty or
            ``None`` for Linear layers. Follows the IO collector convention.
        append_ones_for_bias: Whether to append a ones column for joint
            weight+bias treatment.

    Returns:
        ``(x_prepared, scale)`` where ``x_prepared`` has shape ``[B, d_in]``
        and ``scale`` is the weight-sharing normalization factor.
    """
    # Step 1: Convert to weight sharing format [batch, *sharing, d_in]
    if layer_hyperparams:
        patch_extractor_fn = {
            KFACType.EXPAND: extract_patches,
            KFACType.REDUCE: extract_averaged_patches,
        }[kfac_approx]
        x = patch_extractor_fn(
            x,
            layer_hyperparams["kernel_size"],
            layer_hyperparams["stride"],
            layer_hyperparams["padding"],
            layer_hyperparams["dilation"],
            layer_hyperparams["groups"],
        )

    # Step 2: Collapse sharing dimensions
    if kfac_approx == KFACType.EXPAND:
        scale = x.shape[1:-1].numel()
        x = x.flatten(end_dim=-2)  # [batch, *sharing, d_in] -> [(batch *sharing), d_in]
    else:
        scale = 1.0
        x = reduce(x, "batch ... d_in -> batch d_in", "mean")

    # Step 3: Optionally append ones column for joint weight+bias
    if append_ones_for_bias:
        x = cat([x, x.new_ones(x.shape[0], 1)], dim=1)

    return x, scale


def grad_to_weight_sharing_format(
    g: Tensor,
    kfac_approx: str,
    layer_hyperparams: dict[str, Any] | None = None,
    num_leading_dims: int = 1,
) -> Tensor:
    """Convert a layer's output gradient to weight sharing format and prepare for covariance.

    For Conv2d layers (non-empty ``layer_hyperparams``), moves the channel
    dimension to last position. Then collapses the sharing dimensions
    (flatten for expand, sum for reduce).

    Args:
        g: Output gradient. Hooks: ``[batch, ...]``.
            FX: ``[v, batch, ...]``.
        kfac_approx: KFAC approximation type (``KFACType.EXPAND`` or
            ``KFACType.REDUCE``).
        layer_hyperparams: Convolution hyperparameters. Empty or ``None``
            for Linear layers. Follows the IO collector convention.
        num_leading_dims: Number of leading dims to preserve (1 for hooks,
            2 for FX batched grads).

    Returns:
        Prepared gradient with shape ``[(*leading,) B, d_out]``.
    """
    # Step 1: Convert to weight sharing format [*leading, batch, *sharing, d_out]
    if layer_hyperparams:
        # [leading..., C_out, spatial...] -> [leading..., spatial..., C_out]
        g = g.movedim(num_leading_dims, -1)

    # Step 2: Collapse sharing dimensions
    if kfac_approx == KFACType.EXPAND:
        # hooks: [batch, s1, s2, d] -> [(batch s1 s2), d]
        # FX:    [v, batch, s1, s2, d] -> [v, (batch s1 s2), d]
        leading = g.shape[: num_leading_dims - 1]
        d_out = g.shape[-1]
        g = g.reshape(*leading, -1, d_out)
    else:
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
