"""Shared math for KFAC/EKFAC Kronecker factor computation.

Pure functions for preparing layer inputs/outputs and computing loss corrections,
shared between the hook-based and FX-based backends.

Weight sharing format
---------------------
KFAC treats every supported layer as a linear map applied identically across
zero or more *weight-sharing* positions (spatial locations for Conv2d, sequence
positions for Linear with matrix or higher-dimensional features). The
**weight sharing format** normalises inputs and gradients to::

    [batch, shared, features]

where ``shared`` collapses all weight-sharing positions into a single axis
and ``features`` is ``d_in`` (for inputs) or ``d_out`` (for gradients).
In this layout every layer looks like ``output[b, s] = W @ input[b, s] + bias``,
so the downstream KFAC math (covariance computation, expand/reduce) is uniform.

The collapsing strategy depends on the KFAC approximation type:

* **expand**: flatten all sharing positions (``shared = prod(*sharing)``).
* **reduce**: average (inputs) or sum (gradients) over sharing positions
  (``shared = 1``).

Examples of the conversion (expand):

* **Linear** (vector input): ``[batch, d_in]`` → ``[batch, 1, d_in]``.
* **Linear** (matrix input): ``[batch, seq, d_in]`` → ``[batch, seq, d_in]``.
* **Conv2d input**: ``[batch, C_in, H, W]`` → patch extraction →
  ``[batch, O1*O2, C_in*K1*K2]``.
* **Conv2d gradient**: ``[batch, C_out, H, W]`` → channel-last + flatten →
  ``[batch, H*W, C_out]``.
"""

from typing import Any

from einops import rearrange, reduce
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
    bias_pad: int | None = None,
) -> Tensor:
    """Convert a layer's input to weight sharing format ``[batch, shared, d_in]``.

    Converts the input to ``[batch, *sharing, d_in]``, then collapses the
    sharing dimensions into a single axis (flatten for expand, mean for reduce)
    and optionally appends a constant column for joint weight+bias treatment.

    The weight-sharing normalization factor is encoded in the returned shape
    as ``x.shape[1]`` (the ``shared`` dimension).

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
        bias_pad: Value to append as a constant column for joint weight+bias
            treatment. ``1`` appends ones (usage has bias), ``0`` appends zeros
            (usage lacks bias but joint treatment is active), ``None`` means
            no padding.

    Returns:
        Tensor of shape ``[batch, shared, d_in]`` where ``shared`` is the
        number of weight-sharing positions (expand) or 1 (reduce).
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

    # Step 2: Collapse sharing dimensions into a single axis
    if kfac_approx == KFACType.REDUCE:
        x = reduce(x, "batch ... d_in -> batch 1 d_in", "mean")
    else:
        x = rearrange(x, "batch ... d_in -> batch (...) d_in")

    # Step 3: Optionally append constant column for joint weight+bias
    if bias_pad is not None:
        x = cat([x, x.new_full((*x.shape[:-1], 1), bias_pad)], dim=-1)

    return x


def grad_to_weight_sharing_format(
    g: Tensor,
    kfac_approx: str,
    layer_hyperparams: dict[str, Any] | None = None,
    num_leading_dims: int = 1,
) -> Tensor:
    """Convert a layer's output gradient to weight sharing format.

    For Conv2d layers (non-empty ``layer_hyperparams``), moves the channel
    dimension to last position. Then collapses the sharing dimensions into
    a single axis (flatten for expand, sum for reduce).

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
        Tensor of shape ``[(*leading,) batch, shared, d_out]`` where
        ``shared`` is the number of weight-sharing positions (expand)
        or 1 (reduce).
    """
    # Step 1: Convert to weight sharing format [*leading, batch, *sharing, d_out]
    if layer_hyperparams:
        # [leading..., C_out, spatial...] -> [leading..., spatial..., C_out]
        g = g.movedim(num_leading_dims, -1)

    # Step 2: Collapse sharing dimensions into single axis
    leading = {1: "", 2: "v "}[num_leading_dims]
    if kfac_approx == KFACType.REDUCE:
        g = reduce(g, f"{leading}batch ... d_out -> {leading}batch 1 d_out", "sum")
    else:
        g = rearrange(g, f"{leading}batch ... d_out -> {leading}batch (...) d_out")

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
