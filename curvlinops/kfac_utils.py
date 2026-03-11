"""Utility functions specific to KFAC (patch extraction, canonical space converters).

Also defines ``FisherType`` and ``KFACType`` enums used across the KFAC codebase.
"""

from __future__ import annotations

from enum import Enum, EnumMeta

from einconv import index_pattern
from einconv.utils import get_conv_paddings
from einops import einsum, rearrange, reduce
from torch import Size, Tensor, cat, device, dtype
from torch.nn.functional import unfold
from torch.nn.modules.utils import _pair

from curvlinops._torch_base import PyTorchLinearOperator


def _has_joint_weight_and_bias(
    separate_weight_and_bias: bool, param_pos: dict[str, int]
) -> bool:
    """Check whether weight and bias are treated jointly in a KFAC block.

    Args:
        separate_weight_and_bias: Whether weight and bias are treated separately.
        param_pos: Dictionary mapping parameter names to positions.

    Returns:
        ``True`` if the block has both weight and bias and they should be joint.
    """
    return not separate_weight_and_bias and {"weight", "bias"} == set(param_pos.keys())


class MetaEnum(EnumMeta):
    """Metaclass for the Enum class for desired behavior of the ``in`` operator."""

    def __contains__(cls, item):
        """Return whether ``item`` is a valid Enum value.

        Args:
            item: Candidate value.

        Returns:
            ``True`` if ``item`` is a valid Enum value.
        """
        try:
            cls(item)
        except ValueError:
            return False
        return True


class FisherType(str, Enum, metaclass=MetaEnum):
    """Enum for the Fisher type.

    Attributes:
        TYPE2 (str): ``'type-2'`` - Type-2 Fisher, i.e. the exact Hessian of the
            loss w.r.t. the model outputs is used. This requires as many backward
            passes as the output dimension, i.e. the number of classes for
            classification.
        MC (str): ``'mc'`` - Monte-Carlo approximation of the expectation by sampling
            ``mc_samples`` labels from the model's predictive distribution.
        EMPIRICAL (str): ``'empirical'`` - Empirical gradients are used which
            corresponds to the uncentered gradient covariance, or the empirical Fisher.
        FORWARD_ONLY (str): ``'forward-only'`` - The gradient covariances will be
            identity matrices, see the FOOF method in
            `Benzing, 2022 <https://arxiv.org/abs/2201.12250>`_ or ISAAC in
            `Petersen et al., 2023 <https://arxiv.org/abs/2305.00604>`_.
    """

    TYPE2 = "type-2"
    MC = "mc"
    EMPIRICAL = "empirical"
    FORWARD_ONLY = "forward-only"


class KFACType(str, Enum, metaclass=MetaEnum):
    """Enum for the KFAC approximation type.

    KFAC-expand and KFAC-reduce are defined in
    `Eschenhagen et al., 2023 <https://arxiv.org/abs/2311.00636>`_.

    Attributes:
        EXPAND (str): ``'expand'`` - KFAC-expand approximation.
        REDUCE (str): ``'reduce'`` - KFAC-reduce approximation.
    """

    EXPAND = "expand"
    REDUCE = "reduce"


def extract_patches(
    x: Tensor,
    kernel_size: tuple[int, int] | int,
    stride: tuple[int, int] | int,
    padding: tuple[int, int] | int | str,
    dilation: tuple[int, int] | int,
    groups: int,
) -> Tensor:
    """Extract patches from the input of a 2d-convolution.

    The patches are averaged over channel groups.

    Args:
        x: Input to a 2d-convolution. Has shape ``[batch_size, C_in, I1, I2]``.
        kernel_size: The convolution's kernel size supplied as 2-tuple or integer.
        stride: The convolution's stride supplied as 2-tuple or integer.
        padding: The convolution's padding supplied as 2-tuple, integer, or string.
        dilation: The convolution's dilation supplied as 2-tuple or integer.
        groups: The number of channel groups.

    Returns:
        A tensor of shape ``[batch_size, O1 * O2, C_in // groups * K1 * K2]`` where
        each column ``[b, o1_o2, :]`` contains the flattened patch of sample ``b`` used
        for output location ``(o1, o2)``, averaged over channel groups.

    Raises:
        NotImplementedError: If ``padding`` is a string that would lead to unequal
            padding along a dimension.
    """
    if isinstance(padding, str):  # get padding as integers
        padding_as_int = []
        for k, s, d in zip(_pair(kernel_size), _pair(stride), _pair(dilation)):
            p_left, p_right = get_conv_paddings(k, s, padding, d)
            if p_left != p_right:
                raise NotImplementedError("Unequal padding not supported in unfold.")
            padding_as_int.append(p_left)
        padding = tuple(padding_as_int)

    # average channel groups
    x = rearrange(x, "b (g c_in) i1 i2 -> b g c_in i1 i2", g=groups)
    x = reduce(x, "b g c_in i1 i2 -> b c_in i1 i2", "mean")

    x_unfold = unfold(x, kernel_size, dilation=dilation, padding=padding, stride=stride)
    return rearrange(x_unfold, "b c_in_k1_k2 o1_o2 -> b o1_o2 c_in_k1_k2")


def extract_averaged_patches(
    x: Tensor,
    kernel_size: tuple[int, int] | int,
    stride: tuple[int, int] | int,
    padding: tuple[int, int] | int | str,
    dilation: tuple[int, int] | int,
    groups: int,
) -> Tensor:
    """Extract averaged patches from the input of a 2d-convolution.

    The patches are averaged over channel groups and output locations.

    Uses the tensor network formulation of convolution from
    `Dangel, 2023 <https://arxiv.org/abs/2307.02275>`_.

    Args:
        x: Input to a 2d-convolution. Has shape ``[batch_size, C_in, I1, I2]``.
        kernel_size: The convolution's kernel size supplied as 2-tuple or integer.
        stride: The convolution's stride supplied as 2-tuple or integer.
        padding: The convolution's padding supplied as 2-tuple, integer, or string.
        dilation: The convolution's dilation supplied as 2-tuple or integer.
        groups: The number of channel groups.

    Returns:
        A tensor of shape ``[batch_size, C_in // groups * K1 * K2]`` where each column
        ``[b, :]`` contains the flattened patch of sample ``b`` averaged over all output
        locations and channel groups.
    """
    # average channel groups
    x = rearrange(x, "b (g c_in) i1 i2 -> b g c_in i1 i2", g=groups)
    x = reduce(x, "b g c_in i1 i2 -> b c_in i1 i2", "mean")

    # TODO For convolutions with special structure, we don't even need to compute
    # the index pattern tensors, or can resort to contracting only slices thereof.
    # In order for this to work `einconv`'s TN simplification mechanism must first
    # be refactored to work purely symbolically. Once this is done, it will be
    # possible to do the below even more efficiently (memory and run time) for
    # structured convolutions.

    # compute index pattern tensors, average output dimension
    patterns = []
    input_sizes = x.shape[-2:]
    for i, k, s, p, d in zip(
        input_sizes,
        _pair(kernel_size),
        _pair(stride),
        (padding, padding) if isinstance(padding, str) else _pair(padding),
        _pair(dilation),
    ):
        pi = index_pattern(
            i, k, stride=s, padding=p, dilation=d, dtype=x.dtype, device=x.device
        )
        pi = reduce(pi, "k o i -> k i", "mean")
        patterns.append(pi)

    x = einsum(x, *patterns, "b c_in i1 i2, k1 i1, k2 i2 -> b c_in k1 k2")
    return rearrange(x, "b c_in k1 k2 -> b (c_in k1 k2)")


class _CanonicalizationLinearOperator(PyTorchLinearOperator):
    """Base class for canonical form transformation operators."""

    def __init__(
        self,
        param_shapes: list[Size],
        param_positions: list[dict[str, int]],
        separate_weight_and_bias: bool,
        device: device,
        dtype: dtype,
    ):
        """Initialize the canonical form transformation operator.

        Args:
            param_shapes: List of parameter shapes as Size objects.
            param_positions: List of parameter position dictionaries for each layer.
            separate_weight_and_bias: Whether to treat weights and biases separately.
            device: Device of the parameters.
            dtype: Data type of the parameters.
        """
        self._param_shapes = param_shapes
        self._device = device
        self._dtype = dtype

        self._param_positions = param_positions
        self._separate_weight_and_bias = separate_weight_and_bias

        in_shape, out_shape = self._compute_shapes()
        super().__init__(in_shape, out_shape)

    def _compute_shapes(self) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Compute input and output shapes for the transformation.

        Returns:
            Tuple of (in_shape, out_shape) where each is a list of parameter shapes.
        """
        raise NotImplementedError("Subclasses must implement _compute_shapes")

    def _compute_canonical_shapes(self) -> list[tuple[int, ...]]:
        """Compute the shapes in KFAC's canonical basis.

        Returns:
            List of shapes after canonical transformation.
        """
        canonical_shapes = []

        for param_pos in self._param_positions:
            if _has_joint_weight_and_bias(self._separate_weight_and_bias, param_pos):
                w_pos = param_pos["weight"]
                w_shape = self._param_shapes[w_pos]
                # Combined weight+bias gets flattened to 1D
                total_params = (
                    self._param_shapes[w_pos].numel() + w_shape[0]
                )  # weight + bias
                canonical_shapes.append((total_params,))
            else:
                # Handle separate weight and bias
                for p_name in param_pos:
                    pos = param_pos[p_name]
                    # Each parameter gets flattened to 1D
                    canonical_shapes.append((self._param_shapes[pos].numel(),))

        return canonical_shapes

    @property
    def device(self):
        """Return the stored device.

        Returns:
            The device of the parameters.
        """
        return self._device

    @property
    def dtype(self):
        """Return the stored dtype.

        Returns:
            The dtype of the parameters.
        """
        return self._dtype


class ToCanonicalLinearOperator(_CanonicalizationLinearOperator):
    """Linear operator that transforms parameters from original to canonical form.

    Canonical form orders parameters by layer, with proper grouping and flattening.
    This is the adjoint of FromCanonicalLinearOperator.
    """

    def _compute_shapes(self) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Compute input and output shapes for the transformation.

        Returns:
            Tuple of (in_shape, out_shape) for original to canonical transformation.
        """
        in_shape = [tuple(shape) for shape in self._param_shapes]
        out_shape = self._compute_canonical_shapes()
        return in_shape, out_shape

    def _matmat(self, M: list[Tensor]) -> list[Tensor]:
        """Transform parameter tensors to canonical form.

        Args:
            M: Parameter tensors in original order.

        Returns:
            Parameter tensors in canonical form (flattened and reordered).
        """
        canonical_M = []

        for param_pos in self._param_positions:
            if _has_joint_weight_and_bias(self._separate_weight_and_bias, param_pos):
                w_pos, b_pos = param_pos["weight"], param_pos["bias"]
                # Flatten weight tensor into matrix and concatenate bias
                w_flat = M[w_pos].flatten(start_dim=1, end_dim=-2)
                # Add bias as additional row
                combined = cat([w_flat, M[b_pos].unsqueeze(1)], dim=1)
                # Flatten parameter space dimension
                canonical_M.append(combined.flatten(end_dim=-2))
            else:
                # Handle separate weight and bias
                for p_name in param_pos:
                    pos = param_pos[p_name]
                    canonical_M.append(M[pos].flatten(end_dim=-2))

        return canonical_M

    def _adjoint(self) -> FromCanonicalLinearOperator:
        """Return the adjoint transformation operator.

        Returns:
            Linear operator that transforms from canonical to parameter form.
        """
        return FromCanonicalLinearOperator(
            self._param_shapes,
            self._param_positions,
            self._separate_weight_and_bias,
            self._device,
            self._dtype,
        )


class FromCanonicalLinearOperator(_CanonicalizationLinearOperator):
    """Linear operator that transforms parameters from canonical to original form.

    This is the adjoint of ToCanonicalLinearOperator.
    """

    def _compute_shapes(self) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
        """Compute input and output shapes for the transformation.

        Returns:
            Tuple of (in_shape, out_shape) for canonical to original transformation.
        """
        out_shape = [tuple(shape) for shape in self._param_shapes]
        in_shape = self._compute_canonical_shapes()
        return in_shape, out_shape

    def _matmat(self, M: list[Tensor]) -> list[Tensor]:
        """Transform parameter tensors from canonical form back to original order.

        Args:
            M: Parameter tensors in canonical form.

        Returns:
            Parameter tensors in original order with proper shapes.

        Raises:
            RuntimeError: If parameters were incorrectly processed, likely due
                to an erroneous `self._param_positions`.
        """
        original_M = [None] * len(self._param_shapes)
        (num_columns,) = {m.shape[-1] for m in M}
        processed = 0

        for param_pos in self._param_positions:
            if _has_joint_weight_and_bias(self._separate_weight_and_bias, param_pos):
                w_pos, b_pos = param_pos["weight"], param_pos["bias"]
                combined = M[processed]

                # Get original weight shape
                w_shape = self._param_shapes[w_pos]
                w_rows, w_cols = (
                    w_shape[0],
                    self._param_shapes[w_pos].numel() // w_shape[0],
                )

                # Reshape combined tensor back to (weight + bias) matrix
                combined = combined.reshape(w_rows, w_cols + 1, num_columns)
                w_part, b_part = combined.split([w_cols, 1], dim=1)

                # Reshape into parameter shape
                original_M[w_pos] = w_part.reshape(*w_shape, num_columns)
                original_M[b_pos] = b_part.reshape(w_rows, num_columns)
                processed += 1
            else:
                # Handle separate weight and bias
                for p_name in param_pos:
                    pos = param_pos[p_name]
                    original_M[pos] = M[processed].reshape(
                        *self._param_shapes[pos], num_columns
                    )
                    processed += 1

        if any(m is None for m in original_M) or processed != len(M):
            raise RuntimeError("Mismatch in number of processed parameters.")

        return original_M

    def _adjoint(self) -> ToCanonicalLinearOperator:
        """Return the adjoint transformation operator.

        Returns:
            Linear operator that transforms from parameter to canonical form.
        """
        return ToCanonicalLinearOperator(
            self._param_shapes,
            self._param_positions,
            self._separate_weight_and_bias,
            self._device,
            self._dtype,
        )
