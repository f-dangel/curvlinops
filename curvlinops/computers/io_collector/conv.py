"""Pattern matching for detecting convolution layer operations and parameter usage."""

from typing import Any

from torch.fx import Node
from torch.ops import aten

from curvlinops.computers.io_collector._base import AffineLayerInfo, _PatternMatcher

CONV_STR = "Conv(y=W*x+b)"


class ConvolutionWeightMatcher(_PatternMatcher):
    """Matcher for weight parameters in convolution layers.

    Detects pattern: conv(x, W, b, ...) where the node is W.
    """

    def matches(
        self, p_node: Node
    ) -> tuple[list[AffineLayerInfo], list[tuple[Node, ...]]]:
        """Match a parameter node used as weight in conv layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of AffineLayerInfo for all weight usage matches found.
                - List of paths from parameter node to detected output nodes.
        """
        matches, paths = [], []

        # Check all users of the parameter node
        for p_user in p_node.users:
            if (
                (p_user.op, p_user.target)
                != ("call_function", aten.convolution.default)
                or p_user.kwargs
                or len(p_user.args) < 3  # At least x, weight, bias
            ):
                continue

            inputs, weight, bias = p_user.args[:3]
            # Verify this parameter is the weight (second argument)
            if weight == p_node:
                hyperparams = _extract_conv_hyperparams(p_user.args, weight)
                layer_info = AffineLayerInfo(
                    CONV_STR, p_user, p_node, inputs, bias, hyperparams
                )
                matches.append(layer_info)
                paths.append((p_node, p_user))

        return matches, paths


class ConvolutionBiasMatcher(_PatternMatcher):
    """Matcher for bias parameters in convolution layers.

    Detects pattern: conv(x, W, b, ...) where the node is b.
    """

    def matches(
        self, p_node: Node
    ) -> tuple[list[AffineLayerInfo], list[tuple[Node, ...]]]:
        """Match a parameter node used as bias in convolution layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of AffineLayerInfo for all bias usage matches found.
                - List of paths from parameter node to detected output nodes.
        """
        matches, paths = [], []

        # Check all users of the parameter node
        for p_user in p_node.users:
            if (
                (p_user.op, p_user.target)
                != ("call_function", aten.convolution.default)
                or p_user.kwargs
                or len(p_user.args) < 3  # At least x, weight, bias
            ):
                continue

            inputs, weight, bias = p_user.args[:3]
            # Verify this parameter is the bias (third argument)
            if bias == p_node:
                hyperparams = _extract_conv_hyperparams(p_user.args, weight)
                layer_info = AffineLayerInfo(
                    CONV_STR,
                    p_user,
                    weight,
                    inputs,
                    p_node,
                    hyperparams,
                )
                matches.append(layer_info)
                paths.append((p_node, p_user))

        return matches, paths


def _extract_conv_hyperparams(
    args: tuple[Any, ...], weight_node: Node
) -> dict[str, Any]:
    """Extract hyperparameters from convolution arguments.

    Args:
        args: Arguments tuple from aten.convolution.default call.
        weight_node: The FX node for the weight parameter, used to infer
            ``kernel_size`` from the weight tensor's shape metadata.

    Returns:
        Dictionary of hyperparameters including ``kernel_size``.

    Raises:
        ValueError: If args doesn't have exactly 9 elements.
    """
    if len(args) != 9:
        raise ValueError(
            f"Expected 9 convolution arguments but got {len(args)}. "
            f"Arguments should be: (input, weight, bias, stride, padding, "
            f"dilation, transposed, output_padding, groups)"
        )

    _, _, _, stride, padding, dilation, transposed, output_padding, groups = args

    # Infer kernel_size from weight tensor shape metadata (shape[2:] for Conv2d)
    # Try tensor_meta first (symbolic tracing), fall back to val (real tracing)
    if "tensor_meta" in weight_node.meta:
        kernel_size = list(weight_node.meta["tensor_meta"].shape[2:])
    elif "val" in weight_node.meta:
        kernel_size = list(weight_node.meta["val"].shape[2:])
    else:
        raise ValueError(
            f"Cannot infer kernel_size: weight node {weight_node} has no shape "
            f"metadata. Available meta keys: {list(weight_node.meta.keys())}"
        )

    return {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "transposed": transposed,
        "output_padding": output_padding,
        "groups": groups,
    }
