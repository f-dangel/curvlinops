"""Pattern matching for detecting convolution layer operations and parameter usage."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from torch.fx import Node
from torch.ops import aten

from ._base import PatternMatcher, _create_info_tuple


@dataclass
class ConvolutionLayerInfo:
    """Information about a detected convolution layer operation.

    Attributes:
        y_node: The FX node representing the output of the convolution operation.
        x_node: The FX node representing the input to the convolution operation.
        W_node: The FX node representing the weight/kernel parameter.
        b_node: The FX node representing the bias parameter.
    """

    y_node: Node
    x_node: Node
    W_node: Union[Node, None]
    b_node: Optional[Node]

    def to_info_tuple(
        self, node_name_to_param_name: Dict[str, str]
    ) -> Tuple[str, Node, Node, Optional[str], Optional[str]]:
        """Convert to layer info tuple format.

        Args:
            node_name_to_param_name: Mapping from FX node names to parameter names.

        Returns:
            Tuple containing ("Conv2d", y_node, x_node, weight_name, bias_name).
        """
        return _create_info_tuple(
            "Conv2d(y=x*W+b)",
            self.y_node,
            self.x_node,
            self.W_node,
            self.b_node,
            node_name_to_param_name,
        )


class ConvolutionWeightMatcher(PatternMatcher):
    """Matcher for weight parameters in 2D convolution layers.

    Detects pattern: conv2d(x, W, b, ...) where the node is W.
    """

    def matches(
        self, p_node: Node
    ) -> Tuple[List[ConvolutionLayerInfo], List[Tuple[Node, ...]]]:
        """Match a parameter node used as weight in convolution layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of ConvolutionLayerInfo for all weight usage matches found.
                - List of paths from parameter node to detected output nodes.
        """
        matches, paths = [], []

        # Check all users of the parameter node
        for p_user in p_node.users:
            if (
                (p_user.op, p_user.target)
                != (
                    "call_function",
                    aten.convolution.default,
                )
                or p_user.kwargs
                or len(p_user.args) < 3  # At least x, weight, bias
            ):
                continue

            inputs, weight, bias = p_user.args[:3]
            # Verify this parameter is the weight (second argument)
            if weight == p_node:
                layer_info = ConvolutionLayerInfo(
                    p_user, inputs, W_node=p_node, b_node=bias
                )
                matches.append(layer_info)
                paths.append((p_node, p_user))

        return matches, paths


class ConvolutionBiasMatcher(PatternMatcher):
    """Matcher for bias parameters in convolution layers.

    Detects pattern: conv2d(x, W, b, ...) where the node is b.
    """

    def matches(
        self, p_node: Node
    ) -> Tuple[List[ConvolutionLayerInfo], List[Tuple[Node, ...]]]:
        """Match a parameter node used as bias in convolution layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of ConvolutionLayerInfo for all bias usage matches found.
                - List of paths from parameter node to detected output nodes.
        """
        matches, paths = [], []

        # Check all users of the parameter node
        for p_user in p_node.users:
            if (
                (p_user.op, p_user.target)
                != (
                    "call_function",
                    aten.convolution.default,
                )
                or p_user.kwargs
                or len(p_user.args) < 3  # At least x, weight, bias
            ):
                continue

            inputs, weight, bias = p_user.args[:3]
            # Verify this parameter is the bias (third argument)
            if bias == p_node:
                layer_info = ConvolutionLayerInfo(
                    p_user, inputs, W_node=weight, b_node=p_node
                )
                matches.append(layer_info)
                paths.append((p_node, p_user))

        return matches, paths
