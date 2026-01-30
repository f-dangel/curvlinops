"""Contains patterns for detecting how parameters are used."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch.fx import Node
from torch.ops import aten


@dataclass
class LinearLayerInfo:
    """Information about a detected linear layer operation.

    Attributes:
        weight_node: The FX node representing the weight parameter.
        input_node: The FX node representing the input to the linear operation.
        output_node: The FX node representing the output of the linear operation.
        bias_node: The FX node representing the bias parameter, if present.
    """

    weight_node: Node
    input_node: Node
    output_node: Node
    bias_node: Optional[Node] = None
    not_a_param: str = "__not_a_param"

    def to_info_tuple(
        self, node_name_to_param_name: Dict[str, str]
    ) -> Tuple[str, Node, Node, Optional[str], Optional[str]]:
        """Convert to layer info tuple format.

        Args:
            node_name_to_param_name: Mapping from FX node names to parameter names.

        Returns:
            Tuple containing ("Linear", input_node, output_node, weight_name, bias_name).
        """
        weight_name = node_name_to_param_name.get(
            self.weight_node.name, self.not_a_param
        )
        bias_name = (
            None
            if self.bias_node is None
            else node_name_to_param_name.get(self.bias_node.name, self.not_a_param)
        )
        return (
            "Linear(y=x@W^T+b)",
            self.input_node,
            self.output_node,
            weight_name,
            bias_name,
        )


class PatternMatcher:
    """Base class for matching parameter usage patterns in FX graphs.

    Subclasses implement specific pattern matching logic for different layer types
    (e.g., linear layers with/without bias).
    """

    def matches(self, p_node: Node) -> Optional[LinearLayerInfo]:
        """Attempt to match a parameter node to a known usage pattern.

        Args:
            p_node: A parameter node to check.

        Returns:
            LinearLayerInfo if the node matches this pattern, None otherwise.
        """
        raise NotImplementedError

    @staticmethod
    def _ensure_single_user(node: Node) -> Node:
        """Get the single user of a node, raising if not exactly one.

        Args:
            node: The node to check.

        Returns:
            The single user node.

        Raises:
            ValueError: If the node does not have exactly one user.
        """
        users = list(node.users.keys())
        if len(users) != 1:
            raise ValueError(f"Node {node} is not used once ({len(users)}x).")
        return users[0]


class LinearWeightMatcher(PatternMatcher):
    """Matcher for weight parameters in linear layers.

    Detects patterns: x @ W.T (mm) or x @ W.T + b (addmm).
    """

    def matches(self, p_node: Node) -> Optional[LinearLayerInfo]:
        """Match a parameter node used as weight in a linear layer.

        Args:
            p_node: A parameter node to check.

        Returns:
            LinearLayerInfo if the node is used as a weight, None otherwise.
        """
        user_node = self._ensure_single_user(p_node)

        # Weight must first be transposed
        if not (user_node.op == "call_function" and user_node.target == aten.t.default):
            return None

        T_user_node = self._ensure_single_user(user_node)

        # Case: x @ W.T + b (addmm)
        if (
            T_user_node.op == "call_function"
            and T_user_node.target == aten.addmm.default
        ):
            assert len(T_user_node.args) == 3 and not T_user_node.kwargs
            bias, inputs, _ = T_user_node.args
            layer_info = LinearLayerInfo(p_node, inputs, T_user_node, bias_node=bias)

        # Case: x @ W.T (mm, no bias)
        elif (
            T_user_node.op == "call_function" and T_user_node.target == aten.mm.default
        ):
            assert len(T_user_node.args) == 2 and not T_user_node.kwargs
            inputs, _ = T_user_node.args
            layer_info = LinearLayerInfo(p_node, inputs, T_user_node)

        else:
            layer_info = None

        return layer_info


class LinearBiasMatcher(PatternMatcher):
    """Matcher for bias parameters in linear layers.

    Detects pattern: x @ W.T + b (addmm) where the node is b.
    """

    def matches(self, p_node: Node) -> Optional[LinearLayerInfo]:
        """Match a parameter node used as bias in a linear layer.

        Args:
            p_node: A parameter node to check.

        Returns:
            LinearLayerInfo if the node is used as a bias, None otherwise.
        """
        user_node = self._ensure_single_user(p_node)

        # Bias is used directly in addmm
        if not (
            user_node.op == "call_function" and user_node.target == aten.addmm.default
        ):
            return None

        bias, inputs, WT = user_node.args
        assert WT.op == "call_function" and WT.target == aten.t.default
        (W,) = list(WT.all_input_nodes)
        return LinearLayerInfo(W, inputs, user_node, bias_node=bias)


def match_parameter_usage(
    param_nodes: List[Node],
) -> List[LinearLayerInfo]:
    """Match parameter nodes against known usage patterns.

    Args:
        param_nodes: List of parameter nodes to analyze.

    Returns:
        List of LinearLayerInfo for all matched patterns.

    Raises:
        ValueError: If a parameter node is used in an unsupported operation.
    """
    patterns: List[PatternMatcher] = [LinearWeightMatcher(), LinearBiasMatcher()]
    usage_info: List[LinearLayerInfo] = []

    for p_node in param_nodes:
        matched = False

        for pattern in patterns:
            layer_info = pattern.matches(p_node)
            if layer_info is not None:
                matched = True
                break

        if layer_info is not None and layer_info not in usage_info:
            usage_info.append(layer_info)

        if not matched:
            raise ValueError(
                f"Parameter node {p_node} is used in an unsupported pattern."
            )

    return usage_info
