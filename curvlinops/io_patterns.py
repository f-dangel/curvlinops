"""Contains patterns for detecting how parameters are used."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from torch.fx import Node
from torch.ops import aten

NOT_A_PARAM: str = "__not_a_param"


@dataclass
class LinearLayerInfo:
    """Information about a detected linear layer operation.

    Attributes:
        y_node: The FX node representing the output of the linear operation.
        x_node: The FX node representing the input to the linear operation.
        WT_node: The FX node representing the weight transpose operation.
        W_node: The FX node representing the weight parameter.
        b_node: The FX node representing the bias parameter.
    """

    y_node: Node
    x_node: Node
    WT_node: Node
    W_node: Union[Node, None]
    b_node: Optional[Node]

    def to_info_tuple(
        self, node_name_to_param_name: Dict[str, str]
    ) -> Tuple[str, Node, Node, Optional[str], Optional[str]]:
        """Convert to layer info tuple format.

        Args:
            node_name_to_param_name: Mapping from FX node names to parameter names.

        Returns:
            Tuple containing ("Linear", y_node, x_node, weight_name, bias_name).
        """
        weight_name = (
            node_name_to_param_name.get(self.W_node.name, NOT_A_PARAM)
            if self.W_node is not None
            else NOT_A_PARAM
        )
        bias_name = (
            None
            if self.b_node is None
            else node_name_to_param_name.get(self.b_node.name, NOT_A_PARAM)
        )
        return ("Linear(y=x@W^T+b)", self.y_node, self.x_node, weight_name, bias_name)


class PatternMatcher:
    """Base class for matching parameter usage patterns in FX graphs.

    Subclasses implement specific pattern matching logic for different layer types
    (e.g., linear layers with/without bias).
    """

    def matches(
        self, p_node: Node
    ) -> Tuple[List[LinearLayerInfo], List[Tuple[Node, ...]]]:
        """Attempt to match a parameter node to known usage patterns.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of LinearLayerInfo for all matches found for this pattern.
                - List of paths from parameter node to detected output nodes.
        """
        raise NotImplementedError


class LinearWeightMatcher(PatternMatcher):
    """Matcher for weight parameters in linear layers.

    Detects patterns: x @ W.T (mm) or x @ W.T + b (addmm).
    """

    def matches(
        self, p_node: Node
    ) -> Tuple[List[LinearLayerInfo], List[Tuple[Node, ...]]]:
        """Match a parameter node used as weight in linear layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of LinearLayerInfo for all weight usage matches found.
                - List of paths from parameter node to detected output nodes.

        Raises:
            ValueError: If the detected operation has unexpected arguments or structure.
        """
        matches, paths = [], []

        # Iterate over all usages where p is transposed
        for pT in [
            n
            for n in p_node.users
            if (n.op, n.target) == ("call_function", aten.t.default)
        ]:
            # Check all users of p.T
            for pT_user in pT.users:
                if pT_user.op != "call_function":
                    continue

                target = pT_user.target

                # Case: x @ W.T + b (addmm)
                if (
                    target == aten.addmm.default
                    and len(pT_user.args) == 3
                    and not pT_user.kwargs
                ):
                    bias, inputs, _ = pT_user.args
                    layer_info = LinearLayerInfo(
                        pT_user, inputs, pT, W_node=p_node, b_node=bias
                    )
                    matches.append(layer_info)
                    paths.append((p_node, pT, pT_user))

                # Case: x @ W.T (mm, no bias)
                elif (
                    target == aten.mm.default
                    and len(pT_user.args) == 2
                    and not pT_user.kwargs
                ):
                    inputs, _ = pT_user.args
                    layer_info = LinearLayerInfo(
                        pT_user, inputs, pT, W_node=p_node, b_node=None
                    )
                    matches.append(layer_info)
                    paths.append((p_node, pT, pT_user))

        return matches, paths


class LinearBiasMatcher(PatternMatcher):
    """Matcher for bias parameters in linear layers.

    Detects pattern: x @ W.T + b (addmm) where the node is b.
    """

    def matches(
        self, p_node: Node
    ) -> Tuple[List[LinearLayerInfo], List[Tuple[Node, ...]]]:
        """Match a parameter node used as bias in linear layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of LinearLayerInfo for all bias usage matches found.
                - List of paths from parameter node to detected output nodes.

        Raises:
            ValueError: If the detected operation has unexpected structure.
        """
        matches = []
        paths = []

        # Check all users of the parameter node
        for p_user in p_node.users:
            if p_user.op != "call_function":
                continue

            if p_user.target == aten.addmm.default:
                # Detect the weight
                bias, inputs, WT = p_user.args

                # Check if WT is a valid weight transpose operation
                (W_node,) = (
                    list(WT.all_input_nodes)
                    if (WT.op, WT.target) == ("call_function", aten.t.default)
                    else (None,)
                )
                layer_info = LinearLayerInfo(
                    p_user, inputs, WT, W_node=W_node, b_node=bias
                )
                matches.append(layer_info)
                paths.append((p_node, p_user))

        return matches, paths


def match_parameter_usage(
    param_nodes: List[Node],
) -> Tuple[List[LinearLayerInfo], List[Tuple[Node, ...]]]:
    """Match parameter nodes against known usage patterns.

    Args:
        param_nodes: List of parameter nodes to analyze.

    Returns:
        Tuple containing:
            - List of LinearLayerInfo for all matched patterns.
            - List of paths from parameter nodes to detected output nodes.

    Raises:
        ValueError: If a parameter node is used in an unsupported operation.
    """
    patterns: List[PatternMatcher] = [LinearWeightMatcher(), LinearBiasMatcher()]
    usage_info: List[LinearLayerInfo] = []
    path_info: List[Tuple[Node, ...]] = []

    for p_node in param_nodes:
        p_usage_info = []

        for pattern in patterns:
            layer_infos, detected_paths = pattern.matches(p_node)
            for info in layer_infos:
                if info not in usage_info + p_usage_info:
                    p_usage_info.append(info)

            path_info.extend(detected_paths)

        usage_info.extend(p_usage_info)

    return usage_info, path_info
