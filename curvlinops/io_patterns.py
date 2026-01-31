"""Contains patterns for detecting how parameters are used."""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Set, Tuple

from torch.fx import Graph, Node
from torch.ops import aten

NOT_A_PARAM: str = "__not_a_param"


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

    def to_info_tuple(
        self, node_name_to_param_name: Dict[str, str]
    ) -> Tuple[str, Node, Node, Optional[str], Optional[str]]:
        """Convert to layer info tuple format.

        Args:
            node_name_to_param_name: Mapping from FX node names to parameter names.

        Returns:
            Tuple containing ("Linear", input_node, output_node, weight_name, bias_name).
        """
        weight_name = node_name_to_param_name.get(self.weight_node.name, NOT_A_PARAM)
        bias_name = (
            None
            if self.bias_node is None
            else node_name_to_param_name.get(self.bias_node.name, NOT_A_PARAM)
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

    def matches(self, p_node: Node) -> List[LinearLayerInfo]:
        """Attempt to match a parameter node to known usage patterns.

        Args:
            p_node: A parameter node to check.

        Returns:
            List of LinearLayerInfo for all matches found for this pattern.
        """
        raise NotImplementedError


class LinearWeightMatcher(PatternMatcher):
    """Matcher for weight parameters in linear layers.

    Detects patterns: x @ W.T (mm) or x @ W.T + b (addmm).
    """

    def matches(self, p_node: Node) -> List[LinearLayerInfo]:
        """Match a parameter node used as weight in linear layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            List of LinearLayerInfo for all weight usage matches found.
        """
        matches = []

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
                if target == aten.addmm.default:
                    assert len(pT_user.args) == 3 and not pT_user.kwargs
                    bias, inputs, _ = pT_user.args
                    layer_info = LinearLayerInfo(
                        p_node, inputs, pT_user, bias_node=bias
                    )
                    matches.append(layer_info)

                # Case: x @ W.T (mm, no bias)
                elif target == aten.mm.default:
                    assert len(pT_user.args) == 2 and not pT_user.kwargs
                    inputs, _ = pT_user.args
                    layer_info = LinearLayerInfo(p_node, inputs, pT_user)
                    matches.append(layer_info)

        return matches


class LinearBiasMatcher(PatternMatcher):
    """Matcher for bias parameters in linear layers.

    Detects pattern: x @ W.T + b (addmm) where the node is b.
    """

    def matches(self, p_node: Node) -> List[LinearLayerInfo]:
        """Match a parameter node used as bias in linear layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            List of LinearLayerInfo for all bias usage matches found.
        """
        matches = []

        # Check all users of the parameter node
        for p_user in p_node.users:
            if p_user.op != "call_function":
                continue

            if p_user.target == aten.addmm.default:
                # Detect the weight
                bias, inputs, WT = p_user.args
                assert WT.op == "call_function" and WT.target == aten.t.default
                (W,) = list(WT.all_input_nodes)
                layer_info = LinearLayerInfo(W, inputs, p_user, bias_node=bias)
                matches.append(layer_info)

        return matches


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
        p_usage_info = []

        for pattern in patterns:
            layer_infos = pattern.matches(p_node)
            for info in layer_infos:
                if info not in usage_info + p_usage_info:
                    p_usage_info.append(info)

        usage_info.extend(p_usage_info)

    return usage_info


def _find_undetected_paths(
    node: Node, path: Tuple[Node, ...], endpoints: Set[Node]
) -> Iterator[Tuple[Node, ...]]:
    """Recursively find paths that reach leaf nodes without detection.

    Args:
        node: Current node in the traversal.
        path: Current path from parameter node.
        endpoints: Set of detected layer output nodes.

    Yields:
        Undetected paths from current node.
    """
    if node in endpoints:
        # Reached a detected output, this path is captured
        return

    # Check if this is a leaf node (no users)
    if not node.users:
        # Reached a leaf node - check if we went through a detected output
        if not any(n in endpoints for n in path):
            yield path
        return

    # Continue traversal to users
    for user in node.users:
        yield from _find_undetected_paths(user, path + (user,), endpoints)


def verify_match_complete(
    param_nodes: List[Node],
    usage_info: List[LinearLayerInfo],
    graph: Graph,
) -> None:
    """Verify that all parameter usages were detected by pattern matching.

    For each parameter node, traverses all user paths and verifies that each
    path eventually leads to a detected layer output rather than the final
    graph output without going through detected patterns.

    Args:
        param_nodes: List of parameter nodes to verify.
        usage_info: List of detected layer information.
        graph: The full FX graph.

    Raises:
        ValueError: If any parameter usage was not detected, including the
            graph and the first undetected path in the error message.
    """
    # For each parameter, collect all detected output nodes
    param_to_detected_outputs: Dict[Node, set] = {node: set() for node in param_nodes}

    for info in usage_info:
        weight_node, bias_node = info.weight_node, info.bias_node
        output_node = info.output_node
        if weight_node is not None and weight_node in param_nodes:
            param_to_detected_outputs[weight_node].add(output_node)
        if bias_node is not None and bias_node in param_nodes:
            param_to_detected_outputs[bias_node].add(output_node)

    # For each parameter, verify all usage paths
    for p in param_nodes:
        detected_endpoints = param_to_detected_outputs[p]

        # Start traversal from parameter node and raise error on first undetected path
        for undetected_path in _find_undetected_paths(p, (p,), detected_endpoints):
            path_str = " -> ".join([str(node) for node in undetected_path])
            raise ValueError(
                f"FX Graph:\n{graph}\n\n"
                f"First undetected usage path:\n{path_str}\n\n"
                f"Parameter node {p} is used in an unsupported pattern."
            )
