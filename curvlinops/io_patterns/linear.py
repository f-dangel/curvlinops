"""Pattern matching for detecting linear layer operations and parameter usage."""

from typing import List, Tuple

from torch.fx import Node
from torch.ops import aten

from ._base import NOT_A_PARAM, AffineLayerInfo, _PatternMatcher

LINEAR_STR = "Linear(y=W@x+b)"


class LinearWeightMatcher(_PatternMatcher):
    """Matcher for weight parameters in linear layers.

    Detects patterns: x @ W.T (mm) or x @ W.T + b (addmm).
    """

    def matches(
        self, p_node: Node
    ) -> Tuple[List[AffineLayerInfo], List[Tuple[Node, ...]]]:
        """Match a parameter node used as weight in linear layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of AffineLayerInfo for all weight usage matches found.
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
                    layer_info = AffineLayerInfo(
                        LINEAR_STR, pT_user, p_node, inputs, bias, {}
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
                    layer_info = AffineLayerInfo(
                        LINEAR_STR, pT_user, p_node, inputs, None, {}
                    )
                    matches.append(layer_info)
                    paths.append((p_node, pT, pT_user))

        return matches, paths


class LinearBiasMatcher(_PatternMatcher):
    """Matcher for bias parameters in linear layers.

    Detects pattern: x @ W.T + b (addmm) where the node is b.
    """

    def matches(
        self, p_node: Node
    ) -> Tuple[List[AffineLayerInfo], List[Tuple[Node, ...]]]:
        """Match a parameter node used as bias in linear layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of AffineLayerInfo for all bias usage matches found.
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

                # Check if WT is a valid weight transpose operation and extract W_node
                (W_node,) = (
                    list(WT.all_input_nodes)
                    if (WT.op, WT.target) == ("call_function", aten.t.default)
                    else [NOT_A_PARAM]
                )

                layer_info = AffineLayerInfo(
                    LINEAR_STR, p_user, W_node, inputs, bias, {}
                )
                matches.append(layer_info)
                paths.append((p_node, p_user))

        return matches, paths
