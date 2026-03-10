"""Pattern matching for detecting linear layer operations and parameter usage."""

from torch.fx import Node
from torch.ops import aten

from curvlinops.computers.io_collector._base import (
    NOT_A_PARAM,
    AffineLayerInfo,
    _PatternMatcher,
)

LINEAR_STR = "Linear(y=W@x+b)"

# Reshape ops that may appear between mm and bias addition
_RESHAPE_OPS = frozenset({
    aten._unsafe_view.default,
    aten.view.default,
    aten.reshape.default,
})


def _trace_forward_through_reshapes(node: Node) -> Node:
    """Follow a chain of reshape operations forward (toward outputs).

    Starting from a node, walk through single-user reshape operations
    (``view``, ``_unsafe_view``, ``reshape``).  Return the last node in
    that chain (which may be the input itself if there are no reshapes).

    Args:
        node: Starting node (typically the output of ``aten.mm``).

    Returns:
        The last node after following single-user reshape chains.
    """
    while (
        len(node.users) == 1
        and (user := next(iter(node.users))).op == "call_function"
        and user.target in _RESHAPE_OPS
    ):
        node = user
    return node


def _trace_backward_through_reshapes(node: Node) -> Node:
    """Follow a chain of reshape operations backward (toward inputs).

    Starting from a node, walk backward through reshape operations whose
    first argument is a reshape, finding the original tensor before any
    flattening/reshaping.

    Args:
        node: Starting node (typically the first argument of ``aten.mm``).

    Returns:
        The first node before any reshape chain.
    """
    while (
        node.op == "call_function"
        and node.target in _RESHAPE_OPS
        and len(node.args) >= 1
    ):
        node = node.args[0]
    return node


def _find_add_bias(mm_or_reshape: Node) -> tuple[Node, Node] | None:
    """Check if a node feeds into ``aten.add.Tensor`` with a bias parameter.

    Looks for the pattern ``add(mm_result, bias)`` or ``add(bias, mm_result)``
    among the users of ``mm_or_reshape``.

    Args:
        mm_or_reshape: Node whose users to inspect (mm output or reshape output).

    Returns:
        ``(add_node, bias_node)`` if a matching add is found, else ``None``.
    """
    for user in mm_or_reshape.users:
        if (
            user.op == "call_function"
            and user.target == aten.add.Tensor
            and len(user.args) == 2
        ):
            lhs, rhs = user.args
            # Bias can be either argument of the add
            if lhs == mm_or_reshape:
                return user, rhs
            if rhs == mm_or_reshape:
                return user, lhs
    return None


class LinearWeightMatcher(_PatternMatcher):
    """Matcher for weight parameters in linear layers.

    Detects patterns:
    - ``x @ W.T + b`` (addmm)
    - ``x @ W.T`` (mm, no bias)
    - ``x @ W.T`` (mm) followed by optional reshapes and ``add(result, b)``
    """

    def matches(
        self, p_node: Node
    ) -> tuple[list[AffineLayerInfo], list[tuple[Node, ...]]]:
        """Match a parameter node used as weight in linear layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of AffineLayerInfo for all weight usage matches found.
                - List of paths from parameter node to detected output nodes.
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
                    bias, inputs, mat2 = pT_user.args
                    if mat2 == pT:
                        layer_info = AffineLayerInfo(
                            LINEAR_STR, pT_user, p_node, inputs, bias, {}
                        )
                        matches.append(layer_info)
                        paths.append((p_node, pT, pT_user))

                # Case: x @ W.T (mm)
                elif (
                    target == aten.mm.default
                    and len(pT_user.args) == 2
                    and not pT_user.kwargs
                ):
                    mm_input, mat2 = pT_user.args
                    if mat2 != pT:
                        continue

                    # Check for mm → (optional reshapes) → add(result, bias)
                    last_node = _trace_forward_through_reshapes(pT_user)
                    add_result = _find_add_bias(last_node)

                    if add_result is not None:
                        add_node, bias_node = add_result
                        # Trace back through reshapes on the input side to
                        # recover the original (unflattened) input tensor
                        original_input = _trace_backward_through_reshapes(mm_input)
                        layer_info = AffineLayerInfo(
                            LINEAR_STR,
                            add_node,
                            p_node,
                            original_input,
                            bias_node,
                            {},
                        )
                        matches.append(layer_info)
                        paths.append((p_node, pT, pT_user))
                    else:
                        # No bias addition found — pure mm
                        layer_info = AffineLayerInfo(
                            LINEAR_STR, pT_user, p_node, mm_input, None, {}
                        )
                        matches.append(layer_info)
                        paths.append((p_node, pT, pT_user))

        return matches, paths


class LinearBiasMatcher(_PatternMatcher):
    """Matcher for bias parameters in linear layers.

    Detects patterns:
    - ``x @ W.T + b`` (addmm) where the node is ``b``
    - ``add(mm_result_or_reshape, b)`` where ``b`` is the bias parameter
    """

    def matches(
        self, p_node: Node
    ) -> tuple[list[AffineLayerInfo], list[tuple[Node, ...]]]:
        """Match a parameter node used as bias in linear layers.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of AffineLayerInfo for all bias usage matches found.
                - List of paths from parameter node to detected output nodes.
        """
        matches = []
        paths = []

        # Check all users of the parameter node
        for p_user in p_node.users:
            if p_user.op != "call_function":
                continue

            # Case: addmm(bias, x, W.T)
            if (
                p_user.target == aten.addmm.default
                and len(p_user.args) == 3
                and not p_user.kwargs
            ):
                bias, inputs, WT = p_user.args

                # Verify this parameter is the bias (first argument)
                if bias != p_node:
                    continue

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

            # Case: add(mm_result_or_reshape, bias)
            elif p_user.target == aten.add.Tensor and len(p_user.args) == 2:
                lhs, rhs = p_user.args
                other = lhs if rhs == p_node else rhs if lhs == p_node else None
                if other is None:
                    continue

                # Trace back through reshapes to find the mm node
                mm_node = _trace_backward_through_reshapes(other)

                # Check if we arrived at an mm node
                if not (
                    mm_node.op == "call_function" and mm_node.target == aten.mm.default
                ):
                    continue

                # Extract the weight node and input from mm(x, W.T)
                mm_input, WT = mm_node.args
                (W_node,) = (
                    list(WT.all_input_nodes)
                    if (WT.op, WT.target) == ("call_function", aten.t.default)
                    else [NOT_A_PARAM]
                )

                # Trace back through reshapes on input side to get original input
                original_input = _trace_backward_through_reshapes(mm_input)

                layer_info = AffineLayerInfo(
                    LINEAR_STR, p_user, W_node, original_input, p_node, {}
                )
                matches.append(layer_info)
                paths.append((p_node, p_user))

        return matches, paths
