"""Pattern matching for detecting linear layer operations and parameter usage."""

from torch.fx import Node
from torch.ops import aten

from curvlinops.computers.io_collector._base import (
    NOT_A_PARAM,
    AffineLayerInfo,
    _PatternMatcher,
)

LINEAR_STR = "Linear(y=W@x+b)"

# Reshape ops inserted by F.linear for >2D inputs (view → mm/addmm → view)
_RESHAPE_OPS = {aten._unsafe_view.default, aten.view.default}


def _trace_forward_through_reshapes(node: Node) -> Node:
    """Follow a chain of reshape operations forward (toward outputs).

    Starting from a node, walk through single-user reshape operations
    (``view``, ``_unsafe_view``) that preserve the last dimension (feature
    dimension).  This ensures we only follow F.linear's paired reshapes
    (which unflatten batch dimensions) and stop at model reshapes that
    alter the feature dimension.

    Args:
        node: Starting node (typically the output of ``aten.mm``/``aten.addmm``).

    Returns:
        The last node after following single-user, last-dim-preserving
        reshape chains (which may be the input itself if there are no
        such reshapes).
    """
    while (
        len(node.users) == 1
        and (user := next(iter(node.users))).op == "call_function"
        and user.target in _RESHAPE_OPS
        and len(user.args) >= 2
        and isinstance(user.args[1], list)
        and user.args[1][-1] == node.meta["val"].shape[-1]
    ):
        node = user
    return node


def _trace_backward_through_reshapes(node: Node) -> Node:
    """Follow a chain of reshape operations backward (toward inputs).

    Starting from a node, walk backward through reshape operations that
    preserve the last dimension (feature dimension), finding the original
    tensor before any flattening/reshaping by F.linear.

    Args:
        node: Starting node (typically the first argument of ``aten.mm``).

    Returns:
        The first node before any last-dim-preserving reshape chain.
    """
    while (
        node.op == "call_function"
        and node.target in _RESHAPE_OPS
        and len(node.args) >= 1
        and node.meta["val"].shape[-1] == node.args[0].meta["val"].shape[-1]
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


def _resolve_paired_reshapes(input_node: Node, output_node: Node) -> tuple[Node, Node]:
    """Resolve paired reshapes inserted by ``F.linear`` for >2D inputs.

    ``F.linear`` inserts ``view → mm/addmm → view`` for inputs with more than
    2 dimensions.  This function traces forward from ``output_node`` and
    backward from ``input_node`` through reshapes, but **only** if both sides
    have reshapes (indicating paired F.linear views). If only one side has a
    reshape it belongs to the model (e.g. ``Flatten``) and is left alone.

    Args:
        input_node: Node feeding the ``mm``/``addmm`` input (may be a view).
        output_node: Node produced by ``mm``/``addmm`` (may feed a view).

    Returns:
        ``(original_input, final_output)`` with reshapes resolved on both
        sides, or the original nodes if there are no paired reshapes.
    """
    final_output = _trace_forward_through_reshapes(output_node)
    if final_output is output_node:
        # No forward reshape → no paired F.linear views
        return input_node, output_node
    original_input = _trace_backward_through_reshapes(input_node)
    return original_input, final_output


def _extract_weight_node(WT: Node) -> Node | str:
    """Extract weight node from a transpose node, or return NOT_A_PARAM.

    Args:
        WT: Node expected to be ``aten.t(weight)``.

    Returns:
        The weight node if ``WT`` is a transpose, else ``NOT_A_PARAM``.
    """
    if (WT.op, WT.target) == ("call_function", aten.t.default):
        (W_node,) = WT.all_input_nodes
        return W_node
    return NOT_A_PARAM


def _match_addmm_weight(
    p_node: Node, pT: Node, addmm_node: Node
) -> AffineLayerInfo | None:
    """Try to match ``addmm(bias, x, W.T)`` from the weight side.

    Args:
        p_node: The weight parameter node.
        pT: The transpose node ``aten.t(p_node)``.
        addmm_node: A candidate ``aten.addmm`` node.

    Returns:
        ``AffineLayerInfo`` if matched, else ``None``.
    """
    if len(addmm_node.args) != 3 or addmm_node.kwargs:
        return None
    bias, inputs, mat2 = addmm_node.args
    if mat2 != pT:
        return None
    original_input, output_node = _resolve_paired_reshapes(inputs, addmm_node)
    return AffineLayerInfo(LINEAR_STR, output_node, p_node, original_input, bias, {})


def _match_mm_weight(
    p_node: Node, pT: Node, mm_node: Node
) -> AffineLayerInfo | None:
    """Try to match ``mm(x, W.T)`` optionally followed by reshapes and bias add.

    Args:
        p_node: The weight parameter node.
        pT: The transpose node ``aten.t(p_node)``.
        mm_node: A candidate ``aten.mm`` node.

    Returns:
        ``AffineLayerInfo`` if matched, else ``None``.
    """
    if len(mm_node.args) != 2 or mm_node.kwargs:
        return None
    mm_input, mat2 = mm_node.args
    if mat2 != pT:
        return None

    last_node = _trace_forward_through_reshapes(mm_node)
    has_reshapes = last_node is not mm_node
    add_result = _find_add_bias(last_node)

    if add_result is not None:
        output_node, bias_node = add_result
    else:
        output_node, bias_node = last_node, None

    original_input = (
        _trace_backward_through_reshapes(mm_input) if has_reshapes else mm_input
    )
    return AffineLayerInfo(
        LINEAR_STR, output_node, p_node, original_input, bias_node, {}
    )


def _match_addmm_bias(p_node: Node, addmm_node: Node) -> AffineLayerInfo | None:
    """Try to match ``addmm(bias, x, W.T)`` from the bias side.

    Args:
        p_node: The bias parameter node.
        addmm_node: A candidate ``aten.addmm`` node.

    Returns:
        ``AffineLayerInfo`` if matched, else ``None``.
    """
    if len(addmm_node.args) != 3 or addmm_node.kwargs:
        return None
    bias, inputs, WT = addmm_node.args
    if bias != p_node:
        return None
    W_node = _extract_weight_node(WT)
    original_input, output_node = _resolve_paired_reshapes(inputs, addmm_node)
    return AffineLayerInfo(LINEAR_STR, output_node, W_node, original_input, bias, {})


def _match_add_bias(p_node: Node, add_node: Node) -> AffineLayerInfo | None:
    """Try to match ``add(mm_result_or_reshape, bias)`` from the bias side.

    Args:
        p_node: The bias parameter node.
        add_node: A candidate ``aten.add.Tensor`` node.

    Returns:
        ``AffineLayerInfo`` if matched, else ``None``.
    """
    if len(add_node.args) != 2:
        return None
    lhs, rhs = add_node.args
    other = lhs if rhs == p_node else rhs if lhs == p_node else None
    if other is None:
        return None

    # Trace back through reshapes to find the mm node
    mm_node = _trace_backward_through_reshapes(other)
    if not (mm_node.op == "call_function" and mm_node.target == aten.mm.default):
        return None

    mm_input, WT = mm_node.args
    W_node = _extract_weight_node(WT)
    original_input = _trace_backward_through_reshapes(mm_input)
    return AffineLayerInfo(LINEAR_STR, add_node, W_node, original_input, p_node, {})


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

        for pT in p_node.users:
            if (pT.op, pT.target) != ("call_function", aten.t.default):
                continue

            for pT_user in pT.users:
                if pT_user.op != "call_function":
                    continue

                if pT_user.target == aten.addmm.default:
                    info = _match_addmm_weight(p_node, pT, pT_user)
                elif pT_user.target == aten.mm.default:
                    info = _match_mm_weight(p_node, pT, pT_user)
                else:
                    continue

                if info is not None:
                    matches.append(info)
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
        matches, paths = [], []

        for p_user in p_node.users:
            if p_user.op != "call_function":
                continue

            if p_user.target == aten.addmm.default:
                info = _match_addmm_bias(p_node, p_user)
            elif p_user.target == aten.add.Tensor:
                info = _match_add_bias(p_node, p_user)
            else:
                continue

            if info is not None:
                matches.append(info)
                paths.append((p_node, p_user))

        return matches, paths
