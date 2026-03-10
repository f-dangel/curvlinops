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
_VIEW_OPS = {aten._unsafe_view.default, aten.view.default}


def _is_last_dim_preserving_view(node: Node) -> bool:
    """Check if a node is a last-dim-preserving reshape (``view``/``_unsafe_view``).

    Returns:
        ``True`` if the node is a view that preserves the last dimension.
    """
    return (
        node.op == "call_function"
        and node.target in _VIEW_OPS
        and len(node.args) >= 2
        and isinstance(node.args[1], list)
        and node.args[1][-1] == node.args[0].meta["val"].shape[-1]
    )


def _find_add_bias(node: Node) -> tuple[Node, Node] | None:
    """Check if a node feeds into ``aten.add.Tensor`` with a bias parameter.

    Looks for the pattern ``add(node, bias)`` or ``add(bias, node)``
    among the users of ``node``.

    Args:
        node: Node whose users to inspect.

    Returns:
        ``(add_node, bias_node)`` if a matching add is found, else ``None``.
    """
    for user in node.users:
        if (
            user.op == "call_function"
            and user.target == aten.add.Tensor
            and len(user.args) == 2
        ):
            lhs, rhs = user.args
            if lhs == node:
                return user, rhs
            if rhs == node:
                return user, lhs


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
    """Match ``addmm(bias, x, W.T)`` or ``view → addmm → view`` from weight side.

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

    # Check for paired view → addmm → view (3D F.linear)
    if (
        _is_last_dim_preserving_view(inputs)
        and len(addmm_node.users) == 1
        and _is_last_dim_preserving_view(view_out := next(iter(addmm_node.users)))
    ):
        return AffineLayerInfo(LINEAR_STR, view_out, p_node, inputs.args[0], bias, {})

    # 2D case: no reshapes
    return AffineLayerInfo(LINEAR_STR, addmm_node, p_node, inputs, bias, {})


def _match_mm_weight(p_node: Node, pT: Node, mm_node: Node) -> AffineLayerInfo | None:
    """Match ``mm(x, W.T)`` or ``view → mm → view [→ add]`` from weight side.

    Supported patterns::

        2D no bias:  mm(x, W.T)
        2D + bias:   add(mm(x, W.T), b)
        >2D no bias: view → mm → view
        >2D + bias:  view → mm → view → add(_, b)

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

    # Check for view → mm → view pattern (>2D F.linear)
    if (
        _is_last_dim_preserving_view(mm_input)
        and len(mm_node.users) == 1
        and _is_last_dim_preserving_view(view_out := next(iter(mm_node.users)))
    ):
        original_input = mm_input.args[0]
        add_result = _find_add_bias(view_out)
        if add_result is not None:
            return AffineLayerInfo(
                LINEAR_STR, add_result[0], p_node, original_input, add_result[1], {}
            )
        return AffineLayerInfo(LINEAR_STR, view_out, p_node, original_input, None, {})

    # 2D case: mm [→ add]
    add_result = _find_add_bias(mm_node)
    if add_result is not None:
        return AffineLayerInfo(
            LINEAR_STR, add_result[0], p_node, mm_input, add_result[1], {}
        )
    return AffineLayerInfo(LINEAR_STR, mm_node, p_node, mm_input, None, {})


def _match_addmm_bias(p_node: Node, addmm_node: Node) -> AffineLayerInfo | None:
    """Match ``addmm(bias, x, W.T)`` or ``view → addmm → view`` from bias side.

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

    # Check for paired view → addmm → view (3D F.linear)
    if (
        _is_last_dim_preserving_view(inputs)
        and len(addmm_node.users) == 1
        and _is_last_dim_preserving_view(view_out := next(iter(addmm_node.users)))
    ):
        return AffineLayerInfo(LINEAR_STR, view_out, W_node, inputs.args[0], bias, {})

    # 2D case
    return AffineLayerInfo(LINEAR_STR, addmm_node, W_node, inputs, bias, {})


def _match_add_bias(p_node: Node, add_node: Node) -> AffineLayerInfo | None:
    """Match ``view → mm → view → add(_, bias)`` from the bias side.

    This pattern is produced by F.linear for 4D+ inputs with bias.

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

    # Expect: view → mm → view (= other) → add
    if not _is_last_dim_preserving_view(other):
        return None
    mm_node = other.args[0]
    if not (mm_node.op == "call_function" and mm_node.target == aten.mm.default):
        return None
    mm_input, WT = mm_node.args
    if not _is_last_dim_preserving_view(mm_input):
        return None

    W_node = _extract_weight_node(WT)
    return AffineLayerInfo(LINEAR_STR, add_node, W_node, mm_input.args[0], p_node, {})


class LinearWeightMatcher(_PatternMatcher):
    """Matcher for weight parameters in linear layers.

    Detects patterns:
    - ``addmm(b, x, W.T)`` (2D with bias)
    - ``view → addmm → view`` (3D with bias)
    - ``mm(x, W.T)`` (2D, no bias)
    - ``mm(x, W.T) → add(_, b)`` (2D with bias, rare)
    - ``view → mm → view`` (>2D, no bias)
    - ``view → mm → view → add(_, b)`` (>2D with bias)
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
    - ``addmm(b, x, W.T)`` where the node is ``b`` (2D)
    - ``view → addmm → view`` where the node is ``b`` (3D)
    - ``view → mm → view → add(_, b)`` where the node is ``b`` (>2D)
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
