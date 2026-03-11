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
        ``(add, bias)`` if a matching add is found, else ``None``.
    """
    for user in node.users:
        if (
            user.op == "call_function"
            and user.target == aten.add.Tensor
            and len(user.args) == 2
        ):
            lhs, rhs = user.args
            bias = rhs if lhs == node else lhs if rhs == node else None
            if bias is not None and bias.meta["val"].ndim == 1:
                return user, bias


def _extract_weight(WT: Node) -> Node | str:
    """Extract weight from a transpose, or return NOT_A_PARAM.

    Args:
        WT: Node expected to be ``aten.t(weight)``.

    Returns:
        The weight if ``WT`` is a transpose, else ``NOT_A_PARAM``.
    """
    if (WT.op, WT.target) == ("call_function", aten.t.default):
        (W,) = WT.all_input_nodes
        return W
    return NOT_A_PARAM


def _match_addmm_weight(p: Node, pT: Node, addmm: Node) -> AffineLayerInfo | None:
    """Match ``addmm(bias, x, W.T)`` or ``view → addmm → view`` from weight side.

    Args:
        p: The weight parameter node.
        pT: The transpose node ``aten.t(p)``.
        addmm: A candidate ``aten.addmm`` node.

    Returns:
        ``AffineLayerInfo`` if matched, else ``None``.
    """
    if len(addmm.args) != 3 or addmm.kwargs:
        return None
    bias, x, WT = addmm.args
    if WT != pT:
        return None

    # Check for paired view → addmm → view (3D F.linear)
    x_view = x
    if (
        _is_last_dim_preserving_view(x_view)
        and len(addmm.users) == 1
        and _is_last_dim_preserving_view(y_view := next(iter(addmm.users)))
    ):
        x, y = x_view.args[0], y_view
        return AffineLayerInfo(LINEAR_STR, y, p, x, bias, {})

    # 2D case: no reshapes
    return AffineLayerInfo(LINEAR_STR, addmm, p, x, bias, {})


def _match_mm_weight(p: Node, pT: Node, mm: Node) -> AffineLayerInfo | None:
    """Match ``mm(x, W.T)`` or ``view → mm → view [→ add]`` from weight side.

    Supported patterns::

        2D no bias:  mm(x, W.T)
        2D + bias:   add(mm(x, W.T), b)
        >2D no bias: view → mm → view
        >2D + bias:  view → mm → view → add(_, b)

    Args:
        p: The weight parameter node.
        pT: The transpose node ``aten.t(p)``.
        mm: A candidate ``aten.mm`` node.

    Returns:
        ``AffineLayerInfo`` if matched, else ``None``.
    """
    if len(mm.args) != 2 or mm.kwargs:
        return None
    x, WT = mm.args
    if WT != pT:
        return None

    # Check for view → mm → view pattern (>2D F.linear)
    x_view = x
    if (
        _is_last_dim_preserving_view(x_view)
        and len(mm.users) == 1
        and _is_last_dim_preserving_view(y_view := next(iter(mm.users)))
    ):
        x = x_view.args[0]
        add_result = _find_add_bias(y_view)
        if add_result is not None:
            y, bias = add_result
            return AffineLayerInfo(LINEAR_STR, y, p, x, bias, {})
        y = y_view
        return AffineLayerInfo(LINEAR_STR, y, p, x, None, {})

    # 2D case: mm [→ add]
    add_result = _find_add_bias(mm)
    if add_result is not None:
        y, bias = add_result
        return AffineLayerInfo(LINEAR_STR, y, p, x, bias, {})
    y = mm
    return AffineLayerInfo(LINEAR_STR, y, p, x, None, {})


def _match_addmm_bias(p: Node, addmm: Node) -> AffineLayerInfo | None:
    """Match ``addmm(bias, x, W.T)`` or ``view → addmm → view`` from bias side.

    Args:
        p: The bias parameter node.
        addmm: A candidate ``aten.addmm`` node.

    Returns:
        ``AffineLayerInfo`` if matched, else ``None``.
    """
    if len(addmm.args) != 3 or addmm.kwargs:
        return None
    bias, x, WT = addmm.args
    if bias != p:
        return None
    W = _extract_weight(WT)

    # Check for paired view → addmm → view (3D F.linear)
    x_view = x
    if (
        _is_last_dim_preserving_view(x_view)
        and len(addmm.users) == 1
        and _is_last_dim_preserving_view(y_view := next(iter(addmm.users)))
    ):
        x, y = x_view.args[0], y_view
        return AffineLayerInfo(LINEAR_STR, y, W, x, bias, {})

    # 2D case
    return AffineLayerInfo(LINEAR_STR, addmm, W, x, bias, {})


def _match_add_bias(p: Node, add: Node) -> AffineLayerInfo | None:
    """Match ``view → mm → view → add(_, bias)`` from the bias side.

    This pattern is produced by F.linear for 4D+ inputs with bias.

    Args:
        p: The bias parameter node.
        add: A candidate ``aten.add.Tensor`` node.

    Returns:
        ``AffineLayerInfo`` if matched, else ``None``.
    """
    if len(add.args) != 2 or p.meta["val"].ndim != 1:
        return None
    lhs, rhs = add.args
    y_view = lhs if rhs == p else rhs if lhs == p else None
    if y_view is None:
        return None

    # Expect: view → mm → view (= y_view) → add
    if not _is_last_dim_preserving_view(y_view):
        return None
    mm = y_view.args[0]
    if not (mm.op == "call_function" and mm.target == aten.mm.default):
        return None
    x_view, WT = mm.args
    if not _is_last_dim_preserving_view(x_view):
        return None

    W = _extract_weight(WT)
    x = x_view.args[0]
    return AffineLayerInfo(LINEAR_STR, add, W, x, p, {})


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

    def matches(self, p: Node) -> tuple[list[AffineLayerInfo], list[tuple[Node, ...]]]:
        """Match a parameter node used as weight in linear layers.

        Args:
            p: A parameter node to check.

        Returns:
            Tuple containing:
                - List of AffineLayerInfo for all weight usage matches found.
                - List of paths from parameter node to detected output nodes.
        """
        matches, paths = [], []

        for pT in p.users:
            if (pT.op, pT.target) != ("call_function", aten.t.default):
                continue

            for pT_user in pT.users:
                if pT_user.op != "call_function":
                    continue

                if pT_user.target == aten.addmm.default:
                    info = _match_addmm_weight(p, pT, pT_user)
                elif pT_user.target == aten.mm.default:
                    info = _match_mm_weight(p, pT, pT_user)
                else:
                    continue

                if info is not None:
                    matches.append(info)
                    paths.append((p, pT, pT_user))

        return matches, paths


class LinearBiasMatcher(_PatternMatcher):
    """Matcher for bias parameters in linear layers.

    Detects patterns:
    - ``addmm(b, x, W.T)`` where the node is ``b`` (2D)
    - ``view → addmm → view`` where the node is ``b`` (3D)
    - ``view → mm → view → add(_, b)`` where the node is ``b`` (>2D)
    """

    def matches(self, p: Node) -> tuple[list[AffineLayerInfo], list[tuple[Node, ...]]]:
        """Match a parameter node used as bias in linear layers.

        Args:
            p: A parameter node to check.

        Returns:
            Tuple containing:
                - List of AffineLayerInfo for all bias usage matches found.
                - List of paths from parameter node to detected output nodes.
        """
        matches, paths = [], []

        for p_user in p.users:
            if p_user.op != "call_function":
                continue

            if p_user.target == aten.addmm.default:
                info = _match_addmm_bias(p, p_user)
            elif p_user.target == aten.add.Tensor:
                info = _match_add_bias(p, p_user)
            else:
                continue

            if info is not None:
                matches.append(info)
                paths.append((p, p_user))

        return matches, paths
