"""Base classes and utilities for parameter usage pattern matching in FX graphs."""

from typing import Any, Dict, List, Optional, Tuple, Union

from torch.fx import Node


NOT_A_PARAM: str = "__not_a_param"


def _create_info_tuple(
    operation_name: str,
    y_node: Node,
    x_node: Node,
    W_node: Union[Node, None],
    b_node: Optional[Node],
    hyperparams: Dict[str, Any],
    node_name_to_param_name: Dict[str, str],
) -> Tuple[str, Node, Node, Optional[str], Optional[str], Dict[str, Any]]:
    """Create a layer info tuple with common logic.

    Args:
        operation_name: Name of the operation (e.g., "Linear(y=x@W^T+b)").
        y_node: The output node.
        x_node: The input node.
        W_node: The weight parameter node.
        b_node: The bias parameter node.
        hyperparams: Dictionary of layer hyperparameters.
        node_name_to_param_name: Mapping from FX node names to parameter names.

    Returns:
        Tuple containing:
            (operation_name, y_node, x_node, weight_name, bias_name, hyperparams).
    """
    weight_name = (
        node_name_to_param_name.get(W_node.name, NOT_A_PARAM)
        if W_node is not None
        else NOT_A_PARAM
    )
    bias_name = (
        None
        if b_node is None
        else node_name_to_param_name.get(b_node.name, NOT_A_PARAM)
    )
    return (operation_name, y_node, x_node, weight_name, bias_name, hyperparams)


class PatternMatcher:
    """Base class for matching parameter usage patterns in FX graphs.

    Subclasses implement specific pattern matching logic for different layer types
    (e.g., linear layers with/without bias).
    """

    def matches(
        self, p_node: Node
    ) -> Tuple[List[Any], List[Tuple[Node, ...]]]:
        """Attempt to match a parameter node to known usage patterns.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of layer info objects for all matches found for this pattern.
                - List of paths from parameter node to detected output nodes.
        """
        raise NotImplementedError
