"""Base classes and utilities for parameter usage pattern matching in FX graphs."""

from dataclasses import dataclass
from typing import Any

from torch.fx import Node

NOT_A_PARAM: str = "__not_a_param"


@dataclass
class AffineLayerInfo:
    """Information about a detected affine layer operation.

    An affine operation has the form y = W(x) + b where W acts linearly on the input x
    and b is an additive bias term. This includes linear layers (y = W @ x + b) and
    convolutions (y = W * x + b) where * denotes the convolution operation.

    Attributes:
        operation: Name of the operation (e.g., "Linear(y=W@x+b)").
        y: The FX node representing the output of the affine operation.
        W: The FX node representing the weight parameter, None if no weight,
            or NOT_A_PARAM if weight exists but is not tracked.
        x: The FX node representing the input to the affine operation.
        b: The FX node representing the bias parameter, None if no bias,
            or NOT_A_PARAM if bias exists but is not tracked.
        hyperparams: Dictionary of operation hyperparameters (empty for linear layers).
    """

    operation: str
    y: Node
    W: Node | None | str
    x: Node
    b: Node | None | str
    hyperparams: dict[str, Any]


def as_tuple(
    info: AffineLayerInfo,
    node_name_to_param_name: dict[str, str],
) -> tuple[str, Node, Node, str | None, str | None, dict[str, Any]]:
    """Create a layer info tuple from an AffineLayerInfo object.

    Args:
        info: The AffineLayerInfo object to convert.
        node_name_to_param_name: Mapping from FX node names to parameter names.

    Returns:
        Tuple containing:
            (operation, y, x, weight_name, bias_name, hyperparams).
    """
    if isinstance(info.W, str):
        weight_name = info.W  # Should be NOT_A_PARAM
    elif info.W is not None:
        weight_name = node_name_to_param_name.get(info.W.name, NOT_A_PARAM)
    else:
        weight_name = NOT_A_PARAM

    if isinstance(info.b, str):
        bias_name = info.b  # Should be NOT_A_PARAM
    elif info.b is not None:
        bias_name = node_name_to_param_name.get(info.b.name, NOT_A_PARAM)
    else:
        bias_name = None

    return (
        info.operation,
        info.y,
        info.x,
        weight_name,
        bias_name,
        info.hyperparams,
    )


class _PatternMatcher:
    """Base class for matching parameter usage patterns in FX graphs.

    Subclasses implement specific pattern matching logic for different layer types
    (e.g., linear layers with/without bias).
    """

    def matches(self, p_node: Node) -> tuple[list[Any], list[tuple[Node, ...]]]:
        """Attempt to match a parameter node to known usage patterns.

        Args:
            p_node: A parameter node to check.

        Returns:
            Tuple containing:
                - List of layer info objects for all matches found for this pattern.
                - List of paths from parameter node to detected output nodes.
        """
        raise NotImplementedError
