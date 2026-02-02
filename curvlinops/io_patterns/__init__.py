"""Pattern matching system for detecting parameter usage in PyTorch FX graphs."""

from typing import List, Tuple

from torch.fx import Node

from ._base import AffineLayerInfo
from .conv import ConvolutionBiasMatcher, ConvolutionWeightMatcher
from .linear import LinearBiasMatcher, LinearWeightMatcher


def match_parameter_usage(
    param_nodes: List[Node],
) -> Tuple[List[AffineLayerInfo], List[Tuple[Node, ...]]]:
    """Match parameter nodes against known usage patterns.

    Args:
        param_nodes: List of parameter nodes to analyze.

    Returns:
        Tuple containing:
            - List of affine layer info objects for all matched patterns.
            - List of paths from parameter nodes to detected output nodes.

    Raises:
        ValueError: If a parameter node is used in an unsupported operation.
    """
    patterns = [
        LinearWeightMatcher(),
        LinearBiasMatcher(),
        ConvolutionWeightMatcher(),
        ConvolutionBiasMatcher(),
    ]
    usage_info: List[AffineLayerInfo] = []
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
