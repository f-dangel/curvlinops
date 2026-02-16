"""Pattern matching system for detecting parameter usage in PyTorch FX graphs."""

from torch.fx import Node

from curvlinops.computers.io_collector._base import AffineLayerInfo
from curvlinops.computers.io_collector.conv import (
    ConvolutionBiasMatcher,
    ConvolutionWeightMatcher,
)
from curvlinops.computers.io_collector.linear import (
    LinearBiasMatcher,
    LinearWeightMatcher,
)


def match_parameter_usage(
    param_nodes: list[Node],
) -> tuple[list[AffineLayerInfo], list[tuple[Node, ...]]]:
    """Match parameter nodes against known usage patterns.

    Args:
        param_nodes: List of parameter nodes to analyze.

    Returns:
        Tuple containing:
            - List of affine layer info objects for all matched patterns.
            - List of paths from parameter nodes to detected output nodes.
    """
    patterns = [
        LinearWeightMatcher(),
        LinearBiasMatcher(),
        ConvolutionWeightMatcher(),
        ConvolutionBiasMatcher(),
    ]
    usage_info: list[AffineLayerInfo] = []
    path_info: list[tuple[Node, ...]] = []

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
