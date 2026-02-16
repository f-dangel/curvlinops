"""Function to verify that all usages of parameters were captured."""

from collections.abc import Iterator

from torch.fx import Node


def _truncate_to_mismatch(
    path: tuple[Node, ...], detected_paths_from_p: list[tuple[Node, ...]]
) -> tuple[Node, ...]:
    """Truncate a path to show only up to the first mismatching node.

    Args:
        path: The uncovered path.
        detected_paths_from_p: All detected paths from the same parameter.

    Returns:
        Truncated path up to and including the first mismatching node.
    """
    if not detected_paths_from_p:
        return path

    # Find the longest common prefix with any detected path
    max_common_length = 0
    for detected_path in detected_paths_from_p:
        common_length = 0
        for path_node, detected_node in zip(path, detected_path):
            if path_node == detected_node:
                common_length += 1
            else:
                break
        max_common_length = max(max_common_length, common_length)

    # Return path up to and including the first mismatching node
    return path[: max_common_length + 1] if max_common_length < len(path) else path


def _find_all_paths_from(
    node: Node, max_length: int | None = None
) -> Iterator[tuple[Node, ...]]:
    """Find all usage paths starting from a node.

    Args:
        node: The node to start from.
        max_length: Maximum length of paths to traverse. If None, traverse all
            paths.

    Yields:
        All paths from the node to leaf nodes or up to max_length.
    """

    def _traverse_from_node(
        current_node: Node, path: tuple[Node, ...]
    ) -> Iterator[tuple[Node, ...]]:
        # Always yield the current path (we want all paths up to max_length)
        yield path

        # Stop if we've reached the maximum path length
        if max_length is not None and len(path) >= max_length:
            return

        # Stop if this is a leaf node
        if not current_node.users:
            return

        # Continue to all users
        for user in current_node.users:
            yield from _traverse_from_node(user, path + (user,))

    # Start traversal from node
    yield from _traverse_from_node(node, (node,))


def verify_match_complete(
    param_nodes: list[Node],
    detected_paths: list[tuple[Node, ...]],
) -> None:
    """Verify that all parameter usages were detected by pattern matching.

    Finds all parameter usage paths and verifies that each path starts with
    one of the detected paths (i.e., is covered by pattern matching).

    Args:
        param_nodes: List of parameter nodes to verify.
        detected_paths: List of paths from parameter nodes to detected output nodes.

    Raises:
        ValueError: If any parameter usage path does not start with a detected path.
    """
    all_uncovered_paths = []

    # For each parameter, find all usage paths and check coverage
    for p in param_nodes:
        detected_paths_from_p = [path for path in detected_paths if path[0] == p]

        # We only need to explore paths until the maximum detected path length
        max_length = max([len(path) for path in detected_paths_from_p] + [0])

        for path in _find_all_paths_from(p, max_length):
            # Check if this usage path starts with any detected path
            is_covered = any(
                path == detected_path[: len(path)]
                if len(path) <= len(detected_path)
                else False
                for detected_path in detected_paths_from_p
            )
            if not is_covered:
                # Truncate the path to show only the mismatching part
                truncated_path = _truncate_to_mismatch(path, detected_paths_from_p)

                # Only append the truncated path if it is not already in uncovered paths
                if truncated_path not in all_uncovered_paths:
                    all_uncovered_paths.append(truncated_path)

    # If we found any uncovered paths, raise an error with all of them
    if all_uncovered_paths:
        error_lines = ["Undetected usage paths:"]
        error_lines.extend([
            f"\t{' -> '.join([str(node) for node in path])}"
            for path in all_uncovered_paths
        ])
        error_lines.append("")
        error_lines.append("Some parameters are used in unsupported patterns.")

        raise ValueError("\n".join(error_lines))
