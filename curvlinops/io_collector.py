"""Implements functions to collect layer inputs and outputs.

More specifically we want to collect immediate children (e.g. outputs) of parameters
(e.g. weights, biases), their relation (e.g. fully-connected), and additional inputs
(e.g. input to a fully-connected layer). This information is useful as we can use it
to reconstruct (pseudo-)gradients w.r.t. parameters which can be used to build up all
kinds of curvature approximations (e.g. (E)KFAC).

The entry point is a function f that takes a tensor x representing the input, and a
dictionary containing a mapping from names to parameter tensors. The goal is to de-
tect how these parameters are used, and to return a list of these relations.
This is done by tracing f into a `torch.fx.GraphModule`, and inspecting its graph,
mattern for usage patterns such as fully-connected layers.

With the above mechanism, we can now augment f to not only return f(x), but also
the intermediates that are consumed and produced by the specified parameters.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from torch import Tensor, manual_seed, rand_like, randn
from torch.func import functional_call, functionalize
from torch.fx import GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import Linear, Sequential
from torch.ops import aten


@dataclass
class LinearLayerInfo:
    """Information about a detected linear layer operation.

    Attributes:
        weight_name: Name of the weight parameter.
        input_node: The FX node representing the input to the linear operation.
        output_node: The FX node representing the output of the linear operation.
        bias_name: Name of the bias parameter, if present.
    """

    weight_node: Node
    input_node: Node
    output_node: Node
    bias_node: Optional[Node] = None


class PatternMatcher:
    """Base class for matching parameter usage patterns in FX graphs.

    Subclasses implement specific pattern matching logic for different layer types
    (e.g., linear layers with/without bias).
    """

    def matches(self, p_node: Node) -> Optional[LinearLayerInfo]:
        """Attempt to match a parameter node to a known usage pattern.

        Args:
            p_node: A parameter node to check.

        Returns:
            LinearLayerInfo if the node matches this pattern, None otherwise.
        """
        raise NotImplementedError

    @staticmethod
    def _ensure_single_user(node: Node) -> Node:
        """Get the single user of a node, raising if not exactly one.

        Args:
            node: The node to check.

        Returns:
            The single user node.

        Raises:
            ValueError: If the node does not have exactly one user.
        """
        users = list(node.users.keys())
        if len(users) != 1:
            raise ValueError(f"Node {node} is not used once ({len(users)}x).")
        return users[0]


class LinearWeightMatcher(PatternMatcher):
    """Matcher for weight parameters in linear layers.

    Detects patterns: x @ W.T (mm) or x @ W.T + b (addmm).
    """

    def matches(self, p_node: Node) -> Optional[LinearLayerInfo]:
        """Match a parameter node used as weight in a linear layer.

        Args:
            p_node: A parameter node to check.

        Returns:
            LinearLayerInfo if the node is used as a weight, None otherwise.
        """
        user_node = self._ensure_single_user(p_node)

        # Weight must first be transposed
        if not (user_node.op == "call_function" and user_node.target == aten.t.default):
            return None

        T_user_node = self._ensure_single_user(user_node)

        # Case: x @ W.T + b (addmm)
        if (
            T_user_node.op == "call_function"
            and T_user_node.target == aten.addmm.default
        ):
            assert len(T_user_node.args) == 3 and not T_user_node.kwargs
            bias, inputs, _ = T_user_node.args
            layer_info = LinearLayerInfo(p_node, inputs, T_user_node, bias_node=bias)

        # Case: x @ W.T (mm, no bias)
        elif (
            T_user_node.op == "call_function" and T_user_node.target == aten.mm.default
        ):
            assert len(T_user_node.args) == 2 and not T_user_node.kwargs
            inputs, _ = T_user_node.args
            layer_info = LinearLayerInfo(p_node, inputs, T_user_node)

        else:
            layer_info = None

        return layer_info


class LinearBiasMatcher(PatternMatcher):
    """Matcher for bias parameters in linear layers.

    Detects pattern: x @ W.T + b (addmm) where the node is b.
    """

    def matches(self, p_node: Node) -> Optional[LinearLayerInfo]:
        """Match a parameter node used as bias in a linear layer.

        Args:
            p_node: A parameter node to check.

        Returns:
            LinearLayerInfo if the node is used as a bias, None otherwise.
        """
        user_node = self._ensure_single_user(p_node)

        # Bias is used directly in addmm
        if not (
            user_node.op == "call_function" and user_node.target == aten.addmm.default
        ):
            return None

        bias, inputs, WT = user_node.args
        assert WT.op == "call_function" and WT.target == aten.t.default
        (W,) = list(WT.all_input_nodes)
        return LinearLayerInfo(W, inputs, user_node, bias_node=bias)


def _match_parameter_usage(
    param_nodes: List[Node],
) -> List[LinearLayerInfo]:
    """Match parameter nodes against known usage patterns.

    Args:
        param_nodes: List of parameter nodes to analyze.

    Returns:
        List of LinearLayerInfo for all matched patterns.

    Raises:
        ValueError: If a parameter node is used in an unsupported operation.
    """
    patterns: List[PatternMatcher] = [LinearWeightMatcher(), LinearBiasMatcher()]
    usage_info: List[LinearLayerInfo] = []

    for p_node in param_nodes:
        matched = False

        for pattern in patterns:
            layer_info = pattern.matches(p_node)
            if layer_info is not None:
                matched = True
                break

        if layer_info is not None and layer_info not in usage_info:
            usage_info.append(layer_info)

        if not matched:
            raise ValueError(
                f"Parameter node {p_node} is used in an unsupported operation."
            )

    return usage_info


def with_param_io(
    f: Callable[[Tensor, Dict[str, Tensor]], Tensor],
    x: Tensor,
    named_params: Dict[str, Tensor],
) -> GraphModule:
    """Get a traced module that returns layer inputs and outputs alongside the result.

    This function traces f(x, params) using torch.fx, functionalizes any
    inplace operations, and analyzes the graph to detect linear layer usage
    patterns. The returned GraphModule computes the original function output
    along with the inputs and outputs of each detected layer.

    Args:
        f: A function with signature f(x, params) where x is the input tensor
            and params is a dictionary mapping parameter names to tensors.
        x: Example input tensor for tracing.
        named_params: Dictionary mapping parameter names to parameter tensors.

    Returns:
        A traced FX GraphModule that returns a tuple:
            (original_output, layer_info_1, layer_info_2, ...)
        where each layer_info is a tuple:
            ("Linear", layer_input, layer_output, weight_name, bias_name)
        with bias_name being None if the layer has no bias.

    Raises:
        ValueError: If a parameter is used in an unsupported way, or if the
            number of traced parameter nodes doesn't match named_params.
    """
    # Use functionalize to remove inplace operations, then trace the function
    gm = make_fx(functionalize(f))(x, named_params)

    # Find placeholder nodes (inputs to the graph)
    # The first placeholder is the input x, the rest are parameters
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    param_nodes = placeholders[1:]

    # Establish mapping between param names and node names
    if len(named_params) != len(param_nodes):
        raise ValueError(
            f"Expected {len(named_params)} parameter nodes, got {len(param_nodes)}."
        )
    node_name_to_param_name = {
        node.name: param_name for param_name, node in zip(named_params, param_nodes)
    }

    # Match the parameter nodes to usage patterns, such as linear w/o bias.
    usage_info = _match_parameter_usage(param_nodes)

    # NOTE We may want to verify here that the usage info is complete, i.e. that the parameter nodes are not consumed in additional ways compared to the detected usage

    # Find the original output node
    (output_node,) = [n for n in gm.graph.nodes if n.op == "output"]
    ((output_value_node,),) = output_node.args

    # Build the layer info tuples to return alongside the original output
    # Format: ("Linear", input_node, output_node, weight_name | None, bias_name | None)
    layer_info_tuples = []
    for info in usage_info:
        weight_name = node_name_to_param_name.get(info.weight_node.name, None)
        bias_name = (
            None
            if info.bias_node is None
            else node_name_to_param_name.get(info.bias_node.name, None)
        )
        layer_info_tuples.append(
            ("Linear", info.input_node, info.output_node, weight_name, bias_name)
        )

    output_node.args = (((output_value_node, *layer_info_tuples),),)

    # Recompile the graph after modification
    gm.graph.lint()
    gm.recompile()

    return gm


if __name__ == "__main__":
    # Simple one-layer MLP test case
    manual_seed(0)
    mlp = Sequential(Linear(4, 3, bias=True))
    print(dict(mlp.named_parameters()))

    mlp_functional = lambda x, params: functional_call(mlp, params, args=x)

    x_test = randn(2, 4)
    params_test = {name: rand_like(p) for name, p in mlp.named_parameters()}

    mlp_with_io = with_param_io(mlp_functional, x_test, params_test)

    print(mlp_with_io.graph)
    print(mlp_with_io(x_test, params_test))
