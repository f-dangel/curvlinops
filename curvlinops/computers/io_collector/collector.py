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

from typing import Any, Callable, Dict, List, Tuple, TypeAlias, Union

from torch import Tensor
from torch.func import functionalize
from torch.fx import GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx

from curvlinops.computers.io_collector._base import as_tuple
from curvlinops.computers.io_collector.patterns import match_parameter_usage
from curvlinops.computers.io_collector.verification import verify_match_complete

# Type aliases for complex return types
LayerInfoTuple: TypeAlias = Tuple[
    str, Node, Node, str, Union[str, None], Dict[str, Any]
]
ParamIOFunction: TypeAlias = Callable[
    [Tensor, Dict[str, Tensor]], Tuple[Union[Tensor, Tuple[Any, ...]], ...]
]


def _modify_graph_to_include_layer_info(
    gm: GraphModule, layer_info_tuples: Tuple[Tuple[Any, ...], ...]
) -> None:
    """Modify the graph to return layer info alongside the original output.

    Args:
        gm: The GraphModule to modify.
        layer_info_tuples: Layer information tuples to include in output.
    """
    # Find the original output node and modify its argument
    (output_node,) = [n for n in gm.graph.nodes if n.op == "output"]
    ((output_value_node,),) = output_node.args
    output_node.args = (((output_value_node, *layer_info_tuples),),)

    # Recompile the graph after modification
    gm.graph.lint()
    gm.recompile()


def with_param_io(
    f: Callable[[Tensor, Dict[str, Tensor]], Tensor],
    x: Tensor,
    named_params: Dict[str, Tensor],
) -> ParamIOFunction:
    """Get a traced module that returns layer inputs and outputs alongside the result.

    This function traces f(x, params) using torch.fx, functionalizes any
    inplace operations, and analyzes the graph to detect affine layer usage
    patterns (linear layers and convolutions). The returned GraphModule computes
    the original function output along with the inputs and outputs of each detected layer.

    Args:
        f: A function with signature f(x, params) where x is the input tensor
            and params is a dictionary mapping parameter names to tensors.
        x: Example input tensor for tracing. Must be representative of the actual
            inputs that will be used during execution.
        named_params: Dictionary mapping parameter names to parameter tensors.
            The keys should match the parameter names used in the function f.

    Returns:
        A traced FX GraphModule that returns a tuple:
            (original_output, layer_info_1, layer_info_2, ...)
        where each layer_info is a tuple:
            (operation_name, layer_output, layer_input, weight_name, bias_name, hyperparams)
        The operation_name is either "Linear(y=W@x+b)" or "Conv2d(y=W*x+b)".
        bias_name is None if the layer has no bias, and weight_name or bias_name
        is `NOT_A_PARAM` if the tensor used as weight or bias is not a tracked parameter.
        hyperparams is a dictionary of layer-specific parameters (empty for linear layers).

    Raises:
        ValueError: If a parameter is used in an unsupported way, if the
            number of traced parameter nodes doesn't match named_params, or
            if not all parameter usage paths are detected by the pattern matchers.
    """
    # Use functionalize to remove inplace operations, then trace the function
    gm = make_fx(functionalize(f))(x, named_params)

    # Find placeholder nodes (inputs to the graph)
    # The first placeholder is the input x, the rest are parameters
    placeholders: List[Node] = [
        node for node in gm.graph.nodes if node.op == "placeholder"
    ]
    param_nodes: List[Node] = placeholders[1:]

    # Establish mapping between param names and node names
    if len(named_params) != len(param_nodes):
        raise ValueError(
            f"Expected {len(named_params)} parameter nodes, got {len(param_nodes)}."
        )
    node_name_to_param_name: Dict[str, str] = {
        node.name: param_name for param_name, node in zip(named_params, param_nodes)
    }

    # Match the parameter nodes to usage patterns, such as linear w/o bias.
    usage_info, detected_paths = match_parameter_usage(param_nodes)

    # Verify that we detected all usages
    try:
        verify_match_complete(param_nodes, detected_paths)
    except ValueError as e:
        # Prepend the fx.graph to the error message and raise
        raise ValueError(f"FX Graph:\n{gm.graph}\n\n{str(e)}") from e

    # Build the layer info tuples to return alongside the original output
    # Format: (operation_name, output_node, input_node, weight_name, bias_name, hyperparams)
    layer_info_tuples: Tuple[LayerInfoTuple, ...] = tuple(
        as_tuple(info, node_name_to_param_name) for info in usage_info
    )

    # Modify graph to include layer info in output
    _modify_graph_to_include_layer_info(gm, layer_info_tuples)

    return gm
