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

from typing import Callable, Dict

from torch import Tensor, manual_seed, rand_like, randn
from torch.func import functional_call, functionalize
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import Linear, Sequential

from curvlinops.io_patterns import match_parameter_usage


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
    try:
        usage_info = match_parameter_usage(param_nodes)
    except ValueError as e:
        raise ValueError(f"FX Graph:\n{gm.graph}\n\n{e}") from e

    # NOTE We may want to verify here that the usage info is complete, i.e. that the parameter nodes are not consumed in additional ways compared to the detected usage

    # Find the original output node
    (output_node,) = [n for n in gm.graph.nodes if n.op == "output"]
    ((output_value_node,),) = output_node.args

    # Build the layer info tuples to return alongside the original output
    # Format: ("Linear", input_node, output_node, weight_name | None, bias_name | None)
    layer_info_tuples = []

    for info in usage_info:
        layer_info_tuples.append(info.to_info_tuple(node_name_to_param_name))

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
