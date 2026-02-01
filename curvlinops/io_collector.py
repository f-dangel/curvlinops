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

from typing import Any, Callable, Dict, Tuple, Union

from torch import Tensor
from torch.func import functionalize
from torch.fx.experimental.proxy_tensor import make_fx

from curvlinops.io_patterns import NOT_A_PARAM, match_parameter_usage
from curvlinops.io_verification import verify_match_complete
from curvlinops.kfac import FisherType


def with_param_io(
    f: Callable[[Tensor, Dict[str, Tensor]], Tensor],
    x: Tensor,
    named_params: Dict[str, Tensor],
) -> Callable[[Tensor, Dict[str, Tensor]], Tuple[Union[Tensor, Tuple[Any, ...]], ...]]:
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
            ("Linear", layer_output, layer_input, weight_name, bias_name)
        with bias_name being None if the layer has no bias and weight_name or
        bias_name being `NOT_A_PARAM` if the tensor used as weight or bias is
        not a parameter of `named_params`.

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
    usage_info, detected_paths = match_parameter_usage(param_nodes)

    # Verify that we detected all usages
    try:
        verify_match_complete(param_nodes, detected_paths)
    except ValueError as e:
        # Prepend the fx.graph to the error message and raise
        raise ValueError(f"FX Graph:\n{gm.graph}\n\n{str(e)}") from e

    # Build the layer info tuples to return alongside the original output
    # Format: ("Linear", input_node, output_node, weight_name | None, bias_name | None)
    layer_info_tuples = [
        info.to_info_tuple(node_name_to_param_name) for info in usage_info
    ]

    # Find the original output node and modify its argument
    (output_node,) = [n for n in gm.graph.nodes if n.op == "output"]
    ((output_value_node,),) = output_node.args
    output_node.args = (((output_value_node, *layer_info_tuples),),)

    # Recompile the graph after modification
    gm.graph.lint()
    gm.recompile()

    return gm


def with_kfac_io(
    f: Callable[[Tensor, Dict[str, Tensor]], Tensor],
    x: Tensor,
    named_params: Dict[str, Tensor],
    fisher_type: str,
) -> Callable[
    [Tensor, Dict[str, Tensor]],
    Tuple[
        Tensor,
        Dict[str, Tensor],
        Dict[str, Tensor],
        Dict[str, Dict[str, str]],
        Dict[str, Dict[str, Any]],
    ],
]:
    """Return layers and their relevant inputs/outputs of parameters.

    Args:
        f: Function to trace and augment with KFAC IO collection.
        x: Example input tensor for tracing.
        named_params: Dictionary mapping parameter names to parameter tensors.
        fisher_type: Type of Fisher information computation.

    Returns:
        Traced function that returns:
            - Original function output
            - Layer inputs (dict mapping layer names to input tensors)
            - Layer outputs (dict mapping layer names to output tensors)
            - Layer parameter names (dict mapping layer names to param name dicts)
            - Layer hyperparameters (dict mapping layer names to hyperparameter dicts)
    """
    assert fisher_type in FisherType
    f_with_param_io = with_param_io(f, x, named_params)

    # Extract layer info from the traced function's output structure to check param usage
    # The output node contains ((output_value_node, *layer_info_tuples),)
    (output_node,) = [n for n in f_with_param_io.graph.nodes if n.op == "output"]
    ((output_tuple,),) = output_node.args
    layer_info_tuples = output_tuple[1:]  # Skip the first element (original output)

    # Check parameter usage once during setup
    # Make sure that each parameter is only used in a single layer info
    # (multiple usages are currently unsupported)
    param_usages = dict.fromkeys(named_params, 0)
    for layer_info_tuple in layer_info_tuples:
        # Each layer_info_tuple is:
        # ("Linear", y_node, x_node, weight_name, bias_name, hyperparams)
        _, _, _, weight_name, bias_name, _ = layer_info_tuple
        if weight_name in param_usages:
            param_usages[weight_name] += 1
        if bias_name in param_usages:
            param_usages[bias_name] += 1

    if any(usage > 1 for usage in param_usages.values()):
        raise ValueError(
            f"Parameters used multiple times (currently unsupported): {param_usages}"
        )

    def f_and_kfac_io(
        x: Tensor, params: Dict[str, Tensor]
    ) -> Tuple[
        Tensor,
        Dict[str, Tensor],
        Dict[str, Tensor],
        Dict[str, Dict[str, str]],
        Dict[str, Dict[str, Any]],
    ]:
        """Evaluate the function and return all relevant in/outputs for KFAC.

        Returns:
            Tuple containing:
                - Original function output
                - Layer inputs (dict mapping layer names to input tensors)
                - Layer outputs (dict mapping layer names to output tensors)
                - Layer parameter names (dict mapping layer names to param name dicts)
                - Layer hyperparameters (dict mapping layer names to hyperparameter dicts)
        """
        # Evaluate the function and its param IOs
        out_with_io = f_with_param_io(x, params)
        out, layer_infos = out_with_io[0], out_with_io[1:]

        # Look at the IO and figure out which parameters are one layer.
        # Set up a dictionary that maps layers (e.g. 'Linear0') to their weight
        # and bias names {'weight': weight_name, 'bias': bias_name}.
        # Also set up and fill dictionaries for layer inputs and outputs.
        layer_names: Dict[str, Dict[str, str]] = {}
        layer_inputs: Dict[str, Tensor] = {}
        layer_outputs: Dict[str, Tensor] = {}
        layer_hyperparams: Dict[str, Dict[str, Any]] = {}

        for i, layer_info in enumerate(layer_infos):
            op, y, x, weight_name, bias_name, hyperparams = layer_info
            if op.startswith("Linear("):
                name = f"Linear{i}"
            elif op.startswith("Conv2d("):
                name = f"Conv2d{i}"
            else:
                raise ValueError(f"Unsupported operation: {op}")
            layer_names[name] = {}
            layer_hyperparams[name] = hyperparams

            if weight_name != NOT_A_PARAM:
                layer_inputs[name] = x
                if fisher_type != "forward-only":
                    layer_outputs[name] = y
                layer_names[name]["weight"] = weight_name
            if bias_name not in {None, NOT_A_PARAM}:
                if fisher_type != "forward-only":
                    layer_outputs[name] = y
                layer_names[name]["bias"] = bias_name

        return out, layer_inputs, layer_outputs, layer_names, layer_hyperparams

    return make_fx(f_and_kfac_io)(x, named_params)
