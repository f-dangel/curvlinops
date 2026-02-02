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

from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple, TypeAlias, Union

from torch import Tensor
from torch.func import functionalize
from torch.fx import GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx

from curvlinops.io_patterns import match_parameter_usage
from curvlinops.io_patterns._base import NOT_A_PARAM, as_tuple
from curvlinops.io_patterns.conv import CONV_STR
from curvlinops.io_patterns.linear import LINEAR_STR
from curvlinops.io_verification import verify_match_complete
from curvlinops.kfac import FisherType

# Type aliases for complex return types
LayerInfoTuple: TypeAlias = Tuple[
    str, Node, Node, str, Union[str, None], Dict[str, Any]
]
ParamIOFunction: TypeAlias = Callable[
    [Tensor, Dict[str, Tensor]], Tuple[Union[Tensor, Tuple[Any, ...]], ...]
]
KFACIOFunction: TypeAlias = Callable[
    [Tensor, Dict[str, Tensor]],
    Tuple[
        Tensor,
        Dict[str, Tensor],
        Dict[str, Tensor],
        Dict[str, Dict[str, str]],
        Dict[str, Dict[str, Any]],
    ],
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


def _verify_supported_by_kfac(
    layer_info_tuples: Tuple[LayerInfoTuple, ...], named_params: Dict[str, Tensor]
) -> None:
    """Verify that the detected layer patterns are supported by KFAC.

    Args:
        layer_info_tuples: Tuples containing layer information from pattern matching.
        named_params: Dictionary mapping parameter names to parameter tensors.

    Raises:
        ValueError: If unsupported patterns are detected (multiple parameter usage,
            transposed convolutions, or non-2D convolutions).
    """
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

    # Make sure there is no transposed and no 1D or 3D convolution
    for layer_info_tuple in layer_info_tuples:
        op, _, _, _, _, hyperparams = layer_info_tuple
        if op == CONV_STR:
            if hyperparams["transposed"]:
                raise ValueError("Transposed convolutions are currently unsupported")

            # Determine the convolution's dimension from the hyperparameters
            param_lengths: set[int] = {
                len(hyperparams[key])
                for key in ["stride", "padding", "dilation", "output_padding"]
            }
            if len(param_lengths) != 1:
                raise ValueError("Inconsistent convolution parameter dimensions")
            (conv_dim,) = param_lengths
            if conv_dim != 2:
                raise ValueError(
                    f"{conv_dim}D convolutions are currently unsupported. "
                    f"Only 2D convolutions are supported."
                )


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


def _extract_layer_info_tuples(
    f_with_param_io: GraphModule,
) -> Tuple[LayerInfoTuple, ...]:
    """Extract layer info tuples from traced function's output structure.

    Args:
        f_with_param_io: Traced function module with parameter IO information.

    Returns:
        Tuple of layer info tuples from the graph.
    """
    (output_node,) = [n for n in f_with_param_io.graph.nodes if n.op == "output"]
    ((output_tuple,),) = output_node.args
    return output_tuple[1:]  # Skip the first element (original output)


def _process_layer_info_tuple(
    layer_info_tuple: LayerInfoTuple,
    op_to_prefix: Dict[str, str],
    counts: Dict[str, int],
    fisher_type: str,
) -> Tuple[str, bool, bool, Dict[str, str], Dict[str, Any]]:
    """Process a single layer info tuple to extract configuration.

    Args:
        layer_info_tuple: Tuple containing (op, y, x, weight_name, bias_name, hyperparams).
        op_to_prefix: Mapping from operation strings to layer name prefixes.
        counts: Dictionary tracking layer counts by type.
        fisher_type: Type of Fisher information computation.

    Returns:
        Tuple of (layer_name, store_input, store_output, param_names, hyperparams).
    """
    op, _, _, weight_name, bias_name, hyperparams = layer_info_tuple

    if op not in op_to_prefix:
        raise ValueError(f"Unsupported operation: {op}")

    prefix = op_to_prefix[op]
    layer_name = f"{prefix}{counts[prefix]}"
    counts[prefix] += 1

    # Determine what to store based on parameter types and Fisher type
    store_input = weight_name != NOT_A_PARAM
    store_output = fisher_type != "forward-only" and (
        weight_name != NOT_A_PARAM or bias_name not in {None, NOT_A_PARAM}
    )

    # Build parameter names mapping
    param_names = {}
    if weight_name != NOT_A_PARAM:
        param_names["weight"] = weight_name
    if bias_name not in {None, NOT_A_PARAM}:
        param_names["bias"] = bias_name

    return layer_name, store_input, store_output, param_names, hyperparams


def with_kfac_io(
    f: Callable[[Tensor, Dict[str, Tensor]], Tensor],
    x: Tensor,
    named_params: Dict[str, Tensor],
    fisher_type: str,
) -> KFACIOFunction:
    """Return a function that collects layer inputs/outputs for KFAC computation.

    This function analyzes the provided function to detect affine layer operations
    (linear layers and 2D convolutions) and returns a traced version that collects
    the inputs and outputs needed for KFAC (Kronecker-Factored Approximate Curvature)
    computation alongside the original function output.

    Args:
        f: Function to trace and augment with KFAC IO collection. Should have signature
            f(x, params) -> output where x is the input tensor and params is a parameter dict.
        x: Example input tensor for tracing. Must be representative of actual inputs.
        named_params: Dictionary mapping parameter names to parameter tensors. Keys should
            match parameter names used in function f.
        fisher_type: Type of Fisher information computation. Must be one of the values
            from FisherType enum (e.g., "empirical", "forward-only").

    Returns:
        A traced function with the same signature as f but returning a 5-tuple:
            - Original function output (Tensor)
            - Layer inputs (Dict[str, Tensor]): Maps layer names to input tensors
            - Layer outputs (Dict[str, Tensor]): Maps layer names to output tensors
            - Layer parameter names (Dict[str, Dict[str, str]]): Maps layer names to
              parameter name mappings (e.g., {"weight": "conv1.weight", "bias": "conv1.bias"})
            - Layer hyperparameters (Dict[str, Dict[str, Any]]): Maps layer names to
              hyperparameter dictionaries (empty for linear layers, contains stride/padding/etc
              for convolution layers)

    Raises:
        ValueError: If unsupported layer types are detected (transposed convolutions,
            1D/3D convolutions), if parameters are used multiple times, or if fisher_type
            is not valid.
        AssertionError: If fisher_type is not in the FisherType enum.
    """
    assert fisher_type in FisherType
    f_with_param_io = with_param_io(f, x, named_params)

    # Extract layer info from the traced function's output structure
    layer_info_tuples = _extract_layer_info_tuples(f_with_param_io)

    # Verify that all detected patterns are supported by KFAC
    _verify_supported_by_kfac(layer_info_tuples, named_params)

    # Pre-analyze layers once during setup instead of on every function call
    layer_configs: List[
        Tuple[str, bool, bool]
    ] = []  # [(layer_name, store_input, store_output)]
    layer_names: Dict[str, Dict[str, str]] = {}
    layer_hyperparams: Dict[str, Dict[str, Any]] = {}

    # Create a mapping from operation strings to layer name prefixes
    op_to_prefix: Dict[str, str] = {LINEAR_STR: "Linear", CONV_STR: "Conv"}
    counts: Dict[str, int] = defaultdict(int)

    for layer_info_tuple in layer_info_tuples:
        layer_name, store_input, store_output, param_names, hyperparams = (
            _process_layer_info_tuple(
                layer_info_tuple, op_to_prefix, counts, fisher_type
            )
        )
        layer_configs.append((layer_name, store_input, store_output))
        layer_names[layer_name] = param_names
        layer_hyperparams[layer_name] = hyperparams

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

        # Use pre-computed layer configuration to collect inputs and outputs
        layer_inputs: Dict[str, Tensor] = {}
        layer_outputs: Dict[str, Tensor] = {}

        for (layer_name, store_input, store_output), layer_info in zip(
            layer_configs, layer_infos
        ):
            _, y, x, _, _, _ = layer_info

            if store_input:
                layer_inputs[layer_name] = x
            if store_output:
                layer_outputs[layer_name] = y

        return (out, layer_inputs, layer_outputs, layer_names, layer_hyperparams)

    return make_fx(f_and_kfac_io)(x, named_params)
