"""General utility functions."""

from typing import Callable, List, MutableMapping, Tuple, Union

from einops import rearrange
from numpy import cumsum, ndarray
from torch import Tensor, as_tensor
from torch.func import functional_call
from torch.nn import CrossEntropyLoss, Module, Parameter


def split_list(x: Union[List, Tuple], sizes: List[int]) -> List[List]:
    """Split a list into multiple lists of specified size.

    Args:
        x: List or tuple to be split.
        sizes: Sizes of the resulting lists.

    Returns:
        List of lists. Each sub-list has the size specified in ``sizes``.

    Raises:
        ValueError: If the sum of ``sizes`` does not match the input list's length.
    """
    if len(x) != sum(sizes):
        raise ValueError(
            f"List to be split has length {len(x)}, but requested sub-list with a total"
            + f" of {sum(sizes)} entries."
        )
    boundaries = cumsum([0] + sizes)
    return [list(x[boundaries[i] : boundaries[i + 1]]) for i in range(len(sizes))]


def allclose_report(
    tensor1: Union[Tensor, ndarray],
    tensor2: Union[Tensor, ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Same as ``allclose``, but prints entries that differ.

    Args:
        tensor1: First tensor for comparison.
        tensor2: Second tensor for comparison.
        rtol: Relative tolerance. Default is ``1e-5``.
        atol: Absolute tolerance. Default is ``1e-8``.

    Returns:
        ``True`` if the tensors are close, ``False`` otherwise.
    """
    tensor1 = as_tensor(tensor1)
    tensor2 = as_tensor(tensor2, device=tensor1.device)
    close = tensor1.allclose(tensor2, rtol=rtol, atol=atol)
    if not close:
        # print non-close values
        nonclose_idx = tensor1.isclose(tensor2, rtol=rtol, atol=atol).logical_not_()
        nonclose_entries = 0
        for idx, t1, t2 in zip(
            nonclose_idx.argwhere(),
            tensor1[nonclose_idx].flatten(),
            tensor2[nonclose_idx].flatten(),
        ):
            print(f"at index {idx.tolist()}: {t1:.5e} ≠ {t2:.5e}, ratio: {t1 / t2:.5e}")
            nonclose_entries += 1

        # print largest and smallest absolute entries
        amax1, amax2 = tensor1.abs().max().item(), tensor2.abs().max().item()
        print(f"Abs max: {amax1:.5e} vs. {amax2:.5e}.")
        amin1, amin2 = tensor1.abs().min().item(), tensor2.abs().min().item()
        print(f"Abs min: {amin1:.5e} vs. {amin2:.5e}.")

        # print number of nonclose values and tolerances
        print(f"Non-close entries: {nonclose_entries} / {tensor1.numel()}.")
        print(f"rtol = {rtol}, atol = {atol}.")

    return close


def assert_is_square(A) -> int:
    """Assert that a matrix or linear operator is square.

    Args:
        A: Matrix or linear operator to be checked. Must have a ``.shape`` attribute.

    Returns:
        The dimension of the square matrix.

    Raises:
        ValueError: If the matrix is not square.
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Operator must be square. Got shape {A.shape}.")
    (dim,) = set(A.shape)
    return dim


def assert_matvecs_subseed_dim(A, num_matvecs: int):
    """Assert that the number of matrix-vector products is smaller than the dimension.

    Args:
        A: Matrix or linear operator to be checked. Must have a ``.shape`` attribute.
        num_matvecs: Number of matrix-vector products.

    Raises:
        ValueError: If the number of matrix-vector products is greater than the dimension.
    """
    if any(num_matvecs >= d for d in A.shape):
        raise ValueError(
            f"num_matvecs ({num_matvecs}) must be less than A's size ({A.shape})."
        )


def assert_divisible_by(num: int, divisor: int, name: str):
    """Assert that a number is divisible by another number.

    Args:
        num: Number to be checked.
        divisor: Divisor.
        name: Name of the number.

    Raises:
        ValueError: If the number is not divisible by the divisor.
    """
    if num % divisor != 0:
        raise ValueError(f"{name} ({num}) must be divisible by {divisor}.")


def make_functional_call(
    module: Module, free_param_names: List[str]
) -> Callable[[Tuple[Parameter, ...]], Tensor]:
    """Create a function that calls a module with given free parameters.

    Args:
        module: The PyTorch module to make functional.
        free_param_names: Names of parameters that will be passed as arguments.

    Returns:
        A function that takes free parameters and module inputs, returning the
        module's output. For model functions, inputs are typically (X,). For loss
        functions, inputs are typically (predictions, targets).
    """
    # Detect frozen parameters and buffers not in free_param_names
    frozen_params = {
        n: p for n, p in module.named_parameters() if n not in free_param_names
    }
    frozen_buffers = dict(module.named_buffers())
    num_free_params = len(free_param_names)

    def functional_module(*args: Tuple[Parameter, ...]) -> Tensor:
        """Call the module functionally with free parameters and module inputs.

        Args:
            *args: First len(free_param_names) arguments are free parameters,
                remaining arguments are inputs to the module.

        Returns:
            Module output.
        """
        # Separate free parameters and module inputs
        free_params = dict(zip(free_param_names, args[:num_free_params]))
        module_inputs = args[num_free_params:]

        # Call module with all parameters and buffers
        return functional_call(
            module, {**free_params, **frozen_params, **frozen_buffers}, module_inputs
        )

    return functional_module


def make_functional_model_and_loss(
    model_func: Module, loss_func: Module, params: Tuple[Parameter, ...]
) -> Tuple[Callable[[Tuple[Tensor, ...]], Tensor], Callable[[Tensor, Tensor], Tensor]]:
    """Create functional versions of model and loss functions.

    Args:
        model_func: The neural network model.
        loss_func: The loss function.
        params: A tuple of parameters w.r.t. which the functions are made functional.
            All parameters must be part of ``model_func.parameters()``.

    Returns:
        A tuple containing:
        - f: Functional model with signature (*params, X) -> prediction
        - c: Functional loss with signature (prediction, y) -> loss
    """
    # detect the parameters w.r.t. which the functions are made functional
    free_param_names = []
    for p in params:
        (name,) = [n for n, pp in model_func.named_parameters() if pp is p]
        free_param_names.append(name)

    # Create functional versions of model and loss
    f = make_functional_call(model_func, free_param_names)  # *params, X -> prediction
    c = make_functional_call(loss_func, [])  # prediction, y -> loss

    return f, c


def make_functional_flattened_model_and_loss(
    model_func: Module, loss_func: Module, params: Tuple[Parameter, ...]
) -> Tuple[
    Callable[[Tuple[Tensor, ...], Tensor], Tensor], Callable[[Tensor, Tensor], Tensor]
]:
    """Create flattened versions of model and loss functions.

    This is required for the (empirical) Fisher, for which we don't know how to handle
    additional axes. Therefore, they are flattened into the batch axis.

    Args:
        model_func: The neural network module.
        loss_func: The loss function module.
        params: Free parameters of the model.

    Returns:
        Tuple of (f_flat, c_flat) where:
        - f_flat: Function that executes model and flattens batch and shared axes:
          (*params, X) -> output_flat
        - c_flat: Function that executes loss with flattened labels:
          (output_flat, y) -> loss
    """
    # Create functional versions of model and loss
    f, c = make_functional_model_and_loss(model_func, loss_func, params)

    # Determine how to flatten
    output_flattening = (
        "batch c ... -> (batch ...) c"
        if isinstance(loss_func, CrossEntropyLoss)
        else "batch ... c -> (batch ...) c"
    )
    label_flattening = (
        "batch ... -> (batch ...)"
        if isinstance(loss_func, CrossEntropyLoss)
        else "batch ... c -> (batch ...) c"
    )

    # Set up functions that operate on flattened quantities
    def f_flat(*params_and_X: Union[Tensor, MutableMapping]) -> Tensor:
        """Execute model and flatten batch and shared axes.

        If >2d output we convert to an equivalent 2d output for loss computation.
        For CrossEntropyLoss: (batch, c, ...) -> (batch*..., c)
        For other losses: (batch, ..., c) -> (batch*..., c)

        Args:
            params_and_X: Parameters and input data X.

        Returns:
            Flattened model output.
        """
        output = f(*params_and_X)
        return rearrange(output, output_flattening)

    def c_flat(output_flat: Tensor, y: Tensor) -> Tensor:
        """Execute loss with flattened labels.

        Flattens the labels to match the flattened output format:
        For CrossEntropyLoss: (batch, ...) -> (batch*...)
        For other losses: (batch, ..., c) -> (batch*..., c)

        Args:
            output_flat: Flattened model_output.
            y: Un-flattened labels

        Returns:
            The loss.
        """
        y_flat = rearrange(y, label_flattening)
        return c(output_flat, y_flat)

    return f_flat, c_flat
