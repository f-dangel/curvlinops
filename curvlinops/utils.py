"""General utility functions."""

from collections.abc import Callable, Iterable, MutableMapping
from functools import partial

from einops import rearrange
from numpy import cumsum, ndarray
from torch import Generator, Tensor, as_tensor, device, dtype
from torch.func import functional_call
from torch.nn import CrossEntropyLoss, Module, Parameter


def _infer_device(objects: Iterable) -> device:
    """Infer a single device from objects that have a ``.device`` attribute.

    Args:
        objects: Iterable of objects with a ``.device`` attribute (e.g. tensors,
            parameters, or linear operators).

    Returns:
        The common device.

    Raises:
        RuntimeError: If the objects live on different devices.
    """
    devices = {obj.device for obj in objects}
    if len(devices) != 1:
        raise RuntimeError(f"Expected single device, got {devices}.")
    return devices.pop()


def _infer_dtype(objects: Iterable) -> dtype:
    """Infer a single data type from objects that have a ``.dtype`` attribute.

    Args:
        objects: Iterable of objects with a ``.dtype`` attribute (e.g. tensors,
            parameters, or linear operators).

    Returns:
        The common data type.

    Raises:
        RuntimeError: If the objects have different data types.
    """
    dtypes = {obj.dtype for obj in objects}
    if len(dtypes) != 1:
        raise RuntimeError(f"Expected single dtype, got {dtypes}.")
    return dtypes.pop()


def _seed_generator(generator: Generator | None, dev: device, seed: int) -> Generator:
    """Create (if needed) and seed a random number generator on the given device.

    Re-uses an existing generator when it already lives on the correct device.

    Args:
        generator: An existing generator, or ``None`` to create a new one.
        dev: The device the generator must live on.
        seed: The seed to set on the generator.

    Returns:
        A seeded generator on ``dev``.
    """
    if generator is None or generator.device != dev:
        generator = Generator(device=dev)
    generator.manual_seed(seed)
    return generator


def split_list(x: list | tuple, sizes: list[int]) -> list[list]:
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
    tensor1: Tensor | ndarray,
    tensor2: Tensor | ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Compare tensors like ``allclose`` and print entries that differ.

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


def identify_free_parameters(
    model: Module, params: list[Parameter]
) -> dict[str, Parameter]:
    """Identify free parameters by matching them against a model's named parameters.

    Matches each parameter in ``params`` to its name in ``model`` by data pointer.
    Validates that there are no duplicates in ``params``.

    Args:
        model: The model whose named parameters to search.
        params: List of parameters to identify.

    Returns:
        Ordered dict mapping parameter names to parameter tensors.

    Raises:
        ValueError: If ``params`` contains duplicate tensors or if a parameter
            is not found in the model.
    """
    # Check for duplicates in params
    param_ptrs = [p.data_ptr() for p in params]
    if len(set(param_ptrs)) != len(param_ptrs):
        raise ValueError(
            "params contains duplicate parameters (same tensor passed twice)."
        )

    # Build ptr -> name mapping from model's named parameters
    ptr_to_name = {p.data_ptr(): name for name, p in model.named_parameters()}

    # Match params to names
    named_params: dict[str, Parameter] = {}
    for p in params:
        ptr = p.data_ptr()
        if ptr not in ptr_to_name:
            raise ValueError(
                f"Parameter with data_ptr {ptr} not found in model. "
                f"All free parameters must be part of model.parameters()."
            )
        named_params[ptr_to_name[ptr]] = p

    return named_params


def make_functional_call(module: Module) -> Callable[..., Tensor]:
    """Create a function that calls a module with overridden parameters.

    ``functional_call`` treats the supplied ``params`` as overrides and falls
    back to the module's own parameters and buffers for everything else, so
    there is no need to explicitly capture frozen parameters or buffers.

    Args:
        module: The PyTorch module to make functional.

    Returns:
        A function ``(params, *module_inputs) -> output`` where ``params`` is a
        ``dict[str, Tensor]`` of the parameters to override. For model functions,
        inputs are typically ``(X,)``. For loss functions, inputs are typically
        ``(predictions, targets)``.
    """

    def functional_module(params: dict[str, Tensor], *module_inputs) -> Tensor:
        """Call the module functionally with free parameters and module inputs.

        Args:
            params: Dictionary mapping parameter names to tensor values.
            *module_inputs: Inputs to the module (e.g. X for models, or
                (prediction, target) for loss functions).

        Returns:
            Module output.
        """
        return functional_call(module, params, module_inputs)

    return functional_module


def make_functional_model_and_loss(
    model_func: Module, loss_func: Module
) -> tuple[
    Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
    Callable[[Tensor, tuple], Tensor],
]:
    """Create functional versions of model and loss functions.

    Args:
        model_func: The neural network model.
        loss_func: The loss function.

    Returns:
        A tuple containing:
        - f: Functional model with signature ``(params_dict, X) -> prediction``
        - c: Functional loss with signature ``(prediction, loss_args) -> loss``
    """
    # Create functional versions of model and loss
    f = make_functional_call(model_func)  # (params_dict, X) -> prediction
    c_raw = partial(make_functional_call(loss_func), {})  # (prediction, y) -> loss

    def c(prediction: Tensor, loss_args: tuple) -> Tensor:
        """Evaluate the loss function on a prediction and loss arguments.

        Args:
            prediction: Model prediction.
            loss_args: Tuple of loss function arguments, e.g. ``(y,)``.

        Returns:
            Scalar loss value.
        """
        return c_raw(prediction, *loss_args)

    return f, c


def make_functional_flattened_model_and_loss(
    model_func: Module, loss_func: Module
) -> tuple[
    Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
    Callable[[Tensor, tuple], Tensor],
]:
    """Create flattened versions of model and loss functions.

    This is required for the (empirical) Fisher, for which we don't know how to handle
    additional axes. Therefore, they are flattened into the batch axis.

    Args:
        model_func: The neural network module.
        loss_func: The loss function module.

    Returns:
        Tuple of (f_flat, c_flat) where:
        - f_flat: Function that executes model and flattens batch and shared axes:
          (params, X) -> output_flat
        - c_flat: Function that executes loss with flattened labels:
          (output_flat, loss_args) -> loss
    """
    # Create functional versions of model and loss
    f, c = make_functional_model_and_loss(model_func, loss_func)

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
    def f_flat(params: dict[str, Tensor], X: Tensor | MutableMapping) -> Tensor:
        """Execute model and flatten batch and shared axes.

        If >2d output we convert to an equivalent 2d output for loss computation.
        For CrossEntropyLoss: (batch, c, ...) -> (batch*..., c)
        For other losses: (batch, ..., c) -> (batch*..., c)

        Args:
            params: Model parameters as a dict mapping names to tensors.
            X: Input data.

        Returns:
            Flattened model output.
        """
        output = f(params, X)
        return rearrange(output, output_flattening)

    def c_flat(output_flat: Tensor, loss_args: tuple) -> Tensor:
        """Execute loss with flattened labels.

        Flattens the labels to match the flattened output format:
        For CrossEntropyLoss: (batch, ...) -> (batch*...)
        For other losses: (batch, ..., c) -> (batch*..., c)

        Args:
            output_flat: Flattened model output.
            loss_args: Tuple of ``(y,)`` with un-flattened labels.

        Returns:
            The loss.
        """
        (y,) = loss_args
        y_flat = rearrange(y, label_flattening)
        return c(output_flat, (y_flat,))

    return f_flat, c_flat
