"""General utility functions."""

from collections.abc import Callable, Iterable, Iterator, MutableMapping
from contextlib import contextmanager
from functools import partial

from einops import rearrange
from numpy import ndarray
from torch import Generator, Tensor, as_tensor, device, dtype, manual_seed
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import CrossEntropyLoss, Module
from torch.random import fork_rng

#: Standardized ``make_fx`` with fake tensor tracing. All ``make_fx`` calls in
#: curvlinops should use this to ensure consistent tracing behavior (fake mode
#: avoids materializing real tensors during tracing, reducing memory usage).
_make_fx = partial(make_fx, tracing_mode="fake", _allow_non_fake_inputs=True)


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


@contextmanager
def fork_rng_with_seed(seed: int | None) -> Iterator[None]:
    """Fork the global RNG state and seed it, restoring on exit.

    Used in the FX backends to isolate ``manual_seed`` calls from the caller's
    global RNG state (the hooks backends achieve the same with a dedicated
    ``torch.Generator``, but ``make_fx`` cannot trace through that).

    Args:
        seed: Seed to set inside the forked context. If ``None``, the context
            is a pass-through no-op (neither forking nor seeding).

    Yields:
        None.
    """
    if seed is None:
        yield
    else:
        with fork_rng():
            manual_seed(seed)
            yield


def _has_single_element(iterable: Iterable) -> None:
    """Validate that ``iterable`` yields exactly one element.

    Advances the underlying iterator at most two steps, so a lazy iterable
    (e.g., a generator or ``DataLoader``) accidentally passed in is not
    materialized. Callers that need to use the element afterwards should
    pass a reusable iterable and re-iterate it themselves.

    Args:
        iterable: Iterable expected to yield exactly one element.

    Raises:
        ValueError: If ``iterable`` yields zero or more than one element.
    """
    it = iter(iterable)
    try:
        next(it)
    except StopIteration as err:
        raise ValueError("Iterable must contain exactly one element, got 0.") from err
    try:
        next(it)
    except StopIteration:
        return
    raise ValueError("Iterable must contain exactly one element, got more than one.")


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
    start = 0
    result = []
    for s in sizes:
        result.append(list(x[start : start + s]))
        start += s
    return result


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


def make_functional_loss(loss_func: Module) -> Callable[[Tensor, tuple], Tensor]:
    """Create a functional version of a loss function.

    Args:
        loss_func: The loss function module.

    Returns:
        A function ``(prediction, loss_args) -> loss`` where ``loss_args`` is a
        tuple of additional arguments, e.g. ``(y,)`` for targets.
    """
    c_raw = partial(make_functional_call(loss_func), {})

    def c(prediction: Tensor, loss_args: tuple) -> Tensor:
        """Evaluate the loss function on a prediction and loss arguments.

        Args:
            prediction: Model prediction.
            loss_args: Tuple of loss function arguments, e.g. ``(y,)``.

        Returns:
            Scalar loss value.
        """
        return c_raw(prediction, *loss_args)

    return c


def make_functional_flattened_model_and_loss(
    f: Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
    loss_func: Module,
) -> tuple[
    Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
    Callable[[Tensor, tuple], Tensor],
]:
    """Create flattened versions of model and loss functions.

    This is required for the (empirical) Fisher, for which we don't know how to handle
    additional axes. Therefore, they are flattened into the batch axis.

    Args:
        f: Functional model with signature ``(params_dict, X) -> prediction``.
        loss_func: The loss function module.

    Returns:
        Tuple of (f_flat, c_flat) where:
        - f_flat: Function that executes model and flattens batch and shared axes:
          (params, X) -> output_flat
        - c_flat: Function that executes loss with flattened labels:
          (output_flat, loss_args) -> loss
    """
    c = make_functional_loss(loss_func)

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
