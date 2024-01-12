from __future__ import annotations

import contextlib
from types import TracebackType
from typing import Callable, Iterable, List, Optional, Set, Tuple, Type, Union

from backpack.hessianfree.hvp import hessian_vector_product
from numpy import ndarray
from torch import Tensor, from_numpy
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from curvlinops._base import _LinearOperator


class ActivationHessianLinearOperator(_LinearOperator):
    """Hessian of the loss w.r.t. hidden features in a neural network."""

    def __init__(
        self,
        model_func: Module,
        loss_func: Callable[[Tensor, Tensor], Tensor],
        activation: Tuple[str, str, int],
        data: Iterable[Tuple[Tensor, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        shape: Optional[Tuple[int, int]] = None,
    ):
        self._activation = activation

        # Compute shape of activation and ensure there is only one batch
        data_iter = iter(data)
        X, _ = next(data_iter)
        activation_storage = []
        with store_activation(model_func, *activation, activation_storage):
            model_func(X)
        act = activation_storage.pop()
        shape = (act.numel(), act.numel())
        self._activation_shape = tuple(act.shape)

        with contextlib.suppress(StopIteration):
            next(data_iter)
            raise ValueError(f"{self.__class__.__name__} requires a single batch.")

        super().__init__(
            model_func,
            loss_func,
            list(model_func.parameters()),
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            shape=shape,
        )

    def _matvec_batch(
        self, X: Tensor, y: Tensor, x_list: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """Apply the mini-batch Hessian to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            x_list: Vector in list format (same shape as trainable model parameters).

        Returns:
            Result of Hessian-multiplication in list format.
        """
        activation_storage = []
        with store_activation(self._model_func, *self._activation, activation_storage):
            loss = self._loss_func(self._model_func(X), y)
        activation = activation_storage.pop()

        return hessian_vector_product(loss, [activation], x_list)

    def _preprocess(self, x: ndarray) -> List[Tensor]:
        return [from_numpy(x).to(self._device).reshape(self._activation_shape)]


class store_activation:
    """Temporarily install a hook on a module to store one of its in/outputs.

    Attributes:
        SUPPORTED_IO_TYPES: Supported types of in/outputs.
    """

    SUPPORTED_IO_TYPES: Set[str] = {"input", "output"}

    def __init__(
        self,
        model: Module,
        module_name: str,
        io_type: str,
        io_idx: int,
        destination: List,
    ) -> None:
        """Set up the context.

        Args:
            model: The neural network whose in/output will be stored during a forward
                pass.
            module_name: Name of the module whose in/output will be stored.
            io_type: Whether to store the input or output of the module.
            io_idx: Which tensor of the in/output tuple to store. If the model maps a
                single tensor to a single tensor, this should be ``0``.
            destination: Empty list to which the in/output will be appended.

        Raises:
            ValueError: If the module name is not found in the model.
            ValueError: If the io_type is not supported.
            ValueError: If ``destination`` is not empty.
        """
        self._model = model

        # check that the requested layer exists
        module_names = [name for name, _ in model.named_modules()]
        if module_name not in module_names:
            raise ValueError(
                f"Module {module_name} not found in model. Available modules are: "
                + f"{module_names}."
            )
        self._module_name = module_name

        # check that the requested input/output type exists
        if io_type not in self.SUPPORTED_IO_TYPES:
            raise ValueError(
                f"Unsupported io_type: {io_type}. Supported types are: "
                + f"{self.SUPPORTED_IO_TYPES}."
            )
        self._io_type = io_type
        self._io_idx = io_idx

        if destination:
            raise ValueError(f"`destination` must be empty. Got {destination}.")
        self._destination = destination

        self._hook_handles: List[RemovableHandle] = []

    def __enter__(self) -> None:
        """Install hook that appends the requested activation to the destination."""
        mod = self._model.get_submodule(self._module_name)
        self._hook_handles.append(mod.register_forward_hook(self.hook))

    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> None:
        """Remove hook.

        Args:
            __exc_type: exception type
            __exc_value: exception value
            __traceback: exception traceback
        """
        for handle in self._hook_handles:
            handle.remove()

    def hook(
        self,
        module: Module,
        inputs: Tuple[Tensor],
        output: Union[Tensor, Tuple[Tensor]],
    ):
        """Forward hook that appends the requested activation to the destination.

        Modifies ``self._destination`` in-place.

        Args:
            module: Layer onto which the hook is installed.
            inputs: Layer inputs.
            output: Layer outputs.

        Raises:
            ValueError: If the requested in/output index does not exist.
        """
        idx, io_type = self._io_idx, self._io_type

        if io_type == "output" and isinstance(output, Tensor) and idx != 0:
            raise ValueError(
                f"Output is a single tensor, but specified index is {idx}."
            )

        if io_type == "input":
            store = inputs[idx]
        else:
            store = output if isinstance(output, Tensor) else output[idx]

        self._destination.append(store)
