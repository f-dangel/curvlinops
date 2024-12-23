from __future__ import annotations

import contextlib
from types import TracebackType
from typing import (
    Callable,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from backpack.hessianfree.hvp import hessian_vector_product
from torch import Tensor, zeros_like
from torch.autograd import grad
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from curvlinops._torch_base import CurvatureLinearOperator


class ActivationHessianLinearOperator(CurvatureLinearOperator):
    r"""Hessian of the loss w.r.t. hidden features in a neural network.

    Consider the empirical risk on a single mini-batch
    :math:`\mathbf{X} = \{ \mathbf{x}_1, \dots, \mathbf{x}_N\}`,
    :math:`\mathbf{y} = \{ \mathbf{y}_1, \dots, \mathbf{y}_N\}`:

    .. math::
        \mathcal{L}(\mathbf{\theta})
        =
        \ell(f_{\mathbf{\theta}}(\mathbf{X}), \mathbf{y})

    Let :math:`\mathbf{Z} = \{ \mathbf{z}_1, \dots, \mathbf{z}_N \}` denote a batch
    of some intermediate feature produced inside the neural network's forward pass.
    The Hessian w.r.t. the flattened activations that is represented by this linear
    operator is

    .. math::
        \nabla^2_{\mathbf{Z}} \mathcal{L}
        =
        \nabla^2_{\mathbf{Z}}
        \ell(f_{\mathbf{\theta}}(\mathbf{X}), \mathbf{y})

    and has dimension :math:`\mathrm{dim}(\mathbf{Z}) = N \mathrm{dim}(\mathbf{z})`.

    Attributes:
        SELF_ADJOINT: Whether the linear operator is self-adjoint. ``True`` for the
            activation Hessian.
    """

    SELF_ADJOINT: bool = True

    def __init__(
        self,
        model_func: Module,
        loss_func: Callable[[Union[MutableMapping, Tensor], Tensor], Tensor],
        activation: Tuple[str, str, int],
        data: Iterable[Tuple[Union[MutableMapping, Tensor], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
    ):
        """Linear operator for the loss Hessian w.r.t. intermediate features.

        Args:
            model_func: A neural network.
            loss_func: A loss function.
            activation: A tuple specifying w.r.t. what intermediate feature the Hessian
                shall be computed. Has three entries. The first entry is the layer's
                name inside the model (any string from ``model_func.named_modules``).
                The second entry can either be ``"output"`` or ``"input"`` and
                specifies whether the intermediate feature is the layer's output or the
                layer's input. The third entry specifies what tensor should be taken
                from the tuple of input tensors or the tuple of output tensors. For
                most layers that process a single Tensor into another single Tensor,
                this value is ``0``.
            data: An iterable of mini-batches. Must have a single batch, otherwise the
                activation Hessian is not defined.
            progressbar: Show a progressbar during matrix-multiplication.
                Default: ``False``.
            check_deterministic: Probe that model and data are deterministic, i.e.
                that the data does not use ``drop_last`` or data augmentation. Also, the
                model's forward pass could depend on the order in which mini-batches
                are presented (BatchNorm, Dropout). Default: ``True``. This is a
                safeguard, only turn it off if you know what you are doing.

        Raises:
            ValueError: If ``data`` contains more than one batch.

        Example:
            >>> from torch import manual_seed, rand, eye, allclose
            >>> from torch.nn import Linear, MSELoss, Sequential, ReLU
            >>>
            >>> loss_func = MSELoss()
            >>> model = Sequential(Linear(4, 3), ReLU(), Linear(3, 2))
            >>> [name for name, _ in model.named_modules()] # available layer names
            ['', '0', '1', '2']
            >>> data = [(rand(10, 4), rand(10, 2))]
            >>>
            >>> hessian = ActivationHessianLinearOperator( # Hessian w.r.t. ReLU input
            ...     model, loss_func, ("1", "input", 0), data
            ... )
            >>> hessian.shape # batch size * feature dimension (10 * 3)
            (30, 30)
            >>>
            >>> # The ReLU's input is the first Linear's output, let's check that
            >>> hessian2 = ActivationHessianLinearOperator( # Hessian w.r.t. first output
            ...     model, loss_func, ("0", "output", 0), data
            ... )
            >>> I = eye(hessian.shape[1])
            >>> allclose(hessian @ I, hessian2 @ I)
            True
        """
        self._activation = activation

        # Ensure there is only one batch
        data_iter = iter(data)
        next(data_iter)

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
        )

    def _get_out_shape(self) -> List[Tuple[int, ...]]:
        """Return the output shape of the activation Hessian.

        Returns:
            The dimensions of the activation Hessian's output space.
        """
        if not hasattr(self, "_activation_shape"):
            X, _ = next(iter(self._data))
            dev = self._infer_device()
            if isinstance(X, Tensor):
                X = X.to(dev)
            activation_storage = []
            with store_activation(
                self._model_func, *self._activation, activation_storage
            ):
                self._model_func(X)
            act = activation_storage.pop()

            self._activation_shape = tuple(act.shape)

        return [self._activation_shape]

    def _get_in_shape(self) -> List[Tuple[int, ...]]:
        """Return the input shape of the activation Hessian.

        Returns:
            The dimensions of the activation Hessian's input space.
        """
        return self._get_out_shape()

    def _matmat_batch(self, X: Tensor, y: Tensor, M: List[Tensor]) -> List[Tensor]:
        """Apply the activation Hessian to a matrix in tensor list format.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M: Matrix to be multiplied with in tensor list format.
                Tensors have same shape as trainable model parameters, and an
                additional trailing axis for the matrix columns.

        Returns:
            Result of activation Hessian multiplication in list format. Has the same
            shape as ``M``, i.e. each tensor in the list has the shape of a
            parameter and a trailing dimension of matrix columns.
        """
        activation_storage = []
        with store_activation(self._model_func, *self._activation, activation_storage):
            loss = self._loss_func(self._model_func(X), y)
        activation = activation_storage.pop()

        # Re-cycle first backward pass of the HVP's double-backward
        # between columns the Hessian is multiplied onto
        grad_activation = grad(loss, activation, create_graph=True)

        # collect
        HM = [zeros_like(m) for m in M]
        (num_vectors,) = {m.shape[-1] for m in M}
        for n in range(num_vectors):
            HM_col = hessian_vector_product(
                loss, [activation], [m[..., n] for m in M], grad_params=grad_activation
            )
            for HM_p, HM_col_p in zip(HM, HM_col):
                HM_p[..., n].add_(HM_col_p)

        return HM


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
