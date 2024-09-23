"""Defines a minimal ``LinearOperator`` interface in PyTorch."""

from __future__ import annotations

from typing import Callable, Iterable, List, MutableMapping, Optional, Tuple, Union

import numpy
from scipy.sparse.linalg import LinearOperator
from torch import Size, Tensor, cat, device, dtype, from_numpy, rand, tensor, zeros_like
from torch.autograd import grad
from torch.nn import Module, Parameter
from tqdm import tqdm

from curvlinops.utils import allclose_report


class PyTorchLinearOperator:
    """Interface for linear operators in PyTorch.

    Heavily inspired by the Scipy interface
    (https://github.com/scipy/scipy/blob/v1.13.1/scipy/sparse/linalg/_interface.py),
    but only supports a sub-set of the functionality.

    One main difference is that the linear operators cannot only multiply
    vectors/matrices specified as single PyTorch tensors, but also
    vectors/matrices specified in tensor list format. This is common in
    PyTorch, where the space a linear operator acts on is a tensor product

    Functions that need to be implemented are ``_matmat`` and ``_adjoint``.

    The interface also supports exporting the PyTorch linear operator to a SciPy linear
    operator, which can be useful for interfacing with SciPy routines. To achieve this,
    the functions ``_infer_device`` and ``_infer_dtype`` must be implemented.

    Attributes:
        SELF_ADJOINT: Whether the linear operator is self-adjoint. If ``True``,
            ``_adjoint`` does not need to be implemented. Default: ``False``.

    """

    SELF_ADJOINT: bool = False

    def __init__(
        self, in_shape: List[Tuple[int, ...]], out_shape: List[Tuple[int, ...]]
    ):
        """Store the linear operator's input and output space dimensions.

        Args:
            in_shape: A list of shapes specifying the linear operator's input space.
            out_shape: A list of shapes specifying the linear operator's output space.
        """
        self._in_shape = [Size(s) for s in in_shape]
        self._out_shape = [Size(s) for s in out_shape]

        self._in_shape_flat = [s.numel() for s in self._in_shape]
        self._out_shape_flat = [s.numel() for s in self._out_shape]
        self.shape = (sum(self._out_shape_flat), sum(self._in_shape_flat))

    def __matmul__(self, X: Union[List[Tensor], Tensor]) -> Union[List[Tensor], Tensor]:
        """Multiply onto a vector or matrix given as PyTorch tensor or tensor list.

        Args:
            X: A vector or matrix to multiply onto, represented as a single tensor or a
                tensor list.

                Assume the linear operator has total shape ``[M, N]``:
                If ``X`` is a single tensor, it can be of shape ``[N, K]`` (matrix), or
                ``[N]`` (vector). The result will have shape ``[M, K]`` or ``[M]``.

                Instead, we can also pass ``X`` as tensor list:
                Assume the linear operator's rows are formed by a list of shapes
                ``[M1, M2, ...]`` and the columns by ``[N1, N2, ...]``, such that
                ``M1.numel() + M2.numel() + ... = M`` and ``N1.numel() + N2.numel() +
                ... = N``. Then, ``X`` can also be a list of tensors with shape
                ``[*N1], [*N2], ...`` (vector) or ``[*N1, K], [*N2, K], ...`` (matrix).
                In this case, the output will be tensor list with shapes ``[*M1], [*M2],
                ...`` (vector) or ``[K, *M1], [K, *M2], ...`` (matrix).

        Returns:
            The result of the matrix-vector or matrix-matrix multiplication in the same
            format as ``X``.
        """
        # convert to tensor list format
        X, list_format, is_vec, num_vecs = self._check_input_and_preprocess(X)

        # matrix-matrix-multiply using tensor list format
        AX = self._matmat(X)

        # return same format as ``X`` passed by the user
        return self._check_output_and_postprocess(AX, list_format, is_vec, num_vecs)

    def _matmat(self, X: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication.

        Args:
            X: A list of tensors representing the matrix to multiply onto.
                The list must contain tensors of shape ``[*N1, K], [*N2, K], ...``,
                where ``N1, N2, ...`` are the shapes of the linear operator's columns.

        Returns: # noqa: D402
            A list of tensors with shape ``[*M1, K], [*M2, K], ...``, where ``M1, M2,
            ...`` are the shapes of the linear operator's rows.

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError

    def adjoint(self) -> PyTorchLinearOperator:
        """Return the adjoint of the linear operator.

        Returns:
            The adjoint of the linear operator.
        """
        return self if self.SELF_ADJOINT else self._adjoint()

    def _adjoint(self) -> PyTorchLinearOperator:
        """Adjoint of the linear operator.

        Returns: # noqa: D402
            The adjoint of the linear operator.

        Raises:
            NotImplementedError: Must be implemented by the subclass.
        """
        raise NotImplementedError

    def _check_input_and_preprocess(
        self, X: Union[List[Tensor], Tensor]
    ) -> Tuple[List[Tensor], bool, bool, int]:
        """Check input format and pre-process it to a matrix in tensor list format.

        Args:
            X: The object onto which the linear operator is multiplied.

        Returns:
            X_tensor_list: The input object in tensor list format.
            list_format: Whether the input was specified in tensor list format.
                This is useful for post-processing the multiplication's result.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.

        Raises:
            ValueError: If the input format is invalid.
        """
        if isinstance(X, Tensor):
            list_format = False
            X_tensor_list, is_vec, num_vecs = self.__check_tensor_and_preprocess(X)

        elif isinstance(X, list) and all(isinstance(x, Tensor) for x in X):
            list_format = True
            X_tensor_list, is_vec, num_vecs = self.__check_tensor_list_and_preprocess(X)

        else:
            raise ValueError(f"Input must be tensor or list of tensors. Got {type(X)}.")

        return X_tensor_list, list_format, is_vec, num_vecs

    def __check_tensor_and_preprocess(
        self, X: Tensor
    ) -> Tuple[List[Tensor], bool, int]:
        """Check single-tensor input format and process into a matrix tensor list.

        Args:
            X: The tensor onto which the linear operator is multiplied.

        Returns:
            X_processed: The input tensor as matrix in tensor list format.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.

        Raises:
            ValueError: If the input tensor has an invalid shape.
        """
        if X.ndim > 2 or X.shape[0] != self.shape[1]:
            raise ValueError(
                f"Input tensor must have shape ({self.shape[1]},) or "
                + f"({self.shape[1]}, K), with K arbitrary. Got {X.shape}."
            )

        # determine whether the input is a vector or matrix
        is_vec = X.ndim == 1
        num_vecs = 1 if is_vec else X.shape[1]

        # convert to matrix in tensor list format
        X_processed = [
            x.reshape(*s, num_vecs)
            for x, s in zip(X.split(self._in_shape_flat), self._in_shape)
        ]

        return X_processed, is_vec, num_vecs

    def __check_tensor_list_and_preprocess(
        self, X: List[Tensor]
    ) -> Tuple[List[Tensor], bool, int]:
        """Check tensor list input format and process into a matrix tensor list.

        Args:
            X: The tensor list onto which the linear operator is multiplied.

        Returns:
            X_processed: The input as matrix in tensor list format.
            is_vec: Whether the input is a vector or a matrix.
            num_vecs: The number of vectors represented by the input.

        Raises:
            ValueError: If the tensor entries in the list have invalid shapes.
        """
        if len(X) != len(self._in_shape):
            raise ValueError(
                f"List must contain {len(self._in_shape)} tensors. Got {len(X)}."
            )

        # check if input is a vector or a matrix
        if all(x.shape == s for x, s in zip(X, self._in_shape)):
            is_vec, num_vecs = True, 1
        elif (
            all(
                x.ndim == len(s) + 1 and x.shape[:-1] == s
                for x, s in zip(X, self._in_shape)
            )
            and len({x.shape[-1] for x in X}) == 1
        ):
            is_vec, (num_vecs,) = False, {x.shape[-1] for x in X}
        else:
            raise ValueError(
                f"Input list must contain tensors with shapes {self._in_shape} "
                + "and optional trailing dimension for the matrix columns. "
                + f"Got {[x.shape for x in X]}."
            )

        # convert to matrix in tensor list format
        X_processed = [x.unsqueeze(-1) for x in X] if is_vec else X

        return X_processed, is_vec, num_vecs

    def _check_output_and_postprocess(
        self, AX: List[Tensor], list_format: bool, is_vec: bool, num_vecs: int
    ) -> Union[List[Tensor], Tensor]:
        """Check multiplication output and post-process it to the original format.

        Args:
            AX: The output of the multiplication as matrix in tensor list format.
            list_format: Whether the output should be in tensor list format.
            is_vec: Whether the output should be a vector or a matrix.
            num_vecs: The number of vectors represented by the output.

        Returns:
            AX_processed: The output in the original format, either as single tensor
                or list of tensors.

        Raises:
            ValueError: If the output tensor list has an invalid length or shape.
        """
        # verify output tensor list format
        if len(AX) != len(self._out_shape):
            raise ValueError(
                f"Output list must contain {len(self._out_shape)} tensors. Got {len(AX)}."
            )
        if any(Ax.shape != (*s, num_vecs) for Ax, s in zip(AX, self._out_shape)):
            raise ValueError(
                f"Output tensors must have shapes {self._out_shape} and additional "
                + f"trailing dimension of {num_vecs}. "
                + f"Got {[Ax.shape for Ax in AX]}."
            )

        if list_format:
            AX_processed = [Ax.squeeze(-1) for Ax in AX] if is_vec else AX
        else:
            AX_processed = cat(
                [Ax.reshape(s, num_vecs) for Ax, s in zip(AX, self._out_shape_flat)]
            )
            AX_processed = AX_processed.squeeze(-1) if is_vec else AX_processed

        return AX_processed

    ###############################################################################
    #                                 SCIPY EXPORT                                #
    ###############################################################################

    def to_scipy(self, dtype: Optional[numpy.dtype] = None) -> LinearOperator:
        """Wrap the PyTorch linear operator with a SciPy linear operator.

        Args:
            dtype: The data type of the SciPy linear operator. If ``None``, uses
                NumPy's default data dtype.


        Returns:
            A SciPy linear operator that carries out the matrix-vector products
            in PyTorch.
        """
        dev = self._infer_device()
        dt = self._infer_dtype()

        scipy_matmat = self._scipy_compatible(self.__matmul__, dev, dt)
        A_adjoint = self.adjoint()
        scipy_rmatmat = A_adjoint._scipy_compatible(A_adjoint.__matmul__, dev, dt)

        return LinearOperator(
            self.shape,
            matvec=scipy_matmat,
            rmatvec=scipy_rmatmat,
            matmat=scipy_matmat,
            rmatmat=scipy_rmatmat,
            dtype=numpy.dtype(dtype) if dtype is None else dtype,
        )

    def _infer_device(self) -> device:
        """Infer the linear operator's device.

        Returns:  # noqa: D402
            The device of the linear operator.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _infer_dtype(self) -> dtype:
        """Infer the linear operator's data type.

        Returns: # noqa: D402
            The data type of the linear operator.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def _scipy_compatible(
        f: Callable[[Tensor], Tensor], device: device, dtype: dtype
    ) -> Callable[[numpy.ndarray], numpy.ndarray]:
        """Wrap a PyTorch matrix multiplication function to be compatible with SciPy.

        Args:
            f: The PyTorch matrix multiplication function.
            device: The device on which the PyTorch linear operator is defined.
            dtype: The data type of the PyTorch linear operator.

        Returns:
            A function that takes a NumPy array and returns a NumPy array.
        """

        def f_scipy(X: numpy.ndarray) -> numpy.ndarray:
            """Scipy-compatible matrix multiplication function.

            Args:
                X: The input matrix in NumPy format.

            Returns:
                The output matrix in NumPy format.
            """
            X_dtype = X.dtype
            X_torch = from_numpy(X).to(device, dtype)
            AX_torch = f(X_torch)
            return AX_torch.detach().cpu().numpy().astype(X_dtype)

        return f_scipy


class CurvatureLinearOperator(PyTorchLinearOperator):
    """Base class for PyTorch linear operators of deep learning curvature matrices.

    To implement a new curvature linear operator, subclass this class and implement
    the ``_matmat_batch`` and ``_adjoint`` methods.

    Attributes:
        SUPPORTS_BLOCKS: Whether the linear operator supports multiplication with
            a block-diagonal approximation rather than the full matrix.
            Default: ``False``.
    """

    SUPPORTS_BLOCKS: bool = False

    def __init__(
        self,
        model_func: Callable[[Union[Tensor, MutableMapping]], Tensor],
        loss_func: Union[Callable[[Tensor, Tensor], Tensor], None],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        in_shape: Optional[List[Tuple[int, ...]]] = None,
        out_shape: Optional[List[Tuple[int, ...]]] = None,
        num_data: Optional[int] = None,
        block_sizes: Optional[List[int]] = None,
        batch_size_fn: Optional[Callable[[Union[MutableMapping, Tensor]], int]] = None,
    ):
        """Linear operator for curvature matrices of empirical risks.

        Note:
            f(X; θ) denotes a neural network, parameterized by θ, that maps a mini-batch
            input X to predictions p. ℓ(p, y) maps the prediction to a loss, using the
            mini-batch labels y.

        Args:
            model_func: A function that maps the mini-batch input X to predictions.
                Could be a PyTorch module representing a neural network.
            loss_func: Loss function criterion. Maps predictions and mini-batch labels
                to a scalar value. If ``None``, there is no loss function and the
                represented matrix is independent of the loss function.
            params: List of differentiable parameters used by the prediction function.
            data: Source from which mini-batches can be drawn, for instance a list of
                mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``. Note that ``X``
                could be a ``dict`` or ``UserDict``; this is useful for custom models.
                In this case, you must (i) specify the ``batch_size_fn`` argument, and
                (ii) take care of preprocessing like ``X.to(device)`` inside of your
                ``model.forward()`` function.
            progressbar: Show a progressbar during matrix-multiplication.
                Default: ``False``.
            check_deterministic: Probe that model and data are deterministic, i.e.
                that the data does not use ``drop_last`` or data augmentation. Also, the
                model's forward pass could depend on the order in which mini-batches
                are presented (BatchNorm, Dropout). Default: ``True``. This is a
                safeguard, only turn it off if you know what you are doing.
            in_shape: Shapes of the linear operator's input tensor product space.
                If ``None``, will use the shapes of ``params``.
            out_shape: Shapes of the linear operator's output tensor product space.
                If ``None``, will use the shapes of ``params``.
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.
            block_sizes: This argument will be ignored if the linear operator does not
                support blocks. List of integers indicating the number of
                ``nn.Parameter``s forming a block. Entries must sum to ``len(params)``.
                For instance ``[len(params)]`` considers the full matrix, while
                ``[1, 1, ...]`` corresponds to a block diagonal approximation where
                each parameter forms its own block.
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified. The intended behavior is to consume the first
                entry of the iterates from ``data`` and return their batch size.

        Raises:
            RuntimeError: If the check for deterministic behavior fails.
            ValueError: If ``block_sizes`` is specified but the linear operator does not
                support blocks.
            ValueError: If the sum of blocks does not equal the number of parameters.
            ValueError: If any block size is not positive.
            ValueError: If ``X`` is not a tensor and ``batch_size_fn`` is not specified.
        """
        if isinstance(next(iter(data))[0], MutableMapping) and batch_size_fn is None:
            raise ValueError(
                "When using dict-like custom data, `batch_size_fn` is required."
            )

        in_shape = [tuple(p.shape) for p in params] if in_shape is None else in_shape
        out_shape = [tuple(p.shape) for p in params] if out_shape is None else out_shape
        super().__init__(in_shape, out_shape)

        self._params = params
        if block_sizes is not None:
            if not self.SUPPORTS_BLOCKS:
                raise ValueError(
                    "Block sizes were specified but operator does not support blocking."
                )
            if sum(block_sizes) != len(params):
                raise ValueError("Sum of blocks must equal the number of parameters.")
            if any(s <= 0 for s in block_sizes):
                raise ValueError("Block sizes must be positive.")
        self._block_sizes = [len(params)] if block_sizes is None else block_sizes

        self._model_func = model_func
        self._loss_func = loss_func
        self._data = data
        self._device = self._infer_device()
        self._progressbar = progressbar
        self._batch_size_fn = (
            (lambda X: X.shape[0]) if batch_size_fn is None else batch_size_fn
        )

        self._N_data = (
            sum(
                self._batch_size_fn(X)
                for (X, _) in self._loop_over_data(desc="_N_data")
            )
            if num_data is None
            else num_data
        )

        if check_deterministic:
            old_device = self._device
            self.to_device(device("cpu"))
            try:
                self._check_deterministic()
            except RuntimeError as e:
                raise e
            finally:
                self.to_device(old_device)

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Matrix-matrix multiplication.

        Args:
            M: Matrix for multiplication in tensor list format. Assume the linear
                operator's input tensor product space consists of shapes ``[*N1],
                [*N2], ...``. Then, ``M`` is a list of tensors with shapes
                ``[*N1, K], [*N2, K], ...`` with ``K`` the number of columns.

        Returns:
            Matrix-multiplication result ``mat @ M`` in tensor list format.
            Has same format as the input matrix, but lives in the linear operator's
            output tensor product space.
        """
        AM = [zeros_like(m) for m in M]

        for X, y in self._loop_over_data(desc="_matmat"):
            normalization_factor = self._get_normalization_factor(X, y)
            for AM_current, current in zip(AM, self._matmat_batch(X, y, M)):
                AM_current.add_(current, alpha=normalization_factor)

        return AM

    def _matmat_batch(
        self, X: Union[MutableMapping, Tensor], y: Tensor, M: List[Tensor]
    ) -> List[Tensor]:
        """Apply the mini-batch matrix to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M: Matrix in list format (same shape as trainable model parameters with
                additional trailing dimension of size number of columns).

        Returns: # noqa: D402
           Result of matrix-multiplication in list format.

        Raises:
            NotImplementedError: Must be implemented by descendants.
        """
        raise NotImplementedError

    def _loop_over_data(
        self, desc: Optional[str] = None, add_device_to_desc: bool = True
    ) -> Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]]:
        """Yield batches of the data set, loaded to the correct device.

        Args:
            desc: Description for the progress bar. Will be ignored if progressbar is
                disabled.
            add_device_to_desc: Whether to add the device to the description.
                Default: ``True``.

        Yields:
            Mini-batches ``(X, y)``.
        """
        data_iter = iter(self._data)

        if self._progressbar:
            desc = f"{self.__class__.__name__}{'' if desc is None else f'.{desc}'}"
            if add_device_to_desc:
                desc = f"{desc} (on {str(self._device)})"
            data_iter = tqdm(data_iter, desc=desc)

        for X, y in data_iter:
            # Assume everything is handled by the model
            # if `X` is a custom data format
            if isinstance(X, Tensor):
                X = X.to(self._device)
            y = y.to(self._device)
            yield (X, y)

    def _get_normalization_factor(
        self, X: Union[MutableMapping, Tensor], y: Tensor
    ) -> float:
        """Return the correction factor for correct normalization over the data set.

        Args:
            X: Input to the DNN.
            y: Ground truth.

        Returns:
            Normalization factor
        """
        return {"sum": 1.0, "mean": self._batch_size_fn(X) / self._N_data}[
            self._loss_func.reduction
        ]

    ###############################################################################
    #                             DETERMINISTIC CHECKS                            #
    ###############################################################################

    def gradient_and_loss(self) -> Tuple[List[Tensor], Tensor]:
        """Evaluate the gradient and loss on the data.

        (Not really part of the LinearOperator interface.)

        Returns:
            Gradient and loss on the data set.

        Raises:
            ValueError: If there is no loss function.
        """
        if self._loss_func is None:
            raise ValueError("No loss function specified.")

        total_loss = tensor([0.0], device=self._device, dtype=self._infer_dtype())
        total_grad = [zeros_like(p) for p in self._params]

        for X, y in self._loop_over_data(desc="gradient_and_loss"):
            loss = self._loss_func(self._model_func(X), y)
            normalization_factor = self._get_normalization_factor(X, y)

            for grad_param, current in zip(total_grad, grad(loss, self._params)):
                grad_param.add_(current, alpha=normalization_factor)
            total_loss.add_(loss.detach(), alpha=normalization_factor)

        return total_grad, total_loss

    def to_device(self, device: device):
        """Load linear operator to a device (inplace).

        Args:
            device: Target device.
        """
        self._device = device

        if isinstance(self._model_func, Module):
            self._model_func = self._model_func.to(self._device)
        self._params = [p.to(device) for p in self._params]

        if isinstance(self._loss_func, Module):
            self._loss_func = self._loss_func.to(self._device)

    def _check_deterministic(self):
        """Check that the linear operator is deterministic.

        Non-deterministic behavior is detected if:

        - Two independent applications of matvec onto the same vector yield different
          results
        - Two independent loss/gradient computations yield different results

        Note:
            Deterministic checks should be performed on CPU. We noticed that even when
            it passes on CPU, it can fail on GPU; probably due to non-deterministic
            operations.

        # TODO This can be impractical if the CPU is less powerful than the GPU.
        # Also, it would be desirable to confirm deterministic behavior on the compute
        # device that will be used for matvecs. Try refactoring by using device-agnostic
        # tolerances. Then remove the ``to_device`` method.

        Raises:
            RuntimeError: If non-deterministic behavior is detected.
        """
        v = rand(self.shape[1], device=self._device, dtype=self._infer_dtype())
        Av1 = self @ v
        Av2 = self @ v

        rtol, atol = 5e-5, 1e-6
        if not allclose_report(Av1, Av2, rtol=rtol, atol=atol):
            raise RuntimeError("Check for deterministic matvec failed.")

        if self._loss_func is None:
            return

        # only carried out if there is a loss function
        grad1, loss1 = self.gradient_and_loss()
        grad2, loss2 = self.gradient_and_loss()

        if not allclose_report(loss1, loss2, rtol=rtol, atol=atol):
            raise RuntimeError("Check for deterministic loss failed.")

        if len(grad1) != len(grad2) or any(
            not allclose_report(g1, g2, atol=atol, rtol=rtol)
            for g1, g2 in zip(grad1, grad2)
        ):
            raise RuntimeError("Check for deterministic gradient failed.")

    ###############################################################################
    #                                 SCIPY EXPORT                                #
    ###############################################################################

    def _infer_device(self) -> device:
        """Infer the device onto which to load NumPy vectors for the matrix multiply.

        Returns:
            Inferred device.

        Raises:
            RuntimeError: If the device cannot be inferred.
        """
        devices = {p.device for p in self._params}
        if len(devices) != 1:
            raise RuntimeError(f"Could not infer device. Parameters live on {devices}.")
        return devices.pop()

    def _infer_dtype(self) -> dtype:
        """Infer the data type to which to load NumPy vectors for the matrix multiply.

        Returns:
            Inferred data type.

        Raises:
            RuntimeError: If the data type cannot be inferred.
        """
        dtypes = {p.dtype for p in self._params}
        if len(dtypes) != 1:
            raise RuntimeError(f"Could not infer data type. Parameters have {dtypes}.")
        return dtypes.pop()
