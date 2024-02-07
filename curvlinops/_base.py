"""Contains functionality to analyze Hessian & GGN via matrix-free multiplication."""

from typing import Callable, Iterable, List, Optional, Tuple, Union
from warnings import warn

from einops import rearrange
from numpy import allclose, argwhere, float32, isclose, logical_not, ndarray
from numpy.random import rand
from scipy.sparse.linalg import LinearOperator
from torch import Tensor, cat
from torch import device as torch_device
from torch import from_numpy, tensor, zeros_like
from torch.autograd import grad
from torch.nn import Module, Parameter
from tqdm import tqdm


class _LinearOperator(LinearOperator):
    """Base class for linear operators of DNN matrices.

    Can be used with SciPy.
    """

    def __init__(
        self,
        model_func: Callable[[Tensor], Tensor],
        loss_func: Union[Callable[[Tensor, Tensor], Tensor], None],
        params: List[Parameter],
        data: Iterable[Tuple[Tensor, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        shape: Optional[Tuple[int, int]] = None,
        num_data: Optional[int] = None,
    ):
        """Linear operator for DNN matrices.

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
                mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``.
            progressbar: Show a progressbar during matrix-multiplication.
                Default: ``False``.
            check_deterministic: Probe that model and data are deterministic, i.e.
                that the data does not use ``drop_last`` or data augmentation. Also, the
                model's forward pass could depend on the order in which mini-batches
                are presented (BatchNorm, Dropout). Default: ``True``. This is a
                safeguard, only turn it off if you know what you are doing.
            shape: Shape of the represented matrix. If ``None`` assumes ``(D, D)``
                where ``D`` is the total number of parameters
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.

        Raises:
            RuntimeError: If the check for deterministic behavior fails.
        """
        if shape is None:
            dim = sum(p.numel() for p in params)
            shape = (dim, dim)
        super().__init__(shape=shape, dtype=float32)

        self._params = params
        self._model_func = model_func
        self._loss_func = loss_func
        self._data = data
        self._device = self._infer_device(self._params)
        self._progressbar = progressbar

        self._N_data = (
            sum(X.shape[0] for (X, _) in self._loop_over_data(desc="_N_data"))
            if num_data is None
            else num_data
        )

        if check_deterministic:
            old_device = self._device
            self.to_device(torch_device("cpu"))
            try:
                self._check_deterministic()
            except RuntimeError as e:
                raise e
            finally:
                self.to_device(old_device)

    @staticmethod
    def _infer_device(params: List[Parameter]) -> torch_device:
        """Infer the device on which to carry out matvecs.

        Args:
            params: DNN parameters that define the linear operators.

        Returns:
            Inferred device.

        Raises:
            RuntimeError: If the device cannot be inferred.
        """
        devices = {p.device for p in params}
        if len(devices) != 1:
            raise RuntimeError(f"Could not infer device. Parameters live on {devices}.")
        return devices.pop()

    def to_device(self, device: torch_device):
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
        """Check that the Linear operator is deterministic.

        Non-deterministic behavior is detected if:

        - Two independent applications of matvec onto the same vector yield different
          results
        - Two independent loss/gradient computations yield different results

        Note:
            Deterministic checks should be performed on CPU. We noticed that even when
            it passes on CPU, it can fail on GPU; probably due to non-deterministic
            operations.

        Raises:
            RuntimeError: If non-deterministic behavior is detected.
        """
        v = rand(self.shape[1]).astype(self.dtype)
        mat_v1 = self @ v
        mat_v2 = self @ v

        rtol, atol = 5e-5, 1e-6
        if not allclose(mat_v1, mat_v2, rtol=rtol, atol=atol):
            self.print_nonclose(mat_v1, mat_v2, rtol, atol)
            raise RuntimeError("Check for deterministic matvec failed.")

        if self._loss_func is None:
            return

        # only carried out if there is a loss function
        grad1, loss1 = self.gradient_and_loss()
        grad1, loss1 = (
            self.flatten_and_concatenate(grad1).cpu().numpy(),
            loss1.cpu().numpy(),
        )

        grad2, loss2 = self.gradient_and_loss()
        grad2, loss2 = (
            self.flatten_and_concatenate(grad2).cpu().numpy(),
            loss2.cpu().numpy(),
        )

        if not allclose(loss1, loss2, rtol=rtol, atol=atol):
            self.print_nonclose(loss1, loss2, rtol, atol)
            raise RuntimeError("Check for deterministic loss failed.")

        if not allclose(grad1, grad2, rtol=rtol, atol=atol):
            self.print_nonclose(grad1, grad2, rtol, atol)
            raise RuntimeError("Check for deterministic gradient failed.")

    @staticmethod
    def print_nonclose(array1: ndarray, array2: ndarray, rtol: float, atol: float):
        """Check if the two arrays are element-wise equal within a tolerance and print
        the entries that differ.

        Args:
            array1: First array for comparison.
            array2: Second array for comparison.
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        if not allclose(array1, array2, rtol=rtol, atol=atol):
            nonclose_idx = logical_not(isclose(array1, array2, rtol=rtol, atol=atol))
            for idx, a1, a2 in zip(
                argwhere(nonclose_idx),
                array1[nonclose_idx].flatten(),
                array2[nonclose_idx].flatten(),
            ):
                print(f"at index {idx}: {a1:.5e} ≠ {a2:.5e}, ratio: {a1 / a2:.5e}")

    def _matmat(self, M: ndarray) -> ndarray:
        """Matrix-matrix multiplication.

        Args:
            M: Matrix for multiplication.

        Returns:
            Matrix-multiplication result ``mat @ M``.
        """
        M_list = self._preprocess(M)
        out_list = [zeros_like(M) for M in M_list]

        for X, y in self._loop_over_data(desc="_matmat"):
            normalization_factor = self._get_normalization_factor(X, y)
            for out, current in zip(out_list, self._matmat_batch(X, y, M_list)):
                out.add_(current, alpha=normalization_factor)

        return self._postprocess(out_list)

    def _matmat_batch(
        self, X: Tensor, y: Tensor, M_list: List[Tensor]
    ) -> Tuple[Tensor]:
        """Apply the mini-batch matrix to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M_list: Matrix in list format (same shape as trainable model parameters with
                additional leading dimension of size number of columns).

        Returns: # noqa: D402
           Result of matrix-multiplication in list format.

        Raises:
            NotImplementedError: Must be implemented by descendants.
        """
        raise NotImplementedError

    def _preprocess(self, M: ndarray) -> List[Tensor]:
        """Convert numpy matrix to torch parameter list format.

        Args:
            M: Matrix for multiplication. Has shape ``[D, K]`` where ``D`` is the
                number of parameters, and ``K`` is the number of columns.

        Returns:
            Matrix in list format. Each entry has the same shape as a parameter with
            an additional leading dimension of size ``K`` for the columns, i.e.
            ``[(K,) + p1.shape), (K,) + p2.shape, ...]``.
        """
        if M.dtype != self.dtype:
            warn(
                f"Input matrix is {M.dtype}, while linear operator is {self.dtype}. "
                + f"Converting to {self.dtype}."
            )
            M = M.astype(self.dtype)
        num_vectors = M.shape[1]

        result = from_numpy(M).to(self._device)
        # split parameter blocks
        dims = [p.numel() for p in self._params]
        result = result.split(dims)
        # column-index first + unflatten parameter dimension
        shapes = [(num_vectors,) + p.shape for p in self._params]
        result = [res.T.reshape(shape) for res, shape in zip(result, shapes)]

        return result

    def _postprocess(self, M_list: List[Tensor]) -> ndarray:
        """Convert torch list format of a matrix to flat numpy matrix format.

        Args:
            M_list: Matrix in list format. Each entry has a leading dimension of size
                ``K``. The remaining dimensions are flattened and concatenated.

        Returns:
            Flat matrix. Has shape ``[D, K]`` where ``D`` is sum of flattened and
            concatenated dimensions over all list entries.
        """
        result = [rearrange(M, "k ... -> (...) k") for M in M_list]
        return cat(result, dim=0).cpu().numpy()

    def _loop_over_data(
        self, desc: Optional[str] = None
    ) -> Iterable[Tuple[Tensor, Tensor]]:
        """Yield batches of the data set, loaded to the correct device.

        Args:
            desc: Description for the progress bar. Will be ignored if progressbar is
                disabled.

        Yields:
            Mini-batches ``(X, y)``.
        """
        data_iter = iter(self._data)

        if self._progressbar:
            desc = f"{self.__class__.__name__}{'' if desc is None else f'.{desc}'}"
            data_iter = tqdm(data_iter, desc=desc)

        for X, y in data_iter:
            X, y = X.to(self._device), y.to(self._device)
            yield (X, y)

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

        total_loss = tensor([0.0], device=self._device)
        total_grad = [zeros_like(p) for p in self._params]

        for X, y in self._loop_over_data(desc="gradient_and_loss"):
            loss = self._loss_func(self._model_func(X), y)
            normalization_factor = self._get_normalization_factor(X, y)

            for grad_param, current in zip(total_grad, grad(loss, self._params)):
                grad_param.add_(current, alpha=normalization_factor)
            total_loss.add_(loss.detach(), alpha=normalization_factor)

        return total_grad, total_loss

    def _get_normalization_factor(self, X: Tensor, y: Tensor) -> float:
        """Return the correction factor for correct normalization over the data set.

        Args:
            X: Input to the DNN.
            y: Ground truth.

        Returns:
            Normalization factor

        Raises:
            ValueError: If loss function does not have a ``reduction`` attribute or
                it is not set to ``'mean'`` or ``'sum'``.
        """
        if not hasattr(self._loss_func, "reduction"):
            raise ValueError("Loss must have a 'reduction' attribute.")

        reduction = self._loss_func.reduction
        if reduction == "sum":
            return 1.0
        elif reduction == "mean":
            return X.shape[0] / self._N_data
        else:
            raise ValueError("Loss must have reduction 'mean' or 'sum'.")

    @staticmethod
    def flatten_and_concatenate(tensors: List[Tensor]) -> Tensor:
        """Flatten then concatenate all tensors in a list.

        Args:
            tensors: List of tensors.

        Returns:
            Concatenated flattened tensors.
        """
        return cat([t.flatten() for t in tensors])
