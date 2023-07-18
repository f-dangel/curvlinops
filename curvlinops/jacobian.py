"""Implements linear operators for per-sample Jacobians."""

from typing import Callable, Iterable, List, Tuple

from backpack.hessianfree.rop import jacobian_vector_product as jvp
from backpack.utils.convert_parameters import vector_to_parameter_list
from numpy import allclose, column_stack, float32, ndarray
from numpy.random import rand
from scipy.sparse.linalg import LinearOperator
from torch import Tensor, cat
from torch import device as torch_device
from torch import from_numpy, no_grad
from torch.nn import Module, Parameter
from tqdm import tqdm

from curvlinops._base import _LinearOperator


class JacobianLinearOperator(LinearOperator):
    """Linear operator for the Jacobian.

    Can be used with SciPy.
    """

    def __init__(
        self,
        model_func: Callable[[Tensor], Tensor],
        params: List[Parameter],
        data: Iterable[Tuple[Tensor, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
    ):
        r"""Linear operator for the Jacobian as SciPy linear operator.

        Consider a model :math:`f(\mathbf{x}, \mathbf{\theta}): \mathbb{R}^M
        \times \mathbb{R}^D \to \mathbb{R}^C` with parameters
        :math:`\mathbf{\theta}` and input :math:`\mathbf{x}`. Assume we are
        given a data set :math:`\mathcal{D} = \{ (\mathbf{x}_n, \mathbf{y}_n)
        \}_{n=1}^N` of input-target pairs via batches. The model's Jacobian
        :math:`\mathbf{J}_\mathbf{\theta}\mathbf{f}` is an :math:`NC \times D`
        with elements

        .. math::
            \left[
                \mathbf{J}_\mathbf{\theta}\mathbf{f}
            \right]_{(n,c), d}
            =
            \frac{\partial f(\mathbf{x}_n, \mathbf{\theta})}{\partial \theta_d}\,.

        Note that the data must be supplied in deterministic order.

        Args:
            model_func: Neural network function.
            params: Neural network parameters.
            data: Iterable of batched input-target pairs.
            progressbar: Show progress bar.
            check_deterministic: Check if model and data are deterministic.
        """
        num_data = sum(t.shape[0] for t, _ in data)
        x = next(iter(data))[0]
        num_outputs = model_func(x).shape[1:].numel()
        num_params = sum(p.numel() for p in params)
        super().__init__(shape=(num_data * num_outputs, num_params), dtype=float32)

        self._params = params
        self._model_func = model_func
        self._data = data
        self._device = _LinearOperator._infer_device(self._params)
        self._progressbar = progressbar

        if check_deterministic:
            old_device = self._device
            self.to_device(torch_device("cpu"))
            try:
                self._check_deterministic()
            except RuntimeError as e:
                raise e
            finally:
                self.to_device(old_device)

    def _check_deterministic(self):
        """Verify that the linear operator is deterministic.

        - Checks that the data is loaded in a deterministic fashion (e.g. shuffling).
        - Checks that the model is deterministic (e.g. dropout).
        - Checks that matrix-vector multiplication with a single random vector is
          deterministic.

        Note:
            Deterministic checks are performed on CPU. We noticed that even when it
            passes on CPU, it can fail on GPU; probably due to non-deterministic
            operations.

        Raises:
            RuntimeError: If the linear operator is not deterministic.
        """
        print("Performing deterministic checks")

        pred1, y1 = self.predictions_and_targets()
        pred1, y1 = pred1.cpu().numpy(), y1.cpu().numpy()
        pred2, y2 = self.predictions_and_targets()
        pred2, y2 = pred2.cpu().numpy(), y2.cpu().numpy()

        rtol, atol = 5e-5, 1e-6

        if not allclose(y1, y2, rtol=rtol, atol=atol):
            _LinearOperator.print_nonclose(y1, y2, rtol=rtol, atol=atol)
            raise RuntimeError(
                "Data is not loaded in a deterministic fashion."
                + " Make sure shuffling is turned off."
            )
        if not allclose(pred1, pred2, rtol=rtol, atol=atol):
            _LinearOperator.print_nonclose(pred1, pred2, rtol=rtol, atol=atol)
            raise RuntimeError(
                "Model predictions are not deterministic."
                + " Make sure dropout and batch normalization are in eval mode."
            )

        v = rand(self.shape[1]).astype(self.dtype)
        mat_v1 = self @ v
        mat_v2 = self @ v
        if not allclose(mat_v1, mat_v2, rtol=rtol, atol=atol):
            _LinearOperator.print_nonclose(mat_v1, mat_v2, rtol, atol)
            raise RuntimeError("Check for deterministic matvec failed.")

    def to_device(self, device: torch_device):
        """Load linear operator to a device (inplace).

        Args:
            device: Target device.
        """
        self._device = device

        if isinstance(self._model_func, Module):
            self._model_func = self._model_func.to(self._device)
        self._params = [p.to(device) for p in self._params]

    def _loop_over_data(self) -> Iterable[Tuple[Tensor, Tensor]]:
        """Yield batches of the data set, loaded to the correct device.

        Yields:
            Mini-batches ``(X, y)``.
        """
        data_iter = iter(self._data)

        if self._progressbar:
            data_iter = tqdm(data_iter, desc="matvec")

        for X, y in data_iter:
            X, y = X.to(self._device), y.to(self._device)
            yield (X, y)

    def predictions_and_targets(self) -> Tuple[Tensor, Tensor]:
        """Return the batch-concatenated model predictions and labels.

        Returns:
            Batch-concatenated model predictions of shape ``[N, *]`` where ``*``
            denotes the model's output shape (for instance ``* = C``).
            Batch-concatenated labels of shape ``[N, *]``, where ``*`` denotes
            the dimension of a label.
        """
        total_pred, total_y = [], []

        with no_grad():
            for X, y in self._loop_over_data():
                total_pred.append(self._model_func(X))
                total_y.append(y)
        assert total_pred and total_y

        return cat(total_pred), cat(total_y)

    def _matvec(self, x: ndarray) -> ndarray:
        """Loop over all batches in the data and apply the matrix to vector x.

        Args:
            x: Vector for multiplication. Has shape ``[D]``.

        Returns:
            Matrix-multiplication result ``mat @ x``.
        """
        x_list = vector_to_parameter_list(from_numpy(x).to(self._device), self._params)
        out_list = [
            jvp(self._model_func(X), self._params, x_list, retain_graph=False)[
                0
            ].flatten(start_dim=1)
            for X, _ in self._loop_over_data()
        ]

        return cat(out_list).cpu().numpy()

    def _matmat(self, X: ndarray) -> ndarray:
        """Matrix-matrix multiplication.

        Args:
            X: Matrix for multiplication.

        Returns:
            Matrix-multiplication result ``mat @ X``.
        """
        return column_stack([self @ col for col in X.T])
