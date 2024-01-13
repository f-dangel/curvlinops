"""Implements linear operators for per-sample Jacobians."""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

from backpack.hessianfree.lop import transposed_jacobian_vector_product as vjp
from backpack.hessianfree.rop import jacobian_vector_product as jvp
from numpy import allclose, ndarray
from torch import Tensor, from_numpy, no_grad, zeros_like
from torch.nn import Parameter

from curvlinops._base import _LinearOperator


class JacobianLinearOperator(_LinearOperator):
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
        matrix with elements

        .. math::
            \left[
                \mathbf{J}_\mathbf{\theta}\mathbf{f}
            \right]_{(n,c), d}
            =
            \frac{\partial [f(\mathbf{x}_n, \mathbf{\theta})]_c}{\partial \theta_d}\,.

        Note that the data must be supplied in deterministic order.

        Args:
            model_func: Neural network function.
            params: Neural network parameters.
            data: Iterable of batched input-target pairs.
            progressbar: Show progress bar.
            check_deterministic: Check if model and data are deterministic.
        """
        num_data = sum(t.shape[0] for t, _ in data)
        x = next(iter(data))[0].to(self._infer_device(params))
        num_outputs = model_func(x).shape[1:].numel()
        num_params = sum(p.numel() for p in params)
        super().__init__(
            model_func,
            None,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            shape=(num_data * num_outputs, num_params),
        )

    def _check_deterministic(self):
        """Verify that the linear operator is deterministic.

        In addition to the checks from the base class, checks that the model
        predictions and data are always the same (loaded in the same order, and
        only deterministic operations in the network.

        Note:
            Deterministic checks should be performed on CPU. We noticed that even when
            it passes on CPU, it can fail on GPU; probably due to non-deterministic
            operations.

        Raises:
            RuntimeError: If the linear operator is not deterministic.
        """
        super()._check_deterministic()

        rtol, atol = 5e-5, 1e-6

        with no_grad():
            for (X1, y1), (X2, y2) in zip(
                self._loop_over_data(desc="_check_deterministic_data_pred"),
                self._loop_over_data(desc="_check_deterministic_data_pred2"),
            ):
                pred1, y1 = self._model_func(X1).cpu().numpy(), y1.cpu().numpy()
                pred2, y2 = self._model_func(X2).cpu().numpy(), y2.cpu().numpy()
                X1, X2 = X1.cpu().numpy(), X2.cpu().numpy()

                if not allclose(X1, X2) or not allclose(y1, y2):
                    self.print_nonclose(X1, X2, rtol=rtol, atol=atol)
                    self.print_nonclose(y1, y2, rtol=rtol, atol=atol)
                    raise RuntimeError("Non-deterministic data loading detected.")

                if not allclose(pred1, pred2):
                    self.print_nonclose(pred1, pred2, rtol=rtol, atol=atol)
                    raise RuntimeError("Non-deterministic model detected.")

    def _matvec(self, x: ndarray) -> ndarray:
        """Loop over all batches in the data and apply the matrix to vector x.

        Args:
            x: Vector for multiplication. Has shape ``[D]``.

        Returns:
            Matrix-multiplication result ``mat @ x``. Has shape ``[N * C]``.
        """
        x_list = self._preprocess(x)
        out_list = [
            jvp(self._model_func(X), self._params, x_list, retain_graph=False)[
                0
            ].flatten(start_dim=1)
            for X, _ in self._loop_over_data(desc="_matvec")
        ]

        return self._postprocess(out_list)

    def _adjoint(self) -> TransposedJacobianLinearOperator:
        """Return a linear operator representing the adjoint.

        Returns:
            Linear operator representing the transposed Jacobian.
        """
        return TransposedJacobianLinearOperator(
            self._model_func,
            self._params,
            self._data,
            progressbar=self._progressbar,
            check_deterministic=False,
        )


class TransposedJacobianLinearOperator(_LinearOperator):
    """Linear operator for the transpose Jacobian.

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
        r"""Linear operator for the transpose Jacobian as SciPy linear operator.

        Consider a model :math:`f(\mathbf{x}, \mathbf{\theta}): \mathbb{R}^M
        \times \mathbb{R}^D \to \mathbb{R}^C` with parameters
        :math:`\mathbf{\theta}` and input :math:`\mathbf{x}`. Assume we are
        given a data set :math:`\mathcal{D} = \{ (\mathbf{x}_n, \mathbf{y}_n)
        \}_{n=1}^N` of input-target pairs via batches. The model's transpose
        Jacobian :math:`(\mathbf{J}_\mathbf{\theta}\mathbf{f})^\top` is an
        :math:`D \times NC` matrix with elements

        .. math::
            \left[
                (\mathbf{J}_\mathbf{\theta}\mathbf{f})^\top
            \right]_{d, (n,c)}
            =
            \frac{\partial [f(\mathbf{x}_n, \mathbf{\theta})]_c}{\partial \theta_d}\,.

        Note that the data must be supplied in deterministic order.

        Args:
            model_func: Neural network function.
            params: Neural network parameters.
            data: Iterable of batched input-target pairs.
            progressbar: Show progress bar.
            check_deterministic: Check if model and data are deterministic.
        """
        num_data = sum(t.shape[0] for t, _ in data)
        x = next(iter(data))[0].to(self._infer_device(params))
        num_outputs = model_func(x).shape[1:].numel()
        num_params = sum(p.numel() for p in params)
        super().__init__(
            model_func,
            None,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            shape=(num_params, num_data * num_outputs),
        )

    def _check_deterministic(self):
        """Verify that the linear operator is deterministic.

        In addition to the checks from the base class, checks that the model
        predictions and data are always the same (loaded in the same order, and
        only deterministic operations in the network.

        Note:
            Deterministic checks should be performed on CPU. We noticed that even when
            it passes on CPU, it can fail on GPU; probably due to non-deterministic
            operations.

        Raises:
            RuntimeError: If the linear operator is not deterministic.
        """
        super()._check_deterministic()

        rtol, atol = 5e-5, 1e-6

        with no_grad():
            for (X1, y1), (X2, y2) in zip(
                self._loop_over_data(desc="_check_deterministic_data_pred1"),
                self._loop_over_data(desc="_check_deterministic_data_pred2"),
            ):
                pred1, y1 = self._model_func(X1).cpu().numpy(), y1.cpu().numpy()
                pred2, y2 = self._model_func(X2).cpu().numpy(), y2.cpu().numpy()
                X1, X2 = X1.cpu().numpy(), X2.cpu().numpy()

                if not allclose(X1, X2) or not allclose(y1, y2):
                    self.print_nonclose(X1, X2, rtol=rtol, atol=atol)
                    self.print_nonclose(y1, y2, rtol=rtol, atol=atol)
                    raise RuntimeError("Non-deterministic data loading detected.")

                if not allclose(pred1, pred2):
                    self.print_nonclose(pred1, pred2, rtol=rtol, atol=atol)
                    raise RuntimeError("Non-deterministic model detected.")

    def _matvec(self, x: ndarray) -> ndarray:
        """Loop over all batches in the data and apply the matrix to vector x.

        Args:
            x: Vector for multiplication. Has shape ``[N C]``.

        Returns:
            Matrix-multiplication result ``mat @ x``. Has shape ``[D]``.
        """
        x_torch = from_numpy(x).to(self._device)
        out_list = [zeros_like(p) for p in self._params]

        processed = 0
        for X, _ in self._loop_over_data(desc="_matvec"):
            pred = self._model_func(X)
            v = x_torch[processed : processed + pred.numel()].reshape_as(pred)
            processed += pred.numel()

            for p_res, p_vjp in zip(
                out_list, vjp(pred, self._params, v, retain_graph=False)
            ):
                p_res.add_(p_vjp)

        return self._postprocess(out_list)

    def _adjoint(self) -> JacobianLinearOperator:
        """Return a linear operator representing the adjoint.

        Returns:
            Linear operator representing the Jacobian.
        """
        return JacobianLinearOperator(
            self._model_func,
            self._params,
            self._data,
            progressbar=self._progressbar,
            check_deterministic=False,
        )
