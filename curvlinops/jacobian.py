"""Implements linear operators for per-sample Jacobians."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Callable, Iterable, List, Optional, Tuple, Union

from backpack.hessianfree.lop import transposed_jacobian_vector_product as vjp
from backpack.hessianfree.rop import jacobian_vector_product as jvp
from numpy import allclose, ndarray
from torch import Tensor, cat, from_numpy, no_grad, stack, zeros_like
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
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[MutableMapping], int]] = None,
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
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified. The intended behavior is to consume the first
                entry of the iterates from ``data`` and return their batch size.
        """
        _batch_size_fn = (
            (lambda X: X.shape[0]) if batch_size_fn is None else batch_size_fn
        )
        num_data = (
            sum(_batch_size_fn(t) for t, _ in data) if num_data is None else num_data
        )
        x = next(iter(data))[0]

        if isinstance(x, Tensor):
            x = x.to(self._infer_device(params))

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
            num_data=num_data,
            batch_size_fn=batch_size_fn,
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

        def check_X_y(X1, X2, y1, y2):
            if not allclose(X1, X2) or not allclose(y1, y2):
                self.print_nonclose(X1, X2, rtol=rtol, atol=atol)
                self.print_nonclose(y1, y2, rtol=rtol, atol=atol)
                raise RuntimeError("Non-deterministic data loading detected.")

        with no_grad():
            for (X1, y1), (X2, y2) in zip(
                self._loop_over_data(desc="_check_deterministic_data_pred"),
                self._loop_over_data(desc="_check_deterministic_data_pred2"),
            ):
                pred1, y1 = self._model_func(X1).cpu().numpy(), y1.cpu().numpy()
                pred2, y2 = self._model_func(X2).cpu().numpy(), y2.cpu().numpy()

                if isinstance(X1, Tensor) or isinstance(X2, Tensor):
                    X1, X2 = X1.cpu().numpy(), X2.cpu().numpy()
                    check_X_y(X1, X2, y1, y2)
                else:  # X is a MutableMapping
                    for k in X1.keys():
                        v1, v2 = X1[k], X2[k]

                        if isinstance(v1, Tensor) or isinstance(v2, Tensor):
                            X1, X2 = v1.cpu().numpy(), v2.cpu().numpy()

                        check_X_y(X1, X2, y1, y2)

                if not allclose(pred1, pred2):
                    self.print_nonclose(pred1, pred2, rtol=rtol, atol=atol)
                    raise RuntimeError("Non-deterministic model detected.")

    def _matmat(self, M: ndarray) -> ndarray:
        """Apply the Jacobian to a matrix.

        Args:
            M: Matrix for multiplication. Has shape ``[D, K]``.

        Returns:
            Matrix-multiplication result ``J @ M``. Has shape ``[N * C, K]``.
        """
        num_vecs = M.shape[1]
        M_list = self._preprocess(M)

        result_list = []

        for X, _ in self._loop_over_data(desc="_matmat"):
            output = self._model_func(X)

            # multiply the mini-batch Jacobian onto all vectors
            col = []
            for n in range(num_vecs):
                (col_n,) = jvp(output, self._params, [M[n] for M in M_list])
                col.append(col_n.flatten(start_dim=1))
            # combine columns into a single tensor and append
            result_list.append(stack(col))

        # concatenate over batches
        result_list = [cat(result_list, dim=1)]

        return self._postprocess(result_list)

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
            batch_size_fn=self._batch_size_fn,
        )


class TransposedJacobianLinearOperator(_LinearOperator):
    """Linear operator for the transpose Jacobian.

    Can be used with SciPy.
    """

    def __init__(
        self,
        model_func: Callable[[Tensor], Tensor],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[MutableMapping], int]] = None,
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
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified. The intended behavior is to consume the first
                entry of the iterates from ``data`` and return their batch size.
        """
        _batch_size_fn = (
            (lambda X: X.shape[0]) if batch_size_fn is None else batch_size_fn
        )
        num_data = (
            sum(_batch_size_fn(t) for t, _ in data) if num_data is None else num_data
        )
        x = next(iter(data))[0]

        if isinstance(x, Tensor):
            x = x.to(self._infer_device(params))

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
            num_data=num_data,
            batch_size_fn=batch_size_fn,
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

                def check_X_y(X1, X2, y1, y2):
                    if not allclose(X1, X2) or not allclose(y1, y2):
                        self.print_nonclose(X1, X2, rtol=rtol, atol=atol)
                        self.print_nonclose(y1, y2, rtol=rtol, atol=atol)
                        raise RuntimeError("Non-deterministic data loading detected.")

                with no_grad():
                    for (X1, y1), (X2, y2) in zip(
                        self._loop_over_data(desc="_check_deterministic_data_pred"),
                        self._loop_over_data(desc="_check_deterministic_data_pred2"),
                    ):
                        pred1, y1 = self._model_func(X1).cpu().numpy(), y1.cpu().numpy()
                        pred2, y2 = self._model_func(X2).cpu().numpy(), y2.cpu().numpy()

                        if isinstance(X1, Tensor) or isinstance(X2, Tensor):
                            X1, X2 = X1.cpu().numpy(), X2.cpu().numpy()
                            check_X_y(X1, X2, y1, y2)
                        else:  # X is a MutableMapping
                            for k in X1.keys():
                                v1, v2 = X1[k], X2[k]

                                if isinstance(v1, Tensor) or isinstance(v2, Tensor):
                                    X1, X2 = v1.cpu().numpy(), v2.cpu().numpy()

                                check_X_y(X1, X2, y1, y2)

                if not allclose(pred1, pred2):
                    self.print_nonclose(pred1, pred2, rtol=rtol, atol=atol)
                    raise RuntimeError("Non-deterministic model detected.")

    def _matmat(self, M: ndarray) -> ndarray:
        """Apply the transpose Jacobian to a matrix.

        Args:
            M: Matrix for multiplication. Has shape ``[N C, K]``.

        Returns:
            Matrix-multiplication result ``J^T @ M``. Has shape ``[D, K]``.
        """
        M_torch = from_numpy(M).to(self._device)
        num_vectors = M_torch.shape[1]

        # allocate result tensors
        out_list = []
        for p in self._params:
            repeat = [num_vectors] + [1] * p.ndim
            out_list.append(zeros_like(p).unsqueeze(0).repeat(*repeat))

        processed = 0
        for X, _ in self._loop_over_data(desc="_matmat"):
            pred = self._model_func(X)
            start, end = processed, processed + pred.numel()

            for n in range(num_vectors):
                v = M_torch[start:end, n].reshape_as(pred)
                JT_v = vjp(pred, self._params, v)

                for p_res, p_vjp in zip(out_list, JT_v):
                    p_res[n].add_(p_vjp)

            processed += pred.numel()

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
            batch_size_fn=self._batch_size_fn,
        )
