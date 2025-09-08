"""Implements linear operators for Jacobians."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Callable, Iterable, List, Optional, Tuple, Union

from backpack.hessianfree.lop import transposed_jacobian_vector_product as vjp
from backpack.hessianfree.rop import jacobian_vector_product as jvp
from torch import Tensor, cat, stack, zeros_like
from torch.nn import Parameter

from curvlinops._torch_base import CurvatureLinearOperator


class JacobianLinearOperator(CurvatureLinearOperator):
    """Linear operator of the Jacobian.

    Attributes:
        FIXED_DATA_ORDER: Whether the data order must be fix. ``True`` for Jacobians.
    """

    FIXED_DATA_ORDER: bool = True

    def __init__(
        self,
        model_func: Callable[[Union[MutableMapping, Tensor]], Tensor],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[Union[Tensor, MutableMapping]], int]] = None,
    ):
        r"""Linear operator for the Jacobian as PyTorch linear operator.

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
        super().__init__(
            model_func,
            None,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            num_data=num_data,
            batch_size_fn=batch_size_fn,
        )

    def _get_out_shape(self) -> List[Tuple[int, ...]]:
        """Return the Jacobian's output space dimensions.

        Returns:
            Shapes of the Jacobian's output tensor product space.
            For a model with output of shape ``S``, this is ``[(N, *S)]`` where ``N``
            is the total number of data points.
        """
        x = next(iter(self._data))[0]
        if isinstance(x, Tensor):
            x = x.to(self.device)

        return [(self._N_data,) + self._model_func(x).shape[1:]]

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Apply the Jacobian to a matrix in tensor list format.

        Args:
            M: Matrix for multiplication in tensor list format.

        Returns:
            Matrix-multiplication result ``J @ M`` in tensor list format.
        """
        (num_vecs,) = {m.shape[-1] for m in M}
        JM = []

        for X, _ in self._loop_over_data(desc="_matmat"):
            output = self._model_func(X)

            # multiply the mini-batch Jacobian onto all vectors
            JM_col = [
                jvp(output, self._params, [m[..., n] for m in M])[0]
                for n in range(num_vecs)
            ]
            JM.append(stack(JM_col, dim=-1))

        # concatenate over batches
        return [cat(JM)]

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
            num_data=self._N_data,
        )


class TransposedJacobianLinearOperator(CurvatureLinearOperator):
    """Linear operator for the transpose Jacobian.

    Attributes:
        FIXED_DATA_ORDER: Whether the data order must be fix. ``True`` for Jacobians.
    """

    FIXED_DATA_ORDER: bool = True

    def __init__(
        self,
        model_func: Callable[[Union[MutableMapping, Tensor]], Tensor],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[Union[Tensor, MutableMapping]], int]] = None,
    ):
        r"""Linear operator for the transpose Jacobian as PyTorch linear operator.

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
        super().__init__(
            model_func,
            None,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            num_data=num_data,
            batch_size_fn=batch_size_fn,
        )

    def _get_in_shape(self) -> List[Tuple[int, ...]]:
        """Return the transposed Jacobian's input space dimensions.

        Returns:
            Shapes of the transposed Jacobian's input tensor product space.
            For a model with output of shape ``S``, this is ``[(N, *S)]`` where ``N``
            is the total number of data points.
        """
        x = next(iter(self._data))[0]
        if isinstance(x, Tensor):
            x = x.to(self.device)

        return [(self._N_data,) + self._model_func(x).shape[1:]]

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Apply the transpose Jacobian to a matrix in tensor list format.

        Args:
            M: Matrix for multiplication in tensor list format.

        Returns:
            Matrix-multiplication result ``J^T @ M`` in tensor list format.
        """
        (num_vectors,) = {m.shape[-1] for m in M}

        # allocate result tensors
        JTM = []
        for p in self._params:
            repeat = p.ndim * [1] + [num_vectors]
            JTM.append(zeros_like(p).unsqueeze(-1).repeat(*repeat))

        processed = 0
        for X, _ in self._loop_over_data(desc="_matmat"):
            pred = self._model_func(X)
            start, end = processed, processed + pred.shape[0]

            for n in range(num_vectors):
                (v,) = [m[start:end, ..., n] for m in M]
                JTv = vjp(pred, self._params, v)

                for JTM_p, JTV_p in zip(JTM, JTv):
                    JTM_p[..., n].add_(JTV_p)

            processed += pred.shape[0]

        return JTM

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
            num_data=self._N_data,
            batch_size_fn=self._batch_size_fn,
        )
