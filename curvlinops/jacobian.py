"""Implements linear operators for Jacobians."""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping
from functools import cached_property

from torch import Tensor, cat, no_grad, zeros_like
from torch.func import jvp, vjp, vmap
from torch.nn import Module, Parameter

from curvlinops._torch_base import CurvatureLinearOperator


def make_batch_jacobian_matrix_product(
    f: Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
) -> Callable[[dict[str, Tensor], Tensor | MutableMapping, dict[str, Tensor]], Tensor]:
    """Set up function to multiply with the mini-batch Jacobian.

    Args:
        f: Functional model with signature ``(params_dict, X) -> prediction``.

    Returns:
        A function ``(params_dict, X, M_dict) -> JM`` that takes parameters as a
        dict, input ``X``, and a matrix ``M`` as a dict (each value has an extra
        trailing dimension for columns), and returns the mini-batch Jacobian
        applied to ``M`` as a Tensor.
    """

    @no_grad()
    def jacobian_vector_product(
        params: dict[str, Tensor],
        X: Tensor | MutableMapping,
        v: dict[str, Tensor],
    ) -> Tensor:
        """Multiply the mini-batch Jacobian on a vector in dict format.

        Args:
            params: Parameters of the model as a dict.
            X: Input to the DNN.
            v: Vector as a dict matching the structure of ``params``.

        Returns:
            Result of Jacobian multiplication as a Tensor with shape
            (batch_size, *output_shape).
        """
        # Apply the Jacobian of f onto v: v → Jv
        _, f_jvp = jvp(lambda p: f(p, X), (params,), (v,))
        return f_jvp

    # Vectorize over vectors to multiply onto a matrix in list format
    return vmap(
        jacobian_vector_product,
        # No vmap in params or X; last-axis vmap over the vector tuple
        in_dims=(None, None, -1),
        # Vmapped output axis is last
        out_dims=-1,
        # We want each vector to be multiplied with the same mini-batch Jacobian
        randomness="same",
    )


def make_batch_transposed_jacobian_matrix_product(
    f: Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
) -> Callable[[dict[str, Tensor], Tensor | MutableMapping, Tensor], dict[str, Tensor]]:
    r"""Set up function to multiply with the mini-batch transposed Jacobian.

    Args:
        f: Functional model with signature ``(params_dict, X) -> prediction``.

    Returns:
        A function ``(params_dict, X, v) -> JTv_dict`` that takes parameters as a
        dict, input ``X``, and a matrix ``v`` as a single tensor, and returns the
        mini-batch transposed Jacobian applied to ``v`` as a dict.
    """

    @no_grad()
    def transposed_jacobian_vector_product(
        params: dict[str, Tensor], X: Tensor | MutableMapping, v: Tensor
    ) -> dict[str, Tensor]:
        """Multiply the mini-batch transposed Jacobian on a vector.

        Args:
            params: Parameters of the model as a dict.
            X: Input to the DNN.
            v: Vector to be multiplied with, shape (batch_size, *output_shape).

        Returns:
            Result of transposed Jacobian multiplication as a dict with the same
            keys as ``params``.
        """
        # Apply the transposed Jacobian of f onto v: v → J^T v
        _, vjp_func = vjp(lambda p: f(p, X), params)
        (result_dict,) = vjp_func(v)
        return result_dict

    # Vectorize over vectors to multiply onto a matrix in list format
    return vmap(
        transposed_jacobian_vector_product,
        # No vmap in params or X, assume last axis is vmapped in the input matrix
        in_dims=(None, None, -1),
        # Vmapped output axis is last
        out_dims=-1,
        # We want each vector to be multiplied with the same mini-batch transposed Jacobian
        randomness="same",
    )


class JacobianLinearOperator(CurvatureLinearOperator):
    """Linear operator of the Jacobian.

    Attributes:
        FIXED_DATA_ORDER: Whether the data order must be fix. ``True`` for Jacobians.
    """

    FIXED_DATA_ORDER: bool = True

    @cached_property
    def _mp(
        self,
    ) -> Callable[
        [dict[str, Tensor], Tensor | MutableMapping, dict[str, Tensor]], Tensor
    ]:
        """Lazy initialization of batch-Jacobian matrix product function.

        Returns:
            Function that computes mini-batch Jacobian-matrix products, given
            parameters as a dict, input ``X``, and a matrix ``M`` as a dict.
        """
        return make_batch_jacobian_matrix_product(self._model_func)

    def __init__(
        self,
        model_func: Module,
        params: list[Parameter],
        data: Iterable[tuple[Tensor | MutableMapping, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: int | None = None,
        batch_size_fn: Callable[[Tensor | MutableMapping], int] | None = None,
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

    def _get_out_shape(self) -> list[tuple[int, ...]]:
        """Return the Jacobian's output space dimensions.

        Returns:
            Shapes of the Jacobian's output tensor product space.
            For a model with output of shape ``S``, this is ``[(N, *S)]`` where ``N``
            is the total number of data points.
        """
        x = next(iter(self._data))[0]
        if isinstance(x, Tensor):
            x = x.to(self.device)

        return [(self._N_data,) + self._model_func(self._params, x).shape[1:]]

    def _matmat(self, M: list[Tensor]) -> list[Tensor]:
        """Apply the Jacobian to a matrix in tensor list format.

        Args:
            M: Matrix for multiplication in tensor list format.

        Returns:
            Matrix-multiplication result ``J @ M`` in tensor list format.
        """
        # Apply mini-batch Jacobians and collect results
        M_dict = dict(zip(self._params.keys(), M))
        JM = []
        for X, _ in self._loop_over_data(desc="_matmat"):
            JM.append(self._mp(self._params, X, M_dict))

        # concatenate over batches
        return [cat(JM)]

    def _adjoint(self) -> TransposedJacobianLinearOperator:
        """Return a linear operator representing the adjoint.

        Returns:
            Linear operator representing the transposed Jacobian.
        """
        return TransposedJacobianLinearOperator(
            self._model_module,
            list(self._params.values()),
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

    @cached_property
    def _mp(
        self,
    ) -> Callable[[dict[str, Tensor], Tensor, Tensor], dict[str, Tensor]]:
        """Lazy initialization of batch-transposed-Jacobian matrix product function.

        Returns:
            Function that computes mini-batch transposed Jacobian-matrix products,
            given parameters as a dict, input ``X``, and a matrix ``M``.
        """
        return make_batch_transposed_jacobian_matrix_product(self._model_func)

    def __init__(
        self,
        model_func: Module,
        params: list[Parameter],
        data: Iterable[tuple[Tensor | MutableMapping, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: int | None = None,
        batch_size_fn: Callable[[Tensor | MutableMapping], int] | None = None,
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

    def _get_in_shape(self) -> list[tuple[int, ...]]:
        """Return the transposed Jacobian's input space dimensions.

        Returns:
            Shapes of the transposed Jacobian's input tensor product space.
            For a model with output of shape ``S``, this is ``[(N, *S)]`` where ``N``
            is the total number of data points.
        """
        x = next(iter(self._data))[0]
        if isinstance(x, Tensor):
            x = x.to(self.device)

        return [(self._N_data,) + self._model_func(self._params, x).shape[1:]]

    def _matmat(self, M: list[Tensor]) -> list[Tensor]:
        """Apply the transpose Jacobian to a matrix in tensor list format.

        Args:
            M: Matrix for multiplication in tensor list format.

        Returns:
            Matrix-multiplication result ``J^T @ M`` in tensor list format.
        """
        # allocate result tensors as dict
        (num_vectors,) = {m.shape[-1] for m in M}
        JTM = {}
        for name, p in self._params.items():
            repeat = p.ndim * [1] + [num_vectors]
            JTM[name] = zeros_like(p).unsqueeze(-1).repeat(*repeat)

        processed = 0
        for X, _ in self._loop_over_data(desc="_matmat"):
            processing = self._batch_size_fn(X)
            start, end = processed, processed + processing

            # Extract the relevant slice of the input matrix for this batch
            M_batch = M[0][start:end]  # Shape: (batch_size, *output_shape, num_vectors)

            # Apply mini-batch transposed Jacobian (returns dict)
            JTM_batch = self._mp(self._params, X, M_batch)

            # Accumulate results
            for name in JTM:
                JTM[name].add_(JTM_batch[name])

            processed += processing

        return [JTM[k] for k in self._params]

    def _adjoint(self) -> JacobianLinearOperator:
        """Return a linear operator representing the adjoint.

        Returns:
            Linear operator representing the Jacobian.
        """
        return JacobianLinearOperator(
            self._model_module,
            list(self._params.values()),
            self._data,
            progressbar=self._progressbar,
            check_deterministic=False,
            num_data=self._N_data,
            batch_size_fn=self._batch_size_fn,
        )
