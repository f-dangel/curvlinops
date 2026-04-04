"""Contains functionality for examples in the documentation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping

import torch
from torch import (
    Tensor,
    autograd,
    device,
    dtype,
    einsum,
    ones,
)
from torch import (
    compile as torch_compile,
)
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import Module

from curvlinops._empirical_risk import _EmpiricalRiskMixin
from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.diag import DiagonalLinearOperator


def gradient_and_loss(
    model_func: Module,
    loss_func: Module,
    params: dict[str, Tensor],
    data: Iterable[tuple[Tensor | MutableMapping, Tensor]],
    batch_size_fn: Callable[[MutableMapping | Tensor], int] | None = None,
    num_data: int | None = None,
) -> tuple[list[Tensor], Tensor]:
    """Evaluate the gradient and loss on a data set.

    Note:
        This uses ``torch.autograd.grad`` internally, not ``torch.func.grad``.
        The functional API (``torch.func.grad_and_value``) uses ~2x the peak GPU
        memory of ``autograd.grad`` for the same computation
        (see `pytorch#134612 <https://github.com/pytorch/pytorch/issues/134612>`_).

    Args:
        model_func: The neural network's forward pass (an ``nn.Module``).
        loss_func: The loss function.
        params: The parameter values at which the gradient is evaluated. A
            dictionary mapping parameter names to tensors.
        data: Source from which mini-batches can be drawn, for instance a list of
            mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``.
        batch_size_fn: Function that returns the batch size given an input ``X``.
            If ``None``, defaults to ``X.shape[0]``.
        num_data: Total number of data points. If ``None``, it is inferred from
            the data at the cost of one traversal through the data loader.

    Returns:
        Tuple of (gradient, loss) accumulated over the full data set.
    """
    mixin = _EmpiricalRiskMixin(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        num_data=num_data,
        check_deterministic=False,
    )
    return mixin._gradient_and_loss()


def trace_gradient_and_loss(
    model: Module,
    loss_func: Module,
    params: dict[str, Tensor],
    example_X: Tensor | MutableMapping,
    example_y: Tensor,
) -> torch.fx.GraphModule:
    """Trace per-batch gradient+loss into an FX graph via ``make_fx``.

    The tracing captures the backward pass (``autograd.grad``) as explicit
    forward ops. The resulting ``GraphModule`` contains no autograd calls.

    Args:
        model: Neural network module.
        loss_func: Loss function module.
        params: Parameter dict at which to evaluate.
        example_X: Example input for tracing (determines batch shape).
        example_y: Example target for tracing.

    Returns:
        Traced ``GraphModule`` with signature ``(params, X, y) -> (grads, loss)``.
    """
    dev = next(model.parameters()).device
    if isinstance(example_X, Tensor):
        example_X = example_X.to(dev)
    example_y = example_y.to(dev)

    param_keys = tuple(params.keys())

    def grad_and_loss(
        params: dict[str, Tensor], X: Tensor | MutableMapping, y: Tensor
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        prediction = functional_call(model, params, X)
        loss = loss_func(prediction, y)
        grads = autograd.grad(loss, tuple(params[k] for k in param_keys))
        return grads, loss.detach()

    return make_fx(grad_and_loss)(params, example_X, example_y)


def make_compiled_gradient_and_loss(
    model: Module,
    loss_func: Module,
    params: dict[str, Tensor],
    example_X: Tensor | MutableMapping,
    example_y: Tensor,
) -> Callable[[dict[str, Tensor], Tensor | MutableMapping, Tensor], tuple]:
    """Trace gradient+loss with ``make_fx``, then compile.

    Args:
        model: Neural network module.
        loss_func: Loss function module.
        params: Parameter dict at which to evaluate.
        example_X: Example input for tracing (determines batch shape).
        example_y: Example target for tracing.

    Returns:
        Compiled function ``(params, X, y) -> (grads_tuple, loss)``.
    """
    traced = trace_gradient_and_loss(model, loss_func, params, example_X, example_y)
    compiled = torch_compile(traced)

    dev = next(model.parameters()).device

    def compiled_grad_and_loss(
        params: dict[str, Tensor], X: Tensor | MutableMapping, y: Tensor
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        # Detach params so aot_autograd does not try to differentiate through
        # the already-traced backward ops (would require double-backward).
        params_detached = {k: v.detach() for k, v in params.items()}
        if isinstance(X, Tensor):
            X = X.to(dev)
        return compiled(params_detached, X, y.to(dev))

    return compiled_grad_and_loss


class TensorLinearOperator(PyTorchLinearOperator):
    """Linear operator wrapping a single tensor as a linear operator."""

    def __init__(self, A: Tensor):
        """Initialize linear operator from a 2D tensor.

        Args:
            A: A 2D tensor representing the matrix.

        Raises:
            ValueError: If ``A`` is not a 2D tensor.
        """
        if A.ndim != 2:
            raise ValueError(f"Input tensor must be 2D. Got {A.ndim}D.")
        super().__init__([(A.shape[1],)], [(A.shape[0],)])
        self._A = A
        self.SELF_ADJOINT = A.shape == A.T.shape and A.allclose(A.T)

    @property
    def device(self) -> device:
        """Infer the linear operator's device.

        Returns:
            The linear operator's device.
        """
        return self._A.device

    @property
    def dtype(self) -> dtype:
        """Infer the linear operator's data type.

        Returns:
            The linear operator's data type.
        """
        return self._A.dtype

    def _adjoint(self) -> TensorLinearOperator:
        """Return a linear operator representing the adjoint.

        Returns:
            The adjoint linear operator.
        """
        return TensorLinearOperator(self._A.conj().T)

    def _matmat(self, M: list[Tensor]) -> list[Tensor]:
        """Multiply the linear operator onto a matrix in list format.

        Args:
            M: Matrix for multiplication in list format.

        Returns:
            Result of the matrix-matrix multiplication in list format.
        """
        (M0,) = M
        return [self._A @ M0]

    def trace(self) -> Tensor:
        """Trace of the matrix.

        Returns:
            Trace of the underlying tensor matrix.
        """
        return self._A.trace()

    def det(self) -> Tensor:
        """Compute the determinant of the matrix.

        Returns:
            Determinant of the underlying tensor matrix.
        """
        return self._A.det()

    def logdet(self) -> Tensor:
        """Log determinant of the matrix.

        Returns:
            Log determinant of the underlying tensor matrix.
        """
        return self._A.logdet()

    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the matrix.

        Returns:
            Frobenius norm of the underlying tensor matrix.
        """
        return self._A.norm(p="fro")


class OuterProductLinearOperator(PyTorchLinearOperator):
    """Linear operator for low-rank matrices of the form ``∑ᵢ cᵢ aᵢ aᵢᵀ``.

    ``cᵢ`` is the coefficient for the vector ``aᵢ``.
    """

    SELF_ADJOINT = True

    def __init__(self, c: Tensor, A: Tensor):
        """Store coefficients and vectors for low-rank representation.

        Args:
            c: Coefficients ``cᵢ``. Has shape ``[K]`` where ``K`` is the rank.
            A: Matrix of shape ``[D, K]``, where ``D`` is the linear operators
                dimension, that stores the low-rank vectors columnwise, i.e. ``aᵢ``
                is stored in ``A[:,i]``.
        """
        shape = [(A.shape[0],)]
        super().__init__(shape, shape)
        self._A = A
        self._c = c

    def _matmat(self, M: list[Tensor]) -> list[Tensor]:
        """Apply the linear operator to a matrix in list format.

        Args:
            M: The matrix to multiply onto in list format.

        Returns:
            The result of the multiplication in list format.
        """
        (M0,) = M
        # Compute ∑ᵢ cᵢ aᵢ aᵢᵀ @ X
        return [einsum("ik,k,jk,jl->il", self._A, self._c, self._A, M0)]

    def _adjoint(self) -> OuterProductLinearOperator:
        """Return the linear operator representing the adjoint.

        An outer product is self-adjoint.

        Returns:
            Self.
        """
        return self

    @property
    def dtype(self) -> dtype:
        """Return the data type of the linear operator.

        Returns:
            The data type of the linear operator.
        """
        return self._A.dtype

    @property
    def device(self) -> device:
        """Return the linear operator's device.

        Returns:
            The device on which the linear operator is defined.
        """
        return self._A.device


class IdentityLinearOperator(DiagonalLinearOperator):
    """Linear operator representing the identity matrix."""

    SELF_ADJOINT = True

    def __init__(self, shape: list[tuple[int, ...]], device: device, dtype: dtype):
        """Store the linear operator's input and output space dimensions.

        Args:
            shape: A list of shapes specifying the identity's input and output space.
            device: The device on which the identity operator is defined.
            dtype: The data type of the identity operator.
        """
        # Build a memory-efficient version of the diagonal containing ones
        alloc_shape, expand_shape = [len(s) * (1,) for s in shape], shape
        diagonal = [
            ones(*alloc_s, device=device, dtype=dtype).expand(*expand_s)
            for alloc_s, expand_s in zip(alloc_shape, expand_shape)
        ]
        super().__init__(diagonal)

    def _matmat(self, M: list[Tensor]) -> list[Tensor]:
        """Apply the linear operator to a matrix in list format.

        Args:
            M: The matrix to multiply onto in list format.

        Returns:
            The result of the matrix multiplication in list format.
        """
        return M
