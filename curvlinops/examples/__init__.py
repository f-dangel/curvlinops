"""Utilities used throughout the documentation examples."""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from math import sqrt

from torch import (
    Tensor,
    device,
    dtype,
    einsum,
    ones,
)
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


def gradient_l2_norm(gradient: Iterable[Tensor | None]) -> Tensor:
    """Compute the Euclidean norm of a gradient stored as tensors.

    Args:
        gradient: Iterable containing gradient tensors. Entries can be ``None`` and
            will be ignored.

    Returns:
        Euclidean norm of the concatenated gradient.

    Raises:
        ValueError: If no gradient tensor is provided.
    """
    sq_norm = None
    for g in gradient:
        if g is None:
            continue
        sq_norm = g.pow(2).sum() if sq_norm is None else sq_norm + g.pow(2).sum()

    if sq_norm is None:
        raise ValueError("Expected at least one gradient tensor.")

    return sq_norm.sqrt()


def _check_damping_hyperparameters(damping_scale: float, min_damping: float):
    """Validate the scalar hyperparameters for gradient-norm damping.

    Args:
        damping_scale: Non-negative scale factor in the damping rule.
        min_damping: Non-negative lower bound on the damping value.

    Raises:
        ValueError: If ``damping_scale`` or ``min_damping`` is negative.
    """
    if damping_scale < 0.0:
        raise ValueError(f"Expected damping_scale >= 0. Got {damping_scale = }.")
    if min_damping < 0.0:
        raise ValueError(f"Expected min_damping >= 0. Got {min_damping = }.")


def gradient_norm_damping(
    gradient: Iterable[Tensor | None], damping_scale: float, min_damping: float = 0.0
) -> float:
    r"""Compute adaptive damping from gradient tensors.

    This damping rule appears in
    `Mishchenko, 2023 <https://doi.org/10.1137/22M1488752>`_ as a cheap
    Levenberg-Marquardt interpretation of cubic regularization.

    Args:
        gradient: Iterable containing gradient tensors. Entries can be ``None`` and
            will be ignored.
        damping_scale: Non-negative constant ``c`` in the rule
            ``sqrt(c * ||g||_2)``.
        min_damping: Non-negative lower bound on the damping value.
            Default: ``0.0``.

    Returns:
        Adaptive damping value as a Python ``float``.

    Example:
        If ``gradients`` is a list of tensors with the same structure as the
        model parameters, the damping is simply

        ``gradient_norm_damping(gradients, damping_scale=1e-1, min_damping=1e-4)``.
    """
    _check_damping_hyperparameters(damping_scale, min_damping)
    return max(min_damping, sqrt(damping_scale * gradient_l2_norm(gradient).item()))


@dataclass(frozen=True)
class GradientNormDamping:
    r"""Callable gradient-norm damping policy.

    This is a small convenience wrapper around :func:`gradient_norm_damping`.
    It is useful when the hyperparameters are fixed for an entire run:

    .. code-block:: python

        damping_rule = GradientNormDamping(damping_scale=1e-1, min_damping=1e-4)
        damping = damping_rule(gradients)

    If you need one damping value per KFAC/EKFAC block, use :meth:`per_block`
    with one gradient collection per block.

    The policy computes

    .. math::

        \lambda = \max(\lambda_{\min}, \sqrt{c \lVert g \rVert_2})

    from a collection of gradient tensors.

    Attributes:
        damping_scale: Non-negative constant ``c`` in the damping rule.
        min_damping: Non-negative lower bound on the damping value.
    """

    damping_scale: float
    min_damping: float = 0.0

    def __post_init__(self):
        """Validate the damping policy hyperparameters."""
        _check_damping_hyperparameters(self.damping_scale, self.min_damping)

    def __call__(self, gradient: Iterable[Tensor | None]) -> float:
        """Evaluate the damping rule on a gradient collection.

        Args:
            gradient: Iterable containing gradient tensors. Entries can be ``None`` and
                will be ignored.

        Returns:
            Damping value as a Python ``float``.
        """
        return gradient_norm_damping(gradient, self.damping_scale, self.min_damping)

    def per_block(
        self, block_gradients: Iterable[Iterable[Tensor | None]]
    ) -> tuple[float, ...]:
        """Evaluate the damping rule independently for each block.

        This is useful with APIs that accept one damping value per canonical block,
        such as ``KFACLinearOperator.inverse`` and ``EKFACLinearOperator.inverse``.

        Args:
            block_gradients: Iterable where each entry contains the gradient tensors
                associated with one block.

        Returns:
            Tuple containing one damping value per block.
        """
        return tuple(self(gradient) for gradient in block_gradients)


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
