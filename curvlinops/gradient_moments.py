"""Contains LinearOperator implementation of gradient moment matrices."""

from __future__ import annotations

from typing import List, Tuple

from torch import Tensor, autograd, einsum, zeros_like

from curvlinops._base import _LinearOperator


class EFLinearOperator(_LinearOperator):
    r"""Uncentered gradient covariance as SciPy linear operator.

    The uncentered gradient covariance is often called 'empirical Fisher' (EF).

    Consider the empirical risk

    .. math::
        \mathcal{L}(\mathbf{\theta})
        =
        c \sum_{n=1}^{N}
        \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)

    with :math:`c = \frac{1}{N}` for ``reduction='mean'`` and :math:`c=1` for
    ``reduction='sum'``. The uncentered gradient covariance matrix is

    .. math::
        c \sum_{n=1}^{N}
        \left(
            \nabla_{\mathbf{\theta}}
            \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)
        \right)
        \left(
            \nabla_{\mathbf{\theta}}
            \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)
        \right)^\top\,.

    .. note::
        Multiplication with the empirical Fisher is currently implemented with an
        inefficient for-loop.
    """

    def _matvec_batch(
        self, X: Tensor, y: Tensor, x_list: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """Apply the mini-batch uncentered gradient covariance to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            x_list: Vector in list format (same shape as trainable model parameters).

        Returns:
            Result of uncentered gradient covariance-multiplication in list format.

        Raises:
            ValueError: If the loss function's reduction cannot be determined.
        """
        result_list = [zeros_like(x) for x in x_list]

        for n in range(X.shape[0]):
            X_n, y_n = X[n].unsqueeze(0), y[n].unsqueeze(0)
            loss_n = self._loss_func(self._model_func(X_n), y_n)
            grad_n = autograd.grad(loss_n, self._params)

            c = sum(einsum("...,...->", g, x) for g, x in zip(grad_n, x_list))

            for idx, g in enumerate(grad_n):
                result_list[idx] += c * g

        reduction = self._loss_func.reduction
        if reduction == "mean":
            normalization = X.shape[0]
        elif reduction == "sum":
            normalization = 1.0
        else:
            raise ValueError("Loss must have reduction 'mean' or 'sum'.")

        return tuple(r / normalization for r in result_list)

    def _adjoint(self) -> EFLinearOperator:
        """Return the linear operator representing the adjoint.

        The empirical Fisher is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self
