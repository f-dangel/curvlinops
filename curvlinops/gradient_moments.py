"""Contains LinearOperator implementation of gradient moment matrices."""

from __future__ import annotations

from typing import List, Tuple

from einops import einsum
from torch import Tensor, autograd, zeros_like

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

    def _matmat_batch(
        self, X: Tensor, y: Tensor, M_list: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """Apply the mini-batch empirical Fisher to a matrix.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M_list: Matrix to be multiplied with in list format.
                Tensors have same shape as trainable model parameters, and an
                additional leading axis for the matrix columns.

        Returns:
            Result of EF multiplication in list format. Has the same shape as
            ``M_list``, i.e. each tensor in the list has the shape of a parameter and a
            leading dimension of matrix columns.

        Raises:
        """
        normalization = {"mean": 1.0 / X.shape[0], "sum": 1.0}[
            self._loss_func.reduction
        ]

        result_list = [zeros_like(M) for M in M_list]

        for n in range(X.shape[0]):
            X_n, y_n = X[n].unsqueeze(0), y[n].unsqueeze(0)
            loss_n = self._loss_func(self._model_func(X_n), y_n)
            grad_n = autograd.grad(loss_n, self._params)

            # coefficients per matrix-vector product
            c = sum(einsum(g, M, "..., col ...-> col") for g, M in zip(grad_n, M_list))

            for idx, g in enumerate(grad_n):
                result_list[idx].add_(
                    einsum(c, g, "col, ... -> col ..."), alpha=normalization
                )

        return tuple(result_list)

    def _adjoint(self) -> EFLinearOperator:
        """Return the linear operator representing the adjoint.

        The empirical Fisher is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self
