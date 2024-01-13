"""Contains LinearOperator implementation of the Hessian."""

from __future__ import annotations

from typing import List, Tuple

from backpack.hessianfree.hvp import hessian_vector_product
from torch import Tensor

from curvlinops._base import _LinearOperator


class HessianLinearOperator(_LinearOperator):
    r"""Hessian as SciPy linear operator.

    Consider the empirical risk

    .. math::
        \mathcal{L}(\mathbf{\theta})
        =
        c \sum_{n=1}^{N}
        \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)

    with :math:`c = \frac{1}{N}` for ``reduction='mean'`` and :math:`c=1` for
    ``reduction='sum'``. The Hessian matrix is

    .. math::
        \nabla^2_{\mathbf{\theta}} \mathcal{L}
        =
        c \sum_{n=1}^{N}
        \nabla_{\mathbf{\theta}}^2
        \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)\,.
    """

    def _matvec_batch(
        self, X: Tensor, y: Tensor, x_list: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """Apply the mini-batch Hessian to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            x_list: Vector in list format (same shape as trainable model parameters).

        Returns:
            Result of Hessian-multiplication in list format.
        """
        loss = self._loss_func(self._model_func(X), y)
        return hessian_vector_product(loss, self._params, x_list)

    def _adjoint(self) -> HessianLinearOperator:
        """Return the linear operator representing the adjoint.

        The Hessian is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self
