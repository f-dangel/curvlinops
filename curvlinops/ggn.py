"""Contains LinearOperator implementation of the GGN."""

from __future__ import annotations

from typing import List, Tuple

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from torch import Tensor

from curvlinops._base import _LinearOperator


class GGNLinearOperator(_LinearOperator):
    r"""GGN as SciPy linear operator.

    Consider the empirical risk

    .. math::
        \mathcal{L}(\mathbf{\theta})
        =
        c \sum_{n=1}^{N}
        \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)

    with :math:`c = \frac{1}{N}` for ``reduction='mean'`` and :math:`c=1` for
    ``reduction='sum'``. The GGN matrix is

    .. math::
        c \sum_{n=1}^{N}
        \left(
            \mathbf{J}_{\mathbf{\theta}}
            f_{\mathbf{\theta}}(\mathbf{x}_n)
        \right)^\top
        \left(
            \nabla_{f_\mathbf{\theta}(\mathbf{x}_n)}^2
            \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)
        \right)
        \left(
            \mathbf{J}_{\mathbf{\theta}}
            f_{\mathbf{\theta}}(\mathbf{x}_n)
        \right)\,.
    """

    def _matvec_batch(
        self, X: Tensor, y: Tensor, x_list: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """Apply the mini-batch GGN to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            x_list: Vector in list format (same shape as trainable model parameters).

        Returns:
            Result of GGN-multiplication in list format.
        """
        output = self._model_func(X)
        loss = self._loss_func(output, y)
        return ggn_vector_product_from_plist(loss, output, self._params, x_list)

    def _adjoint(self) -> GGNLinearOperator:
        """Return the linear operator representing the adjoint.

        The GGN is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self
