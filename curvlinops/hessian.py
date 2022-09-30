"""Contains LinearOperator implementation of the Hessian."""

from typing import List, Tuple

from backpack.hessianfree.hvp import hessian_vector_product
from torch import Tensor

from curvlinops._base import _LinearOperator


class HessianLinearOperator(_LinearOperator):
    """Hessian as SciPy linear operator."""

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
