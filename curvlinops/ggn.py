"""Contains LinearOperator implementation of the GGN."""

from typing import List, Tuple

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from torch import Tensor

from curvlinops._base import _LinearOperator


class GGNLinearOperator(_LinearOperator):
    """GGN as SciPy linear operator."""

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
