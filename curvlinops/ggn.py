"""Contains LinearOperator implementation of the GGN."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import List, Union

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from torch import Tensor, zeros_like

from curvlinops._torch_base import CurvatureLinearOperator


class GGNLinearOperator(CurvatureLinearOperator):
    r"""Linear operator for the generalized Gauss-Newton matrix of an empirical risk.

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

    Attributes:
        SELF_ADJOINT: Whether the linear operator is self-adjoint. ``True`` for GGNs.
    """

    SELF_ADJOINT: bool = True

    def _matmat_batch(
        self, X: Union[Tensor, MutableMapping], y: Tensor, M: List[Tensor]
    ) -> List[Tensor]:
        """Apply the mini-batch GGN to a matrix.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M: Matrix to be multiplied with in tensor list format.
                Tensors have same shape as trainable model parameters, and an
                additional trailing axis for the matrix columns.

        Returns:
            Result of GGN multiplication in list format. Has the same shape as
            ``M_``, i.e. each tensor in the list has the shape of a parameter and a
            trailing dimension of matrix columns.
        """
        output = self._model_func(X)
        loss = self._loss_func(output, y)

        # collect matrix-matrix products per parameter
        (num_vecs,) = {m.shape[-1] for m in M}
        GM = [zeros_like(m) for m in M]

        for n in range(num_vecs):
            col_n = ggn_vector_product_from_plist(
                loss, output, self._params, [m[..., n] for m in M]
            )
            for GM_p, col_n_p in zip(GM, col_n):
                GM_p[..., n].add_(col_n_p)

        return GM
