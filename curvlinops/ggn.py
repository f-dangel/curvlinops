"""Contains LinearOperator implementation of the GGN."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import List, Tuple, Union

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from torch import Tensor, zeros_like

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

    def _matmat_batch(
        self, X: Union[Tensor, MutableMapping], y: Tensor, M_list: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """Apply the mini-batch GGN to a matrix.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M_list: Matrix to be multiplied with in list format.
                Tensors have same shape as trainable model parameters, and an
                additional leading axis for the matrix columns.

        Returns:
            Result of GGN multiplication in list format. Has the same shape as
            ``M_list``, i.e. each tensor in the list has the shape of a parameter and a
            leading dimension of matrix columns.
        """
        output = self._model_func(X)
        loss = self._loss_func(output, y)

        # collect matrix-matrix products per parameter
        result_list = [zeros_like(M) for M in M_list]

        num_vecs = M_list[0].shape[0]
        for n in range(num_vecs):
            col_n_list = ggn_vector_product_from_plist(
                loss, output, self._params, [M[n] for M in M_list]
            )
            for result, col_n in zip(result_list, col_n_list):
                result[n].add_(col_n)

        return tuple(result_list)

    def _adjoint(self) -> GGNLinearOperator:
        """Return the linear operator representing the adjoint.

        The GGN is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self
