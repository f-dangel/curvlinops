"""Contains LinearOperator implementation of the Hessian."""

from __future__ import annotations

from typing import List, Tuple

from backpack.hessianfree.hvp import hessian_vector_product
from torch import Tensor, zeros_like
from torch.autograd import grad

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

    def _matmat_batch(
        self, X: Tensor, y: Tensor, M_list: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """Apply the mini-batch Hessian to a matrix.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M_list: Matrix to be multiplied with in list format.
                Tensors have same shape as trainable model parameters, and an
                additional leading axis for the matrix columns.

        Returns:
            Result of Hessian multiplication in list format. Has the same shape as
            ``M_list``, i.e. each tensor in the list has the shape of a parameter and a
            leading dimension of matrix columns.
        """
        loss = self._loss_func(self._model_func(X), y)

        # Re-cycle first backward pass from the HVP's double-backward
        grad_params = grad(loss, self._params, create_graph=True)

        result_list = [zeros_like(M) for M in M_list]

        num_vecs = M_list[0].shape[0]
        for n in range(num_vecs):
            col_n_list = hessian_vector_product(
                loss, self._params, [M[n] for M in M_list], grad_params=grad_params
            )
            for result, col_n in zip(result_list, col_n_list):
                result[n].add_(col_n)

        return result_list

    def _adjoint(self) -> HessianLinearOperator:
        """Return the linear operator representing the adjoint.

        The Hessian is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self
