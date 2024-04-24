"""Contains LinearOperator implementation of the Hessian."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import List, Tuple, Union

from backpack.hessianfree.hvp import hessian_vector_product
from torch import Tensor, zeros_like
from torch.autograd import grad

from curvlinops._base import _LinearOperator
from curvlinops.utils import split_list


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

    Attributes:
        SUPPORTS_BLOCKS: Whether the linear operator supports block operations.
            Default is ``True``.
    """

    SUPPORTS_BLOCKS: bool = True

    def _matmat_batch(
        self, X: Union[Tensor, MutableMapping], y: Tensor, M_list: List[Tensor]
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

        num_vecs = M_list[0].shape[0]
        result = [zeros_like(M) for M in M_list]

        # per-block HMP
        for M_block, p_block, g_block, res_block in zip(
            split_list(M_list, self._block_sizes),
            split_list(self._params, self._block_sizes),
            split_list(grad_params, self._block_sizes),
            split_list(result, self._block_sizes),
        ):
            for n in range(num_vecs):
                col_n = hessian_vector_product(
                    loss, p_block, [M[n] for M in M_block], grad_params=g_block
                )
                for p, col in enumerate(col_n):
                    res_block[p][n].add_(col)

        return tuple(result)

    def _adjoint(self) -> HessianLinearOperator:
        """Return the linear operator representing the adjoint.

        The Hessian is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self
