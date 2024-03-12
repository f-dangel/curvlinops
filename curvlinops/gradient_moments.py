"""Contains LinearOperator implementation of gradient moment matrices."""

from __future__ import annotations

from typing import List, Tuple

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from einops import einsum
from torch import Tensor, zeros_like
from torch.autograd import grad

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
        """
        output = self._model_func(X)
        reduction_factor = {"mean": X.shape[0], "sum": 1.0}[self._loss_func.reduction]

        # compute ∂ℓₙ/∂fₙ without reduction factor of L
        (grad_output,) = grad(self._loss_func(output, y), output)
        grad_output = grad_output.detach() * reduction_factor

        # Compute the pseudo-loss L' := 0.5 / c ∑ₙ fₙᵀ (gₙ gₙᵀ) fₙ where gₙ = ∂ℓₙ/∂fₙ
        # (detached). The GGN of L' linearized at fₙ is the empirical Fisher.
        # We can thus multiply with the EF by computing the GGN-vector products of L'.
        loss = (
            0.5
            / reduction_factor
            * (einsum(output, grad_output, "n ..., n ... -> n") ** 2).sum()
        )

        # Multiply the EF onto each vector in the input matrix
        result_list = [zeros_like(M) for M in M_list]
        num_vectors = M_list[0].shape[0]
        for v in range(num_vectors):
            for idx, ggnvp in enumerate(
                ggn_vector_product_from_plist(
                    loss, output, self._params, [M[v] for M in M_list]
                )
            ):
                result_list[idx][v].add_(ggnvp.detach())

        return tuple(result_list)

    def _adjoint(self) -> EFLinearOperator:
        """Return the linear operator representing the adjoint.

        The empirical Fisher is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self
