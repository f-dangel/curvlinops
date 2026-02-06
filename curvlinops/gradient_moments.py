"""Contains linear operator implementation of gradient moment matrices."""

from collections.abc import MutableMapping
from functools import cached_property, partial
from typing import Callable, List, Tuple, Union

from einops import einsum
from torch import Tensor, vmap
from torch.func import grad
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss, Parameter

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.ggn import make_ggn_vector_product
from curvlinops.utils import make_functional_flattened_model_and_loss


def make_batch_ef_matrix_product(
    model_func: Module, loss_func: Module, params: Tuple[Parameter, ...]
) -> Callable[
    [Union[Tensor, MutableMapping], Tensor, Tuple[Tensor, ...]], Tuple[Tensor, ...]
]:
    r"""Set up function that multiplies the mini-batch empirical Fisher onto a matrix.

    The empirical Fisher is computed as the GGN of a pseudo-loss that is quadratic
    in the gradients of the original loss. Specifically, for loss gradients
    :math:`g_n = \nabla_f \ell(f_n, y_n)`, the pseudo-loss is:

    .. math::
        L'(\mathbf{\theta}) = \frac{1}{2c} \sum_{n=1}^{N} \langle f_n, g_n \rangle^2

    where :math:`c` is the reduction factor and :math:`f_n = f_{\mathbf{\theta}}(x_n)`.
    The GGN of this pseudo-loss equals the empirical Fisher of the original loss.

    Args:
        model_func: The neural network :math:`f_{\mathbf{\theta}}`.
        loss_func: The loss function :math:`\ell`.
        params: A tuple of parameters w.r.t. which the empirical Fisher is computed.
            All parameters must be part of ``model_func.parameters()``.

    Returns:
        A function that takes inputs ``X``, ``y``, and a matrix ``M`` in list
        format, and returns the mini-batch empirical Fisher applied to ``M`` in
        list format.
    """
    f_flat, c_flat = make_functional_flattened_model_and_loss(
        model_func, loss_func, params
    )
    # function that computes gradients of the loss w.r.t. the flattened outputs
    c_flat_grad = grad(c_flat, argnums=0)

    def c_pseudo_flat(output_flat: Tensor, y: Tensor) -> Tensor:
        """Compute pseudo-loss: L' = 0.5 / c * sum_n <f_n, g_n>^2.

        This pseudo-loss L' := 0.5 / c ∑ₙ fₙᵀ (gₙ gₙᵀ) fₙ where gₙ = ∂ℓₙ/∂fₙ
        (detached). The GGN of L' linearized at fₙ is the empirical Fisher.
        We can thus multiply with the EF by computing the GGN-vector products of L'.

        The reduction factor adjusts the scale depending on the loss reduction used.

        Args:
            output_flat: Flattened model outputs for the mini-batch.
            y: Un-flattened labels for the mini-batch.

        Returns:
            The pseudo-loss whose GGN is the empirical Fisher on the batch.
        """
        # Compute ∂ℓₙ/∂fₙ without reduction factor of L (detached)
        grad_output_flat = c_flat_grad(output_flat.detach(), y)

        # Adjust the scale depending on the loss reduction used
        num_loss_terms, C = output_flat.shape
        reduction_factor = {
            "mean": (
                num_loss_terms
                if isinstance(loss_func, CrossEntropyLoss)
                else num_loss_terms * C
            ),
            "sum": 1.0,
        }[loss_func.reduction]

        # compute the pseudo-loss
        grad_output_flat = grad_output_flat * reduction_factor
        inner_products = einsum(output_flat, grad_output_flat, "n ..., n ... -> n")
        return 0.5 / reduction_factor * (inner_products**2).sum()

    # Create the functional EF-vector product using GGN of pseudo-loss
    ef_vp = make_ggn_vector_product(f_flat, c_pseudo_flat)

    # Freeze parameter values
    efvp = partial(ef_vp, params)  # X, y, *v -> *EFv

    # Parallelize over vectors to multiply onto a matrix in list format
    list_format_vmap_dims = tuple(p.ndim for p in params)  # last axis
    return vmap(
        efvp,
        # No vmap in X, y, assume last axis is vmapped in the matrix list
        in_dims=(None, None, *list_format_vmap_dims),
        # Vmapped output axis is last
        out_dims=list_format_vmap_dims,
        # We want each vector to be multiplied with the same mini-batch EF
        randomness="same",
    )


class EFLinearOperator(CurvatureLinearOperator):
    r"""Uncentered gradient covariance as PyTorch linear operator.

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

    Attributes:
        SELF_ADJOINT: Whether the linear operator is self-adjoint. ``True`` for
            empirical Fisher.
    """

    SUPPORTED_LOSSES = (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)
    SELF_ADJOINT: bool = True

    @cached_property
    def _mp(
        self,
    ) -> Callable[
        [Union[Tensor, MutableMapping], Tensor, Tuple[Tensor, ...]], Tuple[Tensor, ...]
    ]:
        """Lazy initialization of the batch empirical Fisher matrix product function.

        Returns:
            Function that computes mini-batch EF-vector products, given inputs ``X``,
            labels ``y``, and the entries ``v1, v2, ...`` of the vector in list format.
            Produces a list of tensors with the same shape as the input vector that re-
            presents the result of the batch-EF multiplication.

        Raises:
            NotImplementedError: If the loss function is not supported.
        """
        if not isinstance(self._loss_func, self.SUPPORTED_LOSSES):
            raise NotImplementedError(
                f"Loss must be one of {self.SUPPORTED_LOSSES}. Got: {self._loss_func}."
            )
        return make_batch_ef_matrix_product(
            self._model_func, self._loss_func, tuple(self._params)
        )

    def _matmat_batch(
        self, X: Union[Tensor, MutableMapping], y: Tensor, M: List[Tensor]
    ) -> List[Tensor]:
        """Apply the mini-batch empirical Fisher to a matrix in tensor list format.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M: Matrix to be multiplied with in tensor list format.
                Tensors have same shape as trainable model parameters, and an
                additional trailing axis for the matrix columns.

        Returns:
            Result of EF multiplication in tensor list format. Has the same shape as
            ``M``, i.e. each tensor in the list has the shape of a parameter and a
            trailing dimension of matrix columns.
        """
        return list(self._mp(X, y, *M))
