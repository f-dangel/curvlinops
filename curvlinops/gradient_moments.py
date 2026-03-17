"""Contains linear operator implementation of gradient moment matrices."""

from collections.abc import Callable, MutableMapping
from functools import cached_property

from einops import einsum
from torch import Tensor
from torch.func import grad
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.ggn import make_ggn_vector_product
from curvlinops.utils import make_functional_flattened_model_and_loss


def make_batch_ef_vector_product(
    model_func: Module, loss_func: Module
) -> Callable[
    [dict[str, Tensor], Tensor | MutableMapping, tuple, dict[str, Tensor]],
    dict[str, Tensor],
]:
    r"""Set up function that multiplies the mini-batch empirical Fisher onto a vector.

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

    Returns:
        A function ``(params_dict, X, loss_args, v_dict) -> EFv`` that takes
        parameters as a dict, model input ``X``, loss arguments
        ``loss_args = (y,)``, and a vector ``v`` as a dict, and returns the
        mini-batch empirical Fisher applied to ``v`` as a dict.
    """
    f_flat, c_flat = make_functional_flattened_model_and_loss(model_func, loss_func)
    # function that computes gradients of the loss w.r.t. the flattened outputs
    c_flat_grad = grad(c_flat, argnums=0)

    def c_pseudo_flat(output_flat: Tensor, loss_args: tuple) -> Tensor:
        """Compute pseudo-loss: L' = 0.5 / c * sum_n <f_n, g_n>^2.

        This pseudo-loss L' := 0.5 / c ∑ₙ fₙᵀ (gₙ gₙᵀ) fₙ where gₙ = ∂ℓₙ/∂fₙ
        (detached). The GGN of L' linearized at fₙ is the empirical Fisher.
        We can thus multiply with the EF by computing the GGN-vector products of L'.

        The reduction factor adjusts the scale depending on the loss reduction used.

        Args:
            output_flat: Flattened model outputs for the mini-batch.
            loss_args: Tuple of ``(y,)`` with un-flattened labels for the mini-batch.

        Returns:
            The pseudo-loss whose GGN is the empirical Fisher on the batch.
        """
        (y,) = loss_args

        # Compute ∂ℓₙ/∂fₙ without reduction factor of L (detached)
        grad_output_flat = c_flat_grad(output_flat.detach(), (y,))

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
    # (params_dict, X, loss_args, v) -> EFv
    return make_ggn_vector_product(f_flat, c_pseudo_flat)


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
    def _vp(
        self,
    ) -> Callable[
        [dict[str, Tensor], Tensor | MutableMapping, tuple, dict[str, Tensor]],
        dict[str, Tensor],
    ]:
        """Lazy initialization of the batch empirical Fisher vector product function.

        Returns:
            Function that computes mini-batch EF-vector products with signature
            ``(params_dict, X, loss_args, v_dict) -> EFv_dict``.

        Raises:
            NotImplementedError: If the loss function is not supported.
        """
        if not isinstance(self._loss_func, self.SUPPORTED_LOSSES):
            raise NotImplementedError(
                f"Loss must be one of {self.SUPPORTED_LOSSES}. Got: {self._loss_func}."
            )
        return make_batch_ef_vector_product(self._model_func, self._loss_func)

    def _matvec_batch(
        self, X: Tensor | MutableMapping, y: Tensor, v: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Apply the mini-batch empirical Fisher to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            v: Vector as a dict keyed by parameter names.

        Returns:
            Result of EF-vector multiplication as a dict.
        """
        return self._vp(self._params, X, (y,), v)
