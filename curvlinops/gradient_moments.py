"""Contains linear operator implementation of gradient moment matrices."""

from collections.abc import MutableMapping
from typing import Callable, Iterable, List, Optional, Tuple, Union

from einops import einsum, rearrange
from torch import Tensor, vmap
from torch.func import grad
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss, Parameter

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.ggn import make_ggn_vector_product
from curvlinops.utils import make_functional_call


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
    # detect the parameters w.r.t. which the empirical Fisher is computed
    free_param_names = []
    for p in params:
        (name,) = [n for n, pp in model_func.named_parameters() if pp is p]
        free_param_names.append(name)

    # Create functional versions of model and loss
    f = make_functional_call(model_func, free_param_names)  # *params, X -> prediction
    c = make_functional_call(loss_func, [])  # prediction, y -> loss

    # Handle networks which output an additional dimension other than batch and output
    # features by combining the additional axes with the batch axis.
    def f_flat(*params_and_X: Union[Tensor, MutableMapping]) -> Tensor:
        """Execute model and flatten batch and shared axes."""
        *params_inner, X = params_and_X
        output = f(*params_inner, X)
        return rearrange(
            output,
            "batch c ... -> (batch ...) c"
            if isinstance(loss_func, CrossEntropyLoss)
            else "batch ... c -> (batch ...) c",
        )

    def c_flat(output_flat: Tensor, y: Tensor) -> Tensor:
        """Execute loss with flattened labels."""
        y_flat = rearrange(
            y,
            "batch ... -> (batch ...)"
            if isinstance(loss_func, CrossEntropyLoss)
            else "batch ... c -> (batch ...) c",
        )
        return c(output_flat, y_flat)

    c_flat_grad = grad(c_flat, argnums=0)

    def c_pseudo_flat(output_flat: Tensor, y: Tensor) -> Tensor:
        """Compute pseudo-loss: L' = 0.5 / c * sum_n <f_n, g_n>^2.

        This pseudo-loss L' := 0.5 / c ∑ₙ fₙᵀ (gₙ gₙᵀ) fₙ where gₙ = ∂ℓₙ/∂fₙ
        (detached). The GGN of L' linearized at fₙ is the empirical Fisher.
        We can thus multiply with the EF by computing the GGN-vector products of L'.

        The reduction factor adjusts the scale depending on the loss reduction used.
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

    def ef_vector_product(
        X: Union[Tensor, MutableMapping], y: Tensor, *v: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]:
        """Multiply the mini-batch empirical Fisher on a vector in list format.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            *v: Vector to be multiplied with in tensor list format.

        Returns:
            Result of empirical Fisher multiplication in list format. Has the same
            shape as ``v``, i.e. each tensor in the list has the shape of a parameter.
        """
        return ef_vp(params, X, y, *v)

    # Vectorize over vectors to multiply onto a matrix in list format
    return vmap(
        ef_vector_product,
        # No vmap in X, y, assume last axis is vmapped in the matrix list
        in_dims=(None, None) + tuple(p.ndim for p in params),
        # Vmapped output axis is last
        out_dims=tuple(p.ndim for p in params),
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

    supported_losses = (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)
    SELF_ADJOINT: bool = True

    def __init__(
        self,
        model_func: Callable[[Union[MutableMapping, Tensor]], Tensor],
        loss_func: Union[Callable[[Tensor, Tensor], Tensor], None],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[Union[Tensor, MutableMapping]], int]] = None,
    ):
        """Linear operator for the uncentered gradient covariance/empirical Fisher (EF).

        Note:
            f(X; θ) denotes a neural network, parameterized by θ, that maps a mini-batch
            input X to predictions p. ℓ(p, y) maps the prediction to a loss, using the
            mini-batch labels y.

        Args:
            model_func: A function that maps the mini-batch input X to predictions.
                Could be a PyTorch module representing a neural network.
            loss_func: Loss function criterion. Maps predictions and mini-batch labels
                to a scalar value.
            params: List of differentiable parameters used by the prediction function.
            data: Source from which mini-batches can be drawn, for instance a list of
                mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``. Note that ``X``
                could be a ``dict`` or ``UserDict``; this is useful for custom models.
                In this case, you must (i) specify the ``batch_size_fn`` argument, and
                (ii) take care of preprocessing like ``X.to(device)`` inside of your
                ``model.forward()`` function. Due to the sequential internal Monte-Carlo
                sampling, batches must be presented in the same deterministic
                order (no shuffling!).
            progressbar: Show a progressbar during matrix-multiplication.
                Default: ``False``.
            check_deterministic: Probe that model and data are deterministic, i.e.
                that the data does not use `drop_last` or data augmentation. Also, the
                model's forward pass could depend on the order in which mini-batches
                are presented (BatchNorm, Dropout). Default: ``True``. This is a
                safeguard, only turn it off if you know what you are doing.
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified. The intended behavior is to consume the first
                entry of the iterates from ``data`` and return their batch size.

        Raises:
             NotImplementedError: If the loss function differs from ``MSELoss``,
                 ``BCEWithLogitsLoss``, or ``CrossEntropyLoss``.
        """
        if not isinstance(loss_func, self.supported_losses):
            raise NotImplementedError(
                f"Loss must be one of {self.supported_losses}. Got: {loss_func}."
            )
        super().__init__(
            model_func,
            loss_func,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            num_data=num_data,
            batch_size_fn=batch_size_fn,
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
        if not hasattr(self, "_efmp"):
            self._efmp = make_batch_ef_matrix_product(
                self._model_func, self._loss_func, tuple(self._params)
            )
        return list(self._efmp(X, y, *M))

        # OLD IMPLEMENTATION (kept as reference):
        # output = self._model_func(X)
        # # If >2d output we convert to an equivalent 2d output
        # if isinstance(self._loss_func, CrossEntropyLoss):
        #     output = rearrange(output, "batch c ... -> (batch ...) c")
        #     y = rearrange(y, "batch ... -> (batch ...)")
        # else:
        #     output = rearrange(output, "batch ... c -> (batch ...) c")
        #     y = rearrange(y, "batch ... c -> (batch ...) c")

        # # Adjust the scale depending on the loss reduction used
        # num_loss_terms, C = output.shape
        # reduction_factor = {
        #     "mean": (
        #         num_loss_terms
        #         if isinstance(self._loss_func, CrossEntropyLoss)
        #         else num_loss_terms * C
        #     ),
        #     "sum": 1.0,
        # }[self._loss_func.reduction]

        # # compute ∂ℓₙ/∂fₙ without reduction factor of L
        # (grad_output,) = grad(self._loss_func(output, y), output)
        # grad_output = grad_output.detach() * reduction_factor

        # # Compute the pseudo-loss L' := 0.5 / c ∑ₙ fₙᵀ (gₙ gₙᵀ) fₙ where gₙ = ∂ℓₙ/∂fₙ
        # # (detached). The GGN of L' linearized at fₙ is the empirical Fisher.
        # # We can thus multiply with the EF by computing the GGN-vector products of L'.
        # loss = (
        #     0.5
        #     / reduction_factor
        #     * (einsum(output, grad_output, "n ..., n ... -> n") ** 2).sum()
        # )

        # # Multiply the EF onto each vector in the input matrix
        # EM = [zeros_like(m) for m in M]
        # (num_vectors,) = {m.shape[-1] for m in M}
        # for v in range(num_vectors):
        #     for idx, ggnvp in enumerate(
        #         ggn_vector_product_from_plist(
        #             loss, output, self._params, [m[..., v] for m in M]
        #         )
        #     ):
        #         EM[idx][..., v].add_(ggnvp.detach())

        # return EM
