"""Contains LinearOperator implementation of gradient moment matrices."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Callable, Iterable, List, Optional, Tuple, Union

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from einops import einsum, rearrange
from torch import Tensor, zeros_like
from torch.autograd import grad
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, Parameter

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

    supported_losses = (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)

    def __init__(
        self,
        model_func: Callable[[Tensor], Tensor],
        loss_func: Union[Callable[[Tensor, Tensor], Tensor], None],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[MutableMapping], int]] = None,
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
                 BCEWithLogitsLoss, or ``CrossEntropyLoss``.
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
        self, X: Union[Tensor, MutableMapping], y: Tensor, M_list: List[Tensor]
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
        # If >2d output we convert to an equivalent 2d output
        if isinstance(self._loss_func, CrossEntropyLoss):
            output = rearrange(output, "batch c ... -> (batch ...) c")
            y = rearrange(y, "batch ... -> (batch ...)")
        else:
            output = rearrange(output, "batch ... c -> (batch ...) c")
            y = rearrange(y, "batch ... c -> (batch ...) c")

        # Adjust the scale depending on the loss reduction used
        num_loss_terms, C = output.shape
        reduction_factor = {
            "mean": (
                num_loss_terms
                if isinstance(self._loss_func, CrossEntropyLoss)
                else num_loss_terms * C
            ),
            "sum": 1.0,
        }[self._loss_func.reduction]

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
