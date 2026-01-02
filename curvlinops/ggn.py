"""Contains LinearOperator implementation of the GGN."""

from collections.abc import MutableMapping
from functools import cached_property, partial
from typing import Callable, List, Tuple, Union

from torch import Tensor, no_grad, vmap
from torch.func import jacrev, jvp, vjp
from torch.nn import Module, Parameter

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.utils import make_functional_model_and_loss


def make_ggn_vector_product(
    f: Callable[..., Tensor], c: Callable[..., Tensor], num_c_extra_args: int = 0
) -> Callable[..., Tuple[Tensor, ...]]:
    """Create a function that computes GGN-vector products for given f and c functions.

    Args:
        f: Function that takes parameters and input, returns prediction.
            Signature: (*params, X) -> prediction
        c: Function that takes prediction, target, and optional additional args.
            Signature: (prediction, y, *args) -> loss
        num_c_extra_args: Number of additional arguments that the loss function c expects
            beyond prediction and target. Used to correctly split the input arguments
            between the vector to multiply and the additional loss function arguments.

    Returns:
        A function that computes GGN-vector products.
        Signature: (params, X, y, *c_args, *v) -> GGN @ v
        where c_args are additional arguments passed to the loss function c.
    """

    @no_grad()
    def ggn_vector_product(
        params: Tuple[Tensor, ...],
        X: Tensor,
        y: Tensor,
        *args_and_v: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, ...]:
        """Multiply the GGN on a vector in list format.

        Args:
            params: Parameters of the model.
            X: Input to the DNN.
            y: Ground truth.
            *args_and_v: Additional arguments for the loss function c,
                followed by vector to be multiplied with in tensor list format.

        Returns:
            Result of GGN multiplication in list format. Has the same shape as
            the vector part of args_and_v.
        """
        # Split args_and_v into additional loss function arguments and vector v
        c_args, v = args_and_v[:num_c_extra_args], args_and_v[num_c_extra_args:]

        # Apply the Jacobian of f onto v: v → Jv
        f_val, f_jvp = jvp(lambda *params_inner: f(*params_inner, X), params, v)

        # Apply the criterion's Hessian onto Jv: Jv → HJv
        c_grad_func = jacrev(lambda pred: c(pred, y, *c_args))
        _, c_hvp = jvp(c_grad_func, (f_val,), (f_jvp,))

        # Apply the transposed Jacobian of f onto HJv: HJv → JᵀHJv
        # NOTE This re-evaluates the net's forward pass. [Unverified] It should be op-
        # timized away by common sub-expression elimination if you compile the function.
        _, f_vjp_func = vjp(lambda *params_inner: f(*params_inner, X), *params)
        return f_vjp_func(c_hvp)

    return ggn_vector_product


def make_batch_ggn_matrix_product(
    model_func: Module, loss_func: Module, params: Tuple[Parameter, ...]
) -> Callable[
    [Union[Tensor, MutableMapping], Tensor, Tuple[Tensor, ...]], Tuple[Tensor, ...]
]:
    r"""Set up function that multiplies the mini-batch GGN onto a matrix in list format.

    Args:
        model_func: The neural network :math:`f_{\mathbf{\theta}}`.
        loss_func: The loss function :math:`\ell`.
        params: A tuple of parameters w.r.t. which the GGN is computed.
            All parameters must be part of ``model_func.parameters()``.

    Returns:
        A function that takes inputs ``X``, ``y``, and a matrix ``M`` in list
        format, and returns the mini-batch GGN applied to ``M`` in list format.
    """
    # Create functional versions of the model (f: *params, X -> prediction) and
    # criterion function (c: prediction, y -> loss)
    f, c = make_functional_model_and_loss(model_func, loss_func, params)

    # Create the functional GGN-vector product
    ggn_vp = make_ggn_vector_product(f, c)  # params, X, y, *v -> *Gv

    # Fix the parameters
    ggnvp = partial(ggn_vp, params)  # X, y, *c_args, *v -> *Gv

    # Parallelize over vectors to multiply onto a matrix in list format
    list_format_vmap_dims = tuple(p.ndim for p in params)  # last axis
    return vmap(
        ggnvp,
        # No vmap in X, y, last-axis vmap over vector in list format
        in_dims=(None, None, *list_format_vmap_dims),
        # Vmapped output axis is last
        out_dims=list_format_vmap_dims,
        # We want each vector to be multiplied with the same mini-batch GGN
        randomness="same",
    )


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

    @cached_property
    def _mp(
        self,
    ) -> Callable[
        [Union[Tensor, MutableMapping], Tensor, Tuple[Tensor, ...]], Tuple[Tensor, ...]
    ]:
        """Lazy initialization of batch-GGN matrix product function.

        Returns:
            Function that computes mini-batch GGN-vector products, given inputs ``X``,
            labels ``y``, and the entries ``v1, v2, ...`` of the vector in list format.
            Produces a list of tensors with the same shape as the input vector that re-
            presents the result of the batch-GGN multiplication.
        """
        return make_batch_ggn_matrix_product(
            self._model_func, self._loss_func, tuple(self._params)
        )

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
            ``M``, i.e. each tensor in the list has the shape of a parameter and a
            trailing dimension of matrix columns.
        """
        return list(self._mp(X, y, *M))
