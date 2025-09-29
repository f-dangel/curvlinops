"""Contains LinearOperator implementation of the GGN."""

from collections.abc import MutableMapping
from functools import cached_property
from typing import Callable, List, Tuple, Union

from torch import Tensor, no_grad, vmap
from torch.func import jacrev, jvp, vjp
from torch.nn import Module, Parameter

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.utils import make_functional_call


def make_ggn_vector_product(
    f: Callable[..., Tensor], c: Callable[[Tensor, Tensor], Tensor]
) -> Callable[
    [Tuple[Tensor, ...], Tensor, Tensor, Tuple[Tensor, ...]], Tuple[Tensor, ...]
]:
    """Create a function that computes GGN-vector products for given f and c functions.

    Args:
        f: Function that takes parameters and input, returns prediction.
            Signature: (*params, X) -> prediction
        c: Function that takes prediction and target, returns loss.
            Signature: (prediction, y) -> loss

    Returns:
        A function that computes GGN-vector products.
        Signature: (params, X, y, *v) -> GGN @ v
    """

    @no_grad()
    def ggn_vector_product(
        params: Tuple[Tensor, ...], X: Tensor, y: Tensor, *v: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]:
        """Multiply the GGN on a vector in list format.

        Args:
            params: Parameters of the model.
            X: Input to the DNN.
            y: Ground truth.
            *v: Vector to be multiplied with in tensor list format.

        Returns:
            Result of GGN multiplication in list format. Has the same shape as
            ``v``, i.e. each tensor in the list has the shape of a parameter.
        """
        # Apply the Jacobian of f onto v: v → Jv
        f_val, f_jvp = jvp(lambda *params_inner: f(*params_inner, X), params, v)

        # Apply the criterion's Hessian onto Jv: Jv → HJv
        c_grad_func = jacrev(lambda pred: c(pred, y))
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
    # detect the parameters w.r.t. which the GGN is computed
    free_param_names = []
    for p in params:
        (name,) = [n for n, pp in model_func.named_parameters() if pp is p]
        free_param_names.append(name)

    # Create functional versions of model and loss
    f = make_functional_call(model_func, free_param_names)  # *params, X -> prediction
    c = make_functional_call(loss_func, [])  # prediction, y -> loss

    # Create the functional GGN-vector product
    ggn_vp = make_ggn_vector_product(f, c)

    def ggn_vector_product(
        X: Union[Tensor, MutableMapping], y: Tensor, *v: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]:
        """Multiply the mini-batch GGN on a vector in list format.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            *v: Vector to be multiplied with in tensor list format.

        Returns:
            Result of GGN multiplication in list format. Has the same shape as
            ``v``, i.e. each tensor in the list has the shape of a parameter.
        """
        return ggn_vp(params, X, y, *v)

    # Vectorize over vectors to multiply onto a matrix in list format
    return vmap(
        ggn_vector_product,
        # No vmap in X, y, assume last axis is vmapped in the matrix list
        in_dims=(None, None) + tuple(p.ndim for p in params),
        # Vmapped output axis is last
        out_dims=tuple(p.ndim for p in params),
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
    def _ggnmp(self):
        """Lazy initialization of GGN matrix product function."""
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
        return list(self._ggnmp(X, y, *M))
