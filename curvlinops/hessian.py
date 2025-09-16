"""Contains a linear operator implementation of the Hessian."""

from collections.abc import MutableMapping
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

from torch import Tensor, no_grad, vmap
from torch.func import jacrev, jvp
from torch.nn import Module, Parameter

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.utils import make_functional_call, split_list


def make_batch_hessian_matrix_product(
    model_func: Module,
    loss_func: Module,
    params: Tuple[Parameter, ...],
    block_sizes: Optional[List[int]] = None,
) -> Callable[[Tensor, Tensor, Tuple[Tensor, ...]], Tuple[Tensor, ...]]:
    r"""Set up function that multiplies the mini-batch Hessian onto a matrix in list format.

    Args:
        model_func: The neural network :math:`f_{\mathbf{\theta}}`.
        loss_func: The loss function :math:`\ell`.
        params: A tuple of parameters w.r.t. which the Hessian is computed.
            All parameters must be part of ``model_func.parameters()``.
        block_sizes: Sizes of parameter blocks for block-diagonal approximation.
            If ``None``, the full Hessian is used.

    Returns:
        A function that takes inputs ``X``, ``y``, and a matrix ``M`` in list
        format, and returns the mini-batch Hessian applied to ``M`` in list format.
    """
    # Determine block structure
    if block_sizes is None:
        block_sizes = [len(params)]

    # Create block-specific functional calls: *block_params, X -> prediction
    block_params = split_list(list(params), block_sizes)
    block_param_names = []
    block_functionals = []

    for block in block_params:
        # Get parameter names for this block
        param_names = []
        for p in block:
            (name,) = [n for n, pp in model_func.named_parameters() if pp is p]
            param_names.append(name)

        # Create functional version for this block
        f_block = make_functional_call(model_func, param_names)
        block_functionals.append(f_block)
        block_param_names.append(param_names)

    # Create loss functional (same for all blocks)
    c = make_functional_call(loss_func, [])

    @no_grad()
    def hessian_vector_product(
        X: Tensor, y: Tensor, *v: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]:
        """Multiply the mini-batch Hessian on a vector in list format.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            *v: Vector to be multiplied with in tensor list format.

        Returns:
            Result of Hessian multiplication in list format. Has the same shape as
            ``v``, i.e. each tensor in the list has the shape of a parameter.
        """
        # Split input vectors by blocks
        v_blocks = split_list(list(v), block_sizes)

        # Set up loss functions for each block
        block_grad_fns = []

        def loss_fn(f, *params):
            return c(f(*params, X), y)

        for f_block, ps in zip(block_functionals, block_params):
            # Define the loss function composition for this block
            block_loss_fn = partial(loss_fn, f_block)
            block_grad_fn = jacrev(block_loss_fn, argnums=tuple(range(len(ps))))
            block_grad_fns.append(block_grad_fn)

        # Compute the HVPs per block and concatenate the results
        hvps = []
        for grad_fn, ps, vs in zip(block_grad_fns, block_params, v_blocks):
            _, hvp_block = jvp(grad_fn, tuple(ps), tuple(vs))
            hvps.extend(hvp_block)

        return tuple(hvps)

    # Vectorize over vectors to multiply onto a matrix in list format
    return vmap(
        hessian_vector_product,
        # No vmap in X, y, assume last axis is vmapped in the matrix list
        in_dims=(None, None) + tuple(p.ndim for p in params),
        # Vmapped output axis is last
        out_dims=tuple(p.ndim for p in params),
        # We want each vector to be multiplied with the same mini-batch Hessian
        randomness="same",
    )


class HessianLinearOperator(CurvatureLinearOperator):
    r"""Linear operator for the Hessian of an empirical risk.

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
        SELF_ADJOINT: Whether the linear operator is self-adjoint (``True`` for
            Hessians).
    """

    SELF_ADJOINT: bool = True
    SUPPORTS_BLOCKS: bool = True

    def _matmat_batch(
        self, X: Union[Tensor, MutableMapping], y: Tensor, M: List[Tensor]
    ) -> List[Tensor]:
        """Apply the mini-batch Hessian to a matrix.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M: Matrix to be multiplied with in tensor list format.
                Tensors have same shape as trainable model parameters, and an
                additional trailing axis for the matrix columns.

        Returns:
            Result of Hessian multiplication in list format. Has the same shape as
            ``M``, i.e. each tensor in the list has the shape of a parameter and a
            trailing dimension of matrix columns.
        """
        if not hasattr(self, "_hmp"):
            self._hmp = make_batch_hessian_matrix_product(
                self._model_func, self._loss_func, self._params, self._block_sizes
            )

        return list(self._hmp(X, y, *M))
