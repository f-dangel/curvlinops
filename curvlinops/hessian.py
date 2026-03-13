"""Contains a linear operator implementation of the Hessian."""

from collections.abc import Callable, MutableMapping
from functools import cached_property, partial

from torch import Tensor, no_grad
from torch.func import jacrev, jvp
from torch.nn import Module

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.utils import make_functional_model_and_loss, split_list


def make_batch_hessian_vector_product(
    model_func: Module,
    loss_func: Module,
    param_names: list[str],
    block_sizes: list[int] | None = None,
) -> Callable[
    [dict[str, Tensor], Tensor | MutableMapping, tuple, tuple[Tensor, ...]],
    tuple[Tensor, ...],
]:
    r"""Set up function that multiplies the mini-batch Hessian onto a vector in list format.

    Args:
        model_func: The neural network :math:`f_{\mathbf{\theta}}`.
        loss_func: The loss function :math:`\ell`.
        param_names: Names of parameters w.r.t. which the Hessian is computed.
        block_sizes: Sizes of parameter blocks for block-diagonal approximation.
            If ``None``, the full Hessian is used.

    Returns:
        A function ``(params_dict, X, loss_args, v) -> Hv`` that takes parameters
        as a dict, model input ``X``, loss arguments ``loss_args = (y,)``, and a
        vector ``v`` as a tuple of tensors in list format, and returns the mini-batch
        Hessian applied to ``v`` in list format.
    """
    # Determine block structure
    block_sizes = [len(param_names)] if block_sizes is None else block_sizes

    # Split param names into blocks
    block_param_names = split_list(param_names, block_sizes)
    block_functionals = []

    for block_names in block_param_names:
        # criterion functional c is the same for all blocks
        f_block, c = make_functional_model_and_loss(model_func, loss_func, block_names)
        block_functionals.append(f_block)

    @no_grad()
    def hessian_vector_product(
        params: dict[str, Tensor],
        X: Tensor | MutableMapping,
        loss_args: tuple,
        v: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Multiply the mini-batch Hessian on a vector in list format.

        Args:
            params: Parameters of the model as a dict.
            X: Input to the model.
            loss_args: Arguments forwarded to the loss function, e.g. ``(y,)``.
            v: Vector to be multiplied with in tensor list format (tuple of tensors).

        Returns:
            Result of Hessian multiplication in list format. Has the same shape as
            ``v``, i.e. each tensor in the list has the shape of a parameter.
        """
        (y,) = loss_args

        # Split input vectors by blocks
        v_blocks = split_list(list(v), block_sizes)

        # Set up loss functions for each block
        block_grad_fns = []

        def loss_fn(
            f: Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
            block_params: dict[str, Tensor],
        ) -> Tensor:
            """Compute the mini-batch loss given the neural net and its parameters.

            Args:
                f: Functional model with signature (params_dict, X) -> prediction
                block_params: Parameters for the functional model as a dict.

            Returns:
                Mini-batch loss.
            """
            return c(f(block_params, X), (y,))

        for f_block in block_functionals:
            # Define the loss function composition for this block
            block_loss_fn = partial(loss_fn, f_block)
            block_grad_fn = jacrev(block_loss_fn)
            block_grad_fns.append(block_grad_fn)

        # Compute the HVPs per block and concatenate the results
        hvps = []
        for grad_fn, names, vs in zip(block_grad_fns, block_param_names, v_blocks):
            block_params = {n: params[n] for n in names}
            v_dict = dict(zip(names, vs))
            _, hvp_block = jvp(grad_fn, (block_params,), (v_dict,))
            hvps.extend(hvp_block.values())

        return tuple(hvps)

    return hessian_vector_product


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

    Example:
        >>> from torch import rand, eye, allclose, kron, manual_seed
        >>> from torch.nn import Linear, MSELoss
        >>> from curvlinops import HessianLinearOperator
        >>>
        >>> # Create a simple linear model without bias
        >>> _ = manual_seed(0) # make deterministic
        >>> D_in, D_out = 4, 2
        >>> num_data, num_batches = 10, 3
        >>> model = Linear(D_in, D_out, bias=False)
        >>> params = list(model.parameters())
        >>> loss_func = MSELoss(reduction='sum')
        >>>
        >>> # Generate synthetic dataset and chunk into batches
        >>> X, y = rand(num_data, D_in),  rand(num_data, D_out)
        >>> data = list(zip(X.split(num_batches), y.split(num_batches)))
        >>>
        >>> # Create Hessian linear operator
        >>> H_op = HessianLinearOperator(model, loss_func, params, data)
        >>>
        >>> # Compare with the known Hessian matrix 2 I ⊗ Xᵀ X
        >>> H_mat = 2 * kron(eye(D_out), X.T @ X)
        >>> P = sum(p.numel() for p in params)
        >>> v = rand(P) # generate a random vector
        >>> (H_mat @ v).allclose(H_op @ v)
        True

    Attributes:
        SUPPORTS_BLOCKS: Whether the linear operator supports block operations.
            Default is ``True``.
        SELF_ADJOINT: Whether the linear operator is self-adjoint (``True`` for
            Hessians).
    """

    SELF_ADJOINT: bool = True
    SUPPORTS_BLOCKS: bool = True

    @cached_property
    def _vp(
        self,
    ) -> Callable[
        [dict[str, Tensor], Tensor | MutableMapping, tuple, tuple[Tensor, ...]],
        tuple[Tensor, ...],
    ]:
        """Lazy initialization of batch-Hessian vector product function.

        Returns:
            Function that computes mini-batch Hessian-vector products with signature
            ``(params_dict, X, loss_args, v) -> Hv``.
        """
        return make_batch_hessian_vector_product(
            self._model_func,
            self._loss_func,
            list(self._params.keys()),
            self._block_sizes,
        )

    def _matvec_batch(
        self, X: Tensor | MutableMapping, y: Tensor, v: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        """Apply the mini-batch Hessian to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            v: Vector in tensor list format.

        Returns:
            Result of Hessian-vector multiplication in tensor list format.
        """
        return self._vp(self._params, X, (y,), v)
