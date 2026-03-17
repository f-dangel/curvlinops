"""Contains a linear operator implementation of the Hessian."""

from collections.abc import Callable, MutableMapping
from functools import cached_property, partial

from torch import Tensor, no_grad
from torch.func import jacrev, jvp
from torch.nn import Module

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.utils import make_functional_call, make_functional_loss, split_list


def make_batch_hessian_vector_product(
    model_func: Module,
    loss_func: Module,
    param_names: list[str],
    block_sizes: list[int] | None = None,
) -> Callable[
    [dict[str, Tensor], Tensor | MutableMapping, tuple, dict[str, Tensor]],
    dict[str, Tensor],
]:
    r"""Set up function that multiplies the mini-batch Hessian onto a vector in dict format.

    Args:
        model_func: The neural network :math:`f_{\mathbf{\theta}}`.
        loss_func: The loss function :math:`\ell`.
        param_names: Names of parameters w.r.t. which the Hessian is computed.
        block_sizes: Sizes of parameter blocks for block-diagonal approximation.
            If ``None``, the full Hessian is used.

    Returns:
        A function ``(params_dict, X, loss_args, v_dict) -> Hv`` that takes
        parameters as a dict, model input ``X``, loss arguments
        ``loss_args = (y,)``, and a vector ``v`` as a dict, and returns the
        mini-batch Hessian applied to ``v`` as a dict.
    """
    # Determine block structure
    block_sizes = [len(param_names)] if block_sizes is None else block_sizes

    # Split param names into blocks
    block_param_names = split_list(param_names, block_sizes)
    f = make_functional_call(model_func)
    c = make_functional_loss(loss_func)

    @no_grad()
    def hessian_vector_product(
        params: dict[str, Tensor],
        X: Tensor | MutableMapping,
        loss_args: tuple,
        v: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Multiply the mini-batch Hessian on a vector in dict format.

        Args:
            params: Parameters of the model as a dict.
            X: Input to the model.
            loss_args: Arguments forwarded to the loss function, e.g. ``(y,)``.
            v: Vector as a dict matching the structure of ``params``.

        Returns:
            Result of Hessian multiplication as a dict with the same keys as
            ``params``.
        """
        (y,) = loss_args

        # Split input vectors by blocks
        v_blocks = [[v[n] for n in names] for names in block_param_names]

        def loss_fn(
            block_params: dict[str, Tensor], frozen: dict[str, Tensor]
        ) -> Tensor:
            """Compute the mini-batch loss with only block params free.

            Args:
                block_params: Free parameters for this block.
                frozen: Detached non-block parameters.

            Returns:
                Mini-batch loss.
            """
            return c(f({**frozen, **block_params}, X), (y,))

        # Compute the HVPs per block and concatenate the results
        hvps = {}
        for names, vs in zip(block_param_names, v_blocks):
            names_set = set(names)
            frozen = {n: params[n].detach() for n in params if n not in names_set}
            block_loss_fn = partial(loss_fn, frozen=frozen)
            block_params = {n: params[n] for n in names}
            v_block_dict = dict(zip(names, vs))
            _, hvp_block = jvp(jacrev(block_loss_fn), (block_params,), (v_block_dict,))
            hvps.update(hvp_block)

        return hvps

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
        [dict[str, Tensor], Tensor | MutableMapping, tuple, dict[str, Tensor]],
        dict[str, Tensor],
    ]:
        """Lazy initialization of batch-Hessian vector product function.

        Returns:
            Function that computes mini-batch Hessian-vector products with signature
            ``(params_dict, X, loss_args, v_dict) -> Hv_dict``.
        """
        return make_batch_hessian_vector_product(
            self._model_func,
            self._loss_func,
            list(self._params.keys()),
            self._block_sizes,
        )

    def _matvec_batch(
        self, X: Tensor | MutableMapping, y: Tensor, v: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Apply the mini-batch Hessian to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            v: Vector as a dict keyed by parameter names.

        Returns:
            Result of Hessian-vector multiplication as a dict.
        """
        return self._vp(self._params, X, (y,), v)
