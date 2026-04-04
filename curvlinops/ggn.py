"""Contains LinearOperator implementation of the GGN."""

from collections.abc import Callable, Iterable, MutableMapping

from einops import einsum
from torch import Generator, Tensor, no_grad
from torch.func import jacrev, jvp, vjp, vmap
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.ggn_utils import make_grad_output_fn
from curvlinops.kfac_utils import FisherType
from curvlinops.utils import _seed_generator, make_functional_loss


def make_ggn_vector_product(
    f: Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
    c: Callable[[Tensor, tuple], Tensor],
) -> Callable[
    [dict[str, Tensor], Tensor | MutableMapping, tuple, tuple[Tensor, ...]],
    tuple[Tensor, ...],
]:
    """Create a function that computes GGN-vector products for given f and c functions.

    Args:
        f: Function that takes parameters (as a dict) and input, returns prediction.
            Signature: ``(params_dict, X) -> prediction``
        c: Function that takes prediction and a tuple of loss arguments.
            Signature: ``(prediction, loss_args) -> loss``

    Returns:
        A function that computes GGN-vector products.
        Signature: ``(params_dict, X, loss_args, v) -> GGN @ v``
        where ``params_dict`` is a dict mapping parameter names to tensors,
        ``X`` is the model input, ``loss_args`` is a tuple of arguments
        passed to the loss function ``c`` (typically ``(y,)`` or
        ``(y, generator)``), and ``v`` is a tuple of tensors in list format.
    """

    @no_grad()
    def ggn_vector_product(
        params: dict[str, Tensor],
        X: Tensor | MutableMapping,
        loss_args: tuple,
        v: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Multiply the GGN on a vector in dict format.

        Args:
            params: Parameters of the model as a dict.
            X: Input to the model.
            loss_args: Arguments forwarded to the loss function ``c``, e.g.
                ``(y,)`` or ``(y, generator)``.
            v: Vector as a dict matching the structure of ``params``.

        Returns:
            Result of GGN multiplication as a dict with the same keys as ``params``.
        """
        # Apply the Jacobian of f onto v: v → Jv
        f_val, f_jvp = jvp(lambda p: f(p, X), (params,), (v,))

        # Apply the criterion's Hessian onto Jv: Jv → HJv
        c_grad_func = jacrev(lambda pred: c(pred, loss_args))
        _, c_hvp = jvp(c_grad_func, (f_val,), (f_jvp,))

        # Apply the transposed Jacobian of f onto HJv: HJv → JᵀHJv
        # NOTE This re-evaluates the net's forward pass. [Unverified] It should be op-
        # timized away by common sub-expression elimination if you compile the function.
        _, f_vjp_func = vjp(lambda p: f(p, X), params)
        (result_dict,) = f_vjp_func(c_hvp)
        return result_dict

    return ggn_vector_product


def make_batch_ggn_vector_product(
    f: Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
    loss_func: Module,
) -> Callable[
    [dict[str, Tensor], Tensor | MutableMapping, tuple, dict[str, Tensor]],
    dict[str, Tensor],
]:
    r"""Set up function that multiplies the mini-batch GGN onto a vector in dict format.

    Args:
        f: Functional model with signature ``(params_dict, X) -> prediction``.
        loss_func: The loss function :math:`\ell`.

    Returns:
        A function ``(params_dict, X, loss_args, v_dict) -> Gv`` that takes
        parameters as a dict, model input ``X``, loss arguments
        ``loss_args = (y,)``, and a vector ``v`` as a dict, and returns the
        mini-batch GGN applied to ``v`` as a dict.
    """
    c = make_functional_loss(loss_func)
    return make_ggn_vector_product(f, c)


def make_batch_ggn_mc_vector_product(
    f: Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
    loss_func: Module,
    mc_samples: int,
) -> Callable[
    [dict[str, Tensor], Tensor | MutableMapping, tuple, dict[str, Tensor]],
    dict[str, Tensor],
]:
    r"""Set up function that multiplies the mini-batch MC-approximated GGN onto a vector.

    The MC approximation replaces the exact loss Hessian with a Monte-Carlo estimate
    by sampling from the model's predictive distribution. For exponential family losses
    (MSE, CrossEntropy, BCEWithLogitsLoss), the MC estimate converges to the exact GGN.

    Internally constructs a pseudo-loss whose GGN equals the MC-approximated GGN,
    using sampled gradient output vectors from
    :func:`curvlinops.ggn_utils.make_grad_output_fn`.

    Args:
        f: Functional model with signature ``(params_dict, X) -> prediction``.
        loss_func: The loss function :math:`\ell`.
        mc_samples: Number of Monte-Carlo samples.

    Returns:
        A function ``(params_dict, X, loss_args, v_dict) -> Gv`` that takes
        parameters as a dict, model input ``X``, loss arguments
        ``loss_args = (y, generator)``, and a vector ``v`` as a dict, and returns
        the mini-batch MC-GGN applied to ``v`` as a dict.
    """
    _grad_output_fn = make_grad_output_fn(loss_func, FisherType.MC, mc_samples)
    # vmap over batch: per-datum grad outputs → batched
    batched_grad_output_fn = vmap(
        _grad_output_fn,
        in_dims=(0, 0, None),
        randomness="different",
    )

    def c_pseudo(prediction: Tensor, loss_args: tuple) -> Tensor:
        r"""Pseudo-loss whose GGN equals the MC-approximated GGN.

        Constructs :math:`L' = \frac{1}{2c} \sum_n \sum_k
        \langle \mathbf{g}'_{nk}, \mathbf{f}_n \rangle^2`
        where :math:`\mathbf{g}'_{nk}` are sampled gradient output vectors (scaled
        by :math:`1/\sqrt{M}`) and :math:`c` is the reduction factor.

        Args:
            prediction: Batched model predictions.
            loss_args: Tuple of ``(y, generator)`` with labels and random generator.

        Returns:
            Scalar pseudo-loss.
        """
        y, generator = loss_args

        # [batch, mc_samples, *output_shape], scaled by 1/sqrt(mc_samples)
        grad_outputs = batched_grad_output_fn(prediction.detach(), y, generator)

        # Inner products: [batch, mc_samples]
        ip = einsum(grad_outputs, prediction, "n k ..., n ... -> n k")

        batch_size = prediction.shape[0]
        reduction_factor = {"mean": batch_size, "sum": 1.0}[loss_func.reduction]

        return 0.5 / reduction_factor * (ip**2).sum()

    # Create GGN-vp of pseudo-loss: (params_dict, X, loss_args, v) -> Gv
    return make_ggn_vector_product(f, c_pseudo)


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

    Denoting :math:`\mathbf{f}_n = f_{\mathbf{\theta}}(\mathbf{x}_n)` and using a
    matrix square root :math:`\mathbf{S}_n \mathbf{S}_n^\top =
    \nabla_{\mathbf{f}_n}^2 \ell(\mathbf{f}_n, \mathbf{y}_n)`, this can be rewritten
    as

    .. math::
        c \sum_{n=1}^{N}
        \left(
            \mathbf{J}_{\mathbf{\theta}} \mathbf{f}_n
        \right)^\top
        \mathbf{S}_n \mathbf{S}_n^\top
        \left(
            \mathbf{J}_{\mathbf{\theta}} \mathbf{f}_n
        \right)\,.

    When ``mc_samples > 0``, the loss Hessian's square root is approximated via
    Monte-Carlo sampling. For exponential family losses (``MSELoss``,
    ``CrossEntropyLoss``, ``BCEWithLogitsLoss``), the loss Hessian equals
    :math:`\mathbb{E}_{\tilde{\mathbf{y}}_n \sim q(\cdot \mid \mathbf{f}_n)}
    [\nabla_{\mathbf{f}_n} \ell(\mathbf{f}_n, \tilde{\mathbf{y}}_n)
    \nabla_{\mathbf{f}_n} \ell(\mathbf{f}_n, \tilde{\mathbf{y}}_n)^\top]`,
    where :math:`q` is the model's predictive distribution. This expectation is
    approximated by drawing :math:`M` samples :math:`\tilde{\mathbf{y}}_n^{(m)}`
    and using the sampled gradients
    :math:`\mathbf{g}_{nm} = \nabla_{\mathbf{f}_n}
    \ell(\mathbf{f}_n, \tilde{\mathbf{y}}_n^{(m)})` as columns of
    :math:`\mathbf{S}_n`:

    .. math::
        \nabla_{\mathbf{f}_n}^2 \ell
        \approx
        \frac{1}{M} \sum_{m=1}^{M}
        \mathbf{g}_{nm} \mathbf{g}_{nm}^\top\,.

    The MC estimate converges to the exact GGN as :math:`M \to \infty`.

    Attributes:
        SELF_ADJOINT: Whether the linear operator is self-adjoint. ``True`` for GGNs.
        MC_SUPPORTED_LOSSES: Loss functions supported by the MC approximation.
    """

    SELF_ADJOINT: bool = True
    MC_SUPPORTED_LOSSES = (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)

    def __init__(
        self,
        model_func: Module
        | Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
        loss_func: Callable[[Tensor, Tensor], Tensor],
        params: dict[str, Tensor],
        data: Iterable[tuple[Tensor | MutableMapping, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_data: int | None = None,
        batch_size_fn: Callable[[MutableMapping], int] | None = None,
        mc_samples: int = 0,
        seed: int = 2147483647,
    ):
        r"""Linear operator for the GGN of an empirical risk.

        Note:
            f(X; θ) denotes a neural network, parameterized by θ, that maps a mini-batch
            input X to predictions p. ℓ(p, y) maps the prediction to a loss, using the
            mini-batch labels y.

        Args:
            model_func: The neural network's forward pass, defining the functional
                relationship ``(params, X) -> prediction``. Either an ``nn.Module``
                (architecture) or a callable ``(params_dict, X) -> prediction``.
            loss_func: Loss function criterion. Maps predictions and mini-batch labels
                to a scalar value.
            params: The parameter values at which the GGN is evaluated. A dictionary
                mapping parameter names to tensors.
            data: Source from which mini-batches can be drawn, for instance a list of
                mini-batches ``[(X, y), ...]`` or a torch ``DataLoader``. Note that ``X``
                could be a ``dict`` or ``UserDict``; this is useful for custom models.
                In this case, you must (i) specify the ``batch_size_fn`` argument, and
                (ii) take care of preprocessing like ``X.to(device)`` inside of your
                ``model.forward()`` function. When using MC sampling, batches must be
                presented in the same deterministic order (no shuffling!).
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
            mc_samples: Number of Monte-Carlo samples to approximate the loss Hessian.
                ``0`` (default) uses the exact GGN. Positive values activate the MC
                approximation, which is only supported for ``MSELoss``,
                ``CrossEntropyLoss``, and ``BCEWithLogitsLoss``.
            seed: Seed for the internal random number generator used for MC sampling.
                Only used when ``mc_samples > 0``. Default: ``2147483647``.

        Raises:
            NotImplementedError: If ``mc_samples > 0`` and the loss function is not
                in ``MC_SUPPORTED_LOSSES``.
        """
        self._mc_samples = mc_samples
        if mc_samples > 0:
            if not isinstance(loss_func, self.MC_SUPPORTED_LOSSES):
                raise NotImplementedError(
                    f"MC-GGN requires loss in {self.MC_SUPPORTED_LOSSES}. "
                    f"Got: {loss_func}."
                )
            self.FIXED_DATA_ORDER = True
            self._seed = seed
            self._generator: None | Generator = None
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

    def _matmat(self, M: list[Tensor]) -> list[Tensor]:
        """Multiply the GGN onto a matrix.

        Seeds the random number generator when using MC sampling.

        Args:
            M: Matrix for multiplication in tensor list format.

        Returns:
            Matrix-multiplication result ``mat @ M`` in tensor list format.
        """
        if self._mc_samples > 0:
            self._generator = _seed_generator(self._generator, self.device, self._seed)
        return super()._matmat(M)

    def _init_mp(self):
        """Set up the batch GGN-vector product function, then build vmap."""
        if self._mc_samples > 0:
            self._vp = make_batch_ggn_mc_vector_product(
                self._model_func, self._loss_func, self._mc_samples
            )
        else:
            self._vp = make_batch_ggn_vector_product(self._model_func, self._loss_func)
        super()._init_mp()

    def _matvec_batch(
        self, X: Tensor | MutableMapping, y: Tensor, v: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Apply the mini-batch GGN to a vector.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            v: Vector as a dict keyed by parameter names.

        Returns:
            Result of GGN-vector multiplication as a dict.
        """
        loss_args = (y, self._generator) if self._mc_samples > 0 else (y,)
        return self._vp(self._params, X, loss_args, v)
