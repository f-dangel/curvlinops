"""Contains LinearOperator implementation of the GGN."""

from collections.abc import Callable, Iterable, MutableMapping
from functools import cached_property, partial

from einops import einsum
from torch import Generator, Tensor, no_grad
from torch.func import jacrev, jvp, vjp, vmap
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss, Parameter

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.ggn_utils import (
    _check_binary_if_BCEWithLogitsLoss,
    make_grad_output_fn,
)
from curvlinops.utils import _seed_generator, make_functional_model_and_loss


def make_ggn_vector_product(
    f: Callable[[tuple[Tensor, ...], Tensor | MutableMapping], Tensor],
    c: Callable[[Tensor, tuple], Tensor],
) -> Callable[
    [tuple[Tensor, ...], Tensor | MutableMapping, tuple, tuple[Tensor, ...]],
    tuple[Tensor, ...],
]:
    """Create a function that computes GGN-vector products for given f and c functions.

    Args:
        f: Function that takes parameters (as a tuple) and input, returns prediction.
            Signature: ``(params, X) -> prediction``
        c: Function that takes prediction and a tuple of loss arguments.
            Signature: ``(prediction, loss_args) -> loss``

    Returns:
        A function that computes GGN-vector products.
        Signature: ``(params, X, loss_args, v) -> GGN @ v``
        where ``X`` is the model input, ``loss_args`` is a tuple of arguments
        passed to the loss function ``c`` (typically ``(y,)`` or
        ``(y, generator)``), and ``v`` is a tuple of tensors in list format.
    """

    @no_grad()
    def ggn_vector_product(
        params: tuple[Tensor, ...],
        X: Tensor | MutableMapping,
        loss_args: tuple,
        v: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Multiply the GGN on a vector in list format.

        Args:
            params: Parameters of the model as a tuple.
            X: Input to the model.
            loss_args: Arguments forwarded to the loss function ``c``, e.g.
                ``(y,)`` or ``(y, generator)``.
            v: Vector to be multiplied with in tensor list format (tuple of tensors).

        Returns:
            Result of GGN multiplication in list format. Has the same shape as ``v``.
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
        (result,) = f_vjp_func(c_hvp)
        return result

    return ggn_vector_product


def make_batch_ggn_matrix_product(
    model_func: Module, loss_func: Module, params: tuple[Parameter, ...]
) -> Callable[[Tensor | MutableMapping, tuple, tuple[Tensor, ...]], tuple[Tensor, ...]]:
    r"""Set up function that multiplies the mini-batch GGN onto a matrix in list format.

    Args:
        model_func: The neural network :math:`f_{\mathbf{\theta}}`.
        loss_func: The loss function :math:`\ell`.
        params: A tuple of parameters w.r.t. which the GGN is computed.
            All parameters must be part of ``model_func.parameters()``.

    Returns:
        A function ``(X, loss_args, M) -> GM`` that takes model input ``X``, loss
        arguments ``loss_args = (y,)``, and a matrix ``M`` as a tuple of tensors in
        list format, and returns the mini-batch GGN applied to ``M`` in list format.
    """
    # Create functional versions of the model (f: (params, X) -> prediction) and
    # criterion function (c: (prediction, loss_args) -> loss)
    f, c = make_functional_model_and_loss(model_func, loss_func, params)

    # Create the functional GGN-vector product: (params, X, loss_args, v) -> Gv
    ggn_vp = make_ggn_vector_product(f, c)

    # Fix the parameters: (X, loss_args, v) -> Gv
    ggnvp = partial(ggn_vp, params)

    # Parallelize over vectors to multiply onto a matrix in list format
    return vmap(
        ggnvp,
        # No vmap in X, loss_args; last-axis vmap over the vector tuple
        in_dims=(None, None, -1),
        # Vmapped output axis is last
        out_dims=-1,
        # We want each vector to be multiplied with the same mini-batch GGN
        randomness="same",
    )


def make_batch_ggn_mc_matrix_product(
    model_func: Module,
    loss_func: Module,
    params: tuple[Parameter, ...],
    mc_samples: int,
) -> Callable[
    [Tensor | MutableMapping, Tensor, Generator, tuple[Tensor, ...]],
    tuple[Tensor, ...],
]:
    r"""Set up function that multiplies the mini-batch MC-approximated GGN onto a matrix.

    The MC approximation replaces the exact loss Hessian with a Monte-Carlo estimate
    by sampling from the model's predictive distribution. For exponential family losses
    (MSE, CrossEntropy, BCEWithLogitsLoss), the MC estimate converges to the exact GGN.

    Internally constructs a pseudo-loss whose GGN equals the MC-approximated GGN,
    using sampled gradient output vectors from
    :func:`curvlinops.ggn_utils.make_grad_output_fn`.

    Args:
        model_func: The neural network :math:`f_{\mathbf{\theta}}`.
        loss_func: The loss function :math:`\ell`.
        params: A tuple of parameters w.r.t. which the GGN is computed.
            All parameters must be part of ``model_func.parameters()``.
        mc_samples: Number of Monte-Carlo samples.

    Returns:
        A function ``(X, y, generator, M) -> GM`` that takes model input ``X``,
        labels ``y``, a random ``generator``, and a matrix ``M`` as a tuple of
        tensors in list format, and returns the mini-batch MC-GGN applied to ``M``.
    """
    f, _ = make_functional_model_and_loss(model_func, loss_func, params)

    _grad_output_fn = make_grad_output_fn(loss_func, "mc", mc_samples)
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

    # Create GGN-vp of pseudo-loss: (params, X, loss_args, v) -> Gv
    mc_ggn_vp = make_ggn_vector_product(f, c_pseudo)
    mc_ggnvp = partial(mc_ggn_vp, params)  # (X, loss_args, v) -> Gv

    # Parallelize over matrix columns
    mc_ggnmp = vmap(
        mc_ggnvp,
        # No vmap in X, loss_args; last-axis vmap over the matrix tuple
        in_dims=(None, None, -1),
        out_dims=-1,
        randomness="same",
    )

    def _mc_ggnmp_with_check(
        X: Tensor | MutableMapping,
        y: Tensor,
        generator: Generator,
        M: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Multiply MC-GGN onto a matrix, with BCEWithLogitsLoss target validation.

        Args:
            X: Input to the model.
            y: Target labels for the batch.
            generator: Random number generator for sampling.
            M: Matrix in tensor list format (tuple of tensors).

        Returns:
            Result of MC-GGN matrix multiplication in tensor list format.
        """
        _check_binary_if_BCEWithLogitsLoss(y, loss_func)
        return mc_ggnmp(X, (y, generator), M)

    return _mc_ggnmp_with_check


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
        model_func: Callable[[Tensor | MutableMapping], Tensor],
        loss_func: Callable[[Tensor, Tensor], Tensor],
        params: list[Parameter],
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

    @cached_property
    def _mp(self) -> Callable:
        """Lazy initialization of batch-GGN matrix product function.

        Returns:
            Function that computes mini-batch GGN-matrix products. In exact mode,
            takes model input ``X``, loss args ``(y,)``, and a matrix ``M`` as a
            tuple of tensors in list format. In MC mode, takes
            ``(X, y, generator, M)`` (the loss args packing is handled
            internally); the number of MC samples is fixed when this function is
            constructed.
        """
        if self._mc_samples > 0:
            return make_batch_ggn_mc_matrix_product(
                self._model_func,
                self._loss_func,
                tuple(self._params),
                self._mc_samples,
            )
        return make_batch_ggn_matrix_product(
            self._model_func, self._loss_func, tuple(self._params)
        )

    def _matmat_batch(
        self, X: Tensor | MutableMapping, y: Tensor, M: list[Tensor]
    ) -> list[Tensor]:
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
        if self._mc_samples > 0:
            return list(self._mp(X, y, self._generator, tuple(M)))
        return list(self._mp(X, (y,), tuple(M)))
