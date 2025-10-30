"""Contains LinearOperator implementation of the approximate Fisher."""

from collections.abc import MutableMapping
from functools import cached_property, partial
from math import sqrt
from typing import Callable, Iterable, List, Optional, Tuple, Union

from einops import einsum, rearrange
from torch import Generator, Tensor, as_tensor, normal, softmax, vmap, zeros
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss, Parameter
from torch.nn.functional import one_hot

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.ggn import make_ggn_vector_product
from curvlinops.utils import make_functional_flattened_model_and_loss


def make_grad_output_sampler(
    loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
) -> Callable[[Tensor, int, Tensor, Generator], Tensor]:
    """Create a function that samples gradients w.r.t. network outputs.

    Args:
        loss_func: The loss function to create the sampler for.

    Returns:
        A function that samples gradients w.r.t. the model prediction.
        Signature: (output, num_samples, y, generator) -> grad_samples
    """

    def sample_grad_output(
        output: Tensor, num_samples: int, y: Tensor, generator: Generator
    ) -> Tensor:
        """Draw would-be gradients ``∇_f log p(·|f)`` with explicit generator.

        For a single data point, the would-be gradient's outer product equals the
        Hessian ``∇²_f log p(·|f)`` in expectation.

        Currently only supports ``MSELoss``, ``CrossEntropyLoss``, and
        ``BCEWithLogitsLoss``.

        The returned gradient does not account for the scaling of the loss function by
        the output dimension ``C`` that ``MSELoss`` and ``BCEWithLogitsLoss`` apply when
        ``reduction='mean'``.

        Args:
            output: model prediction ``f`` for multiple data with batch axis as
                0th dimension.
            num_samples: Number of samples to draw.
            y: Labels of the data on which output was produced.
            generator: Random generator for sampling.

        Returns:
            Samples of the gradient w.r.t. the model prediction.
            Has shape ``[num_samples, *output.shape]``.

        Raises:
            NotImplementedError: For unsupported loss functions.
            NotImplementedError: If the prediction does not have two dimensions.
            NotImplementedError: If binary classification labels are not binary.
        """
        if output.ndim != 2:
            raise NotImplementedError(f"Only 2d outputs supported. Got {output.shape}")

        _, C = output.shape

        if isinstance(loss_func, MSELoss):
            std = as_tensor(sqrt(0.5), device=output.device)
            mean = zeros(
                num_samples, *output.shape, device=output.device, dtype=output.dtype
            )
            return 2 * normal(mean, std, generator=generator)

        elif isinstance(loss_func, CrossEntropyLoss):
            prob = softmax(output, dim=1)
            sample = prob.multinomial(
                num_samples=num_samples, replacement=True, generator=generator
            )
            sample = rearrange(sample, "batch s -> s batch")
            onehot_sample = one_hot(sample, num_classes=C)
            # repeat ``num_sample`` times along a new leading axis to avoid broadcasting
            prob = prob.unsqueeze(0).expand_as(onehot_sample)
            return prob - onehot_sample

        elif isinstance(loss_func, BCEWithLogitsLoss):
            # Check if targets are binary by ensuring all values are 0 or 1
            is_binary = (y == 0).logical_or(y == 1).all()
            if not is_binary:
                raise NotImplementedError(
                    "Only binary targets (0, 1) are currently supported with"
                    + f" BCEWithLogitsLoss. Got non-binary values {y.unique()}."
                )
            prob = output.sigmoid()
            # repeat ``num_sample`` times along a new leading axis
            prob = prob.unsqueeze(0).expand(num_samples, -1, -1)
            sample = prob.bernoulli(generator=generator)
            return prob - sample

        else:
            raise NotImplementedError(
                f"Supported losses: {(MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)}"
            )

    return sample_grad_output


def make_batch_fmc_matrix_product(
    model_func: Module, loss_func: Module, params: Tuple[Parameter, ...]
) -> Callable[
    [Union[Tensor, MutableMapping], Tensor, Tuple[Tensor, ...], int, Generator],
    Tuple[Tensor, ...],
]:
    r"""Set up function that multiplies the mini-batch MC Fisher onto a matrix.

    The MC Fisher approximates the Fisher information matrix by sampling from the
    model's predictive distribution and computing the expected outer product of
    gradients w.r.t. the sampled targets.

    The implementation works by:
    1. Computing model predictions :math:`f_n = f_{\mathbf{\theta}}(\mathbf{x}_n)`
    2. Flattening outputs to 2D format for loss computation:
       - CrossEntropyLoss: ``(batch, c, ...) -> (batch*..., c)``
       - Other losses: ``(batch, ..., c) -> (batch*..., c)``
    3. Sampling gradients :math:`g_{nm} = \nabla_{f_n} \ell(f_n, \tilde{y}_{nm})`
       where :math:`\tilde{y}_{nm} \sim q(\cdot | f_n)` are sampled targets
    4. Computing a pseudo-loss :math:`L' = \frac{1}{2Mc} \sum_n \sum_m \langle f_n, g_{nm} \rangle^2`
       where :math:`M` is the number of MC samples and :math:`c` is the reduction factor
    5. Using the GGN of this pseudo-loss to approximate the MC Fisher matrix-vector products

    The reduction factor :math:`c` adjusts for the loss function's reduction:
    - ``'mean'``: :math:`c = N` (CrossEntropyLoss) or :math:`c = N \times C` (others)
    - ``'sum'``: :math:`c = 1`

    Args:
        model_func: The neural network :math:`f_{\mathbf{\theta}}`.
        loss_func: The loss function :math:`\ell`.
        params: A tuple of parameters w.r.t. which the MC Fisher is computed.
            All parameters must be part of ``model_func.parameters()``.

    Returns:
        A function that takes inputs ``X``, ``y``, a matrix ``M`` in list format,
        ``mc_samples``, and ``generator``, and returns the mini-batch MC Fisher
        applied to ``M`` in list format.
    """
    f_flat, _ = make_functional_flattened_model_and_loss(model_func, loss_func, params)

    # Create the gradient output sampler for this loss function
    sample_grad_output_flat = make_grad_output_sampler(loss_func)

    def c_pseudo_flat(
        output_flat: Tensor, y: Tensor, mc_samples: int, generator: Generator
    ) -> Tensor:
        """Compute MC-Fisher pseudo-loss: L' = 0.5 / (M * c) * sum_n sum_m <f_n, g_nm>^2.

        This pseudo-loss L' := 0.5 / (M * c) ∑ₙ ∑ₘ fₙᵀ (gₙₘ gₙₘᵀ) fₙ where
        gₙₘ = ∂ℓₙ(yₙₘ)/∂fₙ (detached) and M is the number of MC samples.
        The GGN of L' linearized at fₙ is the MC Fisher.
        We can thus multiply with the MC Fisher by computing the GGN-vector products of L'.

        The reduction factor adjusts the scale depending on the loss reduction used.

        Args:
            output_flat: The flattened model predictions.
            y: The un-flattened labels.
            mc_samples: Number of MC samples to use.
            generator: Random generator for sampling.

        Returns:
            The pseudo-loss whose GGN is the MC-Fisher.
        """
        # Sample gradients w.r.t. output using the provided generator
        grad_output_samples = sample_grad_output_flat(
            output_flat.detach(), mc_samples, y, generator
        )

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

        # Compute the pseudo-loss: 0.5 / (M * c) * sum_n sum_m <f_n, g_nm>^2
        inner_products = einsum(
            output_flat, grad_output_samples, "n ..., m n ... -> m n"
        )
        return 0.5 / (mc_samples * reduction_factor) * (inner_products**2).sum()

    # Create the functional MC Fisher-vector product using GGN of pseudo-loss
    fmc_vp = make_ggn_vector_product(f_flat, c_pseudo_flat, num_c_extra_args=2)

    # Fix the parameters: X, y, mc_samples, generator, *v -> *Fv
    fmcvp = partial(fmc_vp, params)

    # Parallelize over vectors to multiply onto a matrix in list format
    list_format_vmap_dims = tuple(p.ndim for p in params)  # last axis
    return vmap(
        fmcvp,
        # X, y, mc_samples, generator are not vmapped, matrix columns are vmapped
        in_dims=(None, None, None, None, *list_format_vmap_dims),
        # Vmapped output axis is last
        out_dims=list_format_vmap_dims,
        # We want each vector to be multiplied with the same mini-batch MC Fisher
        randomness="same",
    )


class FisherMCLinearOperator(CurvatureLinearOperator):
    r"""Monte-Carlo approximation of the Fisher as SciPy linear operator.

    Consider the empirical risk

    .. math::
        \mathcal{L}(\mathbf{\theta})
        =
        c \sum_{n=1}^{N}
        \ell(f_{\mathbf{\theta}}(\mathbf{x}_n), \mathbf{y}_n)

    with :math:`c = \frac{1}{N}` for ``reduction='mean'`` and :math:`c=1` for
    ``reduction='sum'``. Let :math:`\ell(\mathbf{f}, \mathbf{y}) = - \log
    q(\mathbf{y} \mid \mathbf{f})` be a negative log-likelihood loss. Denoting
    :math:`\mathbf{f}_n = f_{\mathbf{\theta}}(\mathbf{x}_n)`, the Fisher
    information matrix is

    .. math::
        c \sum_{n=1}^{N}
        \left(
            \mathbf{J}_{\mathbf{\theta}}
            \mathbf{f}_n
        \right)^\top
        \mathbb{E}_{\mathbf{\tilde{y}}_n \sim q( \cdot  \mid \mathbf{f}_n)}
        \left[
            \nabla_{\mathbf{f}_n}^2
            \ell(\mathbf{f}_n, \mathbf{\tilde{y}}_n)
        \right]
        \left(
            \mathbf{J}_{\mathbf{\theta}}
            \mathbf{f}_n
        \right)
        \\
        =
        c \sum_{n=1}^{N}
        \left(
            \mathbf{J}_{\mathbf{\theta}}
            \mathbf{f}_n
        \right)^\top
        \mathbb{E}_{\mathbf{\tilde{y}}_n \sim q( \cdot  \mid \mathbf{f}_n)}
        \left[
            \left(
            \nabla_{\mathbf{f}_n}
            \ell(\mathbf{f}_n, \mathbf{\tilde{y}}_n)
            \right)
            \left(
            \nabla_{\mathbf{f}_n}
            \ell(\mathbf{f}_n, \mathbf{\tilde{y}}_n)
            \right)^{\top}
        \right]
        \left(
            \mathbf{J}_{\mathbf{\theta}}
            \mathbf{f}_n
        \right)
        \\
        \approx
        c \sum_{n=1}^{N}
        \left(
            \mathbf{J}_{\mathbf{\theta}}
            \mathbf{f}_n
        \right)^\top
        \frac{1}{M}
        \sum_{m=1}^M
        \left[
            \left(
            \nabla_{\mathbf{f}_n}
            \ell(\mathbf{f}_n, \mathbf{\tilde{y}}_{n}^{(m)})
            \right)
            \left(
            \nabla_{\mathbf{f}_n}
            \ell(\mathbf{f}_n, \mathbf{\tilde{y}}_{n}^{(m)})
            \right)^{\top}
        \right]
        \left(
            \mathbf{J}_{\mathbf{\theta}}
            \mathbf{f}_n
        \right)

    with sampled targets :math:`\mathbf{\tilde{y}}_{n}^{(m)} \sim q( \cdot \mid
    \mathbf{f}_n)`. The expectation over the model's likelihood is approximated
    via a Monte-Carlo estimator with :math:`M` samples.

    The linear operator represents a deterministic sample from this MC Fisher estimator.
    To generate different samples, you have to create instances with varying random
    seed argument.

    Attributes:
        SELF_ADJOINT: Whether the operator is self-adjoint. ``True`` for the Fisher.
        supported_losses: Supported loss functions.
        FIXED_DATA_ORDER: Whether the data order must be fix. ``True`` for MC-Fisher.
    """

    SELF_ADJOINT: bool = True
    FIXED_DATA_ORDER: bool = True
    supported_losses = (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)

    def __init__(
        self,
        model_func: Callable[[Union[Tensor, MutableMapping]], Tensor],
        loss_func: Union[MSELoss, CrossEntropyLoss],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        seed: int = 2147483647,
        mc_samples: int = 1,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[MutableMapping], int]] = None,
    ):
        """Linear operator for the Monte-Carlo approximation of the type-I Fisher.

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
            seed: Seed used to construct the internal random number generator used to
                draw samples at the beginning of each matrix-vector product.
                Default: ``2147483647``
            mc_samples: Number of samples to use. Default: ``1``.
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
        self._seed = seed
        self._generator: Union[None, Generator] = None
        self._mc_samples = mc_samples
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

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Multiply the MC-Fisher onto a matrix.

        Create and seed the random number generator.

        Args:
            M: Matrix for multiplication in tensor list format.

        Returns:
            Matrix-multiplication result ``mat @ M`` in tensor list format.
        """
        if self._generator is None or self._generator.device != self.device:
            self._generator = Generator(device=self.device)
        self._generator.manual_seed(self._seed)

        return super()._matmat(M)

    @cached_property
    def _mp(
        self,
    ) -> Callable[
        [Union[Tensor, MutableMapping], Tensor, int, Generator, Tuple[Tensor, ...]],
        Tuple[Tensor, ...],
    ]:
        """Lazy initialization of batch MC-Fisher matrix product function.

        Returns:
            Function that computes mini-batch MC-Fisher-vector products, given inputs
            ``X``, labels ``y``, number of MC samples, a random generator, and the
            entries ``v1, v2, ...`` of the vector in list format. Produces a list of
            tensors with the same shape as the input vector that represents the result
            of the mini-batch MC-Fisher multiplication.
        """
        return make_batch_fmc_matrix_product(
            self._model_func, self._loss_func, tuple(self._params)
        )

    def _matmat_batch(
        self, X: Union[Tensor, MutableMapping], y: Tensor, M: List[Tensor]
    ) -> List[Tensor]:
        """Apply the mini-batch MC-Fisher to a matrix.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M: Matrix to be multiplied with in tensor list format.
                Tensors have same shape as trainable model parameters, and an
                additional trailing axis for the matrix columns.

        Returns:
            Result of MC-Fisher multiplication in tensor list format. Has the same shape
            as ``M``, i.e. each tensor in the list has the shape of a parameter and a
            trailing dimension of matrix columns.
        """
        return list(self._mp(X, y, self._mc_samples, self._generator, *M))
