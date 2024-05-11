"""Contains LinearOperator implementation of the (approximate) Fisher."""

from __future__ import annotations

from collections.abc import MutableMapping
from math import sqrt
from typing import Callable, Iterable, List, Optional, Tuple, Union

from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from einops import einsum, rearrange
from numpy import ndarray
from torch import Generator, Tensor, as_tensor, normal, softmax, zeros, zeros_like
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, Parameter
from torch.nn.functional import one_hot

from curvlinops._base import _LinearOperator


class FisherMCLinearOperator(_LinearOperator):
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
    """

    supported_losses = (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)

    def __init__(
        self,
        model_func: Callable[[Tensor], Tensor],
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
        """Linear operator for the MC approximation of the Fisher.

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

    def _matmat(self, M: ndarray) -> ndarray:
        """Multiply the MC-Fisher onto a matrix.

        Create and seed the random number generator.

        Args:
            M: Matrix for multiplication.

        Returns:
            Matrix-multiplication result ``mat @ M``.
        """
        if self._generator is None or self._generator.device != self._device:
            self._generator = Generator(device=self._device)
        self._generator.manual_seed(self._seed)

        return super()._matmat(M)

    def _matmat_batch(
        self, X: Union[Tensor, MutableMapping], y: Tensor, M_list: List[Tensor]
    ) -> Tuple[Tensor, ...]:
        """Apply the mini-batch MC-Fisher to a matrix.

        Args:
            X: Input to the DNN.
            y: Ground truth.
            M_list: Matrix to be multiplied with in list format.
                Tensors have same shape as trainable model parameters, and an
                additional leading axis for the matrix columns.

        Returns:
            Result of MC-Fisher multiplication in list format. Has the same shape as
            ``M_list``, i.e. each tensor in the list has the shape of a parameter and a
            leading dimension of matrix columns.
        """
        # compute ∂ℓₙ(yₙₘ)/∂fₙ where fₙ is the prediction for datum n and
        # yₙₘ is the m-th sampled label for datum n
        output = self._model_func(X)
        # If >2d output we convert to an equivalent 2d output
        if isinstance(self._loss_func, CrossEntropyLoss):
            output = rearrange(output, "batch c ... -> (batch ...) c")
            y = rearrange(y, "batch ... -> (batch ...)")
        else:
            output = rearrange(output, "batch ... c -> (batch ...) c")
            y = rearrange(y, "batch ... c -> (batch ...) c")

        grad_output = self.sample_grad_output(output, self._mc_samples, y)

        # Adjust the scale depending on the loss function and reduction used
        num_loss_terms, C = output.shape
        reduction_factor = {
            "mean": (
                num_loss_terms
                if isinstance(self._loss_func, CrossEntropyLoss)
                else num_loss_terms * C
            ),
            "sum": 1.0,
        }[self._loss_func.reduction]

        # Compute the pseudo-loss L' := 0.5 / (M * c) ∑ₙ ∑ₘ fₙᵀ (gₙₘ gₙₘᵀ) fₙ where
        # gₙₘ = ∂ℓₙ(yₙₘ)/∂fₙ (detached) and M is the number of MC samples.
        # The GGN of L' linearized at fₙ is the MC Fisher.
        # We can thus multiply with it by computing the GGN-vector products of L'.
        loss = (
            0.5
            / reduction_factor
            / self._mc_samples
            * (einsum(output, grad_output, "n ..., m n ... -> m n") ** 2).sum()
        )

        # Multiply the MC Fisher onto each vector in the input matrix
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

    def sample_grad_output(self, output: Tensor, num_samples: int, y: Tensor) -> Tensor:
        """Draw would-be gradients ``∇_f log p(·|f)``.

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

        C = output.shape[1]

        if isinstance(self._loss_func, MSELoss):
            std = as_tensor(sqrt(0.5), device=output.device)
            mean = zeros(
                num_samples, *output.shape, device=output.device, dtype=output.dtype
            )
            return 2 * normal(mean, std, generator=self._generator)

        elif isinstance(self._loss_func, CrossEntropyLoss):
            prob = softmax(output, dim=1)
            sample = prob.multinomial(
                num_samples=num_samples, replacement=True, generator=self._generator
            )
            sample = rearrange(sample, "batch s -> s batch")
            onehot_sample = one_hot(sample, num_classes=C)
            # repeat ``num_sample`` times along a new leading axis to avoid broadcasting
            prob = prob.unsqueeze(0).expand_as(onehot_sample)
            return prob - onehot_sample

        elif isinstance(self._loss_func, BCEWithLogitsLoss):
            unique = set(y.unique().flatten().tolist())
            if not unique.issubset({0, 1}):
                raise NotImplementedError(
                    "Only binary targets (0, 1) are currently supported with"
                    + f" BCEWithLogitsLoss. Got {unique}."
                )
            prob = output.sigmoid()
            # repeat ``num_sample`` times along a new leading axis
            prob = prob.unsqueeze(0).expand(num_samples, -1, -1)
            sample = prob.bernoulli(generator=self._generator)
            return prob - sample

        else:
            raise NotImplementedError(f"Supported losses: {self.supported_losses}")

    def _adjoint(self) -> FisherMCLinearOperator:
        """Return the linear operator representing the adjoint.

        The Fisher MC-approximation is real symmetric, and hence self-adjoint.

        Returns:
            Self.
        """
        return self
