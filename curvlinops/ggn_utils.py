"""Utility functions related to the GGN and its approximations (KFAC, diagonal GGN)."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import sqrt

from einops import einsum, rearrange
from torch import (
    Generator,
    Tensor,
    as_tensor,
    block_diag,
    diag,
    normal,
    softmax,
    zeros,
    zeros_like,
)
from torch.func import grad, vmap
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import one_hot

from curvlinops.utils import make_functional_call


def loss_hessian_matrix_sqrt(
    output_one_datum: Tensor,
    target_one_datum: Tensor,
    loss_func: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
) -> Tensor:
    r"""Compute the loss function's matrix square root for a sample's output.

    Args:
        output_one_datum: The model's prediction on a single datum.
            Has shape ``[C, *D]`` for CE where ``C`` is the number of classes,
            or ``[*D]`` for MSE/BCE with ``*D`` optional (and potentially multiple)
            sequence dimensions. Has no batch axis.
        target_one_datum: The label of the single datum. Has shape ``[*D]``.
            Has no batch axis.
        loss_func: The loss function.

    Returns:
        The matrix square root
        :math:`\mathbf{S}` of the Hessian. Has shape ``[C, *D, C, *D]`` for CE and
        ``[*D, *D]`` for BCE/MSE loss. Its matrix view satisfies

        .. math::
            \mathbf{S} \mathbf{S}^\top
            =
            \nabla^2_{\mathbf{f}} \ell(\mathbf{f}, \mathbf{y})

        where :math:`\mathbf{f} := f(\mathbf{x})` is the model's prediction on a single
        datum :math:`\mathbf{x}` and :math:`\mathbf{y}` is the label.

    Below, we list the Hessian square roots for vector-valued predictions of shape ``[C]``.

    Note:
        For :class:`torch.nn.MSELoss` (with :math:`c = 1` for ``reduction='sum'``
        and :math:`c = 1/C` for ``reduction='mean'``), we have:

        .. math::
            \ell(\mathbf{f}) &= c \sum_{i=1}^C (f_i - y_i)^2
            \\
            \nabla^2_{\mathbf{f}} \ell(\mathbf{f}, \mathbf{y}) &= 2 c \mathbf{I}_C
            \\
            \mathbf{S} &= \sqrt{2 c} \mathbf{I}_C

    Note:
        For :class:`torch.nn.CrossEntropyLoss` (with :math:`c = 1` irrespective of the
        reduction, :math:`\mathbf{p}:=\mathrm{softmax}(\mathbf{f}) \in \mathbb{R}^C`,
        and the element-wise natural logarithm :math:`\log`) we have:

        .. math::
            \ell(\mathbf{f}, y) = - c \log(\mathbf{p})^\top \mathrm{onehot}(y)
            \\
            \nabla^2_{\mathbf{f}} \ell(\mathbf{f}, y)
            =
            c \left(
            \mathrm{diag}(\mathbf{p}) - \mathbf{p} \mathbf{p}^\top
            \right)
            \\
            \mathbf{S} = \sqrt{c} \left(
            \mathrm{diag}(\sqrt{\mathbf{p}}) - \sqrt{\mathbf{p}} \mathbf{p}^\top
            \right)\,,

       where the square root is applied element-wise. See for instance Example 5.1 of
       `this thesis <https://d-nb.info/1280233206/34>`_ or equations (5) and (6) of
       `this paper <https://arxiv.org/abs/1901.08244>`_.

    Note:
        For :class:`torch.nn.BCEWithLogitsLoss` (with :math:`c = 1` for ``reduction='sum'``
        and :math:`c = 1/C` for ``reduction='mean'``) we have (:math:`\sigma` is the sigmoid
        function; targets may be any value in :math:`[0, 1]`):

        .. math::
            \ell(\mathbf{f})
            &=
            c \sum_{i=1}^C - y_i \log(\sigma(f_i)) - (1 - y_i) \log(1 - \sigma(f_i))
            \\
            \nabla^2_{\mathbf{f}} \ell(\mathbf{f}, \mathbf{y})
            &=
            c \mathrm{diag}( \sigma(f_i) \odot (1 - \sigma(f_i)) )
            \\
            \mathbf{S}
            &=
            \sqrt{c} \mathrm{diag}(\sqrt{\sigma(f_i) \odot (1 - \sigma(f_i))})\,,

        where the square root is applied element-wise.

    Raises:
        NotImplementedError: If the loss function is not supported.
    """
    # Number of losses contributed from a datum's sequence-valued prediction
    num_features = (
        output_one_datum.numel() / output_one_datum.shape[0]
        if isinstance(loss_func, CrossEntropyLoss)
        else output_one_datum.numel()
    )
    # Reduction factor from accumulation over losses in a sequence
    reduction = loss_func.reduction
    c = {"sum": 1.0, "mean": 1.0 / num_features}[reduction]

    # Construct the Hessian square root as matrix (w.r.t. the flattened outputs)
    if isinstance(loss_func, MSELoss):
        hess_sqrt_flat = (
            zeros_like(output_one_datum).fill_(sqrt(2 * c)).flatten().diag()
        )

    elif isinstance(loss_func, CrossEntropyLoss):
        # Output has shape [C, d1, d2, ...], flatten into [C, d1 * d2 * ...]
        output_flat = output_one_datum.unsqueeze(-1).flatten(start_dim=1)
        C, D = output_flat.shape
        p = output_flat.softmax(dim=0)

        def hess_sqrt_element(p: Tensor) -> Tensor:
            """Compute the Hessian square root for a single element of the sequence.

            Args:
                p: Vector of probabilities for a single sequence. Has shape ``[C]``.

            Returns:
                The Hessian square root matrix. Has shape ``[C, C]``.
            """
            p_sqrt = sqrt(c) * p.sqrt()
            return diag(p_sqrt) - einsum(p, p_sqrt, "i, j -> i j")

        # Compute the per-element Hessian square root
        blocks_stacked = vmap(hess_sqrt_element, in_dims=-1)(p)  # [D, C, C]

        # This is the Hessian square root in a rearranged basis [d1 * d2 * ... , C]
        blocks = block_diag(*blocks_stacked)
        # Rearrange into the basis [C, d1 * d2 * ...]
        hess_sqrt_flat = rearrange(
            blocks, "(d1 c1) (d2 c2) -> (c1 d1) (c2 d2)", d1=D, d2=D, c1=C, c2=C
        )
        hess_sqrt_flat = hess_sqrt_flat.reshape(C * D, C * D)

    elif isinstance(loss_func, BCEWithLogitsLoss):
        p = output_one_datum.flatten().sigmoid()
        hess_sqrt_diag = sqrt(c) * (p * (1 - p)).sqrt()
        hess_sqrt_flat = hess_sqrt_diag.diag()

    else:
        raise NotImplementedError(f"Loss function {loss_func} not supported.")

    # Un-flatten the output dimensions
    output_shape = output_one_datum.shape
    return hess_sqrt_flat.reshape(*output_shape, *output_shape)


def _make_single_datum_sampler(
    loss_func: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
) -> Callable[[Tensor, int, Tensor, Generator], Tensor]:
    """Create a function that samples gradients w.r.t. a single datum's output.

    The expectation of the sampled gradient outer product is the loss function's
    Hessian, including scaling from reductions over non-batch axes.

    Args:
        loss_func: The loss function to create the sampler for.

    Returns:
        A function that samples gradients w.r.t. the model prediction for one datum.
        Signature: ``(output, num_samples, target, generator) -> grad_samples``.
        The returned gradient samples have shape ``[num_samples, *output.shape]``.
    """

    def sample_grad_output(
        output_one_datum: Tensor,
        num_samples: int,
        target_one_datum: Tensor,
        generator: Generator,
    ) -> Tensor:
        """Draw would-be gradients ``nabla_f log p(.|f)`` with explicit generator.

        Handles a single data point.
        The would-be gradient's outer product equals the Hessian ``nabla^2_f log p(.|f)``
        in expectation.
        Currently supports ``MSELoss``, ``CrossEntropyLoss``, and
        ``BCEWithLogitsLoss`` with arbitrary output dimensions.
        The returned gradients include proper scaling based on the loss function's
        reduction type over the feature dimensions.

        Args:
            output_one_datum: model prediction ``f`` for one datum. Has no batch axis.
            num_samples: Number of samples to draw.
            target_one_datum: Labels of the datum. Has no batch axis.
            generator: Random generator for sampling.

        Returns:
            Samples of the gradient w.r.t. the model prediction for one datum.
            Has shape ``[num_samples, *output.shape]``.

        Raises:
            NotImplementedError: For unsupported loss functions.
        """
        # Number of losses contributed from a datum's sequence-valued prediction
        num_features = (
            output_one_datum.numel() / output_one_datum.shape[0]
            if isinstance(loss_func, CrossEntropyLoss)
            else output_one_datum.numel()
        )
        # Reduction factor from accumulation over losses in a sequence
        reduction = loss_func.reduction
        c = {"sum": 1.0, "mean": 1.0 / num_features}[reduction]

        if isinstance(loss_func, MSELoss):
            dev, dt = output_one_datum.device, output_one_datum.dtype
            std = as_tensor(sqrt(2 * c), device=dev, dtype=dt)
            mean = zeros(num_samples, *output_one_datum.shape, device=dev, dtype=dt)
            grad_samples = normal(mean, std, generator=generator)

        elif isinstance(loss_func, CrossEntropyLoss):
            # Flatten sequence dimensions: [C, *seq] -> [C, seq_flat]
            C = output_one_datum.shape[0]
            output_flat = output_one_datum.unsqueeze(-1).flatten(start_dim=1)
            prob = softmax(output_flat, dim=0)  # [C, seq_flat]

            # Sample for each sequence position independently
            # Rearrange to [seq_flat, C] for multinomial sampling
            prob_for_sampling = rearrange(prob, "c s -> s c")
            samples = prob_for_sampling.multinomial(
                num_samples=num_samples, replacement=True, generator=generator
            )  # [seq_flat, num_samples]
            samples = rearrange(samples, "s n -> n s")  # [num_samples, seq_flat]
            onehot_samples = one_hot(samples, num_classes=C)
            # [num_samples, seq_flat, C] -> [num_samples, C, seq_flat]
            onehot_samples = rearrange(onehot_samples, "n s c -> n c s")

            # Expand prob to match: [C, seq_flat] -> [num_samples, C, seq_flat]
            prob_expanded = prob.unsqueeze(0).expand_as(onehot_samples)
            grad_samples_flat = sqrt(c) * (prob_expanded - onehot_samples)

            # Reshape back to original sequence dimensions
            out_shape = (num_samples, *output_one_datum.shape)
            grad_samples = grad_samples_flat.reshape(out_shape)

        elif isinstance(loss_func, BCEWithLogitsLoss):
            prob = output_one_datum.sigmoid()
            # repeat ``num_sample`` times along a new leading axis
            prob = prob.unsqueeze(0).expand(num_samples, *prob.shape)
            sample = prob.bernoulli(generator=generator)
            grad_samples = sqrt(c) * (prob - sample)

        else:
            raise NotImplementedError(
                f"Supported losses: {(MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)}"
            )

        return grad_samples

    return sample_grad_output


def make_grad_output_fn(
    loss_func: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
    mode: str,
    mc_samples: int = 1,
) -> Callable[[Tensor, Tensor, Generator | None], Tensor]:
    """Create a function computing gradient output vectors for a single datum.

    For exact mode, returns the columns of the loss Hessian's matrix square root.
    For MC mode, returns Monte-Carlo sampled gradient vectors.
    For empirical mode, returns the gradient of the loss w.r.t. the output.
    For forward-only mode, returns an empty tensor (no backward passes needed).

    Note:
        For MC mode, the returned vectors are scaled by ``1 / sqrt(mc_samples)``
        so that the sum of their outer products approximates the Hessian, matching
        the exact mode contract.

    Args:
        loss_func: The loss function.
        mode: ``'exact'`` for Hessian square root, ``'mc'`` for Monte-Carlo sampling,
            ``'empirical'`` for empirical gradients, ``'forward-only'`` for no
            backward passes.
        mc_samples: Number of Monte-Carlo samples (only used when ``mode='mc'``).
            Default: ``1``.

    Returns:
        A function with signature
        ``(output, target, generator=None) -> [num_vectors, *output.shape]``
        operating on a single datum (no batch axis). ``num_vectors`` is
        ``output.numel()`` for exact mode, ``mc_samples`` for MC mode, ``1``
        for empirical mode, or ``0`` for forward-only mode.

    Raises:
        ValueError: If ``mode`` is not ``'exact'``, ``'mc'``, ``'empirical'``,
            or ``'forward-only'``.
    """
    if mode not in ("exact", "mc", "empirical", "forward-only"):
        raise ValueError(
            f"Invalid mode {mode!r}. "
            "Must be 'exact', 'mc', 'empirical', or 'forward-only'."
        )

    sample_grad_output = _make_single_datum_sampler(loss_func)

    if mode == "empirical":
        functional_loss_func = partial(make_functional_call(loss_func), {})

        def _scaled_datum_loss(prediction: Tensor, target: Tensor) -> Tensor:
            """Compute a scaled loss for one sample, adjusting for mean reduction.

            For ``MSELoss`` and ``BCEWithLogitsLoss`` with ``reduction='mean'``,
            the loss averages over both batch and output dimensions. Since we
            operate on a single datum (no batch), the output-dimension averaging
            produces an extra ``1/C`` factor. We want only ``1/sqrt(C)`` so that
            the gradient outer product gives the correct contribution to the
            empirical Fisher.

            Args:
                prediction: Model prediction for one sample, without batch dim.
                target: Target for one sample, without batch dim.

            Returns:
                Scaled loss for one sample.
            """
            (C,) = prediction.shape
            scale = (
                sqrt(C)
                if (
                    isinstance(loss_func, (BCEWithLogitsLoss, MSELoss))
                    and loss_func.reduction == "mean"
                )
                else 1.0
            )
            return scale * functional_loss_func(
                prediction.unsqueeze(0), target.unsqueeze(0)
            )

        _empirical_grad = grad(_scaled_datum_loss, argnums=0)

    def grad_output_fn(
        output: Tensor, target: Tensor, generator: Generator | None = None
    ) -> Tensor:
        """Compute gradient output vectors for a single datum.

        Args:
            output: Model prediction for one datum (no batch axis).
            target: Label for the datum (no batch axis).
            generator: Random generator (used for MC mode, ignored otherwise).

        Returns:
            Gradient vectors of shape ``[num_vectors, *output.shape]``.
        """
        if mode == "forward-only":
            return output.new_empty(0, *output.shape)
        elif mode == "exact":
            hessian_sqrt = loss_hessian_matrix_sqrt(output, target, loss_func)
            return hessian_sqrt.reshape(*output.shape, output.numel()).movedim(-1, 0)
        elif mode == "mc":
            return sample_grad_output(output, mc_samples, target, generator).div_(
                sqrt(mc_samples)
            )
        else:  # mode == "empirical"
            return _empirical_grad(output, target).unsqueeze(0)

    return grad_output_fn
