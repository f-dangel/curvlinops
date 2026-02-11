"""Utility functions related to KFAC."""

from math import sqrt
from typing import Callable, Tuple, Union
from warnings import warn

from einconv import index_pattern
from einconv.utils import get_conv_paddings
from einops import einsum, rearrange, reduce
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
from torch.func import vmap
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import one_hot, unfold
from torch.nn.modules.utils import _pair


def _warn_BCEWithLogitsLoss_targets_unchecked(
    loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
) -> None:
    """Warn that BCEWithLogitsLoss targets are not verified to be binary.

    Args:
        loss_func: The loss function being used.
    """
    if isinstance(loss_func, BCEWithLogitsLoss):
        warn(
            "BCEWithLogitsLoss only supports binary targets (0, 1), but this is "
            "not being verified. Ensure your targets are binary to avoid "
            "incorrect results (using _check_binary_if_BCEWithLogitsLoss).",
            UserWarning,
            stacklevel=3,
        )


def _check_binary_if_BCEWithLogitsLoss(
    target: Tensor, loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss]
) -> None:
    """Check if targets are binary (0 or 1) when using BCEWithLogitsLoss.

    Args:
        target: Target tensor.
        loss_func: The loss function being used.

    Raises:
        NotImplementedError: If the loss function is BCEWithLogitsLoss but targets
            are not binary (0 or 1).
    """
    if isinstance(loss_func, BCEWithLogitsLoss):
        unique = set(target.unique().tolist())
        if not unique.issubset({0, 1}):
            raise NotImplementedError(
                "Only binary targets (0, 1) are currently supported with"
                + f" BCEWithLogitsLoss. Got values {unique}."
            )


def loss_hessian_matrix_sqrt(
    output_one_datum: Tensor,
    target_one_datum: Tensor,
    loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
    warn_BCEWithLogitsLoss_targets_unchecked: bool = True,
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
        warn_BCEWithLogitsLoss_targets_unchecked: Whether to warn that targets are
            not verified to be binary for BCEWithLogitsLoss. Default: ``True``.

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
        function, and assuming binary labels):

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
        NotImplementedError: If the loss function is ``BCEWithLogitsLoss`` but the
            target is not binary.
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
        if warn_BCEWithLogitsLoss_targets_unchecked:
            _warn_BCEWithLogitsLoss_targets_unchecked(loss_func)

        p = output_one_datum.flatten().sigmoid()
        hess_sqrt_diag = sqrt(c) * (p * (1 - p)).sqrt()
        hess_sqrt_flat = hess_sqrt_diag.diag()

    else:
        raise NotImplementedError(f"Loss function {loss_func} not supported.")

    # Un-flatten the output dimensions
    output_shape = output_one_datum.shape
    return hess_sqrt_flat.reshape(*output_shape, *output_shape)


def make_grad_output_sampler(
    loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
    warn_BCEWithLogitsLoss_targets_unchecked: bool = True,
) -> Callable[[Tensor, int, Tensor, Generator], Tensor]:
    """Create a function that samples gradients w.r.t. network outputs.

    The expectation of the sampled gradient outer product is the loss function's
    Hessian, including scaling from reductions over non-batch axes.

    Args:
        loss_func: The loss function to create the sampler for.
        warn_BCEWithLogitsLoss_targets_unchecked: Whether to warn that targets are
            not verified to be binary for BCEWithLogitsLoss. Default: ``True``.

    Returns:
        A function that samples gradients w.r.t. the model prediction.
        Signature: (output, num_samples, y, generator) -> grad_samples.
        The predictions (output) and labels (y) both have a batch axis, and the
        returned gradient samples will have the shape ``[num_samples, *output.shape]``.
    """

    def sample_grad_output(
        output_one_datum: Tensor,
        num_samples: int,
        target_one_datum: Tensor,
        generator: Generator,
    ) -> Tensor:
        """Draw would-be gradients ``∇_f log p(·|f)`` with explicit generator.

        Handles a single data point.
        The would-be gradient's outer product equals the Hessian ``∇²_f log p(·|f)``
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
            if warn_BCEWithLogitsLoss_targets_unchecked:
                _warn_BCEWithLogitsLoss_targets_unchecked(loss_func)

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

    # Parallelize over predictions and targets
    return vmap(
        sample_grad_output,
        in_dims=(0, None, 0, None),
        out_dims=1,
        randomness="different",
    )


def extract_patches(
    x: Tensor,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int],
    padding: Union[Tuple[int, int], int, str],
    dilation: Union[Tuple[int, int], int],
    groups: int,
) -> Tensor:
    """Extract patches from the input of a 2d-convolution.

    The patches are averaged over channel groups.

    Args:
        x: Input to a 2d-convolution. Has shape ``[batch_size, C_in, I1, I2]``.
        kernel_size: The convolution's kernel size supplied as 2-tuple or integer.
        stride: The convolution's stride supplied as 2-tuple or integer.
        padding: The convolution's padding supplied as 2-tuple, integer, or string.
        dilation: The convolution's dilation supplied as 2-tuple or integer.
        groups: The number of channel groups.

    Returns:
        A tensor of shape ``[batch_size, O1 * O2, C_in // groups * K1 * K2]`` where
        each column ``[b, o1_o2, :]`` contains the flattened patch of sample ``b`` used
        for output location ``(o1, o2)``, averaged over channel groups.

    Raises:
        NotImplementedError: If ``padding`` is a string that would lead to unequal
            padding along a dimension.
    """
    if isinstance(padding, str):  # get padding as integers
        padding_as_int = []
        for k, s, d in zip(_pair(kernel_size), _pair(stride), _pair(dilation)):
            p_left, p_right = get_conv_paddings(k, s, padding, d)
            if p_left != p_right:
                raise NotImplementedError("Unequal padding not supported in unfold.")
            padding_as_int.append(p_left)
        padding = tuple(padding_as_int)

    # average channel groups
    x = rearrange(x, "b (g c_in) i1 i2 -> b g c_in i1 i2", g=groups)
    x = reduce(x, "b g c_in i1 i2 -> b c_in i1 i2", "mean")

    x_unfold = unfold(x, kernel_size, dilation=dilation, padding=padding, stride=stride)
    return rearrange(x_unfold, "b c_in_k1_k2 o1_o2 -> b o1_o2 c_in_k1_k2")


def extract_averaged_patches(
    x: Tensor,
    kernel_size: Union[Tuple[int, int], int],
    stride: Union[Tuple[int, int], int],
    padding: Union[Tuple[int, int], int, str],
    dilation: Union[Tuple[int, int], int],
    groups: int,
) -> Tensor:
    """Extract averaged patches from the input of a 2d-convolution.

    The patches are averaged over channel groups and output locations.

    Uses the tensor network formulation of convolution from
    `Dangel, 2023 <https://arxiv.org/abs/2307.02275>`_.

    Args:
        x: Input to a 2d-convolution. Has shape ``[batch_size, C_in, I1, I2]``.
        kernel_size: The convolution's kernel size supplied as 2-tuple or integer.
        stride: The convolution's stride supplied as 2-tuple or integer.
        padding: The convolution's padding supplied as 2-tuple, integer, or string.
        dilation: The convolution's dilation supplied as 2-tuple or integer.
        groups: The number of channel groups.

    Returns:
        A tensor of shape ``[batch_size, C_in // groups * K1 * K2]`` where each column
        ``[b, :]`` contains the flattened patch of sample ``b`` averaged over all output
        locations and channel groups.
    """
    # average channel groups
    x = rearrange(x, "b (g c_in) i1 i2 -> b g c_in i1 i2", g=groups)
    x = reduce(x, "b g c_in i1 i2 -> b c_in i1 i2", "mean")

    # TODO For convolutions with special structure, we don't even need to compute
    # the index pattern tensors, or can resort to contracting only slices thereof.
    # In order for this to work `einconv`'s TN simplification mechanism must first
    # be refactored to work purely symbolically. Once this is done, it will be
    # possible to do the below even more efficiently (memory and run time) for
    # structured convolutions.

    # compute index pattern tensors, average output dimension
    patterns = []
    input_sizes = x.shape[-2:]
    for i, k, s, p, d in zip(
        input_sizes,
        _pair(kernel_size),
        _pair(stride),
        (padding, padding) if isinstance(padding, str) else _pair(padding),
        _pair(dilation),
    ):
        pi = index_pattern(
            i, k, stride=s, padding=p, dilation=d, dtype=x.dtype, device=x.device
        )
        pi = reduce(pi, "k o i -> k i", "mean")
        patterns.append(pi)

    x = einsum(x, *patterns, "b c_in i1 i2, k1 i1, k2 i2 -> b c_in k1 k2")
    return rearrange(x, "b c_in k1 k2 -> b (c_in k1 k2)")
