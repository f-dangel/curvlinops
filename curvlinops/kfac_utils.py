"""Utility functions related to KFAC."""

from math import sqrt
from typing import Callable, Tuple, Union

from einconv import index_pattern
from einconv.utils import get_conv_paddings
from einops import einsum, rearrange, reduce
from torch import (
    Generator,
    zeros_like,
    Tensor,
    as_tensor,
    diag,
    normal,
    softmax,
    zeros,
    block_diag,
    vmap,
    stack,
    eye,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import one_hot, unfold
from torch.nn.modules.utils import _pair


def _check_binary_if_BCEWithLogitsLoss(
    target: Tensor,
    loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
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
        unique = set(u for u in target.unique().tolist())
        if not unique.issubset({0, 1}):
            raise NotImplementedError(
                "Only binary targets (0, 1) are currently supported with"
                + f" BCEWithLogitsLoss. Got values {unique}."
            )


def loss_hessian_matrix_sqrt(
    output_one_datum: Tensor,
    target_one_datum: Tensor,
    loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
    check_binary_if_BCEWithLogitsLoss: bool = True,
) -> Tensor:
    r"""Compute the loss function's matrix square root for a sample's output.

    Args:
        output_one_datum: The model's prediction on a single datum. Has shape
            ``[1, *, C]`` where ``C`` is the number of classes (outputs of the neural
            network) and * is arbitrary (e.g. empty or sequence length).
        target_one_datum: The label of the single datum.
            Has shape ``[1, *]`` (CE) or ``[1, *, C]`` (BCE, MSE).
        loss_func: The loss function.
        check_binary_if_BCEWithLogitsLoss: Whether to check if targets are binary
            for BCEWithLogitsLoss. Default: ``True``.

    Returns:
        The matrix square root
        :math:`\mathbf{S}` of the Hessian. Has shape
        ``[*, C, *, C]`` and satisfies the relation

        .. math::
            \mathbf{S} \mathbf{S}^\top
            =
            \nabla^2_{\mathbf{f}} \ell(\mathbf{f}, \mathbf{y})
            \in \mathbb{R}^{C \times C}

        where :math:`\mathbf{f} := f(\mathbf{x}) \in \mathbb{R}^C` is the model's
        prediction on a single datum :math:`\mathbf{x}` and :math:`\mathbf{y}` is
        the label.

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
        ValueError: If the batch size is not one.
        NotImplementedError: If the loss function is not supported.
    """
    if output_one_datum.shape[0] != 1:
        raise ValueError(
            f"Expected output_one_datum to have batch size 1, got {output_one_datum.shape}."
        )
    if target_one_datum.shape[0] != 1:  # targets for 2d predictions are sometimes 1d
        raise ValueError(
            f"Expected target_one_datum to have batch_size 1, got {target_one_datum.shape}."
        )
    output_dim = output_one_datum.numel()
    # Construct the Hessian square root as matrix (w.r.t. the flattened outputs)
    reduction = loss_func.reduction

    if isinstance(loss_func, MSELoss):
        c = {"sum": 1.0, "mean": 1.0 / output_dim}[reduction]
        hess_sqrt_flat = (
            zeros_like(output_one_datum).fill_(sqrt(2 * c)).flatten().diag()
        )

    elif isinstance(loss_func, CrossEntropyLoss):
        # Output has shape [1, C, d1, d2, ...], flatten into [C, d1 * d2 * ...]
        output_flat = output_one_datum.squeeze(0).unsqueeze(-1).flatten(start_dim=1)
        p = output_flat.softmax(dim=0)
        c = {"sum": 1.0, "mean": 1.0 / p.shape[1]}[reduction]

        def hess_sqrt_element(p: Tensor) -> Tensor:
            """Compute the Hessian square root for a single element of the sequence.

            Args:
                p: Vector of probabilities for a single sequence. Has shape ``[C]``.

            Returns:
                The Hessian square root matrix. Has shape ``[C, C]``.
            """
            p_sqrt = p.sqrt()
            return (diag(p_sqrt) - einsum(p, p_sqrt, "i, j -> i j")).mul_(sqrt(c))

        # Compute the per-element Hessian square root
        blocks_stacked = vmap(hess_sqrt_element, in_dims=-1)(p)  # [D, C, C]

        C, D = output_flat.shape
        # Create identity matrix for the D dimension to select block-diagonal
        eye_D = eye(D, device=p.device, dtype=p.dtype)  # [D, D]
        # Construct [C, D, C, D] tensor with blocks on the (d, d) diagonal
        # blocks_stacked[d, c1, c2] * eye_D[d, d2] gives non-zero only when d == d2
        hess_sqrt_flat = einsum(blocks_stacked, eye_D, "d c1 c2, d d2 -> c1 d c2 d2")
        hess_sqrt_flat = hess_sqrt_flat.reshape(C * D, C * D)

    elif isinstance(loss_func, BCEWithLogitsLoss):
        if check_binary_if_BCEWithLogitsLoss:
            _check_binary_if_BCEWithLogitsLoss(target_one_datum, loss_func)

        c = {"sum": 1.0, "mean": 1.0 / output_dim}[reduction]
        p = output_one_datum.flatten().sigmoid()
        hess_diag = sqrt(c) * (p * (1 - p)).sqrt()
        hess_sqrt_flat = hess_diag.diag()
    else:
        raise NotImplementedError(f"Loss function {loss_func} not supported.")

    # Un-flatten the output dimensions
    output_shape = output_one_datum.shape[1:]
    return hess_sqrt_flat.reshape(*output_shape, *output_shape)


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


def make_grad_output_sampler(
    loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
    check_binary_if_BCEWithLogitsLoss: bool = True,
) -> Callable[[Tensor, int, Tensor, Generator], Tensor]:
    """Create a function that samples gradients w.r.t. network outputs.

    Args:
        loss_func: The loss function to create the sampler for.
        check_binary_if_BCEWithLogitsLoss: Whether to check if targets are binary
            for BCEWithLogitsLoss. Default: ``True``.

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
            if check_binary_if_BCEWithLogitsLoss:
                _check_binary_if_BCEWithLogitsLoss(y, loss_func)
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
