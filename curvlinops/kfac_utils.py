"""Utility functions related to KFAC."""

from math import sqrt
from typing import Tuple, Union

from einconv import index_pattern
from einconv.utils import get_conv_paddings
from einops import einsum, rearrange, reduce
from torch import Tensor, diag, eye
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import unfold
from torch.nn.modules.utils import _pair


def loss_hessian_matrix_sqrt(
    output_one_datum: Tensor,
    target_one_datum: Tensor,
    loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
) -> Tensor:
    r"""Compute the loss function's matrix square root for a sample's output.

    Args:
        output_one_datum: The model's prediction on a single datum. Has shape
            ``[1, C]`` where ``C`` is the number of classes (outputs of the neural
            network).
        target_one_datum: The label of the single datum.
        loss_func: The loss function.

    Returns:
        The matrix square root
        :math:`\mathbf{S}` of the Hessian. Has shape
        ``[C, C]`` and satisfies the relation

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
        ValueError: If the batch size is not one, or the output is not 2d.
        NotImplementedError: If the loss function is not supported.
        NotImplementedError: If the loss function is ``BCEWithLogitsLoss`` but the
            target is not binary.
    """
    if output_one_datum.ndim != 2 or output_one_datum.shape[0] != 1:
        raise ValueError(
            f"Expected 'output_one_datum' to be 2d with shape [1, C], got "
            f"{output_one_datum.shape}"
        )
    if target_one_datum.shape[0] != 1:  # targets for 2d predictions are sometimes 1d
        raise ValueError(
            "Expected 'target_one_datum' to have batch_size 1."
            + f" Got {target_one_datum.shape}."
        )
    output = output_one_datum.squeeze(0)
    output_dim = output.numel()

    if isinstance(loss_func, MSELoss):
        c = {"sum": 1.0, "mean": 1.0 / output_dim}[loss_func.reduction]
        return eye(output_dim, device=output.device, dtype=output.dtype).mul_(
            sqrt(2 * c)
        )

    elif isinstance(loss_func, CrossEntropyLoss):
        c = 1.0
        p = output_one_datum.softmax(dim=1).squeeze()
        p_sqrt = p.sqrt()
        return (diag(p_sqrt) - einsum(p, p_sqrt, "i, j -> i j")).mul_(sqrt(c))

    elif isinstance(loss_func, BCEWithLogitsLoss):
        unique = set(target_one_datum.unique().flatten().tolist())
        if not unique.issubset({0, 1}):
            raise NotImplementedError(
                "Only binary targets (0, 1) are currently supported with"
                + f"BCEWithLogitsLoss. Got {unique}."
            )

        c = {"sum": 1.0, "mean": 1.0 / output_dim}[loss_func.reduction]
        p = output_one_datum.sigmoid().squeeze(0)
        hess_diag = sqrt(c) * (p * (1 - p)).sqrt()
        return hess_diag.diag()
    else:
        raise NotImplementedError(f"Loss function {loss_func} not supported.")


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
