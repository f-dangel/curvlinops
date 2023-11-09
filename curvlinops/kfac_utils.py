"""Utility functions related to KFAC."""

from math import sqrt
from typing import Union

from torch import Tensor, diag, einsum, eye
from torch.nn import CrossEntropyLoss, MSELoss


def loss_hessian_matrix_sqrt(
    output_one_datum: Tensor, loss_func: Union[MSELoss, CrossEntropyLoss]
) -> Tensor:
    r"""Compute the loss function's matrix square root for a sample's output.

    Args:
        output_one_datum: The model's prediction on a single datum. Has shape
            ``[1, C]`` where ``C`` is the number of classes (outputs of the neural
            network).
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

    Raises:
        ValueError: If the batch size is not one, or the output is not 2d.
        NotImplementedError: If the loss function is not supported.
    """
    if output_one_datum.ndim != 2 or output_one_datum.shape[0] != 1:
        raise ValueError(
            f"Expected 'output_one_datum' to be 2d with shape [1, C], got "
            f"{output_one_datum.shape}"
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
        return (diag(p_sqrt) - einsum("i,j->ij", p, p_sqrt)).mul_(sqrt(c))
    else:
        raise NotImplementedError(f"Loss function {loss_func} not supported.")
