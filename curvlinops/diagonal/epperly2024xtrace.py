"""Implements the XDiag algorithm from Epperly 2024."""

from typing import Union

from torch import Tensor, column_stack, dot, einsum
from torch.linalg import inv, qr

from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.sampling import random_vector
from curvlinops.utils import (
    assert_divisible_by,
    assert_is_square,
    assert_matvecs_subseed_dim,
)


def xdiag(A: Union[PyTorchLinearOperator, Tensor], num_matvecs: int) -> Tensor:
    """Estimate a linear operator's diagonal using the XDiag algorithm.

    The method is presented in `this paper <https://arxiv.org/pdf/2301.07825>`_:

    - Epperly, E. N., Tropp, J. A., & Webber, R. J. (2024). Xtrace: making the most
      of every sample in stochastic trace estimation. SIAM Journal on Matrix Analysis
      and Applications (SIMAX).

    It combines the variance reduction from Diag++ with the exchangeability principle.

    Args:
        A: A square linear operator.
        num_matvecs: Total number of matrix-vector products to use. Must be even and
            less than the dimension of the linear operator (because otherwise one can
            evaluate the true diagonal directly at the same cost).

    Returns:
        The estimated diagonal of the linear operator.
    """
    dim = assert_is_square(A)
    assert_matvecs_subseed_dim(A, num_matvecs)
    assert_divisible_by(num_matvecs, 2, "num_matvecs")

    # draw random vectors and compute their matrix-vector products
    num_vecs = num_matvecs // 2
    W = column_stack(
        [random_vector(dim, "rademacher", A.device, A.dtype) for _ in range(num_vecs)]
    )
    A_W = A @ W

    # compute the orthogonal basis for all test vectors, and its associated diagonal
    Q, R = qr(A_W)
    QT_A = (A.adjoint() @ Q).T
    diag_Q_QT_A = einsum("ij,ji->i", Q, QT_A)

    # Compute and average the diagonals in the bases {Q_i} that would result had we left
    # out the i-th test vector in the QR decomposition. This follows by considering
    # diag(Q_i QT_i A) and using the relation Q_i QT_i = Q (I - s_i sT_i) QT, where the
    # s_i are given by:
    RT_inv = inv(R.T)
    D = 1 / (RT_inv**2).sum(0) ** 0.5
    S = einsum("ij,j->ij", RT_inv, D)
    # Further simplification then leads to
    diagonal = diag_Q_QT_A - einsum("ij,jk,lk,li->i", Q, S, S, QT_A) / num_vecs

    def deflate(v: Tensor, s: Tensor) -> Tensor:
        """Apply (I - s sT) to a vector.

        Args:
            v: Vector to deflate.
            s: Deflation vector.

        Returns:
            Deflated vector.
        """
        return v - dot(s, v) * s

    # estimate the diagonal on the complement of Q_i with vanilla Hutchinson using the
    # i-th test vector
    for i in range(num_vecs):
        w_i = W[:, i]
        s_i = S[:, i]
        A_w_i = A_W[:, i]

        # Compute (I - Q_i QT_i) A w_i
        #       = A w_i - (I - Q_i QT_i) A w_i
        # ( using that Q_i QT_i = Q (I - s_i sT_i) QT )
        #       = A w_i - Q (I - s_i sT_i) QT A w_i
        A_comp_w_i = A_w_i - Q @ deflate(QT_A @ w_i, s_i)

        diag_w_i = w_i * A_comp_w_i / w_i**2
        diagonal += diag_w_i / num_vecs

    return diagonal
