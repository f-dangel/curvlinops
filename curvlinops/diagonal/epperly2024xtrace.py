"""Implements the XDiag algorithm from Epperly 2024."""

from numpy import column_stack, dot, einsum, ndarray
from numpy.linalg import inv, qr
from scipy.sparse.linalg import LinearOperator

from curvlinops.sampling import random_vector


def xdiag(A: LinearOperator, num_matvecs: int) -> ndarray:
    """Estimate a linear operator's diagonal using the XDiag algorithm.

    The method is presented in `this paper <https://arxiv.org/pdf/2301.07825>`_:

    - Epperly, E. N., Tropp, J. A., & Webber, R. J. (2024). Xtrace: making the most
      of every sample in stochastic trace estimation. SIAM Journal on Matrix Analysis
      and Applications (SIMAX).

    It combines the variance reduction from Diag++ with the exchangeability principle.

    Args:
        A: A square linear operator.
        num_matvecs: Total number of matrix-vector products to use. Must be even and
            less than the dimension of the linear operator.

    Returns:
        The estimated diagonal of the linear operator.

    Raises:
        ValueError: If the linear operator is not square or if the number of matrix-
            vector products is not even or is greater than the dimension of the linear
            operator.
    """
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square. Got shape {A.shape}.")
    dim = A.shape[1]
    if num_matvecs % 2 != 0 or num_matvecs >= dim:
        raise ValueError(
            "num_matvecs must be even and less than the dimension of A.",
            f" Got {num_matvecs}.",
        )

    # draw random vectors and compute their matrix-vector products
    num_vecs = num_matvecs // 2
    W = column_stack([random_vector(dim, "rademacher") for _ in range(num_vecs)])
    A_W = A @ W

    # compute the orthogonal basis for all test vectors, and its associated diagonal
    Q, R = qr(A_W)
    QT_A = Q.T @ A
    diag_Q_QT_A = einsum("ij,ji->i", Q, QT_A)

    # Compute and average the diagonals in the bases {Q_i} that would result had we left
    # out the i-th test vector in the QR decomposition. This follows by considering
    # diag(Q_i QT_i A) and using the relation Q_i QT_i = Q (I - s_i sT_i) QT, where the
    # s_i are given by:
    RT_inv = inv(R.T)
    D = 1 / (RT_inv**2).sum(0) ** 0.5
    S = einsum("ij,j->ij", RT_inv, D)
    # Further simplification then leads to
    diagonal = (
        diag_Q_QT_A
        - einsum("ij,jk,lk,li->i", Q, S, S, QT_A, optimize="optimal") / num_vecs
    )

    def deflate(v: ndarray, s: ndarray) -> ndarray:
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
