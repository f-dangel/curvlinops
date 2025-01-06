"""Implements the XTrace algorithm from Epperly 2024."""

from numpy import column_stack, dot, einsum, mean, ndarray
from numpy.linalg import inv, qr
from scipy.sparse.linalg import LinearOperator

from curvlinops.sampling import random_vector


def xtrace(
    A: LinearOperator, num_matvecs: int, distribution: str = "rademacher"
) -> float:
    """Estimate a linear operator's trace using the XTrace algorithm.

    The method is presented in `this paper<https://arxiv.org/pdf/2301.07825>`_:

    - Epperly, E. N., Tropp, J. A., & Webber, R. J. (2024). Xtrace: making the most
      of every sample in stochastic trace estimation. SIAM Journal on Matrix Analysis
      and Applications (SIMAX).

    It combines the variance reduction from Hutch++ with the exchangeability principle.

    Args:
        A: A square linear operator.
        num_matvecs: Total number of matrix-vector products to use. Must be even and
            less than the dimension of the linear operator.
        distribution: Distribution of the random vectors used for the trace estimation.
            Can be either ``'rademacher'`` or ``'normal'``. Default: ``'rademacher'``.

    Returns:
        The estimated trace of the linear operator.

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
    W = column_stack([random_vector(dim, distribution) for _ in range(num_vecs)])
    A_W = A @ W

    # compute the orthogonal basis for all test vectors, and its associated trace
    Q, R = qr(A_W)
    A_Q = A @ Q
    tr_QT_A_Q = einsum("ij,ji->", Q.T, A_Q)

    # compute the traces in the bases that would result had we left out the i-th
    # test vector in the QR decomposition
    RT_inv = inv(R.T)
    D = 1 / (RT_inv**2).sum(0) ** 0.5
    S = einsum("ij,j->ij", RT_inv, D)
    tr_QT_i_A_Q_i = einsum("ij,jk,kl,li->i", S.T, Q.T, A_Q, S)

    # Traces in the bases {Q_i}. This follows by writing Tr(QT_i A Q_i) = Tr(A Q_i QT_i)
    # then using the relation that Q_i QT_i = Q (I - s_i sT_i) QT. Further
    # simplification then leads to
    traces = tr_QT_A_Q - tr_QT_i_A_Q_i

    def deflate(v: ndarray, s: ndarray) -> ndarray:
        """Apply (I - s sT) to a vector.

        Args:
            v: Vector to deflate.
            s: Deflation vector.

        Returns:
            Deflated vector.
        """
        return v - dot(s, v) * s

    # estimate the trace on the complement of Q_i with vanilla Hutchinson using the
    # i-th test vector
    for i in range(num_vecs):
        w_i = W[:, i]
        s_i = S[:, i]
        A_w_i = A_W[:, i]

        # Compute (I - Q_i QT_i) A (I - Q_i QT_i) w_i
        #       = (I - Q_i QT_i) (Aw - AQ_i QT_i w_i)
        # ( using that Q_i QT_i = Q (I - s_i sT_i) QT )
        #       = (I - Q_i QT_i) (Aw - AQ (I - s_i sT_i) QT w)
        #       = (I - Q (I - s_i sT_i) QT) (Aw - AQ (I - s_i sT_i) QT w)
        #                                   |--------- A_p_w_i ---------|
        #         |-------------------- PT_A_P_w_i----------------------|
        A_P_w_i = A_w_i - A_Q @ deflate(Q.T @ w_i, s_i)
        PT_A_P_w_i = A_P_w_i - Q @ deflate(Q.T @ A_P_w_i, s_i)

        tr_w_i = dot(w_i, PT_A_P_w_i)
        traces[i] += tr_w_i

    return mean(traces)
