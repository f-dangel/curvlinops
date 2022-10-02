"""Contains tests for ``curvlinops/inverse``."""

from numpy import eye, random
from numpy.linalg import inv
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator

from curvlinops import CGInverseLinearOperator, GGNLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.examples.utils import report_nonclose


def test_inverse_damped_GGN_matvec(case, delta: float = 1e-2):
    """Test matrix-vector multiplication by the inverse damped GGN."""
    model_func, loss_func, params, data = case

    GGN = GGNLinearOperator(model_func, loss_func, params, data)
    damping = aslinearoperator(delta * sparse.eye(GGN.shape[0]))

    inv_GGN = CGInverseLinearOperator(GGN + damping)
    inv_GGN_functorch = inv(
        functorch_ggn(model_func, loss_func, params, data).detach().cpu().numpy()
        + delta * eye(GGN.shape[1])
    )

    x = random.rand(GGN.shape[1])
    report_nonclose(inv_GGN @ x, inv_GGN_functorch @ x, rtol=5e-3, atol=1e-5)


def test_inverse_damped_GGN_matmat(case, delta: float = 1e-2, num_vecs: int = 3):
    """Test matrix-matrix multiplication by the inverse damped GGN."""
    model_func, loss_func, params, data = case

    GGN = GGNLinearOperator(model_func, loss_func, params, data)
    damping = aslinearoperator(delta * sparse.eye(GGN.shape[0]))

    inv_GGN = CGInverseLinearOperator(GGN + damping)
    inv_GGN_functorch = inv(
        functorch_ggn(model_func, loss_func, params, data).detach().cpu().numpy()
        + delta * eye(GGN.shape[1])
    )

    X = random.rand(GGN.shape[1], num_vecs)
    report_nonclose(inv_GGN @ X, inv_GGN_functorch @ X, rtol=5e-3, atol=1e-5)
