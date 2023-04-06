"""Contains tests for ``curvlinops/inverse``."""

from numpy import array, eye, random
from numpy.linalg import eigh, inv
from pytest import raises
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator

from curvlinops import (
    CGInverseLinearOperator,
    GGNLinearOperator,
    NeumannInverseLinearOperator,
)
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.examples.utils import report_nonclose


def test_CG_inverse_damped_GGN_matvec(case, delta: float = 1e-2):
    """Test matrix-vector multiplication by the inverse damped GGN with CG."""
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


def test_CG_inverse_damped_GGN_matmat(case, delta: float = 1e-2, num_vecs: int = 3):
    """Test matrix-matrix multiplication by the inverse damped GGN with CG."""
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


def test_Neumann_inverse_damped_GGN_matvec(case, delta: float = 1e-2):
    """Test matrix-vector multiplication by the inverse damped GGN with Neumann."""
    model_func, loss_func, params, data = case

    GGN = GGNLinearOperator(model_func, loss_func, params, data)
    damping = aslinearoperator(delta * sparse.eye(GGN.shape[0]))

    damped_GGN_functorch = functorch_ggn(
        model_func, loss_func, params, data
    ).detach().cpu().numpy() + delta * eye(GGN.shape[1])
    inv_GGN_functorch = inv(damped_GGN_functorch)

    # set scale such that Neumann series converges
    eval_max = eigh(damped_GGN_functorch)[0][-1]
    scale = 1.0 if eval_max < 2 else 1.9 / eval_max

    # NOTE This may break when other cases are added because slow convergence
    inv_GGN = NeumannInverseLinearOperator(GGN + damping, num_terms=5_000, scale=scale)

    x = random.rand(GGN.shape[1])
    report_nonclose(inv_GGN @ x, inv_GGN_functorch @ x, rtol=1e-1, atol=1e-1)


def test_NeumannInverseLinearOperator_toy():
    """Test NeumannInverseLinearOperator on a toy example.

    The example is from
    https://en.wikipedia.org/w/index.php?title=Neumann_series&oldid=1131424698#Example
    """
    random.seed(1234)
    A = array(
        [
            [0.0, 1.0 / 2.0, 1.0 / 4.0],
            [5.0 / 7.0, 0.0, 1.0 / 7.0],
            [3.0 / 10.0, 3.0 / 5.0, 0.0],
        ]
    )
    A += eye(A.shape[1])
    # eigenvalues of A: [1.82122892 0.47963837 0.69913271]

    inv_A = inv(A)
    inv_A_neumann = NeumannInverseLinearOperator(aslinearoperator(A))
    inv_A_neumann.set_neumann_hyperparameters(num_terms=1_000)

    x = random.rand(A.shape[1])
    report_nonclose(inv_A @ x, inv_A_neumann @ x, rtol=1e-3, atol=1e-5)

    # If we double the matrix, the Neumann series won't converge anymore ...
    B = 2 * A
    inv_B = inv(B)
    inv_B_neumann = NeumannInverseLinearOperator(aslinearoperator(B))
    inv_B_neumann.set_neumann_hyperparameters(num_terms=1_000)

    y = random.rand(B.shape[1])

    # ... therefore, we should get NaNs during the iteration
    with raises(ValueError):
        inv_B_neumann @ y

    # ... but if we scale the matrix back internally, the Neumann series converges
    inv_B_neumann.set_neumann_hyperparameters(num_terms=1_000, scale=0.5)
    report_nonclose(inv_B @ y, inv_B_neumann @ y, rtol=1e-3, atol=1e-5)

    inv_ground_truth = inv(B)
    inv_neumann = eye(B.shape[1])

    temp = eye(B.shape[1])
    scale = 0.5
    for _ in range(1000):
        temp = temp - scale * B @ temp
        inv_neumann = inv_neumann + temp

    inv_neumann = scale * inv_neumann

    report_nonclose(inv_neumann, inv_ground_truth, rtol=1e-3, atol=1e-5)
    report_nonclose(inv_neumann @ y, inv_ground_truth @ y, rtol=1e-3, atol=1e-5)
