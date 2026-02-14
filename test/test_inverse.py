"""Contains tests for ``curvlinops/inverse``."""

from pytest import mark, raises
from torch import Tensor, cuda, float64, manual_seed, rand, randn
from torch.linalg import inv

from curvlinops import (
    CGInverseLinearOperator,
    GGNLinearOperator,
    LSMRInverseLinearOperator,
    NeumannInverseLinearOperator,
)
from curvlinops.examples import IdentityLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from test.test__torch_base import TensorLinearOperator
from test.utils import (
    change_dtype,
    compare_consecutive_matmats,
    compare_matmat,
    eye_like,
)


class _CountingTensorLinearOperator(TensorLinearOperator):
    """Tensor linear operator that counts `_matmat` calls for efficiency tests."""

    def __init__(self, A: Tensor):
        super().__init__(A)
        self.num_matmats = 0

    def _matmat(self, X):
        self.num_matmats += 1
        return super()._matmat(X)


def test_CGInverseLinearOperator_damped_GGN(inv_case, delta_rel: float = 2e-2):
    """Test matrix multiplication with the inverse damped GGN with CG.

    Args:
        inv_case: Tuple of model, loss function, parameters, data, batch size getter.
        delta_rel: Relative damping factor that is multiplied onto the average trace
            to obtain the damping value.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(inv_case, float64)
    (dev,), (dt,) = {p.device for p in params}, {p.dtype for p in params}

    GGN_naive = functorch_ggn(
        model_func, loss_func, params, data, input_key="x"
    ).detach()
    # add damping proportional to average trace
    delta = delta_rel * GGN_naive.diag().mean().item()
    damping = delta * IdentityLinearOperator([p.shape for p in params], dev, dt)
    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    inv_GGN_naive = inv(GGN_naive + delta * eye_like(GGN_naive))

    # specify tolerance and turn off internal damping to get solution with accuracy
    inv_GGN = CGInverseLinearOperator(GGN + damping, eps=0, tolerance=1e-5)
    compare_consecutive_matmats(inv_GGN)
    # Need to use larger tolerances on GPU, despite float64
    atol, rtol = (1e-8, 1e-5) if "cpu" in str(dev) else (1e-7, 1e-4)
    compare_matmat(inv_GGN, inv_GGN_naive, atol=atol, rtol=rtol)


def test_LSMRInverseLinearOperator_damped_GGN(inv_case, delta: float = 2e-2):
    """Test matrix multiplication with the inverse damped GGN with LSMR."""
    model_func, loss_func, params, data, batch_size_fn = change_dtype(inv_case, float64)
    (dev,), (dt,) = {p.device for p in params}, {p.dtype for p in params}

    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    damping = delta * IdentityLinearOperator([p.shape for p in params], dev, dt)

    # set hyperparameters such that LSMR is accurate enough
    inv_GGN = LSMRInverseLinearOperator(
        GGN + damping, atol=0, btol=0, maxiter=2 * GGN.shape[0]
    )
    inv_GGN_naive = inv(
        functorch_ggn(model_func, loss_func, params, data, input_key="x").detach()
        + delta * eye_like(GGN)
    )

    compare_consecutive_matmats(inv_GGN)
    compare_matmat(inv_GGN, inv_GGN_naive)


def test_NeumannInverseLinearOperator_toy():
    """Test NeumannInverseLinearOperator on a toy example.

    The example is from
    https://en.wikipedia.org/w/index.php?title=Neumann_series&oldid=1131424698#Example
    """
    manual_seed(1234)
    A = Tensor([
        [0.0, 1.0 / 2.0, 1.0 / 4.0],
        [5.0 / 7.0, 0.0, 1.0 / 7.0],
        [3.0 / 10.0, 3.0 / 5.0, 0.0],
    ]).double()
    A += eye_like(A)
    # eigenvalues of A: [1.82122892 0.47963837 0.69913271]

    inv_A = inv(A)
    inv_A_neumann = NeumannInverseLinearOperator(
        TensorLinearOperator(A), num_terms=1_000
    )

    tols = {"rtol": 1e-3, "atol": 1e-5}
    compare_consecutive_matmats(inv_A_neumann)
    compare_matmat(inv_A_neumann, inv_A, **tols)

    # If we double the matrix, the Neumann series won't converge anymore ...
    B = 2 * A
    inv_B = inv(B)
    inv_B_neumann = NeumannInverseLinearOperator(
        TensorLinearOperator(B), num_terms=1_000
    )

    # ... therefore, we should get NaNs during the iteration
    with raises(ValueError):
        compare_consecutive_matmats(inv_B_neumann)

    # ... but if we scale the matrix back internally, the Neumann series converges
    inv_B_neumann = NeumannInverseLinearOperator(
        TensorLinearOperator(B), num_terms=1_000, scale=0.5
    )

    compare_consecutive_matmats(inv_B_neumann)
    compare_matmat(inv_B_neumann, inv_B, **tols)


def test_NeumannInverseLinearOperator_preconditioner_toy():
    """Test preconditioned Neumann inverse on a toy SPD matrix."""
    manual_seed(0)
    A = Tensor(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    ).double()
    inv_A = inv(A)

    # With an exact preconditioner and zero Neumann updates,
    # the approximation should already be exact: A^{-1} ≈ P.
    inv_A_neumann = NeumannInverseLinearOperator(
        TensorLinearOperator(A),
        num_terms=0,
        preconditioner=TensorLinearOperator(inv_A),
    )

    compare_consecutive_matmats(inv_A_neumann)
    compare_matmat(inv_A_neumann, inv_A, rtol=1e-10, atol=1e-10)


def test_CGInverseLinearOperator_preconditioner_toy():
    """Test that CGInverseLinearOperator wires preconditioners into linear_cg."""
    manual_seed(0)
    A = Tensor(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    ).double()
    inv_A = inv(A)
    A_linop = TensorLinearOperator(A)

    inv_A_cg = CGInverseLinearOperator(
        A_linop,
        preconditioner=TensorLinearOperator(inv_A),
        max_iter=1,
        max_tridiag_iter=1,
        tolerance=0,
        eps=0,
    )

    compare_consecutive_matmats(inv_A_cg)
    compare_matmat(inv_A_cg, inv_A, rtol=1e-8, atol=1e-10)


def test_inverse_preconditioner_shape_mismatch_raises():
    """Test both inverse operators reject incompatible preconditioner shapes."""
    manual_seed(0)
    A = Tensor(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    ).double()

    with raises(ValueError, match="Preconditioner must have same in-/out-shapes"):
        CGInverseLinearOperator(
            TensorLinearOperator(A), preconditioner=TensorLinearOperator(rand(2, 2))
        )

    with raises(ValueError, match="Preconditioner must have same in-/out-shapes"):
        NeumannInverseLinearOperator(
            TensorLinearOperator(A), preconditioner=TensorLinearOperator(rand(2, 2))
        )


def test_CGInverseLinearOperator_preconditioner_efficiency_cpu():
    """CPU stress test: preconditioning should reduce CG work at equal quality."""
    manual_seed(0)
    n, num_rhs = 192, 16

    # Diagonal-dominant SPD system with broad spectrum.
    D = rand(n, dtype=float64) * 1_000 + 1.0
    U = randn(n, n, dtype=float64)
    A = 0.01 * (U + U.T) / 2.0
    A = A + D.diag()

    X = rand(n, num_rhs, dtype=float64)

    # Jacobi preconditioner P = diag(A)^{-1}
    P = (1.0 / A.diag()).diag()

    A_no_pre = _CountingTensorLinearOperator(A)
    inv_no_pre = CGInverseLinearOperator(
        A_no_pre,
        eps=0,
        tolerance=1e-7,
        max_iter=400,
        max_tridiag_iter=400,
    )
    Y_no_pre = inv_no_pre @ X

    A_with_pre = _CountingTensorLinearOperator(A)
    inv_with_pre = CGInverseLinearOperator(
        A_with_pre,
        preconditioner=TensorLinearOperator(P),
        eps=0,
        tolerance=1e-7,
        max_iter=400,
        max_tridiag_iter=400,
    )
    Y_with_pre = inv_with_pre @ X

    # Comparable solve quality.
    rel_res_no_pre = (A @ Y_no_pre - X).norm() / X.norm()
    rel_res_with_pre = (A @ Y_with_pre - X).norm() / X.norm()
    assert rel_res_no_pre < 1e-5
    assert rel_res_with_pre < 1e-5

    # Efficiency target: fewer A @ v applications with preconditioning.
    assert A_with_pre.num_matmats < A_no_pre.num_matmats


def test_NeumannInverseLinearOperator_preconditioner_efficiency_cpu():
    """CPU stress test: exact preconditioner should avoid A matmats in Neumann."""
    manual_seed(0)
    n, num_rhs = 128, 8
    M = randn(n, n, dtype=float64)
    A = M.T @ M + 0.1 * eye_like(M)
    inv_A = inv(A)
    X = rand(n, num_rhs, dtype=float64)

    A_no_pre = _CountingTensorLinearOperator(A)
    inv_no_pre = NeumannInverseLinearOperator(A_no_pre, num_terms=200, scale=1e-3)
    _ = inv_no_pre @ X

    A_with_pre = _CountingTensorLinearOperator(A)
    inv_with_pre = NeumannInverseLinearOperator(
        A_with_pre,
        num_terms=0,
        preconditioner=TensorLinearOperator(inv_A),
    )
    Y_with_pre = inv_with_pre @ X

    # Exact (up to numerical precision) solve and less A-work.
    assert ((A @ Y_with_pre - X).norm() / X.norm()) < 1e-10
    assert A_with_pre.num_matmats < A_no_pre.num_matmats


@mark.cuda
@mark.skipif(not cuda.is_available(), reason="CUDA not available")
def test_CGInverseLinearOperator_preconditioner_toy_cuda():
    """CUDA test for CG preconditioner support."""
    manual_seed(0)
    A = Tensor(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    ).double().cuda()
    inv_A = inv(A)

    inv_A_cg = CGInverseLinearOperator(
        TensorLinearOperator(A),
        preconditioner=TensorLinearOperator(inv_A),
        max_iter=1,
        max_tridiag_iter=1,
        tolerance=0,
        eps=0,
    )

    compare_consecutive_matmats(inv_A_cg)
    compare_matmat(inv_A_cg, inv_A, rtol=1e-7, atol=1e-9)


@mark.cuda
@mark.skipif(not cuda.is_available(), reason="CUDA not available")
def test_NeumannInverseLinearOperator_preconditioner_toy_cuda():
    """CUDA test for preconditioned Neumann inverse."""
    manual_seed(0)
    A = Tensor(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ]
    ).double().cuda()
    inv_A = inv(A)

    inv_A_neumann = NeumannInverseLinearOperator(
        TensorLinearOperator(A),
        num_terms=0,
        preconditioner=TensorLinearOperator(inv_A),
    )

    compare_consecutive_matmats(inv_A_neumann)
    compare_matmat(inv_A_neumann, inv_A, rtol=1e-9, atol=1e-10)


@mark.cuda
@mark.skipif(not cuda.is_available(), reason="CUDA not available")
def test_NeumannInverseLinearOperator_scale_toy_cuda():
    """CUDA test for scaled Neumann convergence rescue."""
    manual_seed(0)
    A = Tensor(
        [
            [0.0, 1.0 / 2.0, 1.0 / 4.0],
            [5.0 / 7.0, 0.0, 1.0 / 7.0],
            [3.0 / 10.0, 3.0 / 5.0, 0.0],
        ]
    ).double().cuda()
    A = A + eye_like(A)
    B = 2 * A
    inv_B = inv(B)

    inv_B_neumann = NeumannInverseLinearOperator(
        TensorLinearOperator(B), num_terms=1_000, scale=0.5
    )
    compare_consecutive_matmats(inv_B_neumann)
    compare_matmat(inv_B_neumann, inv_B, rtol=1e-3, atol=1e-5)


@mark.cuda
@mark.skipif(not cuda.is_available(), reason="CUDA not available")
def test_CGInverseLinearOperator_damped_GGN_cuda(inv_case, delta_rel: float = 2e-2):
    """CUDA test for damped GGN inverse with CG and optional preconditioner API."""
    model_func, loss_func, params, data, batch_size_fn = change_dtype(inv_case, float64)
    # move test case to CUDA
    model_func = model_func.cuda()
    loss_func = loss_func.cuda()
    params = list(model_func.parameters())
    data = [
        (({"x": X["x"].cuda()} if isinstance(X, dict) else X.cuda()), y.cuda())
        for X, y in data
    ]

    GGN_naive = functorch_ggn(
        model_func, loss_func, params, data, input_key="x"
    ).detach()
    delta = delta_rel * GGN_naive.diag().mean().item()
    damping = delta * IdentityLinearOperator([p.shape for p in params], "cuda", float64)

    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    inv_GGN_naive = inv(GGN_naive + delta * eye_like(GGN_naive))

    inv_GGN = CGInverseLinearOperator(GGN + damping, eps=0, tolerance=1e-5)
    compare_consecutive_matmats(inv_GGN)
    compare_matmat(inv_GGN, inv_GGN_naive, atol=1e-7, rtol=1e-4)
