"""Contains tests for ``curvlinops/inverse``."""

from pytest import raises
from torch import Tensor, float64, manual_seed, cat
from torch.linalg import inv, eigvalsh

from curvlinops import (
    CGInverseLinearOperator,
    GGNLinearOperator,
    LSMRInverseLinearOperator,
    NeumannInverseLinearOperator,
)
from curvlinops.examples import IdentityLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from test.test__torch_base import (
    TensorLinearOperator,
    PyTorchLinearOperator,
)
from test.utils import (
    change_dtype,
    compare_consecutive_matmats,
    compare_matmat,
    eye_like,
)


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

def test_NeumannInverseLinearOperator_preconditioner():
    """Test NeumannInverseLinearOperator with a preconditioner on a toy example.

    We consider three different preconditioners for matrix:
    1. Richardson iteration: P = I / theta, where theta is a scalar, this is equivalent to the `scale` argument of NeumannInverseLinearOperator.
    2. Jacobi Iteration: P = diag(A)^{-1}, where diag(A) is the diagonal of A.
    3. Gauss-Seidel Iteration: P = (L + D)^{-1}, where L is the lower triangular part of A and D is the diagonal of A.
    
    The test is inspired from
    https://student.cs.uwaterloo.ca/~cs475/CS475-Lecture-Notes.pdf
    """
    manual_seed(1234)
    A = Tensor([
        [5.0, 1.0, 1.0],
        [1.0, 4.0, 1.0],
        [1.0, 1.0, 3.0],
    ]).double()


    inv_A = inv(A)
    inv_A_neumann = NeumannInverseLinearOperator(
        TensorLinearOperator(A), num_terms=1_000
    )
    inv_A_neumann_scaled_20terms = NeumannInverseLinearOperator(
        TensorLinearOperator(A), num_terms=20, scale=0.3
    )
    inv_A_neumann_scaled_100terms = NeumannInverseLinearOperator(
        TensorLinearOperator(A), num_terms=100, scale=0.3
    )

    # Directly applying the Neumann sereis will diverge.
    with raises(ValueError):
        compare_consecutive_matmats(inv_A_neumann)

    tols = {"rtol": 1e-3, "atol": 1e-5}
    # Only 20 terms with scaling is not enough to get a good approximation.
    with raises(AssertionError):
        compare_matmat(inv_A_neumann_scaled_20terms, inv_A, **tols)
    # But 100 terms with scaling is enough to get a good approximation.
    compare_matmat(inv_A_neumann_scaled_100terms, inv_A, **tols)

    # We can use Richardson preconditioner, then we don't need scale
    theta = 0.3
    A_linop = TensorLinearOperator(A)
    preconditioner_richardson = IdentityLinearOperator(A_linop._in_shape, A.device, A.dtype) * theta
    inv_A_neumann_richardson = NeumannInverseLinearOperator(
        TensorLinearOperator(A), num_terms=100, preconditioner=preconditioner_richardson
    )
    compare_consecutive_matmats(inv_A_neumann_richardson)
    compare_matmat(inv_A_neumann_richardson, inv_A, **tols)

    # Jacobi preconditioner, then can converge with only 20 terms
    diag_A = A.diag()
    preconditioner_jacobi = TensorLinearOperator(diag_A.reciprocal().diag())
    inv_A_neumann_jacobi = NeumannInverseLinearOperator(
        TensorLinearOperator(A), num_terms=20, preconditioner=preconditioner_jacobi
    )
    compare_consecutive_matmats(inv_A_neumann_jacobi)
    compare_matmat(inv_A_neumann_jacobi, inv_A, **tols)

    # Gauss-Seidel preconditioner, then can converge with only 20 terms
    L = A.tril(-1)
    D = A.diag().diag()
    preconditioner_gauss_seidel = TensorLinearOperator((L + D).inverse())
    inv_A_neumann_gauss_seidel = NeumannInverseLinearOperator(
        TensorLinearOperator(A), num_terms=20, preconditioner=preconditioner_gauss_seidel
    )
    compare_consecutive_matmats(inv_A_neumann_gauss_seidel) 
    compare_matmat(inv_A_neumann_gauss_seidel, inv_A, **tols)


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



