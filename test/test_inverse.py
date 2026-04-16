"""Contains tests for ``curvlinops/inverse``."""

from pytest import mark, raises
from torch import Tensor, float64, manual_seed, randn
from torch.linalg import inv
from torch.nn import Linear, MSELoss

from curvlinops import (
    CGInverseLinearOperator,
    EKFACLinearOperator,
    FisherType,
    GGNLinearOperator,
    KFACLinearOperator,
    LSMRInverseLinearOperator,
    NeumannInverseLinearOperator,
)
from curvlinops.examples import IdentityLinearOperator, TensorLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.diag import DiagonalLinearOperator
from torch.nn import Linear, MSELoss
from test.utils import (
    change_dtype,
    compare_consecutive_matmats,
    compare_matmat,
    eye_like,
)


@mark.parametrize("precondition", [False, True], ids=["", "preconditioned"])
def test_CGInverseLinearOperator_damped_GGN(
    inv_case, precondition: bool, delta_rel: float = 2e-2
):
    """Test matrix multiplication with the inverse damped GGN with CG.

    Args:
        inv_case: Tuple of model, loss function, parameters, data, batch size getter.
        precondition: Whether to use a Jacobi preconditioner.
        delta_rel: Relative damping factor that is multiplied onto the average trace
            to obtain the damping value.
    """
    model_func, loss_func, params, data, batch_size_fn = change_dtype(inv_case, float64)
    ((dev, dt),) = {(p.device, p.dtype) for p in params.values()}

    GGN_naive = functorch_ggn(
        model_func, loss_func, params, data, input_key="x"
    ).detach()
    # add damping proportional to average trace
    delta = delta_rel * GGN_naive.diag().mean().item()
    damping = delta * IdentityLinearOperator(
        [p.shape for p in params.values()], dev, dt
    )
    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    damped_GGN_naive = GGN_naive + delta * eye_like(GGN_naive)
    inv_GGN_naive = inv(damped_GGN_naive)

    # specify tolerance and turn off internal damping to get solution with accuracy
    jacobi_preconditioner = DiagonalLinearOperator([
        damped_GGN_naive.diag().reciprocal()
    ])
    cg_kwargs = {"eps": 0, "tolerance": 1e-8}
    preconditioner = None if not precondition else jacobi_preconditioner.__matmul__
    inv_GGN = CGInverseLinearOperator(
        GGN + damping, **cg_kwargs, preconditioner=preconditioner
    )
    compare_consecutive_matmats(inv_GGN)
    # Need to use larger tolerances on GPU, despite float64
    atol, rtol = (5e-8, 5e-5) if "cpu" in str(dev) else (5e-7, 5e-4)
    compare_matmat(inv_GGN, inv_GGN_naive, atol=atol, rtol=rtol)


def test_LSMRInverseLinearOperator_damped_GGN(inv_case, delta: float = 2e-2):
    """Test matrix multiplication with the inverse damped GGN with LSMR."""
    model_func, loss_func, params, data, batch_size_fn = change_dtype(inv_case, float64)
    ((dev, dt),) = {(p.device, p.dtype) for p in params.values()}

    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    damping = delta * IdentityLinearOperator(
        [p.shape for p in params.values()], dev, dt
    )

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


def test_KFAC_EKFAC_preconditioners_for_CG_and_Neumann(delta: float = 0.0):
    """Test KFAC and EKFAC inverses as exact preconditioners on linear regression.

    For a single linear layer with MSE loss, the GGN coincides with the Hessian, and
    both KFAC and EKFAC are exact. Their damped inverses can therefore be used as
    exact preconditioners for CG and Neumann.
    """
    manual_seed(1234)
    model = Linear(3, 2, bias=False).double()
    loss_func = MSELoss(reduction="mean")
    X = randn(6, 3).double()
    y = randn(6, 2).double()
    data = [(X, y)]
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}

    GGN_naive = functorch_ggn(model, loss_func, params, data).detach()
    damped_GGN_naive = GGN_naive + delta * eye_like(GGN_naive)
    inv_GGN_naive = inv(damped_GGN_naive)
    inv_GGN_naive_linop = TensorLinearOperator(inv_GGN_naive)

    GGN = GGNLinearOperator(model, loss_func, params, data)
    damping = delta * IdentityLinearOperator(
        [p.shape for n, p in params.items()], X.device, X.dtype
    )

    KFAC = KFACLinearOperator(
        model, loss_func, params, data, fisher_type=FisherType.TYPE2
    )
    EKFAC = EKFACLinearOperator(
        model, loss_func, params, data, fisher_type=FisherType.TYPE2
    )
    inv_KFAC = KFAC.inverse(damping=delta, use_exact_damping=True)
    inv_EKFAC = EKFAC.inverse(damping=delta)

    # check that for linear regression, KFAC and EKFAC are exact.
    for inv_preconditioner in [inv_GGN_naive_linop, inv_KFAC, inv_EKFAC]:
        compare_consecutive_matmats(inv_preconditioner)
        compare_matmat(inv_preconditioner, inv_GGN_naive)

    preconditioners = [
        (inv_GGN_naive_linop, {"num_terms": 0}),
        (inv_KFAC, {"num_terms": 0}),
        (inv_EKFAC, {"num_terms": 0}),
    ]
    inverse_constructors = [
        (
            CGInverseLinearOperator,
            lambda preconditioner, _: {
                "eps": 0,
                "tolerance": 1e-8,
                "preconditioner": preconditioner,
            },
        ),
        (
            NeumannInverseLinearOperator,
            lambda preconditioner, neumann_kwargs: {
                **neumann_kwargs,
                "preconditioner": preconditioner,
            },
        ),
    ]

    for inv_preconditioner, neumann_kwargs in preconditioners:
        preconditioner = inv_preconditioner.__matmul__
        for inverse_cls, kwargs_fn in inverse_constructors:
            inv_GGN = inverse_cls(
                GGN + damping, **kwargs_fn(preconditioner, neumann_kwargs)
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
    https://student.cs.uwaterloo.ca/~cs475/CS475-Lecture-Notes.pdf page 78-82.
    """
    manual_seed(1234)
    A = Tensor([
        [5.0, 1.0, 1.0],
        [1.0, 4.0, 1.0],
        [1.0, 1.0, 3.0],
    ]).double()
    A_linop = TensorLinearOperator(A)

    theta = 0.3

    inv_A = inv(A)
    inv_A_neumann = NeumannInverseLinearOperator(A_linop, num_terms=1_000)
    inv_A_neumann_20terms = NeumannInverseLinearOperator(A_linop, num_terms=20)
    inv_A_neumann_scaled_20terms = NeumannInverseLinearOperator(
        A_linop, num_terms=20, scale=theta
    )
    inv_A_neumann_scaled_100terms = NeumannInverseLinearOperator(
        A_linop, num_terms=100, scale=theta
    )

    # Directly applying the Neumann series will diverge.
    with raises(ValueError, match="Detected NaNs after application of"):
        compare_consecutive_matmats(inv_A_neumann)

    tols = {"rtol": 1e-3, "atol": 1e-5}
    # Without preconditioning, 20 terms are not enough to get a good approximation.
    with raises(AssertionError):
        compare_matmat(inv_A_neumann_20terms, inv_A, **tols)
    # No matter scaled or not, only 20 terms with scaling is not enough to get a good approximation.
    with raises(AssertionError):
        compare_matmat(inv_A_neumann_scaled_20terms, inv_A, **tols)
    # But 100 terms with scaling is enough to get a good approximation.
    compare_matmat(inv_A_neumann_scaled_100terms, inv_A, **tols)

    # We can use Richardson preconditioner, then we don't need scale
    preconditioner_richardson = (
        IdentityLinearOperator(A_linop._in_shape, A.device, A.dtype) * theta
    )

    # Jacobi preconditioner, then can converge with only 20 terms
    preconditioner_jacobi = DiagonalLinearOperator([A.diag().reciprocal()])

    # Gauss-Seidel preconditioner, then can converge with only 20 terms
    L = A.tril(-1)
    D = A.diag().diag()
    preconditioner_gauss_seidel = TensorLinearOperator((L + D).inverse())

    for num_terms, preconditioner in [
        (100, preconditioner_richardson),
        (20, preconditioner_jacobi),
        (20, preconditioner_gauss_seidel),
    ]:
        inv_A_neumann_preconditioned = NeumannInverseLinearOperator(
            A_linop, num_terms=num_terms, preconditioner=preconditioner.__matmul__
        )
        compare_consecutive_matmats(inv_A_neumann_preconditioned)
        compare_matmat(inv_A_neumann_preconditioned, inv_A, **tols)


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
