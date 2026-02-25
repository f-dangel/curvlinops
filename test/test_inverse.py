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


class _DenseMatrixStructuredLinearOperator(PyTorchLinearOperator):
    """Apply a dense matrix on the flattened space with structured tensor IO."""

    def __init__(
        self,
        A: Tensor,
        in_shape: list[tuple[int, ...]],
        out_shape: list[tuple[int, ...]],
    ):
        super().__init__(in_shape, out_shape)
        self._A = A

    @property
    def device(self):
        return self._A.device

    @property
    def dtype(self):
        return self._A.dtype

    def _adjoint(self) -> "_DenseMatrixStructuredLinearOperator":
        return _DenseMatrixStructuredLinearOperator(
            self._A.conj().T, self._out_shape, self._in_shape
        )

    def _matmat(self, X: list[Tensor]) -> list[Tensor]:
        num_vecs = X[0].shape[-1]
        X_flat = cat([x.reshape(-1, num_vecs) for x in X], dim=0)
        AX_flat = self._A @ X_flat
        return [
            y.reshape(*s, num_vecs)
            for y, s in zip(AX_flat.split(self._out_shape_flat, dim=0), self._out_shape)
        ]


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


def test_CGInverseLinearOperator_preconditioner_damped_GGN(inv_case, delta_rel: float = 2e-2):
    """Test CG preconditioners on damped GGN: identity=no-pre, exact=exact inverse."""
    model_func, loss_func, params, data, batch_size_fn = change_dtype(inv_case, float64)
    (dev,), (dt,) = {p.device for p in params}, {p.dtype for p in params}

    GGN_naive = functorch_ggn(
        model_func, loss_func, params, data, input_key="x"
    ).detach()
    delta = delta_rel * GGN_naive.diag().mean().item()
    damping = delta * IdentityLinearOperator([p.shape for p in params], dev, dt)
    A = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    ) + damping
    A_naive = GGN_naive + delta * eye_like(GGN_naive)
    inv_A_naive = inv(A_naive)

    inv_A_cg = CGInverseLinearOperator(
        A, eps=0, tolerance=0, max_iter=10, max_tridiag_iter=10
    )
    inv_A_cg_id = CGInverseLinearOperator(
        A,
        preconditioner=IdentityLinearOperator([p.shape for p in params], dev, dt),
        eps=0,
        tolerance=0,
        max_iter=10,
        max_tridiag_iter=10,
    )
    inv_A_cg_exact = CGInverseLinearOperator(
        A,
        preconditioner=_DenseMatrixStructuredLinearOperator(
            inv_A_naive, A._in_shape, A._out_shape
        ),
        eps=0,
        tolerance=0,
        max_iter=1,
        max_tridiag_iter=1,
    )
    atol, rtol = (1e-8, 1e-5) if "cpu" in str(dev) else (1e-7, 1e-4)
    compare_consecutive_matmats(inv_A_cg_id)
    compare_consecutive_matmats(inv_A_cg_exact)
    compare_matmat(inv_A_cg_exact, inv_A_naive, atol=atol, rtol=rtol)
    compare_matmat(inv_A_cg_id, inv_A_cg, atol=atol, rtol=rtol)


def test_NeumannInverseLinearOperator_preconditioner_damped_GGN(inv_case, delta_rel: float = 2e-2):
    """Test Neumann preconditioners on damped GGN: identity=no-pre, exact=exact inverse."""
    model_func, loss_func, params, data, batch_size_fn = change_dtype(inv_case, float64)
    (dev,), (dt,) = {p.device for p in params}, {p.dtype for p in params}

    GGN_naive = functorch_ggn(
        model_func, loss_func, params, data, input_key="x"
    ).detach()
    delta = delta_rel * GGN_naive.diag().mean().item()
    damping = delta * IdentityLinearOperator([p.shape for p in params], dev, dt)
    A = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    ) + damping
    A_naive = GGN_naive + delta * eye_like(GGN_naive)
    inv_A_naive = inv(A_naive)

    # Safe scale for the undamped Neumann/Richardson iteration on SPD matrix A_naive.
    scale = 0.9 / eigvalsh(A_naive).max().item()

    inv_A_neumann = NeumannInverseLinearOperator(A, num_terms=5, scale=scale)
    inv_A_neumann_id = NeumannInverseLinearOperator(
        A,
        num_terms=5,
        scale=scale,
        preconditioner=IdentityLinearOperator([p.shape for p in params], dev, dt),
    )
    inv_A_neumann_exact = NeumannInverseLinearOperator(
        A,
        num_terms=0,
        scale=1.0,
        preconditioner=_DenseMatrixStructuredLinearOperator(
            inv_A_naive, A._in_shape, A._out_shape
        ),
    )
    atol, rtol = (1e-8, 1e-5) if "cpu" in str(dev) else (1e-7, 1e-4)
    compare_consecutive_matmats(inv_A_neumann_id)
    compare_consecutive_matmats(inv_A_neumann_exact)
    compare_matmat(inv_A_neumann_exact, inv_A_naive, atol=atol, rtol=rtol)
    compare_matmat(inv_A_neumann_id, inv_A_neumann, atol=atol, rtol=rtol)


