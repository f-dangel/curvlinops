"""Contains tests for ``curvlinops/inverse``."""

import torch
from numpy import array, eye, random
from numpy.linalg import eigh, inv
from pytest import mark, raises
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator

from curvlinops import (
    CGInverseLinearOperator,
    GGNLinearOperator,
    KFACInverseLinearOperator,
    KFACLinearOperator,
    NeumannInverseLinearOperator,
)
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.examples.utils import report_nonclose


def test_CG_inverse_damped_GGN_matvec(case, delta: float = 2e-2):
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


@mark.parametrize("cache", [True, False], ids=["cached", "uncached"])
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
def test_KFAC_inverse_damped_matmat(
    case, cache: bool, exclude: str, separate_weight_and_bias: bool, delta: float = 1e-2
):
    """Test matrix-matrix multiplication by an inverse damped KFAC approximation."""
    model_func, loss_func, params, data = case

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model_func.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    loss_average = "batch" if loss_func.reduction == "mean" else None
    KFAC = KFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        loss_average=loss_average,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    KFAC._compute_kfac()

    # add damping manually
    for aaT in KFAC._input_covariances.values():
        aaT.add_(torch.eye(aaT.shape[0], device=aaT.device), alpha=delta)
    for ggT in KFAC._gradient_covariances.values():
        ggT.add_(torch.eye(ggT.shape[0], device=ggT.device), alpha=delta)
    inv_KFAC_naive = torch.inverse(torch.as_tensor(KFAC @ eye(KFAC.shape[0])))

    # remove damping and pass it on as an argument instead
    for aaT in KFAC._input_covariances.values():
        aaT.sub_(torch.eye(aaT.shape[0], device=aaT.device), alpha=delta)
    for ggT in KFAC._gradient_covariances.values():
        ggT.sub_(torch.eye(ggT.shape[0], device=ggT.device), alpha=delta)
    inv_KFAC = KFACInverseLinearOperator(KFAC, damping=(delta, delta), cache=cache)

    num_vectors = 2
    X = random.rand(KFAC.shape[1], num_vectors)
    report_nonclose(inv_KFAC @ X, inv_KFAC_naive @ X, rtol=5e-2)

    assert inv_KFAC._cache == cache
    if cache:
        # test that the cache is not empty
        assert len(inv_KFAC._inverse_input_covariances) > 0
        assert len(inv_KFAC._inverse_gradient_covariances) > 0
    else:
        # test that the cache is empty
        assert len(inv_KFAC._inverse_input_covariances) == 0
        assert len(inv_KFAC._inverse_gradient_covariances) == 0
