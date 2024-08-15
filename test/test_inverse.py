"""Contains tests for ``curvlinops/inverse``."""

import os
from math import sqrt
from test.utils import cast_input, compare_state_dicts
from typing import Iterable, List, Tuple, Union

import torch
from einops import rearrange
from numpy import array, eye, float64, random
from numpy.linalg import eigh, inv
from pytest import mark, raises
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator
from torch.nn import CrossEntropyLoss, Module, MSELoss, Parameter

from curvlinops import (
    CGInverseLinearOperator,
    GGNLinearOperator,
    KFACInverseLinearOperator,
    KFACLinearOperator,
    LSMRInverseLinearOperator,
    NeumannInverseLinearOperator,
)
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.examples.utils import report_nonclose

KFAC_MIN_DAMPING = 1e-8


def test_CG_inverse_damped_GGN_matvec(inv_case, delta: float = 2e-2):
    """Test matrix-vector multiplication by the inverse damped GGN with CG."""
    model_func, loss_func, params, data, batch_size_fn = inv_case

    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    damping = aslinearoperator(delta * sparse.eye(GGN.shape[0]))

    inv_GGN = CGInverseLinearOperator(GGN + damping)
    inv_GGN_functorch = inv(
        functorch_ggn(model_func, loss_func, params, data, input_key="x")
        .detach()
        .cpu()
        .numpy()
        + delta * eye(GGN.shape[1])
    )

    x = random.rand(GGN.shape[1])
    report_nonclose(inv_GGN @ x, inv_GGN_functorch @ x, rtol=5e-3, atol=1e-5)


def test_CG_inverse_damped_GGN_matmat(inv_case, delta: float = 1e-2, num_vecs: int = 3):
    """Test matrix-matrix multiplication by the inverse damped GGN with CG."""
    model_func, loss_func, params, data, batch_size_fn = inv_case

    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    damping = aslinearoperator(delta * sparse.eye(GGN.shape[0]))

    inv_GGN = CGInverseLinearOperator(GGN + damping)
    inv_GGN_functorch = inv(
        functorch_ggn(model_func, loss_func, params, data, input_key="x")
        .detach()
        .cpu()
        .numpy()
        + delta * eye(GGN.shape[1])
    )

    X = random.rand(GGN.shape[1], num_vecs)
    report_nonclose(inv_GGN @ X, inv_GGN_functorch @ X, rtol=5e-3, atol=1e-5)


def test_LSMR_inverse_damped_GGN_matvec(inv_case, delta: float = 2e-2):
    """Test matrix-vector multiplication by the inverse damped GGN with LSMR."""
    model_func, loss_func, params, data, batch_size_fn = inv_case

    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    damping = aslinearoperator(delta * sparse.eye(GGN.shape[0]))

    inv_GGN = LSMRInverseLinearOperator(GGN + damping)
    # set hyperparameters such that LSMR is accurate enough
    inv_GGN.set_lsmr_hyperparameters(atol=0, btol=0, conlim=0, maxiter=2 * GGN.shape[0])
    inv_GGN_functorch = inv(
        functorch_ggn(model_func, loss_func, params, data, input_key="x")
        .detach()
        .cpu()
        .numpy()
        + delta * eye(GGN.shape[1])
    )

    x = random.rand(GGN.shape[1])
    report_nonclose(inv_GGN @ x, inv_GGN_functorch @ x, rtol=5e-3, atol=1e-5)


def test_LSMR_inverse_damped_GGN_matmat(
    inv_case, delta: float = 1e-2, num_vecs: int = 3
):
    """Test matrix-matrix multiplication by the inverse damped GGN with LSMR."""
    model_func, loss_func, params, data, batch_size_fn = inv_case

    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    damping = aslinearoperator(delta * sparse.eye(GGN.shape[0]))

    inv_GGN = LSMRInverseLinearOperator(GGN + damping)
    # set hyperparameters such that LSMR is accurate enough
    inv_GGN.set_lsmr_hyperparameters(atol=0, btol=0, conlim=0, maxiter=2 * GGN.shape[0])
    inv_GGN_functorch = inv(
        functorch_ggn(model_func, loss_func, params, data, input_key="x")
        .detach()
        .cpu()
        .numpy()
        + delta * eye(GGN.shape[1])
    )

    X = random.rand(GGN.shape[1], num_vecs)
    report_nonclose(inv_GGN @ X, inv_GGN_functorch @ X, rtol=5e-3, atol=1e-5)


def test_Neumann_inverse_damped_GGN_matvec(inv_case, delta: float = 1e-2):
    """Test matrix-vector multiplication by the inverse damped GGN with Neumann."""
    model_func, loss_func, params, data, batch_size_fn = inv_case

    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    damping = aslinearoperator(delta * sparse.eye(GGN.shape[0]))

    damped_GGN_functorch = functorch_ggn(
        model_func, loss_func, params, data, input_key="x"
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


@mark.parametrize("fisher_type", KFACLinearOperator._SUPPORTED_FISHER_TYPE)
@mark.parametrize("cache", [True, False], ids=["cached", "uncached"])
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
def test_KFAC_inverse_damped_matmat(
    case: Tuple[
        Module,
        Union[MSELoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ],
    fisher_type: str,
    cache: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by an inverse damped KFAC approximation."""
    model_func, loss_func, params, data, batch_size_fn = case
    dtype = torch.float64  # use double precision for better numerical stability
    model_func = model_func.to(dtype=dtype)
    loss_func = loss_func.to(dtype=dtype)
    params = [p.to(dtype=dtype) for p in params]
    data = [
        (
            (cast_input(x, dtype), y)
            if isinstance(loss_func, CrossEntropyLoss)
            else (cast_input(x, dtype), y.to(dtype=dtype))
        )
        for x, y in data
    ]

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model_func.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    KFAC = KFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=fisher_type,
        check_deterministic=False,
    )
    KFAC.dtype = float64
    KFAC._compute_kfac()

    # add damping manually
    for aaT in KFAC._input_covariances.values():
        aaT.add_(
            torch.eye(aaT.shape[0], dtype=aaT.dtype, device=aaT.device), alpha=delta
        )
    for ggT in KFAC._gradient_covariances.values():
        ggT.add_(
            torch.eye(ggT.shape[0], dtype=ggT.dtype, device=ggT.device), alpha=delta
        )
    inv_KFAC_naive = (
        torch.inverse(torch.as_tensor(KFAC @ eye(KFAC.shape[0]))).cpu().numpy()
    )

    # remove damping and pass it on as an argument instead
    for aaT in KFAC._input_covariances.values():
        aaT.sub_(
            torch.eye(aaT.shape[0], dtype=aaT.dtype, device=aaT.device), alpha=delta
        )
    for ggT in KFAC._gradient_covariances.values():
        ggT.sub_(
            torch.eye(ggT.shape[0], dtype=ggT.dtype, device=ggT.device), alpha=delta
        )
    # as a single scalar
    inv_KFAC = KFACInverseLinearOperator(KFAC, damping=delta, cache=cache)
    # and as a tuple
    inv_KFAC_tuple = KFACInverseLinearOperator(
        KFAC, damping=(delta, delta), cache=cache
    )

    num_vectors = 2
    X = random.rand(KFAC.shape[1], num_vectors)
    report_nonclose(inv_KFAC @ X, inv_KFAC_naive @ X)
    report_nonclose(inv_KFAC_tuple @ X, inv_KFAC_naive @ X)
    report_nonclose(inv_KFAC_tuple @ X, inv_KFAC @ X)

    assert inv_KFAC._cache == cache
    if cache:
        # test that the cache is not empty
        assert len(inv_KFAC._inverse_input_covariances) > 0
        assert len(inv_KFAC._inverse_gradient_covariances) > 0
    else:
        # test that the cache is empty
        assert len(inv_KFAC._inverse_input_covariances) == 0
        assert len(inv_KFAC._inverse_gradient_covariances) == 0


@mark.parametrize("cache", [True, False], ids=["cached", "uncached"])
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
def test_KFAC_inverse_heuristically_damped_matmat(  # noqa: C901
    case: Tuple[
        Module,
        Union[MSELoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ],
    cache: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by an inverse (heuristically) damped KFAC
    approximation."""
    model_func, loss_func, params, data, batch_size_fn = case
    dtype = torch.float64  # use double precision for better numerical stability
    model_func = model_func.to(dtype=dtype)
    loss_func = loss_func.to(dtype=dtype)
    params = [p.to(dtype=dtype) for p in params]
    data = [
        (
            (cast_input(x, dtype), y)
            if isinstance(loss_func, CrossEntropyLoss)
            else (cast_input(x, dtype), y.to(dtype=dtype))
        )
        for x, y in data
    ]

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model_func.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    KFAC = KFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        check_deterministic=False,
    )
    KFAC.dtype = float64
    KFAC._compute_kfac()

    # add heuristic damping manually
    heuristic_damping = {}
    for mod_name in KFAC._mapping.keys():
        aaT = KFAC._input_covariances.get(mod_name)
        ggT = KFAC._gradient_covariances.get(mod_name)
        if aaT is not None and ggT is not None:
            aaT_eig_mean = aaT.trace() / aaT.shape[0]
            ggT_eig_mean = ggT.trace() / ggT.shape[0]
            if aaT_eig_mean >= 0.0 and ggT_eig_mean > 0.0:
                sqrt_eig_mean_ratio = (aaT_eig_mean / ggT_eig_mean).sqrt()
                sqrt_damping = sqrt(delta)
                damping_aaT = max(sqrt_damping * sqrt_eig_mean_ratio, KFAC_MIN_DAMPING)
                damping_ggT = max(sqrt_damping / sqrt_eig_mean_ratio, KFAC_MIN_DAMPING)
                heuristic_damping[mod_name] = (damping_aaT, damping_ggT)
            else:
                damping_aaT, damping_ggT = delta, delta
        else:
            damping_aaT, damping_ggT = delta, delta
        if aaT is not None:
            aaT.add_(
                torch.eye(aaT.shape[0], dtype=aaT.dtype, device=aaT.device),
                alpha=damping_aaT,
            )
        if ggT is not None:
            ggT.add_(
                torch.eye(ggT.shape[0], dtype=ggT.dtype, device=ggT.device),
                alpha=damping_ggT,
            )

    # manual heuristically damped inverse
    inv_KFAC_naive = (
        torch.inverse(torch.as_tensor(KFAC @ eye(KFAC.shape[0]))).cpu().numpy()
    )

    # remove heuristic damping
    for mod_name in KFAC._mapping.keys():
        aaT = KFAC._input_covariances.get(mod_name)
        ggT = KFAC._gradient_covariances.get(mod_name)
        damping_aaT, damping_ggT = heuristic_damping.get(mod_name, (delta, delta))
        if aaT is not None:
            aaT.sub_(
                torch.eye(aaT.shape[0], dtype=aaT.dtype, device=aaT.device),
                alpha=damping_aaT,
            )
        if ggT is not None:
            ggT.sub_(
                torch.eye(ggT.shape[0], dtype=ggT.dtype, device=ggT.device),
                alpha=damping_ggT,
            )

    # check that passing a tuple for heuristic damping will fail
    with raises(ValueError):
        inv_KFAC = KFACInverseLinearOperator(
            KFAC, damping=(delta, delta), use_heuristic_damping=True
        )

    # use heuristic damping with KFACInverseLinearOperator
    inv_KFAC = KFACInverseLinearOperator(
        KFAC,
        damping=delta,
        cache=cache,
        use_heuristic_damping=True,
        min_damping=KFAC_MIN_DAMPING,
    )

    num_vectors = 2
    X = random.rand(KFAC.shape[1], num_vectors)
    # test for equivalence
    report_nonclose(inv_KFAC @ X, inv_KFAC_naive @ X)

    assert inv_KFAC._cache == cache
    if cache:
        # test that the cache is not empty
        assert len(inv_KFAC._inverse_input_covariances) > 0
        assert len(inv_KFAC._inverse_gradient_covariances) > 0
    else:
        # test that the cache is empty
        assert len(inv_KFAC._inverse_input_covariances) == 0
        assert len(inv_KFAC._inverse_gradient_covariances) == 0


@mark.parametrize("cache", [True, False], ids=["cached", "uncached"])
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
def test_KFAC_inverse_exactly_damped_matmat(
    case: Tuple[
        Module,
        Union[MSELoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ],
    cache: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by an inverse (exactly) damped KFAC approximation."""
    model_func, loss_func, params, data, batch_size_fn = case
    dtype = torch.float64  # use double precision for better numerical stability
    model_func = model_func.to(dtype=dtype)
    loss_func = loss_func.to(dtype=dtype)
    params = [p.to(dtype=dtype) for p in params]
    data = [
        (
            (cast_input(x, dtype), y)
            if isinstance(loss_func, CrossEntropyLoss)
            else (cast_input(x, dtype), y.to(dtype=dtype))
        )
        for x, y in data
    ]

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model_func.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    KFAC = KFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        check_deterministic=False,
    )
    KFAC.dtype = float64

    # manual exactly damped inverse
    inv_KFAC_naive = (
        torch.inverse(
            KFAC.torch_matmat(
                torch.eye(KFAC.shape[0], dtype=dtype, device=KFAC._device)
            )
            + delta * torch.eye(KFAC.shape[0], dtype=dtype, device=KFAC._device)
        )
        .cpu()
        .numpy()
    )

    # check that passing a tuple for exact damping will fail
    with raises(ValueError):
        inv_KFAC = KFACInverseLinearOperator(
            KFAC, damping=(delta, delta), use_exact_damping=True
        )

    # use exact damping with KFACInverseLinearOperator
    inv_KFAC = KFACInverseLinearOperator(
        KFAC, damping=delta, cache=cache, use_exact_damping=True
    )

    num_vectors = 2
    X = random.rand(KFAC.shape[1], num_vectors)
    # test for equivalence
    report_nonclose(inv_KFAC @ X, inv_KFAC_naive @ X)

    assert inv_KFAC._cache == cache
    if cache:
        # test that the cache is not empty
        assert len(inv_KFAC._inverse_input_covariances) > 0
        assert len(inv_KFAC._inverse_gradient_covariances) > 0
    else:
        # test that the cache is empty
        assert len(inv_KFAC._inverse_input_covariances) == 0
        assert len(inv_KFAC._inverse_gradient_covariances) == 0


def test_KFAC_inverse_damped_torch_matmat(
    case: Tuple[
        Module,
        Union[MSELoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ],
    delta: float = 1e-2,
):
    """Test torch matrix-matrix multiplication by an inverse damped KFAC approximation."""
    model_func, loss_func, params, data, batch_size_fn = case
    dtype = torch.float64  # use double precision for better numerical stability
    model_func = model_func.to(dtype=dtype)
    loss_func = loss_func.to(dtype=dtype)
    params = [p.to(dtype=dtype) for p in params]
    data = [
        (
            (cast_input(x, dtype), y)
            if isinstance(loss_func, CrossEntropyLoss)
            else (cast_input(x, dtype), y.to(dtype=dtype))
        )
        for x, y in data
    ]

    KFAC = KFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        check_deterministic=False,
    )
    KFAC.dtype = float64
    inv_KFAC = KFACInverseLinearOperator(KFAC, damping=(delta, delta))
    device = KFAC._device

    num_vectors = 2
    X = torch.rand(KFAC.shape[1], num_vectors, dtype=dtype, device=device)
    inv_KFAC_X = inv_KFAC.torch_matmat(X)
    assert inv_KFAC_X.dtype == X.dtype
    assert inv_KFAC_X.device == X.device
    assert inv_KFAC_X.shape == (KFAC.shape[0], num_vectors)
    inv_KFAC_X = inv_KFAC_X.cpu().numpy()

    # Test list input format
    x_list = KFAC._torch_preprocess(X)
    inv_KFAC_x_list = inv_KFAC.torch_matmat(x_list)
    inv_KFAC_x_list = torch.cat(
        [rearrange(M, "k ... -> (...) k") for M in inv_KFAC_x_list]
    )
    report_nonclose(inv_KFAC_X, inv_KFAC_x_list.cpu().numpy())

    # Test against multiplication with dense matrix
    identity = torch.eye(inv_KFAC.shape[1], dtype=dtype, device=device)
    inv_KFAC_mat = inv_KFAC.torch_matmat(identity)
    inv_KFAC_mat_x = inv_KFAC_mat @ X
    report_nonclose(inv_KFAC_X, inv_KFAC_mat_x.cpu().numpy())

    # Test against _matmat
    kfac_x_numpy = inv_KFAC @ X.cpu().numpy()
    report_nonclose(inv_KFAC_X, kfac_x_numpy)


def test_KFAC_inverse_damped_torch_matvec(
    case: Tuple[
        Module,
        Union[MSELoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ],
    delta: float = 1e-2,
):
    """Test torch matrix-vector multiplication by an inverse damped KFAC approximation."""
    model_func, loss_func, params, data, batch_size_fn = case
    dtype = torch.float64  # use double precision for better numerical stability
    model_func = model_func.to(dtype=dtype)
    loss_func = loss_func.to(dtype=dtype)
    params = [p.to(dtype=dtype) for p in params]
    data = [
        (
            (cast_input(x, dtype), y)
            if isinstance(loss_func, CrossEntropyLoss)
            else (cast_input(x, dtype), y.to(dtype=dtype))
        )
        for x, y in data
    ]

    KFAC = KFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        check_deterministic=False,
    )
    KFAC.dtype = float64
    inv_KFAC = KFACInverseLinearOperator(KFAC, damping=(delta, delta))
    device = KFAC._device

    x = torch.rand(KFAC.shape[1], dtype=dtype, device=device)
    inv_KFAC_x = inv_KFAC.torch_matvec(x)
    assert inv_KFAC_x.dtype == x.dtype
    assert inv_KFAC_x.device == x.device
    assert inv_KFAC_x.shape == x.shape

    # Test list input format
    # split parameter blocks
    dims = [p.numel() for p in KFAC._params]
    split_x = x.split(dims)
    # unflatten parameter dimension
    assert len(split_x) == len(KFAC._params)
    x_list = [res.reshape(p.shape) for res, p in zip(split_x, KFAC._params)]
    inv_kfac_x_list = inv_KFAC.torch_matvec(x_list)
    inv_kfac_x_list = torch.cat([rearrange(M, "... -> (...)") for M in inv_kfac_x_list])
    report_nonclose(inv_KFAC_x.cpu().numpy(), inv_kfac_x_list.cpu().numpy())

    # Test against multiplication with dense matrix
    identity = torch.eye(inv_KFAC.shape[1], dtype=dtype, device=device)
    inv_KFAC_mat = inv_KFAC.torch_matmat(identity)
    inv_KFAC_mat_x = inv_KFAC_mat @ x
    report_nonclose(inv_KFAC_x.cpu().numpy(), inv_KFAC_mat_x.cpu().numpy())

    # Test against _matmat
    report_nonclose(inv_KFAC @ x.cpu().numpy(), inv_KFAC_x.cpu().numpy())


def test_KFAC_inverse_save_and_load_state_dict():
    """Test that KFACInverseLinearOperator can be saved and loaded from state dict."""
    torch.manual_seed(0)
    batch_size, D_in, D_out = 4, 3, 2
    X = torch.rand(batch_size, D_in)
    y = torch.rand(batch_size, D_out)
    model = torch.nn.Linear(D_in, D_out)

    params = list(model.parameters())
    # create and compute KFAC
    kfac = KFACLinearOperator(
        model,
        MSELoss(reduction="sum"),
        params,
        [(X, y)],
    )

    # create inverse KFAC
    inv_kfac = KFACInverseLinearOperator(
        kfac, damping=1e-2, use_heuristic_damping=True, retry_double_precision=False
    )
    _ = inv_kfac @ eye(kfac.shape[1])  # to trigger inverse computation

    # save state dict
    state_dict = inv_kfac.state_dict()
    torch.save(state_dict, "inv_kfac_state_dict.pt")

    # create new inverse KFAC with different linop input and try to load state dict
    wrong_kfac = KFACLinearOperator(model, CrossEntropyLoss(), params, [(X, y)])
    inv_kfac_wrong = KFACInverseLinearOperator(wrong_kfac)
    with raises(ValueError, match="mismatch"):
        inv_kfac_wrong.load_state_dict(torch.load("inv_kfac_state_dict.pt"))

    # create new inverse KFAC and load state dict
    inv_kfac_new = KFACInverseLinearOperator(kfac)
    inv_kfac_new.load_state_dict(torch.load("inv_kfac_state_dict.pt"))
    # clean up
    os.remove("inv_kfac_state_dict.pt")

    # check that the two inverse KFACs are equal
    compare_state_dicts(inv_kfac.state_dict(), inv_kfac_new.state_dict())
    test_vec = torch.rand(inv_kfac.shape[1])
    report_nonclose(inv_kfac @ test_vec, inv_kfac_new @ test_vec)


def test_KFAC_inverse_from_state_dict():
    """Test that KFACInverseLinearOperator can be created from state dict."""
    torch.manual_seed(0)
    batch_size, D_in, D_out = 4, 3, 2
    X = torch.rand(batch_size, D_in)
    y = torch.rand(batch_size, D_out)
    model = torch.nn.Linear(D_in, D_out)

    params = list(model.parameters())
    # create and compute KFAC
    kfac = KFACLinearOperator(
        model,
        MSELoss(reduction="sum"),
        params,
        [(X, y)],
    )

    # create inverse KFAC and save state dict
    inv_kfac = KFACInverseLinearOperator(
        kfac, damping=1e-2, use_heuristic_damping=True, retry_double_precision=False
    )
    state_dict = inv_kfac.state_dict()

    # create new KFAC from state dict
    inv_kfac_new = KFACInverseLinearOperator.from_state_dict(state_dict, kfac)

    # check that the two inverse KFACs are equal
    compare_state_dicts(inv_kfac.state_dict(), inv_kfac_new.state_dict())
    test_vec = torch.rand(kfac.shape[1])
    report_nonclose(inv_kfac @ test_vec, inv_kfac_new @ test_vec)


def test_torch_matvec_list_output_shapes(cnn_case):
    """Test output shapes with list input format (issue #124)."""
    model, loss_func, params, data, batch_size_fn = cnn_case
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )
    inv_kfac = KFACInverseLinearOperator(kfac, damping=1e-2)
    vec = [torch.rand_like(p) for p in kfac._params]
    out_list = inv_kfac.torch_matvec(vec)
    assert len(out_list) == len(kfac._params)
    for out_i, p_i in zip(out_list, kfac._params):
        assert out_i.shape == p_i.shape
