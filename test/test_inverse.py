"""Contains tests for ``curvlinops/inverse``."""

import os
from math import sqrt
from typing import Iterable, List, Tuple, Union

from pytest import mark, raises
from torch import Tensor, float64, load, manual_seed, rand, save
from torch.linalg import eigh, inv
from torch.nn import (
    CrossEntropyLoss,
    Linear,
    Module,
    MSELoss,
    Parameter,
    ReLU,
    Sequential,
)

from curvlinops import (
    CGInverseLinearOperator,
    EKFACLinearOperator,
    GGNLinearOperator,
    KFACLinearOperator,
    LSMRInverseLinearOperator,
    NeumannInverseLinearOperator,
)
from curvlinops.examples import IdentityLinearOperator
from curvlinops.examples.functorch import functorch_ggn
from curvlinops.utils import allclose_report
from test.test__torch_base import TensorLinearOperator
from test.utils import (
    cast_input,
    compare_consecutive_matmats,
    compare_matmat,
    eye_like,
    maybe_exclude_or_shuffle_parameters,
)

KFAC_MIN_DAMPING = 1e-8


def test_CGInverseLinearOperator_damped_GGN(inv_case, delta_rel: float = 2e-2):
    """Test matrix multiplication with the inverse damped GGN with CG.

    Args:
        inv_case: Tuple of model, loss function, parameters, data, batch size getter.
        delta_rel: Relative damping factor that is multiplied onto the average trace
            to obtain the damping value.
    """
    model_func, loss_func, params, data, batch_size_fn = inv_case
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
    compare_consecutive_matmats(inv_GGN, adjoint=False, is_vec=False)
    compare_matmat(inv_GGN, inv_GGN_naive, adjoint=False, is_vec=False, rtol=1.5e-2)


def test_LSMRInverseLinearOperator_damped_GGN(inv_case, delta: float = 2e-2):
    """Test matrix multiplication with the inverse damped GGN with LSMR."""
    model_func, loss_func, params, data, batch_size_fn = inv_case
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

    compare_consecutive_matmats(inv_GGN, adjoint=False, is_vec=False)
    compare_matmat(
        inv_GGN, inv_GGN_naive, adjoint=False, is_vec=False, rtol=5e-3, atol=1e-5
    )


def test_NeumannInverseLinearOperator_damped_GGN(inv_case, delta: float = 2e-2):
    """Test matrix multiplication by the inverse damped GGN with Neumann."""
    model_func, loss_func, params, data, batch_size_fn = inv_case
    (dev,), (dt,) = {p.device for p in params}, {p.dtype for p in params}

    GGN = GGNLinearOperator(
        model_func, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    damping = delta * IdentityLinearOperator([p.shape for p in params], dev, dt)

    damped_GGN_naive = functorch_ggn(
        model_func, loss_func, params, data, input_key="x"
    ).detach() + delta * eye_like(GGN)
    inv_GGN_naive = inv(damped_GGN_naive)

    # set scale such that Neumann series converges
    eval_max = eigh(damped_GGN_naive)[0][-1]
    scale = 1.0 if eval_max < 2 else 1.9 / eval_max

    # NOTE This may break when other cases are added because slow convergence
    inv_GGN = NeumannInverseLinearOperator(GGN + damping, num_terms=2_500, scale=scale)

    compare_consecutive_matmats(inv_GGN, adjoint=False, is_vec=False)
    compare_matmat(
        inv_GGN, inv_GGN_naive, adjoint=False, is_vec=False, rtol=1e-1, atol=1e-1
    )


def test_NeumannInverseLinearOperator_toy():
    """Test NeumannInverseLinearOperator on a toy example.

    The example is from
    https://en.wikipedia.org/w/index.php?title=Neumann_series&oldid=1131424698#Example
    """
    manual_seed(1234)
    A = Tensor(
        [
            [0.0, 1.0 / 2.0, 1.0 / 4.0],
            [5.0 / 7.0, 0.0, 1.0 / 7.0],
            [3.0 / 10.0, 3.0 / 5.0, 0.0],
        ]
    ).double()
    A += eye_like(A)
    # eigenvalues of A: [1.82122892 0.47963837 0.69913271]

    inv_A = inv(A)
    inv_A_neumann = NeumannInverseLinearOperator(
        TensorLinearOperator(A), num_terms=1_000
    )

    compare_consecutive_matmats(inv_A_neumann, adjoint=False, is_vec=False)
    compare_matmat(
        inv_A_neumann, inv_A, adjoint=False, is_vec=False, rtol=1e-3, atol=1e-5
    )

    # If we double the matrix, the Neumann series won't converge anymore ...
    B = 2 * A
    inv_B = inv(B)
    inv_B_neumann = NeumannInverseLinearOperator(
        TensorLinearOperator(B), num_terms=1_000
    )

    # ... therefore, we should get NaNs during the iteration
    with raises(ValueError):
        compare_consecutive_matmats(inv_B_neumann, adjoint=False, is_vec=False)

    # ... but if we scale the matrix back internally, the Neumann series converges
    inv_B_neumann = NeumannInverseLinearOperator(
        TensorLinearOperator(B), num_terms=1_000, scale=0.5
    )

    compare_consecutive_matmats(inv_B_neumann, adjoint=False, is_vec=False)
    compare_matmat(
        inv_B_neumann, inv_B, adjoint=False, is_vec=False, rtol=1e-3, atol=1e-5
    )


"""KFACInverseLinearOperator with KFACLinearOperator tests."""


@mark.parametrize("fisher_type", KFACLinearOperator._SUPPORTED_FISHER_TYPE)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_KFAC_inverse_damped_matmat(
    case: Tuple[
        Module,
        Union[MSELoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[Tensor, Tensor]],
    ],
    fisher_type: str,
    exclude: str,
    separate_weight_and_bias: bool,
    adjoint: bool,
    is_vec: bool,
    shuffle: bool,
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by an inverse damped KFAC approximation.

    Args:
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = case
    params = maybe_exclude_or_shuffle_parameters(params, model_func, exclude, shuffle)
    dtype = float64  # use double precision for better numerical stability
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
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=fisher_type,
    )
    # Add damping manually
    for block in KFAC._block_diagonal_operator._blocks:
        for i, S in enumerate(block._factors):
            block._factors[i] = S + delta * eye_like(S)
    inv_KFAC_naive = inv(KFAC @ eye_like(KFAC))

    # Remove damping and pass it on as an argument instead
    for block in KFAC._block_diagonal_operator._blocks:
        for i, S in enumerate(block._factors):
            block._factors[i] = S - delta * eye_like(S)

    inv_KFAC = KFAC.inverse(damping=delta)
    compare_consecutive_matmats(inv_KFAC, adjoint, is_vec)
    compare_matmat(inv_KFAC, inv_KFAC_naive, adjoint, is_vec)


@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_KFAC_inverse_heuristically_damped_matmat(  # noqa: C901, PLR0912, PLR0915
    case: Tuple[
        Module,
        Union[MSELoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[Tensor, Tensor]],
    ],
    exclude: str,
    separate_weight_and_bias: bool,
    adjoint: bool,
    is_vec: bool,
    shuffle: bool,
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by an inverse (heuristically) damped KFAC
    approximation.

    Args:
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = case
    params = maybe_exclude_or_shuffle_parameters(params, model_func, exclude, shuffle)
    dtype = float64  # use double precision for better numerical stability
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
        separate_weight_and_bias=separate_weight_and_bias,
    )

    # Compute heuristic damping values
    heuristic_damping = []
    for block in KFAC._block_diagonal_operator._blocks:
        if len(block._factors) == 1:
            block_damping = (delta,)
        else:
            ggT, aaT = block._factors
            aaT_eig_mean = aaT.trace() / aaT.shape[0]
            ggT_eig_mean = ggT.trace() / ggT.shape[0]
            if aaT_eig_mean >= 0.0 and ggT_eig_mean > 0.0:
                sqrt_eig_mean_ratio = (aaT_eig_mean / ggT_eig_mean).sqrt()
                sqrt_damping = sqrt(delta)
                damping_aaT = max(sqrt_damping * sqrt_eig_mean_ratio, KFAC_MIN_DAMPING)
                damping_ggT = max(sqrt_damping / sqrt_eig_mean_ratio, KFAC_MIN_DAMPING)
                block_damping = (damping_ggT, damping_aaT)

        heuristic_damping.append(block_damping)

    # Add heuristic damping manually
    for damping, block in zip(heuristic_damping, KFAC._block_diagonal_operator._blocks):
        for i, (damping_i, S_i) in enumerate(zip(damping, block._factors)):
            block._factors[i] = S_i + damping_i * eye_like(S_i)

    # manual heuristically damped inverse
    inv_KFAC_naive = inv(KFAC @ eye_like(KFAC))

    # Remove heuristic damping manually
    for damping, block in zip(heuristic_damping, KFAC._block_diagonal_operator._blocks):
        for i, (damping_i, S_i) in enumerate(zip(damping, block._factors)):
            block._factors[i] = S_i - damping_i * eye_like(S_i)

    # Check that using exact and heuristic damping at the same time fails
    with raises(ValueError, match="Either use heuristic damping or exact damping"):
        KFAC.inverse(use_exact_damping=True, use_heuristic_damping=True)

    # use heuristic damping with KFACInverseLinearOperator
    inv_KFAC = KFAC.inverse(
        damping=delta, use_heuristic_damping=True, min_damping=KFAC_MIN_DAMPING
    )

    compare_consecutive_matmats(inv_KFAC, adjoint, is_vec)
    compare_matmat(inv_KFAC, inv_KFAC_naive, adjoint, is_vec)


@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_KFAC_inverse_exactly_damped_matmat(
    case: Tuple[
        Module,
        Union[MSELoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[Tensor, Tensor]],
    ],
    exclude: str,
    separate_weight_and_bias: bool,
    adjoint: bool,
    is_vec: bool,
    shuffle: bool,
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by an inverse (exactly) damped KFAC approximation.

    Args:
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = case
    params = maybe_exclude_or_shuffle_parameters(params, model_func, exclude, shuffle)
    dtype = float64  # use double precision for better numerical stability
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
        separate_weight_and_bias=separate_weight_and_bias,
        check_deterministic=False,
    )

    # manual exactly damped inverse
    KFAC_mat = KFAC @ eye_like(KFAC)
    inv_KFAC_naive = inv(KFAC_mat + delta * eye_like(KFAC_mat))

    # check that using exact and heuristic damping at the same time fails
    with raises(ValueError, match="Either use heuristic damping or exact damping"):
        KFAC.inverse(use_exact_damping=True, use_heuristic_damping=True)

    # use exact damping
    inv_KFAC = KFAC.inverse(damping=delta, use_exact_damping=True)

    compare_consecutive_matmats(inv_KFAC, adjoint, is_vec)
    compare_matmat(inv_KFAC, inv_KFAC_naive, adjoint, is_vec)


@mark.parametrize("use_exact_damping", [True, False], ids=["exact_damping", ""])
@mark.parametrize("use_heuristic_damping", [True, False], ids=["heuristic_damping", ""])
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_KFAC_inverse_save_and_load(
    use_exact_damping: bool,
    use_heuristic_damping: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    shuffle: bool,
):
    """Test that KFACInverseLinearOperator can be saved and loaded."""
    manual_seed(0)
    batch_size, D_in, D_hidden, D_out = 4, 3, 5, 2
    X = rand(batch_size, D_in)
    y = rand(batch_size, D_out)
    model = Sequential(
        Linear(D_in, D_hidden),
        ReLU(),
        Linear(D_hidden, D_hidden, bias=False),
        ReLU(),
        Linear(D_hidden, D_out),
    )

    params = list(model.parameters())
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    # create and compute KFAC
    kfac = KFACLinearOperator(
        model,
        MSELoss(reduction="sum"),
        params,
        [(X, y)],
        separate_weight_and_bias=separate_weight_and_bias,
    )

    # create inverse KFAC
    kwargs = {
        "use_exact_damping": use_exact_damping,
        "use_heuristic_damping": use_heuristic_damping,
        "retry_double_precision": False,
    }
    if use_exact_damping and use_heuristic_damping:
        return
    inv_kfac = kfac.inverse(damping=1e-2, **kwargs)
    inv_kfac_as_mat = inv_kfac @ eye_like(kfac)

    # save state dict
    INV_KFAC_PATH = "inv_kfac_state_dict.pt"
    save(inv_kfac, INV_KFAC_PATH)
    del inv_kfac

    # create new inverse KFAC and load state dict
    inv_kfac_loaded = load(INV_KFAC_PATH, weights_only=False)
    # clean up
    os.remove(INV_KFAC_PATH)

    # check that the two inverse KFACs are equal
    assert allclose_report(inv_kfac_as_mat, inv_kfac_loaded @ eye_like(kfac))


"""KFACInverseLinearOperator with EKFACLinearOperator tests."""


@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_EKFAC_inverse_matmat(
    inv_case: Tuple[
        Module,
        Union[MSELoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[Tensor, Tensor]],
    ],
    exclude: str,
    separate_weight_and_bias: bool,
    adjoint: bool,
    is_vec: bool,
    shuffle: bool,
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by an inverse (exactly) damped EKFAC approximation.

    Args:
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
    """
    model_func, loss_func, params, data, batch_size_fn = inv_case
    params = maybe_exclude_or_shuffle_parameters(params, model_func, exclude, shuffle)
    dtype = float64  # use double precision for better numerical stability
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

    EKFAC = EKFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
    )

    # manual exactly damped inverse
    inv_EKFAC_naive = inv(EKFAC @ eye_like(EKFAC) + delta * eye_like(EKFAC))

    # use exact damping with KFACInverseLinearOperator
    inv_EKFAC = EKFAC.inverse(damping=delta)

    compare_consecutive_matmats(inv_EKFAC, adjoint, is_vec)
    compare_matmat(inv_EKFAC, inv_EKFAC_naive, adjoint, is_vec)


def test_EKFAC_inverse_save_and_load():
    """Test that KFACInverseLinearOperator can be saved and loaded."""
    manual_seed(0)
    batch_size, D_in, D_out = 4, 3, 2
    X = rand(batch_size, D_in)
    y = rand(batch_size, D_out)
    model = Linear(D_in, D_out)

    params = list(model.parameters())
    # create and compute EKFAC
    ekfac = EKFACLinearOperator(
        model,
        # use non-default loss reduction to verify if it is correctly saved and loaded
        MSELoss(reduction="sum"),
        params,
        [(X, y)],
    )

    # create inverse KFAC
    inv_ekfac = ekfac.inverse(damping=1e-2)
    inv_ekfac_as_mat = inv_ekfac @ eye_like(ekfac)  # to trigger inverse computation

    # save state dict
    INV_EKFAC_PATH = "inv_ekfac_state_dict.pt"
    save(inv_ekfac, INV_EKFAC_PATH)
    del inv_ekfac

    # create new inverse KFAC and load state dict
    inv_ekfac_loaded = load(INV_EKFAC_PATH, weights_only=False)
    # clean up
    os.remove(INV_EKFAC_PATH)

    # check that the two inverse KFACs are equal
    assert allclose_report(inv_ekfac_as_mat, inv_ekfac_loaded @ eye_like(ekfac))
