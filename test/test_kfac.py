"""Contains tests for ``curvlinops.kfac``."""

from collections.abc import Iterable
from math import sqrt
from pathlib import Path

from einops.layers.torch import Rearrange
from pytest import mark
from torch import (
    Tensor,
    allclose,
    device,
    float64,
    load,
    manual_seed,
    rand,
    rand_like,
    save,
)
from torch.linalg import inv
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Flatten,
    Linear,
    Module,
    MSELoss,
    Parameter,
    Sequential,
)

from curvlinops import EFLinearOperator, GGNLinearOperator
from curvlinops.computers.kfac import KFACComputer
from curvlinops.kfac import KFACLinearOperator
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import allclose_report
from test.cases import DEVICES, DEVICES_IDS
from test.utils import (
    Conv2dModel,
    SplitConcatModel,
    UnetModel,
    WeightShareModel,
    _test_inplace_activations,
    _test_property,
    block_diagonal,
    change_dtype,
    classification_targets,
    compare_consecutive_matmats,
    compare_matmat,
    eye_like,
    maybe_exclude_or_shuffle_parameters,
    regression_targets,
)

# Constants for MC tests
MC_SAMPLES = 3_000
MC_TOLS = {"rtol": 1e-1, "atol": 1.5e-2}

# Backend parametrization
BACKENDS = list(KFACLinearOperator._BACKENDS)
BACKENDS_IDS = BACKENDS


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac_type2(
    kfac_exact_case: tuple[
        Module, MSELoss, list[Parameter], Iterable[tuple[Tensor, Tensor]]
    ],
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    backend: str,
):
    """Test the KFAC implementation against the exact GGN.

    Args:
        kfac_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the KFAC matrix.
        backend: The backend to use for computing Kronecker factors.
    """
    model, loss_func, params, data, batch_size_fn = kfac_exact_case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.TYPE2,
        separate_weight_and_bias=separate_weight_and_bias,
        backend=backend,
    )
    kfac_mat = kfac @ eye_like(kfac)

    assert allclose_report(ggn, kfac_mat)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
@mark.parametrize("setting", [KFACType.EXPAND, KFACType.REDUCE])
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac_type2_weight_sharing(
    kfac_weight_sharing_exact_case: tuple[
        WeightShareModel | Conv2dModel,
        MSELoss,
        list[Parameter],
        dict[str, Iterable[tuple[Tensor, Tensor]]],
    ],
    setting: str,
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    backend: str,
):
    """Test KFAC for linear weight-sharing layers against the exact GGN.

    Args:
        kfac_weight_sharing_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        setting: The weight-sharing setting to use. Can be ``KFACType.EXPAND`` or
            ``KFACType.REDUCE``.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the KFAC matrix.
        backend: The backend to use for computing Kronecker factors.
    """
    model, loss_func, params, data, batch_size_fn = kfac_weight_sharing_exact_case
    model.setting = setting
    if isinstance(model, Conv2dModel):
        # parameters are only initialized after the setting property is set
        params = [p for p in model.parameters() if p.requires_grad]
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    data = data[setting]
    model, loss_func, params, data, batch_size_fn = change_dtype(
        (model, loss_func, params, data, batch_size_fn), float64
    )

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.TYPE2,
        kfac_approx=setting,  # choose KFAC approximation consistent with setting
        separate_weight_and_bias=separate_weight_and_bias,
        backend=backend,
    )
    kfac_mat = kfac @ eye_like(kfac)

    assert allclose_report(ggn, kfac_mat)


def _test_weight_tying_type2(
    linop_cls: type[KFACLinearOperator],
    reduction: str,
    bias: bool,
    separate_weight_and_bias: bool,
):
    """Test (E)KFAC with weight tying against the exact block-diagonal GGN.

    Uses a split-concat model where the same module is applied to two halves
    of the input. With N=1, (E)KFAC-expand is exact because the two paths are
    independent.

    For ``KFACLinearOperator``, also verifies that the hooks backend gives wrong
    results (hooks fire twice per forward, each with ``scale=1``). EKFAC hooks
    may still be correct because the eigenvalue correction recomputes eigenvalues
    from data, compensating for the wrong Kronecker factors.

    Args:
        linop_cls: The linear operator class to test.
        reduction: Loss reduction mode.
        bias: Whether the shared linear layer has a bias.
        separate_weight_and_bias: Whether to treat weight and bias separately.
    """
    manual_seed(0)
    D = 4

    model = SplitConcatModel(D, bias=bias)
    loss_func = MSELoss(reduction=reduction)
    params = [p for p in model.parameters() if p.requires_grad]
    data = [
        (rand(1, 2 * D), regression_targets((1, 2 * D))),
    ]
    model, loss_func, params, data, _ = change_dtype(
        (model, loss_func, params, data, None), float64
    )

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        separate_weight_and_bias=separate_weight_and_bias,
    )

    for backend in BACKENDS:
        linop = linop_cls(
            model,
            loss_func,
            params,
            data,
            fisher_type=FisherType.TYPE2,
            kfac_approx=KFACType.EXPAND,
            separate_weight_and_bias=separate_weight_and_bias,
            backend=backend,
        )
        linop_mat = linop @ eye_like(linop)

        if backend == "make_fx":
            # make_fx backend: exact for weight tying
            assert allclose_report(ggn, linop_mat)
        elif backend == "hooks" and linop_cls is KFACLinearOperator:
            # hooks backend: incorrect KFAC factors (known limitation).
            # EKFAC hooks may still be correct because the eigenvalue
            # correction recomputes eigenvalues, compensating for the
            # wrong Kronecker factors.
            assert not allclose(ggn, linop_mat)


@mark.parametrize("reduction", ["mean", "sum"])
@mark.parametrize("bias", [False, True], ids=["no_bias", "with_bias"])
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
def test_kfac_type2_weight_tying(
    reduction: str, bias: bool, separate_weight_and_bias: bool
):
    """Test KFAC with weight tying against exact GGN (make_fx only)."""
    _test_weight_tying_type2(
        KFACLinearOperator, reduction, bias, separate_weight_and_bias
    )


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_kfac_mc(
    kfac_exact_case: tuple[
        Module, MSELoss, list[Parameter], Iterable[tuple[Tensor, Tensor]]
    ],
    backend: str,
):
    """Test the KFAC implementation using MC samples against the exact GGN.

    Args:
        kfac_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        backend: The backend to use for computing Kronecker factors.
    """
    model, loss_func, params, data, batch_size_fn = change_dtype(
        kfac_exact_case, float64
    )

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.MC,
        mc_samples=MC_SAMPLES,
        backend=backend,
    )
    kfac_mat = kfac @ eye_like(kfac)

    # Normalize so we can share tolerances across reductions
    scale = ggn.abs().max()
    assert allclose_report(ggn / scale, kfac_mat / scale, **MC_TOLS)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
@mark.parametrize("setting", [KFACType.EXPAND, KFACType.REDUCE])
def test_kfac_mc_weight_sharing(
    kfac_weight_sharing_exact_case: tuple[
        WeightShareModel | Conv2dModel,
        MSELoss,
        list[Parameter],
        dict[str, Iterable[tuple[Tensor, Tensor]]],
    ],
    setting: str,
    backend: str,
):
    """Test KFAC-MC for linear layers with weight sharing against the exact GGN.

    Args:
        kfac_weight_sharing_exact_case: A fixture that returns a model, loss function,
            list of parameters, and data.
        setting: The weight-sharing setting to use. Can be ``KFACType.EXPAND`` or
            ``KFACType.REDUCE``.
        backend: The backend to use for computing Kronecker factors.
    """
    model, loss_func, params, data, batch_size_fn = kfac_weight_sharing_exact_case
    model.setting = setting
    if isinstance(model, Conv2dModel):
        # parameters are only initialized after the setting property is set
        params = [p for p in model.parameters() if p.requires_grad]
    data = data[setting]
    model, loss_func, params, data, batch_size_fn = change_dtype(
        (model, loss_func, params, data, batch_size_fn), float64
    )

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.MC,
        mc_samples=MC_SAMPLES,
        kfac_approx=setting,  # choose KFAC approximation consistent with setting
        backend=backend,
    )
    kfac_mat = kfac @ eye_like(kfac)

    # Normalize so we can share tolerances across reductions
    scale = ggn.abs().max()
    assert allclose_report(ggn / scale, kfac_mat / scale, **MC_TOLS)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_kfac_one_datum(
    kfac_exact_one_datum_case: tuple[
        Module,
        BCEWithLogitsLoss | CrossEntropyLoss,
        list[Parameter],
        Iterable[tuple[Tensor, Tensor]],
    ],
    backend: str,
):
    """Test KFAC for the one-datum exact case."""
    model, loss_func, params, data, batch_size_fn = kfac_exact_one_datum_case

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.TYPE2,
        backend=backend,
    )
    kfac_mat = kfac @ eye_like(kfac)

    assert allclose_report(ggn, kfac_mat)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_kfac_mc_one_datum(
    kfac_exact_one_datum_case: tuple[
        Module,
        BCEWithLogitsLoss | CrossEntropyLoss,
        list[Parameter],
        Iterable[tuple[Tensor, Tensor]],
    ],
    backend: str,
):
    """Test KFAC-MC for the one-datum exact case."""
    model, loss_func, params, data, batch_size_fn = change_dtype(
        kfac_exact_one_datum_case, float64
    )

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.MC,
        mc_samples=MC_SAMPLES,
        backend=backend,
    )
    kfac_mat = kfac @ eye_like(kfac)

    # Normalize so we can share tolerances across reductions
    scale = ggn.abs().max()
    # Need to use larger tolerances on GPU, despite float64
    tols = (
        MC_TOLS
        if "cpu" in str(params[0].device)
        else {k: 2 * v for k, v in MC_TOLS.items()}
    )
    assert allclose_report(ggn / scale, kfac_mat / scale, **tols)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_kfac_ef_one_datum(
    kfac_exact_one_datum_case: tuple[
        Module,
        BCEWithLogitsLoss | CrossEntropyLoss,
        list[Parameter],
        Iterable[tuple[Tensor, Tensor]],
    ],
    backend: str,
):
    """Test empirical Fisher KFAC for the one-datum exact case."""
    model, loss_func, params, data, batch_size_fn = kfac_exact_one_datum_case

    ef = block_diagonal(
        EFLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )

    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.EMPIRICAL,
        backend=backend,
    )
    kfac_mat = kfac @ eye_like(kfac)

    assert allclose_report(ef, kfac_mat)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_kfac_inplace_activations(dev: device, backend: str):
    """Test that KFAC works if the network has in-place activations.

    We use a test case with a single datum as KFAC becomes exact as the number of
    MC samples increases.

    Args:
        dev: The device to run the test on.
        backend: The backend to use for computing Kronecker factors.
    """
    _test_inplace_activations(KFACLinearOperator, dev, backend)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
@mark.parametrize("fisher_type", KFACComputer._SUPPORTED_FISHER_TYPE)
@mark.parametrize(
    "loss", [MSELoss, CrossEntropyLoss, BCEWithLogitsLoss], ids=["mse", "ce", "bce"]
)
@mark.parametrize("reduction", ["mean", "sum"])
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_multi_dim_output(
    fisher_type: str,
    loss: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
    reduction: str,
    dev: device,
    backend: str,
):
    """Test the KFAC implementation for >2d outputs (using a 3d and 4d output).

    Args:
        fisher_type: The type of Fisher matrix to use.
        loss: The loss function to use.
        reduction: The reduction to use for the loss function.
        dev: The device to run the test on.
        backend: The backend to use for computing Kronecker factors.
    """
    manual_seed(0)
    # set up loss function, data, and model
    loss_func = loss(reduction=reduction).to(dev)
    X1 = rand(2, 7, 5, 5)
    X2 = rand(4, 7, 5, 5)
    if isinstance(loss_func, MSELoss):
        data = [
            (X1, regression_targets((2, 7, 5, 3))),
            (X2, regression_targets((4, 7, 5, 3))),
        ]
        manual_seed(711)
        model = Sequential(Linear(5, 4), Linear(4, 3)).to(dev)
    elif issubclass(loss, BCEWithLogitsLoss):
        data = [
            (X1, rand(2, 7, 5, 3)),
            (X2, rand(4, 7, 5, 3)),
        ]
        manual_seed(711)
        model = Sequential(Linear(5, 4), Linear(4, 3)).to(dev)
    else:
        data = [
            (X1, classification_targets((2, 7, 5), 3)),
            (X2, classification_targets((4, 7, 5), 3)),
        ]
        manual_seed(711)
        # rearrange is necessary to get the expected output shape for ce loss
        model = Sequential(
            Linear(5, 4),
            Linear(4, 3),
            Rearrange("batch ... c -> batch c ..."),
        ).to(dev)

    # KFAC for deep linear network with 4d input and output
    params = list(model.parameters())
    kfac = KFACLinearOperator(
        model, loss_func, params, data, fisher_type=fisher_type, backend=backend
    )
    kfac_mat = kfac @ eye_like(kfac)

    # KFAC for deep linear network with 4d input and equivalent 2d output
    manual_seed(711)
    model_flat = Sequential(
        Linear(5, 4),
        Linear(4, 3),
        Flatten(start_dim=0, end_dim=-2),
    ).to(dev)
    params_flat = list(model_flat.parameters())
    data_flat = [
        (
            (x, y.flatten(start_dim=0, end_dim=-2))
            if isinstance(loss_func, (MSELoss, BCEWithLogitsLoss))
            else (x, y.flatten(start_dim=0))
        )
        for x, y in data
    ]
    kfac_flat = KFACLinearOperator(
        model_flat,
        loss_func,
        params_flat,
        data_flat,
        fisher_type=fisher_type,
        backend=backend,
    )
    kfac_flat_mat = kfac_flat @ eye_like(kfac_flat)

    assert allclose_report(kfac_mat, kfac_flat_mat)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
@mark.parametrize("fisher_type", KFACComputer._SUPPORTED_FISHER_TYPE)
@mark.parametrize(
    "loss", [MSELoss, CrossEntropyLoss, BCEWithLogitsLoss], ids=["mse", "ce", "bce"]
)
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_expand_setting_scaling(
    fisher_type: str,
    loss: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
    dev: device,
    backend: str,
):
    """Test KFAC for correct scaling for expand setting with mean reduction loss.

    See #107 for details.

    Args:
        fisher_type: The type of Fisher matrix to use.
        loss: The loss function to use.
        dev: The device to run the test on.
        backend: The backend to use for computing Kronecker factors.
    """
    manual_seed(0)

    # set up data, loss function, and model (use float64 for numerical precision)
    S = 8  # spatial size (small for speed; UnetModel bottleneck is S/2)
    X1 = rand(2, 3, S, S, dtype=float64)
    X2 = rand(4, 3, S, S, dtype=float64)
    if issubclass(loss, MSELoss):
        data = [
            (X1, regression_targets((2, S, S, 3)).double()),
            (X2, regression_targets((4, S, S, 3)).double()),
        ]
    elif issubclass(loss, BCEWithLogitsLoss):
        data = [
            (X1, rand(2, S, S, 3, dtype=float64)),
            (X2, rand(4, S, S, 3, dtype=float64)),
        ]
    else:
        data = [
            (X1, classification_targets((2, S, S), 3)),
            (X2, classification_targets((4, S, S), 3)),
        ]
    model = UnetModel(loss).to(dev).double()
    params = list(model.parameters())

    # KFAC with sum reduction
    loss_func = loss(reduction="sum").to(dev)
    kfac_sum = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type=fisher_type,
        backend=backend,
    )
    # FOOF does not scale the gradient covariances, even when using a mean reduction
    if fisher_type != FisherType.FORWARD_ONLY:
        # Simulate a mean reduction by manually scaling the gradient covariances
        loss_term_factor = S * S  # number of spatial locations of model output
        if issubclass(loss, (MSELoss, BCEWithLogitsLoss)):
            output_random_variable_size = 3
            # MSE loss averages over number of output channels
            loss_term_factor *= output_random_variable_size

        num_data = sum(X.shape[0] for X, _ in data)
        _, K, _ = kfac_sum
        for block in K:
            # Gradient covariance is always the first Kronecker factor
            block[0] = block[0] / (num_data * loss_term_factor)
    kfac_simulated_mean_mat = kfac_sum @ eye_like(kfac_sum)

    # KFAC with mean reduction
    loss_func = loss(reduction="mean").to(dev)
    kfac_mean = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type=fisher_type,
        backend=backend,
    )
    kfac_mean_mat = kfac_mean @ eye_like(kfac_mean)

    assert allclose_report(kfac_simulated_mean_mat, kfac_mean_mat)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_KFACLinearOperator(
    case,
    backend: str,
):
    """Test matrix multiplication with KFAC.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        backend: The backend to use for computing Kronecker factors.
    """
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        backend=backend,
    )
    kfac_mat = kfac @ eye_like(kfac)

    compare_consecutive_matmats(kfac)
    compare_matmat(kfac, kfac_mat)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_trace(case, backend):
    """Test that the trace property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    _test_property(
        KFACLinearOperator,
        "trace",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        backend=backend,
    )


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_frobenius_norm(case, backend):
    """Test that the Frobenius norm property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    _test_property(
        KFACLinearOperator,
        "frobenius_norm",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        backend=backend,
    )


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_det(case, backend):
    """Test that the determinant property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    _test_property(
        KFACLinearOperator,
        "det",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        backend=backend,
    )


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_logdet(case, backend):
    """Test that the log determinant property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    _test_property(
        KFACLinearOperator,
        "logdet",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        backend=backend,
    )


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_forward_only_fisher_type(
    case: tuple[Module, MSELoss, list[Parameter], Iterable[tuple[Tensor, Tensor]]],
    backend: str,
):
    """Test the KFAC with forward-only Fisher (used for FOOF) implementation.

    Args:
        case: A fixture that returns a model, loss function, list of parameters, and
            data.
        backend: The backend to use for computing Kronecker factors.
    """
    model, loss_func, params, data, batch_size_fn = case

    # Compute KFAC with `fisher_type=FisherType.EMPIRICAL`
    # (could be any but `FisherType.FORWARD_ONLY`)
    foof_simulated = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.EMPIRICAL,
        backend=backend,
    )
    # Manually set all gradient covariances to the identity to simulate FOOF
    _, K, _ = foof_simulated
    for block in K:
        # Gradient covariance is always the first Kronecker factor
        block[0] = eye_like(block[0])
    simulated_foof_mat = foof_simulated @ eye_like(foof_simulated)

    # Compute KFAC with `fisher_type=FisherType.FORWARD_ONLY`
    foof = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.FORWARD_ONLY,
        backend=backend,
    )
    foof_mat = foof @ eye_like(foof)

    assert allclose_report(simulated_foof_mat, foof_mat)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_forward_only_fisher_type_exact_case(
    single_layer_case: tuple[
        Module, MSELoss, list[Parameter], Iterable[tuple[Tensor, Tensor]]
    ],
    backend: str,
):
    r"""Test KFAC with forward-only Fisher (FOOF) against exact GGN for one-layer model.

    Consider linear regression with square loss, L =  R * \sum_n^N || W x_n - y_n ||^2,
    where R is the reduction factor from the MSELoss. Per definition,
    FOOF(W) = I \otimes (\sum_n x_n x_n^T / N). Hence, if R = 1 [reduction='sum'], we
    have that GGN(W) = 2 * [I \otimes (\sum_n x_n x_n^T)] = 2 * N * FOOF(W).
    If R = 1 / (N * C) [reduction='mean'], where C is the output dimension, we have
    GGN(W) = 2 * R * [I \otimes (\sum_n x_n x_n^T)] = 2 / C * FOOF(W).

    Args:
        single_layer_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        backend: The backend to use for computing Kronecker factors.
    """
    model, loss_func, params, data, batch_size_fn = single_layer_case

    # Compute exact block-diagonal GGN
    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )

    # Compute KFAC with `fisher_type=FisherType.FORWARD_ONLY`
    foof = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.FORWARD_ONLY,
        backend=backend,
    )
    foof_mat = foof @ eye_like(foof)

    # Check for equivalence
    num_data = sum(X.shape[0] for X, _ in data)
    y: Tensor = data[0][1]
    out_dim = y.shape[1]
    # See the docstring for the explanation of the scale
    scale = num_data if loss_func.reduction == "sum" else 1 / out_dim
    assert allclose_report(ggn, 2 * scale * foof_mat)


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
@mark.parametrize("setting", [KFACType.EXPAND, KFACType.REDUCE])
def test_forward_only_fisher_type_exact_weight_sharing_case(
    single_layer_weight_sharing_case: tuple[
        WeightShareModel | Conv2dModel,
        MSELoss,
        list[Parameter],
        dict[str, Iterable[tuple[Tensor, Tensor]]],
    ],
    setting: str,
    backend: str,
):
    r"""Test KFAC with forward-only Fisher (FOOF) against GGN for weight-sharing models.

    Expand setting: Consider linear regression with square loss,
    L =  R * \sum_n^N \sum_s^S || W x_{n,s} - y_{n,s} ||^2, where R is the reduction
    factor from the MSELoss and S is the weight-sharing dimension size. Per definition,
    FOOF(W) = I \otimes (\sum_n^N \sum_s^S x_{n,s} x_{n,s}^T / (N * S)).
    Hence, if R = 1 [reduction='sum'], we have that
    GGN(W) = 2 * [I \otimes (\sum_n^N \sum_s^S x_{n,s} x_{n,s}^T)] = 2 * N * S * FOOF(W).
    If R = 1 / (N * C * S) [reduction='mean'], where C is the output dimension, we have
    GGN(W) = 2 * R * [I \otimes (\sum_n^N \sum_s^S x_{n,s} x_{n,s}^T)] = 2 / C * FOOF(W).

    Reduce setting: Consider linear regression with square loss,
    L =  R * \sum_n^N || W x_n - y_n ||^2, where R is the reduction factor from the
    MSELoss. Per definition,
    FOOF(W) = I \otimes (\sum_n^N (\sum_s^S x_{n,s} \sum_s^S x_{n,s}^T) / (N * S^2)),
    where S is the weight-sharing dimension size. Hence, if R = 1 [reduction='sum'], we
    have that
    GGN(W) = 2 * [I \otimes (\sum_n^N \sum_s^S x_{n,s} \sum_s^S x_{n,s}^T) / S^2]
    = 2 * N * FOOF(W) (assumes the mean/average pooling as reduction function).
    If R = 1 / (N * C) [reduction='mean'], where C is the output dimension, we have
    GGN(W) = 2 * R * [I \otimes (\sum_n^N \sum_s^S x_{n,s} \sum_s^S x_{n,s}^T) / S^2]
    = 2 / C * FOOF(W) (assumes the mean/average pooling as reduction function).

    Args:
        single_layer_weight_sharing_case: A fixture that returns a model, loss function,
            list of parameters, and data.
        setting: The weight-sharing setting to use. Can be ``KFACType.EXPAND`` or
            ``KFACType.REDUCE``.
        backend: The backend to use for computing Kronecker factors.
    """
    model, loss_func, params, data, batch_size_fn = single_layer_weight_sharing_case
    model.setting = setting
    if isinstance(model, Conv2dModel):
        # parameters are only initialized after the setting property is set
        params = [p for p in model.parameters() if p.requires_grad]
    data = data[setting]

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )
    foof = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.FORWARD_ONLY,
        kfac_approx=setting,  # choose KFAC approximation consistent with setting
        backend=backend,
    )
    foof_mat = foof @ eye_like(foof)

    # Check for equivalence
    num_data = sum(X.shape[0] for X, _ in data)
    X, y = next(iter(data))
    out_dim = y.shape[-1]
    # See the docstring for the explanation of the scale
    scale = num_data if loss_func.reduction == "sum" else 1 / out_dim
    if loss_func.reduction == "sum" and setting == KFACType.EXPAND:
        sequence_length = (
            (X.shape[-2] + 1) * (X.shape[-1] + 1)
            if isinstance(model, Conv2dModel)
            else X.shape[1:-1].numel()
        )
        scale *= sequence_length
    assert allclose_report(ggn, 2 * scale * foof_mat, rtol=1e-4)


def _check_does_not_affect_grad(linop_cls):
    """Make sure that computing a linear operator does not affect `.grad`.

    Args:
        linop_cls: The linear operator class to test.
    """
    manual_seed(0)
    batch_size, D_in, D_out = 4, 3, 2
    X = rand(batch_size, D_in)
    y = rand(batch_size, D_out)
    model = Linear(D_in, D_out)

    params = list(model.parameters())
    # set gradients to random numbers
    for p in params:
        p.grad = rand_like(p)
    # make independent copies
    grads_before = [p.grad.clone() for p in params]

    # create and compute the linear operator
    _ = linop_cls(model, MSELoss(), params, [(X, y)])

    # make sure gradients are unchanged
    for grad_before, p in zip(grads_before, params):
        assert allclose(grad_before, p.grad)


def test_kfac_does_not_affect_grad():
    """Make sure KFAC computation does not write to `.grad`."""
    _check_does_not_affect_grad(KFACLinearOperator)


def _check_torch_save_load(linop_cls: type, tmp_path: Path) -> None:
    """Test that an (E)KFAC operator can be saved and loaded with torch.save/load.

    Args:
        linop_cls: The linear operator class to test.
        tmp_path: Temporary directory provided by pytest.
    """
    manual_seed(0)
    model = Linear(3, 2)
    params = list(model.parameters())
    data = [(rand(4, 3), rand(4, 2))]

    linop = linop_cls(model, MSELoss(), params, data)
    mat_before = linop @ eye_like(linop)

    path = tmp_path / "linop.pt"
    save(linop, path)
    linop_loaded = load(path, weights_only=False)

    mat_after = linop_loaded @ eye_like(linop_loaded)
    assert allclose(mat_before, mat_after)

    path.unlink()


def test_kfac_torch_save_load(tmp_path: Path) -> None:
    """Test that KFACLinearOperator can be saved and loaded with torch.save/load."""
    _check_torch_save_load(KFACLinearOperator, tmp_path)


@mark.parametrize("fisher_type", ["type-2", "mc", "empirical", "forward-only"])
@mark.parametrize("kfac_approx", ["expand", "reduce"])
def test_string_in_enum(fisher_type: str, kfac_approx: str):
    """Test whether checking if a string is contained in enum works.

    To reproduce issue #118.
    """
    model = Linear(2, 2)
    KFACLinearOperator(
        model,
        MSELoss(),
        list(model.parameters()),
        [(rand(2, 2), rand(2, 2))],
        fisher_type=fisher_type,
        kfac_approx=kfac_approx,
    )


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_bug_132_dtype_deterministic_checks(dev: device, backend: str):
    """Test whether the vectors used in the deterministic checks have correct data type.

    This bug was reported in https://github.com/f-dangel/curvlinops/issues/132.

    Args:
        dev: The device to run the test on.
        backend: The backend to use for computing Kronecker factors.
    """
    # make deterministic
    manual_seed(0)

    # create a toy problem, load everything to float64
    dt = float64
    N = 4
    D_in = 3
    D_out = 2

    X = rand(N, D_in, dtype=dt, device=dev)
    y = rand(N, D_out, dtype=dt, device=dev)
    data = [(X, y)]

    model = Linear(D_in, D_out).to(dev, dt)
    params = [p for p in model.parameters() if p.requires_grad]

    loss_func = MSELoss().to(dev, dt)

    # run deterministic checks
    KFACLinearOperator(
        model, loss_func, params, data, check_deterministic=True, backend=backend
    )


"""KFACLinearOperator.inverse() tests."""

KFAC_MIN_DAMPING = 1e-8


@mark.parametrize("fisher_type", KFACComputer._SUPPORTED_FISHER_TYPE)
def test_KFAC_inverse_damped_matmat(
    case: tuple[
        Module,
        MSELoss | CrossEntropyLoss,
        list[Parameter],
        Iterable[tuple[Tensor, Tensor]],
    ],
    fisher_type: str,
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by an inverse damped KFAC approximation."""
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    KFAC = KFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=fisher_type,
    )

    # Invert KFAC linear operators
    inv_KFAC = KFAC.inverse(damping=delta)

    # Manually add damping to each Kronecker factor, materialize, invert
    _, K, _ = KFAC
    for block in K:
        for idx in range(len(block)):
            # NOTE Needs out-of-place addition because some factors correspond to
            # the same tensors that would otherwise be damped multiple times
            block[idx] = block[idx] + delta * eye_like(block[idx])
    inv_KFAC_naive = inv(KFAC @ eye_like(KFAC))

    compare_consecutive_matmats(inv_KFAC)
    compare_matmat(inv_KFAC, inv_KFAC_naive)


def test_KFAC_inverse_heuristically_damped_matmat(  # noqa: C901
    case: tuple[
        Module,
        MSELoss | CrossEntropyLoss,
        list[Parameter],
        Iterable[tuple[Tensor, Tensor]],
    ],
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by a heuristically damped KFAC inverse."""
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    KFAC = KFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        check_deterministic=False,
    )

    inv_KFAC = KFAC.inverse(
        damping=delta, use_heuristic_damping=True, min_damping=KFAC_MIN_DAMPING
    )

    # Manually add heuristic damping to each Kronecker factor
    # NOTE We cannot use in-place operations for this because some Kronecker factors
    # may correspond to identical tensors that would otherwise be damped multiple times.
    _, K, _ = KFAC
    for block in K:
        if len(block) == 2:
            S1, S2 = block[0], block[1]  # ggT, aaT
            mean_eig1, mean_eig2 = S1.diag().mean(), S2.diag().mean()
            if mean_eig1 > 0 and mean_eig2 >= 0:
                sqrt_eig_mean_ratio = (mean_eig2 / mean_eig1).sqrt()
                sqrt_damping = sqrt(delta)
                damping1 = max(sqrt_damping / sqrt_eig_mean_ratio, KFAC_MIN_DAMPING)
                damping2 = max(sqrt_damping * sqrt_eig_mean_ratio, KFAC_MIN_DAMPING)
            else:
                damping1, damping2 = delta, delta
            block[0] = S1 + damping1 * eye_like(S1)
            block[1] = S2 + damping2 * eye_like(S2)
        else:
            block[0] = block[0] + delta * eye_like(block[0])

    inv_KFAC_naive = inv(KFAC @ eye_like(KFAC))

    compare_consecutive_matmats(inv_KFAC)
    compare_matmat(inv_KFAC, inv_KFAC_naive)


def test_KFAC_inverse_exactly_damped_matmat(
    case: tuple[
        Module,
        MSELoss | CrossEntropyLoss,
        list[Parameter],
        Iterable[tuple[Tensor, Tensor]],
    ],
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by an inverse (exactly) damped KFAC approximation."""
    model_func, loss_func, params, data, batch_size_fn = change_dtype(case, float64)

    KFAC = KFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )

    # Exact damped inverse: inv(KFAC + delta * I)
    inv_KFAC_naive = inv(KFAC @ eye_like(KFAC) + delta * eye_like(KFAC))
    inv_KFAC = KFAC.inverse(damping=delta, use_exact_damping=True)

    compare_consecutive_matmats(inv_KFAC)
    compare_matmat(inv_KFAC, inv_KFAC_naive)


###############################################################################
#                     make_fx backend specific tests                          #
###############################################################################


def _check_make_fx_flatten_different_batch_sizes(linop_cls):
    """Check make_fx with nn.Flatten and different batch sizes.

    ``nn.Flatten`` produces ``aten.view`` with a baked-in batch size during
    real-mode ``make_fx`` tracing, so a single traced function cannot handle
    multiple batch sizes. This verifies that the per-batch-size caching
    handles this correctly.

    Args:
        linop_cls: The linear operator class to test.
    """
    manual_seed(0)
    model = Sequential(Flatten(), Linear(6, 3))
    loss_func = MSELoss()
    params = list(model.parameters())
    # Two batches with different sizes to exercise the per-batch-size cache
    data = [
        (rand(2, 2, 3), regression_targets((2, 3))),
        (rand(5, 2, 3), regression_targets((5, 3))),
    ]

    common_kwargs = dict(
        check_deterministic=False,
        fisher_type=FisherType.EMPIRICAL,
    )
    hooks = linop_cls(model, loss_func, params, data, backend="hooks", **common_kwargs)
    make_fx = linop_cls(
        model, loss_func, params, data, backend="make_fx", **common_kwargs
    )

    hooks_mat = hooks @ eye_like(hooks)
    make_fx_mat = make_fx @ eye_like(make_fx)
    assert allclose_report(hooks_mat, make_fx_mat)


def test_kfac_make_fx_flatten_different_batch_sizes():
    """Test make_fx KFAC with nn.Flatten and different batch sizes."""
    _check_make_fx_flatten_different_batch_sizes(KFACLinearOperator)
