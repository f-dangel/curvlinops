"""Contains tests for ``EKFACLinearOperator`` in ``curvlinops.kfac``."""

from collections.abc import Iterable
from pathlib import Path

from einops.layers.torch import Rearrange
from pytest import mark, raises
from torch import Tensor, device, float64, manual_seed, rand
from torch.linalg import inv
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Linear,
    Module,
    MSELoss,
    Sequential,
)

from curvlinops import EFLinearOperator, GGNLinearOperator
from curvlinops.computers.ekfac_hooks import HooksEKFACComputer
from curvlinops.ekfac import EKFACLinearOperator
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import allclose_report
from test.cases import DEVICES, DEVICES_IDS
from test.test_kfac import (
    BACKENDS,
    BACKENDS_IDS,
    MC_SAMPLES,
    MC_TOLS,
    _check_callable_model_func,
    _check_does_not_affect_grad,
    _check_does_not_affect_requires_grad,
    _check_make_fx_flatten_different_batch_sizes,
    _check_torch_save_load,
    _test_weight_tying_type2,
)
from test.utils import (
    Conv2dModel,
    UnetModel,
    WeightShareModel,
    _test_ekfac_closer_to_exact_than_kfac,
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


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_ekfac_type2(
    kfac_exact_case: tuple[
        Module, MSELoss, dict[str, Tensor], Iterable[tuple[Tensor, Tensor]]
    ],
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    backend: str,
):
    """Test the EKFAC implementation against the exact GGN.

    Args:
        kfac_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        shuffle: Whether to shuffle the parameters before computing the EKFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the EKFAC matrix.
        backend: The backend to use for computing Kronecker factors.
    """
    model, loss_func, params, data, batch_size_fn = change_dtype(
        kfac_exact_case, float64
    )
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
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.TYPE2,
        separate_weight_and_bias=separate_weight_and_bias,
        backend=backend,
    )
    ekfac_mat = ekfac @ eye_like(ekfac)

    assert allclose_report(ggn, ekfac_mat)


@mark.parametrize("setting", [KFACType.EXPAND, KFACType.REDUCE])
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_ekfac_type2_weight_sharing(
    kfac_weight_sharing_exact_case: tuple[
        WeightShareModel | Conv2dModel,
        MSELoss,
        dict[str, Tensor],
        dict[str, Iterable[tuple[Tensor, Tensor]]],
    ],
    setting: str,
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    backend: str,
):
    """Test EKFAC for linear weight-sharing layers against the exact GGN.

    Args:
        kfac_weight_sharing_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        setting: The weight-sharing setting to use. Can be ``KFACType.EXPAND`` or
            ``KFACType.REDUCE``.
        shuffle: Whether to shuffle the parameters before computing the EKFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the EKFAC matrix.
        backend: The backend to use for computing Kronecker factors.
    """
    model, loss_func, params, data, batch_size_fn = kfac_weight_sharing_exact_case
    # The model outputs have to be flattened assuming only the first dimension is the
    # batch dimension since EKFAC only supports 2d outputs.
    model.setting = "expand-flatten" if "expand" in setting else setting
    if isinstance(model, Conv2dModel):
        # For `Conv2dModel` the parameters are only initialized after the setting
        # property is set, so we have to redefine `params` after `model.setting = ...`.
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    data = data[setting]
    # Flatten targets assuming only the first dimension is the batch dimension
    # since EKFAC only supports 2d targets.
    data = [(X, y.flatten(start_dim=1)) for X, y in data]
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
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.TYPE2,
        kfac_approx=setting,  # choose EKFAC approximation consistent with setting
        separate_weight_and_bias=separate_weight_and_bias,
        backend=backend,
    )
    ekfac_mat = ekfac @ eye_like(ekfac)

    assert allclose_report(ggn, ekfac_mat)


@mark.parametrize("reduction", ["mean", "sum"])
@mark.parametrize("bias", [False, True], ids=["no_bias", "with_bias"])
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
def test_ekfac_type2_weight_tying(
    reduction: str, bias: bool, separate_weight_and_bias: bool
):
    """Test EKFAC with weight tying against exact GGN (make_fx only)."""
    _test_weight_tying_type2(
        EKFACLinearOperator, reduction, bias, separate_weight_and_bias
    )


def test_ekfac_mc(
    kfac_exact_case: tuple[
        Module, MSELoss, dict[str, Tensor], Iterable[tuple[Tensor, Tensor]]
    ],
):
    """Test the EKFAC implementation using MC samples against the exact GGN.

    Args:
        kfac_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
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
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.MC,
        mc_samples=MC_SAMPLES,
    )
    ekfac_mat = ekfac @ eye_like(ekfac)

    # Normalize so we can share tolerances across reductions
    scale = ggn.abs().max()
    assert allclose_report(ggn / scale, ekfac_mat / scale, **MC_TOLS)


@mark.parametrize("setting", [KFACType.EXPAND, KFACType.REDUCE])
def test_ekfac_mc_weight_sharing(
    kfac_weight_sharing_exact_case: tuple[
        WeightShareModel | Conv2dModel,
        MSELoss,
        dict[str, Tensor],
        dict[str, Iterable[tuple[Tensor, Tensor]]],
    ],
    setting: str,
):
    """Test EKFAC-MC for linear layers with weight sharing against the exact GGN.

    Args:
        kfac_weight_sharing_exact_case: A fixture that returns a model, loss function,
            dict of parameters, and data.
        setting: The weight-sharing setting to use. Can be ``KFACType.EXPAND`` or
            ``KFACType.REDUCE``.
    """
    model, loss_func, params, data, batch_size_fn = kfac_weight_sharing_exact_case
    # The model outputs have to be flattened assuming only the first dimension is the
    # batch dimension since EKFAC only supports 2d outputs.
    model.setting = "expand-flatten" if "expand" in setting else setting
    if isinstance(model, Conv2dModel):
        # For `Conv2dModel` the parameters are only initialized after the setting
        # property is set, so we have to redefine `params` after `model.setting = ...`.
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    data = data[setting]
    # Flatten targets assuming only the first dimension is the batch dimension
    # since EKFAC only supports 2d targets.
    data = [(X, y.flatten(start_dim=1)) for X, y in data]
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
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.MC,
        mc_samples=MC_SAMPLES,
        kfac_approx=setting,  # choose EKFAC approximation consistent with setting
    )
    ekfac_mat = ekfac @ eye_like(ekfac)

    # Normalize so we can share tolerances across reductions
    scale = ggn.abs().max()
    assert allclose_report(ggn / scale, ekfac_mat / scale, **MC_TOLS)


def test_ekfac_one_datum(
    kfac_exact_one_datum_case: tuple[
        Module,
        BCEWithLogitsLoss | CrossEntropyLoss,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
    ],
):
    """Test EKFAC for the one-datum exact case."""
    model, loss_func, params, data, batch_size_fn = kfac_exact_one_datum_case

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.TYPE2,
    )
    ekfac_mat = ekfac @ eye_like(ekfac)

    assert allclose_report(ggn, ekfac_mat)


def test_ekfac_mc_one_datum(
    kfac_exact_one_datum_case: tuple[
        Module,
        BCEWithLogitsLoss | CrossEntropyLoss,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
    ],
):
    """Test EKFAC-MC for the one-datum exact case."""
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
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.MC,
        mc_samples=MC_SAMPLES,
    )
    ekfac_mat = ekfac @ eye_like(ekfac)

    # Normalize so we can share tolerances across reductions
    scale = ggn.abs().max()
    # Need to use larger tolerances on GPU despite float64
    tols = (
        MC_TOLS
        if "cpu" in str(next(iter(params.values())).device)
        else {k: 2 * v for k, v in MC_TOLS.items()}
    )
    assert allclose_report(ggn / scale, ekfac_mat / scale, **tols)


def test_ekfac_ef_one_datum(
    kfac_exact_one_datum_case: tuple[
        Module,
        BCEWithLogitsLoss | CrossEntropyLoss,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
    ],
):
    """Test EKFAC empirical Fisher for the one-datum exact case."""
    model, loss_func, params, data, batch_size_fn = change_dtype(
        kfac_exact_one_datum_case, float64
    )

    ef = block_diagonal(
        EFLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )

    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.EMPIRICAL,
    )
    ekfac_mat = ekfac @ eye_like(ekfac)

    assert allclose_report(ef, ekfac_mat)


@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_ekfac_inplace_activations(dev: device, backend: str):
    """Test that EKFAC works if the network has in-place activations.

    We use a test case with a single datum as EKFAC becomes exact as the number of
    MC samples increases.

    Args:
        dev: The device to run the test on.
        backend: The backend to use for computing Kronecker factors.
    """
    _test_inplace_activations(EKFACLinearOperator, dev, backend=backend)


@mark.parametrize("fisher_type", HooksEKFACComputer._SUPPORTED_FISHER_TYPE)
@mark.parametrize(
    "loss", [MSELoss, CrossEntropyLoss, BCEWithLogitsLoss], ids=["mse", "ce", "bce"]
)
@mark.parametrize("reduction", ["mean", "sum"])
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_multi_dim_output(
    fisher_type: str,
    loss: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
    reduction: str,
    dev: device,
    backend: str,
):
    """Test the EKFAC implementation for >2d outputs (using a 3d and 4d output).

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

    # EKFAC for deep linear network with 4d input and output
    params = dict(model.named_parameters())
    with raises(ValueError, match="Only 2d output"):
        EKFACLinearOperator(
            model,
            loss_func,
            params,
            data,
            fisher_type=fisher_type,
            backend=backend,
        )


@mark.parametrize("fisher_type", HooksEKFACComputer._SUPPORTED_FISHER_TYPE)
@mark.parametrize(
    "loss", [MSELoss, CrossEntropyLoss, BCEWithLogitsLoss], ids=["mse", "ce", "bce"]
)
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
def test_expand_setting_scaling(
    fisher_type: str,
    loss: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
    dev: device,
    backend: str,
):
    """Test EKFAC for correct scaling for expand setting with mean reduction loss.

    See #107 for details.

    Args:
        fisher_type: The type of Fisher matrix to use.
        loss: The loss function to use.
        dev: The device to run the test on.
        backend: The backend to use for computing Kronecker factors.
    """
    manual_seed(0)

    # set up data, loss function, and model
    S = 8  # spatial size (small for speed; UnetModel bottleneck is S/2)
    X1 = rand(2, 3, S, S)
    X2 = rand(4, 3, S, S)
    # only 2d target is supported for MSE/BCE and 1d output for CE loss
    if issubclass(loss, MSELoss):
        data = [
            (X1, regression_targets((2, S * S * 3))),
            (X2, regression_targets((4, S * S * 3))),
        ]
    elif issubclass(loss, BCEWithLogitsLoss):
        data = [
            (X1, rand(2, S * S * 3)),
            (X2, rand(4, S * S * 3)),
        ]
    else:
        data = [
            (X1, classification_targets((2 * S * S,), 3)),
            (X2, classification_targets((4 * S * S,), 3)),
        ]
    model = UnetModel(loss, flatten=True).to(dev)
    params = dict(model.named_parameters())

    # EKFAC with sum reduction
    loss_func = loss(reduction="sum").to(dev)
    model, loss_func, params, data, _ = change_dtype(
        (model, loss_func, params, data, None), float64
    )
    ekfac_sum = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type=fisher_type,
        backend=backend,
    )
    # Simulate a mean reduction by manually scaling the gradient covariances
    loss_term_factor = S * S  # number of spatial locations of model output
    if issubclass(loss, (MSELoss, BCEWithLogitsLoss)):
        output_random_variable_size = 3
        # MSE loss averages over number of output channels
        loss_term_factor *= output_random_variable_size
    num_data = sum(X.shape[0] for X, _ in data)
    correction = num_data * loss_term_factor
    _, K, _ = ekfac_sum
    for block in K:
        block.eigenvalues = block.eigenvalues / correction
    ekfac_simulated_mean_mat = ekfac_sum @ eye_like(ekfac_sum)

    # EKFAC with mean reduction
    loss_func = loss(reduction="mean").to(dev)
    model, loss_func, params, data, _ = change_dtype(
        (model, loss_func, params, data, None), float64
    )
    ekfac_mean = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type=fisher_type,
        backend=backend,
    )
    ekfac_mean_mat = ekfac_mean @ eye_like(ekfac_mean)

    assert allclose_report(ekfac_simulated_mean_mat, ekfac_mean_mat)


def test_trace(inv_case):
    """Test that the trace property of EKFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = inv_case
    _test_property(
        EKFACLinearOperator,
        "trace",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
    )


def test_frobenius_norm(inv_case):
    """Test that the Frobenius norm property of EKFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = inv_case
    _test_property(
        EKFACLinearOperator,
        "frobenius_norm",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
    )


def test_det(inv_case):
    """Test that the determinant property of EKFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = inv_case
    _test_property(
        EKFACLinearOperator,
        "det",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        rtol=1e-4,
    )


def test_logdet(inv_case):
    """Test that the log determinant property of EKFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = inv_case
    _test_property(
        EKFACLinearOperator,
        "logdet",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        rtol=1e-4,
    )


def test_ekfac_does_not_affect_grad():
    """Make sure EKFAC computation does not write to `.grad`."""
    _check_does_not_affect_grad(EKFACLinearOperator)


def test_ekfac_make_fx_preserves_requires_grad():
    """EKFAC's FX backend must not mutate the user's ``requires_grad`` flags."""
    _check_does_not_affect_requires_grad(EKFACLinearOperator)


def test_ekfac_torch_save_load(tmp_path: Path) -> None:
    """Test that EKFACLinearOperator can be saved and loaded with torch.save/load."""
    _check_torch_save_load(EKFACLinearOperator, tmp_path)


# TODO: Add test for FisherType.MC once tests are in float64.
@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
@mark.parametrize("fisher_type", [FisherType.TYPE2, FisherType.EMPIRICAL])
@mark.parametrize("kfac_approx", HooksEKFACComputer._SUPPORTED_KFAC_APPROX)
def test_ekfac_closer_to_exact_than_kfac(
    inv_case,
    fisher_type: FisherType,
    kfac_approx: KFACType,
    backend: str,
):
    """Test that EKFAC is closer in Frobenius norm to the exact quantity than KFAC."""
    model, loss_func, params, data, batch_size_fn = change_dtype(inv_case, float64)
    _test_ekfac_closer_to_exact_than_kfac(
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        fisher_type=fisher_type,
        kfac_approx=kfac_approx,
        backend=backend,
    )


@mark.parametrize("backend", BACKENDS, ids=BACKENDS_IDS)
@mark.parametrize("fisher_type", HooksEKFACComputer._SUPPORTED_FISHER_TYPE)
@mark.parametrize("kfac_approx", HooksEKFACComputer._SUPPORTED_KFAC_APPROX)
def test_ekfac_closer_to_exact_than_kfac_weight_sharing(
    cnn_case,
    kfac_approx: KFACType,
    fisher_type: FisherType,
    backend: str,
):
    """Test that EKFAC is closer in Frobenius norm to the exact quantity than KFAC.

    For models with weight sharing.
    """
    model, loss_func, params, data, batch_size_fn = change_dtype(cnn_case, float64)
    _test_ekfac_closer_to_exact_than_kfac(
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        fisher_type=fisher_type,
        kfac_approx=kfac_approx,
        backend=backend,
    )


"""EKFACLinearOperator.inverse() tests."""


def test_EKFAC_inverse_exactly_damped_matmat(
    inv_case: tuple[
        Module,
        MSELoss | CrossEntropyLoss,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
    ],
    delta: float = 1e-2,
):
    """Test matrix-matrix multiplication by an inverse (exactly) damped EKFAC approximation."""
    model_func, loss_func, params, data, batch_size_fn = change_dtype(inv_case, float64)

    EKFAC = EKFACLinearOperator(
        model_func,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
    )

    # Exact damped inverse: inv(EKFAC + delta * I)
    inv_EKFAC_naive = inv(EKFAC @ eye_like(EKFAC) + delta * eye_like(EKFAC))
    inv_EKFAC = EKFAC.inverse(damping=delta)

    compare_consecutive_matmats(inv_EKFAC)
    compare_matmat(inv_EKFAC, inv_EKFAC_naive)


def test_ekfac_make_fx_flatten_different_batch_sizes():
    """Test make_fx EKFAC with nn.Flatten and different batch sizes."""
    _check_make_fx_flatten_different_batch_sizes(EKFACLinearOperator)


def test_ekfac_callable_model_func():
    """Test EKFAC make_fx with a plain Callable model_func."""
    _check_callable_model_func(EKFACLinearOperator)
