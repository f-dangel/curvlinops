"""Contains tests for ``curvlinops.kfac``."""

from typing import Dict, Iterable, List, Tuple, Union

from einops.layers.torch import Rearrange
from pytest import mark
from torch import (
    Tensor,
    allclose,
    device,
    float64,
    manual_seed,
    rand,
    rand_like,
)
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
from curvlinops.kfac import FisherType, KFACLinearOperator, KFACType
from curvlinops.utils import allclose_report
from test.cases import DEVICES, DEVICES_IDS
from test.utils import (
    Conv2dModel,
    UnetModel,
    WeightShareModel,
    _test_from_state_dict,
    _test_inplace_activations,
    _test_property,
    _test_save_and_load_state_dict,
    binary_classification_targets,
    block_diagonal,
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
def test_kfac_type2(
    kfac_exact_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ],
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
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
    )
    kfac_mat = kfac @ eye_like(kfac)

    assert allclose_report(ggn, kfac_mat)

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(kfac._input_covariances) == 0


@mark.parametrize("setting", [KFACType.EXPAND, KFACType.REDUCE])
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac_type2_weight_sharing(
    kfac_weight_sharing_exact_case: Tuple[
        Union[WeightShareModel, Conv2dModel],
        MSELoss,
        List[Parameter],
        Dict[str, Iterable[Tuple[Tensor, Tensor]]],
    ],
    setting: str,
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
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
    """
    model, loss_func, params, data, batch_size_fn = kfac_weight_sharing_exact_case
    model.setting = setting
    if isinstance(model, Conv2dModel):
        # parameters are only initialized after the setting property is set
        params = [p for p in model.parameters() if p.requires_grad]
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    data = data[setting]

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
    )
    kfac_mat = kfac @ eye_like(kfac)

    assert allclose_report(ggn, kfac_mat, rtol=1e-4)

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(kfac._input_covariances) == 0


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac_mc(
    kfac_exact_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ],
    separate_weight_and_bias: bool,
    exclude: str,
    shuffle: bool,
):
    """Test the KFAC implementation using MC samples against the exact GGN.

    Args:
        kfac_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the KFAC matrix.
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
        fisher_type=FisherType.MC,
        mc_samples=2_000,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac_mat = kfac @ eye_like(kfac)

    atol = {"sum": 5e-1, "mean": 1e-2}[loss_func.reduction]
    rtol = {"sum": 2e-2, "mean": 2e-2}[loss_func.reduction]

    assert allclose_report(ggn, kfac_mat, rtol=rtol, atol=atol)


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("setting", [KFACType.EXPAND, KFACType.REDUCE])
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac_mc_weight_sharing(
    kfac_weight_sharing_exact_case: Tuple[
        Union[WeightShareModel, Conv2dModel],
        MSELoss,
        List[Parameter],
        Dict[str, Iterable[Tuple[Tensor, Tensor]]],
    ],
    separate_weight_and_bias: bool,
    exclude: str,
    setting: str,
    shuffle: bool,
):
    """Test KFAC-MC for linear layers with weight sharing against the exact GGN.

    Args:
        kfac_weight_sharing_exact_case: A fixture that returns a model, loss function,
            list of parameters, and data.
        setting: The weight-sharing setting to use. Can be ``KFACType.EXPAND`` or
            ``KFACType.REDUCE``.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
    """
    model, loss_func, params, data, batch_size_fn = kfac_weight_sharing_exact_case
    model.setting = setting
    if isinstance(model, Conv2dModel):
        # parameters are only initialized after the setting property is set
        params = [p for p in model.parameters() if p.requires_grad]
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    data = data[setting]

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
        fisher_type=FisherType.MC,
        mc_samples=2_000,
        kfac_approx=setting,  # choose KFAC approximation consistent with setting
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac_mat = kfac @ eye_like(kfac)

    atol = {"sum": 5e-1, "mean": 1e-2}[loss_func.reduction]
    rtol = {"sum": 2e-2, "mean": 2e-2}[loss_func.reduction]

    assert allclose_report(ggn, kfac_mat, rtol=rtol, atol=atol)


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac_one_datum(
    kfac_exact_one_datum_case: Tuple[
        Module,
        Union[BCEWithLogitsLoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[Tensor, Tensor]],
    ],
    separate_weight_and_bias: bool,
    exclude: str,
    shuffle: bool,
):
    model, loss_func, params, data, batch_size_fn = kfac_exact_one_datum_case
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
    )
    kfac_mat = kfac @ eye_like(kfac)

    assert allclose_report(ggn, kfac_mat)


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac_mc_one_datum(
    kfac_exact_one_datum_case: Tuple[
        Module,
        Union[BCEWithLogitsLoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[Tensor, Tensor]],
    ],
    separate_weight_and_bias: bool,
    exclude: str,
    shuffle: bool,
):
    model, loss_func, params, data, batch_size_fn = kfac_exact_one_datum_case
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
        fisher_type=FisherType.MC,
        mc_samples=11_000,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac_mat = kfac @ eye_like(kfac)

    atol = {"sum": 1e-3, "mean": 1e-3}[loss_func.reduction]
    rtol = {"sum": 3e-2, "mean": 3e-2}[loss_func.reduction]

    assert allclose_report(ggn, kfac_mat, rtol=rtol, atol=atol)


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac_ef_one_datum(
    kfac_exact_one_datum_case: Tuple[
        Module,
        Union[BCEWithLogitsLoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[Tensor, Tensor]],
    ],
    separate_weight_and_bias: bool,
    exclude: str,
    shuffle: bool,
):
    model, loss_func, params, data, batch_size_fn = kfac_exact_one_datum_case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    ef = block_diagonal(
        EFLinearOperator,
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
        fisher_type=FisherType.EMPIRICAL,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac_mat = kfac @ eye_like(kfac)

    assert allclose_report(ef, kfac_mat)


@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_kfac_inplace_activations(dev: device):
    """Test that KFAC works if the network has in-place activations.

    We use a test case with a single datum as KFAC becomes exact as the number of
    MC samples increases.

    Args:
        dev: The device to run the test on.
    """
    _test_inplace_activations(KFACLinearOperator, dev)


@mark.parametrize("fisher_type", KFACLinearOperator._SUPPORTED_FISHER_TYPE)
@mark.parametrize(
    "loss", [MSELoss, CrossEntropyLoss, BCEWithLogitsLoss], ids=["mse", "ce", "bce"]
)
@mark.parametrize("reduction", ["mean", "sum"])
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_multi_dim_output(
    fisher_type: str,
    loss: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
    reduction: str,
    dev: device,
):
    """Test the KFAC implementation for >2d outputs (using a 3d and 4d output).

    Args:
        fisher_type: The type of Fisher matrix to use.
        loss: The loss function to use.
        reduction: The reduction to use for the loss function.
        dev: The device to run the test on.
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
            (X1, binary_classification_targets((2, 7, 5, 3))),
            (X2, binary_classification_targets((4, 7, 5, 3))),
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
    kfac = KFACLinearOperator(model, loss_func, params, data, fisher_type=fisher_type)
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
    )
    kfac_flat_mat = kfac_flat @ eye_like(kfac_flat)

    assert allclose_report(kfac_mat, kfac_flat_mat)


@mark.parametrize("fisher_type", KFACLinearOperator._SUPPORTED_FISHER_TYPE)
@mark.parametrize(
    "loss", [MSELoss, CrossEntropyLoss, BCEWithLogitsLoss], ids=["mse", "ce", "bce"]
)
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_expand_setting_scaling(
    fisher_type: str,
    loss: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
    dev: device,
):
    """Test KFAC for correct scaling for expand setting with mean reduction loss.

    See #107 for details.

    Args:
        fisher_type: The type of Fisher matrix to use.
        loss: The loss function to use.
        dev: The device to run the test on.
    """
    manual_seed(0)

    # set up data, loss function, and model
    X1 = rand(2, 3, 32, 32)
    X2 = rand(4, 3, 32, 32)
    if issubclass(loss, MSELoss):
        data = [
            (X1, regression_targets((2, 32, 32, 3))),
            (X2, regression_targets((4, 32, 32, 3))),
        ]
    elif issubclass(loss, BCEWithLogitsLoss):
        data = [
            (X1, binary_classification_targets((2, 32, 32, 3))),
            (X2, binary_classification_targets((4, 32, 32, 3))),
        ]
    else:
        data = [
            (X1, classification_targets((2, 32, 32), 3)),
            (X2, classification_targets((4, 32, 32), 3)),
        ]
    model = UnetModel(loss).to(dev)
    params = list(model.parameters())

    # KFAC with sum reduction
    loss_func = loss(reduction="sum").to(dev)
    kfac_sum = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type=fisher_type,
    )
    # FOOF does not scale the gradient covariances, even when using a mean reduction
    if fisher_type != FisherType.FORWARD_ONLY:
        # Simulate a mean reduction by manually scaling the gradient covariances
        loss_term_factor = 32 * 32  # number of spatial locations of model output
        if issubclass(loss, (MSELoss, BCEWithLogitsLoss)):
            output_random_variable_size = 3
            # MSE loss averages over number of output channels
            loss_term_factor *= output_random_variable_size
        for ggT in kfac_sum._gradient_covariances.values():
            ggT.div_(kfac_sum._N_data * loss_term_factor)
    kfac_simulated_mean_mat = kfac_sum @ eye_like(kfac_sum)

    # KFAC with mean reduction
    loss_func = loss(reduction="mean").to(dev)
    kfac_mean = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type=fisher_type,
    )
    kfac_mean_mat = kfac_mean @ eye_like(kfac_mean)

    assert allclose_report(kfac_simulated_mean_mat, kfac_mean_mat)


@mark.parametrize("separate_weight_and_bias", [False], ids=["joint_bias"])
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_KFACLinearOperator(
    case,
    adjoint: bool,
    is_vec: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    shuffle: bool,
):
    """Test matrix multiplication with KFAC.

    Args:
        case: Tuple of model, loss function, parameters, data, and batch size getter.
        adjoint: Whether to test the adjoint operator.
        is_vec: Whether to test matrix-vector or matrix-matrix multiplication.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the KFAC matrix.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
    """
    model, loss_func, params, data, batch_size_fn = case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac_mat = kfac @ eye_like(kfac)

    compare_consecutive_matmats(kfac, adjoint, is_vec)
    compare_matmat(kfac, kfac_mat, adjoint, is_vec, rtol=1e-5, atol=1e-7)


@mark.parametrize(
    "check_deterministic",
    [True, False],
    ids=["check_deterministic", "dont_check_deterministic"],
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_trace(case, exclude, separate_weight_and_bias, check_deterministic, shuffle):
    """Test that the trace property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    _test_property(
        KFACLinearOperator,
        "trace",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        separate_weight_and_bias,
        check_deterministic,
    )


@mark.parametrize(
    "check_deterministic",
    [True, False],
    ids=["check_deterministic", "dont_check_deterministic"],
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_frobenius_norm(
    case, exclude, separate_weight_and_bias, check_deterministic, shuffle
):
    """Test that the Frobenius norm property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    _test_property(
        KFACLinearOperator,
        "frobenius_norm",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        separate_weight_and_bias,
        check_deterministic,
    )


@mark.parametrize(
    "check_deterministic",
    [True, False],
    ids=["check_deterministic", "dont_check_deterministic"],
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_det(case, exclude, separate_weight_and_bias, check_deterministic, shuffle):
    """Test that the determinant property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    _test_property(
        KFACLinearOperator,
        "det",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        separate_weight_and_bias,
        check_deterministic,
        rtol=1e-4,
    )


@mark.parametrize(
    "check_deterministic",
    [True, False],
    ids=["check_deterministic", "dont_check_deterministic"],
)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_logdet(case, exclude, separate_weight_and_bias, check_deterministic, shuffle):
    """Test that the log determinant property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    _test_property(
        KFACLinearOperator,
        "logdet",
        model,
        loss_func,
        params,
        data,
        batch_size_fn,
        separate_weight_and_bias,
        check_deterministic,
        rtol=1e-4,
    )


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_forward_only_fisher_type(
    case: Tuple[Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]],
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
):
    """Test the KFAC with forward-only Fisher (used for FOOF) implementation.

    Args:
        case: A fixture that returns a model, loss function, list of parameters, and
            data.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the KFAC matrix.
    """
    model, loss_func, params, data, batch_size_fn = case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    # Compute KFAC with `fisher_type=FisherType.EMPIRICAL`
    # (could be any but `FisherType.FORWARD_ONLY`)
    foof_simulated = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=FisherType.EMPIRICAL,
    )
    # Manually set all gradient covariances to the identity to simulate FOOF
    for name, block in foof_simulated._gradient_covariances.items():
        foof_simulated._gradient_covariances[name] = eye_like(block)
    simulated_foof_mat = foof_simulated @ eye_like(foof_simulated)

    # Compute KFAC with `fisher_type=FisherType.FORWARD_ONLY`
    foof = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=FisherType.FORWARD_ONLY,
    )
    foof_mat = foof @ eye_like(foof)

    # Check for equivalence
    assert len(foof_simulated._input_covariances) == len(foof._input_covariances)
    assert len(foof_simulated._gradient_covariances) == len(foof._gradient_covariances)
    assert allclose_report(simulated_foof_mat, foof_mat)

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(foof_simulated._input_covariances) == 0
        assert len(foof._input_covariances) == 0


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_forward_only_fisher_type_exact_case(
    single_layer_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ],
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
):
    r"""Test KFAC with forward-only Fisher (FOOF) against exact GGN for one-layer model.

    Consider linear regression with square loss, L =  R * \sum_n^N || W x_n - y_n ||^2,
    where R is the reduction factor from the MSELoss. Per definition,
    FOOF(W) = I \otimes (\sum_n x_n x_n^T / N). Hence, if R = 1 [reduction='sum'], we
    have that GGN(W) = 2 * [I \otimes (\sum_n x_n x_n^T)] = 2 * N * FOOF(W).
    If R = 1 / (N * C) [reduction='mean'], where C is the output dimension, we have
    GGN(W) = 2 * R * [I \otimes (\sum_n x_n x_n^T)] = 2 / C * FOOF(W).

    Args:
        kfac_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the KFAC matrix.
    """
    model, loss_func, params, data, batch_size_fn = single_layer_case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    # Compute exact block-diagonal GGN
    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
    )

    # Compute KFAC with `fisher_type=FisherType.FORWARD_ONLY`
    foof = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=FisherType.FORWARD_ONLY,
    )
    foof_mat = foof @ eye_like(foof)

    # Check for equivalence
    num_data = sum(X.shape[0] for X, _ in data)
    y: Tensor = data[0][1]
    out_dim = y.shape[1]
    # See the docstring for the explanation of the scale
    scale = num_data if loss_func.reduction == "sum" else 1 / out_dim
    assert allclose_report(ggn, 2 * scale * foof_mat)

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(foof._input_covariances) == 0


@mark.parametrize("setting", [KFACType.EXPAND, KFACType.REDUCE])
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_forward_only_fisher_type_exact_weight_sharing_case(
    single_layer_weight_sharing_case: Tuple[
        Union[WeightShareModel, Conv2dModel],
        MSELoss,
        List[Parameter],
        Dict[str, Iterable[Tuple[Tensor, Tensor]]],
    ],
    setting: str,
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
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
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the KFAC matrix.
    """
    model, loss_func, params, data, batch_size_fn = single_layer_weight_sharing_case
    model.setting = setting
    if isinstance(model, Conv2dModel):
        # parameters are only initialized after the setting property is set
        params = [p for p in model.parameters() if p.requires_grad]
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    data = data[setting]

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    foof = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.FORWARD_ONLY,
        kfac_approx=setting,  # choose KFAC approximation consistent with setting
        separate_weight_and_bias=separate_weight_and_bias,
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

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(foof._input_covariances) == 0


def test_kfac_does_not_affect_grad():
    """Make sure KFAC computation does not write to `.grad`."""
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

    # create and compute KFAC
    kfac = KFACLinearOperator(
        model,
        MSELoss(),
        params,
        [(X, y)],
        # suppress computation of KFAC matrices
        check_deterministic=False,
    )
    kfac.compute_kronecker_factors()

    # make sure gradients are unchanged
    for grad_before, p in zip(grads_before, params):
        assert allclose(grad_before, p.grad)


def test_save_and_load_state_dict():
    """Test that KFACLinearOperator can be saved and loaded from state dict."""
    _test_save_and_load_state_dict(KFACLinearOperator)


def test_from_state_dict():
    """Test that KFACLinearOperator can be created from state dict."""
    _test_from_state_dict(KFACLinearOperator)


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


@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_bug_132_dtype_deterministic_checks(dev: device):
    """Test whether the vectors used in the deterministic checks have correct data type.

    This bug was reported in https://github.com/f-dangel/curvlinops/issues/132.

    Args:
        dev: The device to run the test on.
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
    KFACLinearOperator(model, loss_func, params, data, check_deterministic=True)
