"""Contains tests for ``EKFACLinearOperator`` in ``curvlinops.kfac``."""

import os
from typing import Dict, Iterable, List, Tuple, Union

from einops.layers.torch import Rearrange
from pytest import mark, raises
from torch import (
    Tensor,
    allclose,
    device,
    eye,
    isinf,
    isnan,
    linalg,
    load,
    manual_seed,
    rand,
    rand_like,
    save,
)
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Linear,
    Module,
    MSELoss,
    Parameter,
    ReLU,
    Sequential,
)

from curvlinops import EFLinearOperator, GGNLinearOperator
from curvlinops.ekfac import EKFACLinearOperator, FisherType, KFACType
from curvlinops.kfac import KFACLinearOperator
from curvlinops.utils import allclose_report
from test.cases import DEVICES, DEVICES_IDS
from test.utils import (
    Conv2dModel,
    UnetModel,
    WeightShareModel,
    binary_classification_targets,
    block_diagonal,
    classification_targets,
    compare_state_dicts,
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
def test_ekfac_type2(
    kfac_exact_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ],
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
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
        return_numpy=False,
    )
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.TYPE2,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    ekfac_mat = ekfac @ eye(ekfac.shape[1])

    assert allclose_report(ggn, ekfac_mat, atol=1e-6)

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(ekfac._input_covariances_eigenvectors) == 0


@mark.parametrize("setting", [KFACType.EXPAND, KFACType.REDUCE])
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_ekfac_type2_weight_sharing(
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
    """
    model, loss_func, params, data, batch_size_fn = kfac_weight_sharing_exact_case
    # The model outputs have to be flattened assuming only the first dimension is the
    # batch dimension since EKFAC only supports 2d outputs.
    model.setting = "expand-flatten" if "expand" in setting else setting
    if isinstance(model, Conv2dModel):
        # For `Conv2dModel` the parameters are only initialized after the setting
        # property is set, so we have to redefine `params` after `model.setting = ...`.
        params = [p for p in model.parameters() if p.requires_grad]
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    data = data[setting]
    # Flatten targets assuming only the first dimension is the batch dimension
    # since EKFAC only supports 2d targets.
    data = [(X, y.flatten(start_dim=1)) for X, y in data]

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        return_numpy=False,
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
    )
    ekfac_mat = ekfac @ eye(ekfac.shape[1])

    assert allclose_report(ggn, ekfac_mat, rtol=1e-4)

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(ekfac._input_covariances_eigenvectors) == 0


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_ekfac_mc(
    kfac_exact_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ],
    separate_weight_and_bias: bool,
    exclude: str,
    shuffle: bool,
):
    """Test the EKFAC implementation using MC samples against the exact GGN.

    Args:
        kfac_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        shuffle: Whether to shuffle the parameters before computing the EKFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the EKFAC matrix.
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
        return_numpy=False,
    )
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.MC,
        mc_samples=2_000,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    ekfac_mat = ekfac @ eye(ekfac.shape[1])

    atol = {"sum": 5e-1, "mean": 5e-3}[loss_func.reduction]
    rtol = {"sum": 2e-2, "mean": 2e-2}[loss_func.reduction]

    assert allclose_report(ggn, ekfac_mat, rtol=rtol, atol=atol)


@mark.parametrize("setting", [KFACType.EXPAND, KFACType.REDUCE])
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_ekfac_mc_weight_sharing(
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
    """Test EKFAC-MC for linear layers with weight sharing against the exact GGN.

    Args:
        kfac_weight_sharing_exact_case: A fixture that returns a model, loss function,
            list of parameters, and data.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``, or
            ``None``.
        setting: The weight-sharing setting to use. Can be ``KFACType.EXPAND`` or
            ``KFACType.REDUCE``.
        shuffle: Whether to shuffle the parameters before computing the EKFAC matrix.
    """
    model, loss_func, params, data, batch_size_fn = kfac_weight_sharing_exact_case
    # The model outputs have to be flattened assuming only the first dimension is the
    # batch dimension since EKFAC only supports 2d outputs.
    model.setting = "expand-flatten" if "expand" in setting else setting
    if isinstance(model, Conv2dModel):
        # For `Conv2dModel` the parameters are only initialized after the setting
        # property is set, so we have to redefine `params` after `model.setting = ...`.
        params = [p for p in model.parameters() if p.requires_grad]
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)
    data = data[setting]
    # Flatten targets assuming only the first dimension is the batch dimension
    # since EKFAC only supports 2d targets.
    data = [(X, y.flatten(start_dim=1)) for X, y in data]

    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        return_numpy=False,
    )
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.MC,
        mc_samples=2_000,
        kfac_approx=setting,  # choose EKFAC approximation consistent with setting
        separate_weight_and_bias=separate_weight_and_bias,
        check_deterministic=False,
    )
    ekfac_mat = ekfac @ eye(ekfac.shape[1])

    # Scale absolute tolerance by the number of outputs when using sum reduction.
    num_outputs = sum(y.numel() for _, y in data)
    atol = {"sum": 5e-3 * num_outputs, "mean": 5e-3}[loss_func.reduction]
    rtol = {"sum": 2e-2, "mean": 2e-2}[loss_func.reduction]

    assert allclose_report(ggn, ekfac_mat, rtol=rtol, atol=atol)


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_ekfac_one_datum(
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
        return_numpy=False,
    )
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.TYPE2,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    ekfac_mat = ekfac @ eye(ekfac.shape[1])

    assert allclose_report(ggn, ekfac_mat)


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_ekfac_mc_one_datum(
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
        return_numpy=False,
    )
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.MC,
        mc_samples=11_000,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    ekfac_mat = ekfac @ eye(ekfac.shape[1])

    atol = {"sum": 1e-3, "mean": 1e-3}[loss_func.reduction]
    rtol = {"sum": 3e-2, "mean": 3e-2}[loss_func.reduction]

    assert allclose_report(ggn, ekfac_mat, rtol=rtol, atol=atol)


@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_ekfac_ef_one_datum(
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
        return_numpy=False,
    )

    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type=FisherType.EMPIRICAL,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    ekfac_mat = ekfac @ eye(ekfac.shape[1])

    assert allclose_report(ef, ekfac_mat, atol=1e-7)


@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_ekfac_inplace_activations(dev: device):
    """Test that EKFAC works if the network has in-place activations.

    We use a test case with a single datum as EKFAC becomes exact as the number of
    MC samples increases.

    Args:
        dev: The device to run the test on.
    """
    manual_seed(0)
    model = Sequential(Linear(6, 3), ReLU(inplace=True), Linear(3, 2)).to(dev)
    loss_func = MSELoss().to(dev)
    batch_size = 1
    data = [(rand(batch_size, 6), regression_targets((batch_size, 2)))]
    params = list(model.parameters())

    # 1) compare EKFAC and GGN
    ggn = block_diagonal(
        GGNLinearOperator, model, loss_func, params, data, return_numpy=False
    )

    ekfac = EKFACLinearOperator(model, loss_func, params, data, mc_samples=2_000)
    ekfac_mat = ekfac @ eye(ekfac.shape[1])

    atol = {"sum": 5e-1, "mean": 2e-3}[loss_func.reduction]
    rtol = {"sum": 2e-2, "mean": 2e-2}[loss_func.reduction]

    assert allclose_report(ggn, ekfac_mat, rtol=rtol, atol=atol)

    # 2) Compare GGN (inplace=True) and GGN (inplace=False)
    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False
    ggn_no_inplace = block_diagonal(
        GGNLinearOperator, model, loss_func, params, data, return_numpy=False
    )

    assert allclose_report(ggn, ggn_no_inplace)


@mark.parametrize("fisher_type", EKFACLinearOperator._SUPPORTED_FISHER_TYPE)
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
    """Test the EKFAC implementation for >2d outputs (using a 3d and 4d output).

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

    # EKFAC for deep linear network with 4d input and output
    params = list(model.parameters())
    with raises(ValueError, match="Only 2d output"):
        EKFACLinearOperator(
            model,
            loss_func,
            params,
            data,
            fisher_type=fisher_type,
        )


@mark.parametrize("fisher_type", EKFACLinearOperator._SUPPORTED_FISHER_TYPE)
@mark.parametrize(
    "loss", [MSELoss, CrossEntropyLoss, BCEWithLogitsLoss], ids=["mse", "ce", "bce"]
)
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_expand_setting_scaling(
    fisher_type: str,
    loss: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
    dev: device,
):
    """Test EKFAC for correct scaling for expand setting with mean reduction loss.

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
    # only 2d target is supported for MSE/BCE and 1d output for CE loss
    if issubclass(loss, MSELoss):
        data = [
            (X1, regression_targets((2, 32 * 32 * 3))),
            (X2, regression_targets((4, 32 * 32 * 3))),
        ]
    elif issubclass(loss, BCEWithLogitsLoss):
        data = [
            (X1, binary_classification_targets((2, 32 * 32 * 3))),
            (X2, binary_classification_targets((4, 32 * 32 * 3))),
        ]
    else:
        data = [
            (X1, classification_targets((2 * 32 * 32,), 3)),
            (X2, classification_targets((4 * 32 * 32,), 3)),
        ]
    model = UnetModel(loss, flatten=True).to(dev)
    params = list(model.parameters())

    # EKFAC with sum reduction
    loss_func = loss(reduction="sum").to(dev)
    ekfac_sum = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type=fisher_type,
    )
    # Simulate a mean reduction by manually scaling the gradient covariances
    loss_term_factor = 32 * 32  # number of spatial locations of model output
    if issubclass(loss, (MSELoss, BCEWithLogitsLoss)):
        output_random_variable_size = 3
        # MSE loss averages over number of output channels
        loss_term_factor *= output_random_variable_size
    correction = ekfac_sum._N_data * loss_term_factor
    for eigenvalues in ekfac_sum._corrected_eigenvalues.values():
        if isinstance(eigenvalues, dict):
            for eigenvals in eigenvalues.values():
                eigenvals /= correction
        else:
            eigenvalues /= correction
    ekfac_simulated_mean_mat = ekfac_sum @ eye(ekfac_sum.shape[1], device=dev)

    # EKFAC with mean reduction
    loss_func = loss(reduction="mean").to(dev)
    ekfac_mean = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type=fisher_type,
    )
    ekfac_mean_mat = ekfac_mean @ eye(ekfac_mean.shape[1], device=dev)

    assert allclose_report(ekfac_simulated_mean_mat, ekfac_mean_mat, atol=1e-4)


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
def test_trace(
    inv_case, exclude, separate_weight_and_bias, check_deterministic, shuffle
):
    """Test that the trace property of EKFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = inv_case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        check_deterministic=check_deterministic,
    )

    # Check for equivalence of trace property and naive trace computation
    trace = ekfac.trace
    trace_naive = (ekfac @ eye(ekfac.shape[1])).trace()
    assert allclose_report(trace, trace_naive)

    # Check that the trace property is properly cached and reset
    assert ekfac._trace == trace
    ekfac.compute_kronecker_factors()
    assert ekfac._trace is None


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
    inv_case, exclude, separate_weight_and_bias, check_deterministic, shuffle
):
    """Test that the Frobenius norm property of EKFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = inv_case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        check_deterministic=check_deterministic,
    )

    # Check for equivalence of frobenius_norm property and the naive computation
    frobenius_norm = ekfac.frobenius_norm
    frobenius_norm_naive = linalg.matrix_norm(ekfac @ eye(ekfac.shape[1]))
    assert allclose_report(frobenius_norm, frobenius_norm_naive)

    # Check that the frobenius_norm property is properly cached and reset
    assert ekfac._frobenius_norm == frobenius_norm
    ekfac.compute_kronecker_factors()
    assert ekfac._frobenius_norm is None


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
def test_det(inv_case, exclude, separate_weight_and_bias, check_deterministic, shuffle):
    """Test that the determinant property of EKFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = inv_case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        check_deterministic=check_deterministic,
    )

    # add damping manually to avoid singular matrices
    if not check_deterministic:
        ekfac.compute_kronecker_factors()
        ekfac.compute_eigenvalue_correction()
    assert ekfac._corrected_eigenvalues
    delta = 1.0  # requires much larger damping value compared to ``logdet``
    for eigenvalues in ekfac._corrected_eigenvalues.values():
        if isinstance(eigenvalues, dict):
            for eigenvals in eigenvalues.values():
                eigenvals.add_(delta)
        else:
            eigenvalues.add_(delta)

    # Check for equivalence of the det property and naive determinant computation
    determinant = ekfac.det
    # verify that the determinant is not trivial as this would make the test useless
    assert determinant not in {0.0, 1.0}
    det_naive = (ekfac @ eye(ekfac.shape[1])).det()
    assert allclose_report(determinant, det_naive, rtol=1e-4)

    # Check that the det property is properly cached and reset
    assert ekfac._det == determinant
    ekfac.compute_kronecker_factors()
    ekfac.compute_eigenvalue_correction()
    assert ekfac._det is None


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
def test_logdet(
    inv_case, exclude, separate_weight_and_bias, check_deterministic, shuffle
):
    """Test that the log determinant property of EKFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = inv_case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        check_deterministic=check_deterministic,
    )

    # add damping manually to avoid singular matrices
    if not check_deterministic:
        ekfac.compute_kronecker_factors()
        ekfac.compute_eigenvalue_correction()
    assert ekfac._corrected_eigenvalues
    delta = 1e-3  # only requires much smaller damping value compared to ``det``
    for eigenvalues in ekfac._corrected_eigenvalues.values():
        if isinstance(eigenvalues, dict):
            for eigenvals in eigenvalues.values():
                eigenvals.add_(delta)
        else:
            eigenvalues.add_(delta)

    # Check for equivalence of the logdet property and naive log determinant computation
    log_det = ekfac.logdet
    # verify that the log determinant is finite and not nan
    assert not isinf(log_det) and not isnan(log_det)
    log_det_naive = (ekfac @ eye(ekfac.shape[1])).logdet()
    assert allclose_report(log_det, log_det_naive, rtol=1e-4)

    # Check that the logdet property is properly cached and reset
    assert ekfac._logdet == log_det
    ekfac.compute_kronecker_factors()
    assert ekfac._logdet is None


def test_ekfac_does_not_affect_grad():
    """Make sure EKFAC computation does not write to `.grad`."""
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

    # create and compute EKFAC
    ekfac = EKFACLinearOperator(
        model,
        MSELoss(),
        params,
        [(X, y)],
        # suppress computation of EKFAC matrices
        check_deterministic=False,
    )
    ekfac.compute_kronecker_factors()
    ekfac.compute_eigenvalue_correction()

    # make sure gradients are unchanged
    for grad_before, p in zip(grads_before, params):
        assert allclose(grad_before, p.grad)


def test_save_and_load_state_dict():
    """Test that EKFACLinearOperator can be saved and loaded from state dict."""
    manual_seed(0)
    batch_size, D_in, D_out = 4, 3, 2
    X = rand(batch_size, D_in)
    y = rand(batch_size, D_out)
    model = Linear(D_in, D_out)

    params = list(model.parameters())
    # create and compute EKFAC
    ekfac = EKFACLinearOperator(
        model,
        MSELoss(reduction="sum"),
        params,
        [(X, y)],
    )

    # save state dict
    state_dict = ekfac.state_dict()
    EKFAC_PATH = "ekfac_state_dict.pt"
    save(state_dict, EKFAC_PATH)

    # create new EKFAC with different loss function and try to load state dict
    ekfac_new = EKFACLinearOperator(
        model,
        CrossEntropyLoss(),
        params,
        [(X, y)],
    )
    with raises(ValueError, match="loss"):
        ekfac_new.load_state_dict(load(EKFAC_PATH, weights_only=False))

    # create new EKFAC with different loss reduction and try to load state dict
    ekfac_new = EKFACLinearOperator(
        model,
        MSELoss(),
        params,
        [(X, y)],
    )
    with raises(ValueError, match="reduction"):
        ekfac_new.load_state_dict(load(EKFAC_PATH, weights_only=False))

    # create new EKFAC with different model and try to load state dict
    wrong_model = Sequential(Linear(D_in, 10), ReLU(), Linear(10, D_out))
    wrong_params = list(wrong_model.parameters())
    ekfac_new = EKFACLinearOperator(
        wrong_model,
        MSELoss(reduction="sum"),
        wrong_params,
        [(X, y)],
    )
    with raises(RuntimeError, match="loading state_dict"):
        ekfac_new.load_state_dict(load(EKFAC_PATH, weights_only=False))

    # create new EKFAC and load state dict
    ekfac_new = EKFACLinearOperator(
        model,
        MSELoss(reduction="sum"),
        params,
        [(X, y)],
        check_deterministic=False,  # turn off to avoid computing EKFAC again
    )
    ekfac_new.load_state_dict(load(EKFAC_PATH, weights_only=False))
    # clean up
    os.remove(EKFAC_PATH)

    # check that the two EKFACs are equal
    compare_state_dicts(ekfac.state_dict(), ekfac_new.state_dict())
    test_vec = rand(ekfac.shape[1])
    assert allclose_report(ekfac @ test_vec, ekfac_new @ test_vec)


def test_from_state_dict():
    """Test that EKFACLinearOperator can be created from state dict."""
    manual_seed(0)
    batch_size, D_in, D_out = 4, 3, 2
    X = rand(batch_size, D_in)
    y = rand(batch_size, D_out)
    model = Linear(D_in, D_out)

    params = list(model.parameters())
    # create and compute EKFAC
    ekfac = EKFACLinearOperator(
        model,
        MSELoss(reduction="sum"),
        params,
        [(X, y)],
    )

    # save state dict
    state_dict = ekfac.state_dict()

    # create new EKFAC from state dict
    kfac_new = EKFACLinearOperator.from_state_dict(state_dict, model, params, [(X, y)])

    # check that the two EKFACs are equal
    compare_state_dicts(ekfac.state_dict(), kfac_new.state_dict())
    test_vec = rand(ekfac.shape[1])
    assert allclose_report(ekfac @ test_vec, kfac_new @ test_vec)


@mark.parametrize("fisher_type", EKFACLinearOperator._SUPPORTED_FISHER_TYPE)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_ekfac_closer_to_exact_than_kfac(
    inv_case,
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    fisher_type: FisherType,
):
    """Test that EKFAC is closer in Frobenius norm to the exact quantity than KFAC."""
    model, loss_func, params, data, batch_size_fn = inv_case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    # Compute exact block-wise ground truth quantity.
    linop = (
        EFLinearOperator if fisher_type == FisherType.EMPIRICAL else GGNLinearOperator
    )
    exact = block_diagonal(
        linop,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        return_numpy=False,
    )

    # Compute KFAC and EKFAC.
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=fisher_type,
        mc_samples=1_000 if fisher_type == FisherType.MC else 1,
    )
    kfac_mat = kfac @ eye(kfac.shape[1])
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=fisher_type,
        mc_samples=1_000 if fisher_type == FisherType.MC else 1,
    )
    ekfac_mat = ekfac @ eye(ekfac.shape[1])

    # Compute and compare (relative) distances to the exact quantity.
    exact_norm = linalg.matrix_norm(exact)
    exact_kfac_dist = linalg.matrix_norm(exact - kfac_mat) / exact_norm
    exact_ekfac_dist = linalg.matrix_norm(exact - ekfac_mat) / exact_norm
    if exclude == "weight":
        # For no_weights the numerical error dominates.
        assert allclose_report(exact_kfac_dist, exact_ekfac_dist, atol=1e-5)
    else:
        assert exact_kfac_dist > exact_ekfac_dist


# TODO: FisherType.MC is too expensive, but could be added as optional test?
@mark.parametrize("fisher_type", [FisherType.TYPE2, FisherType.EMPIRICAL])
@mark.parametrize("kfac_approx", EKFACLinearOperator._SUPPORTED_KFAC_APPROX)
@mark.parametrize(
    "separate_weight_and_bias", [True, False], ids=["separate_bias", "joint_bias"]
)
@mark.parametrize(
    "exclude", [None, "weight", "bias"], ids=["all", "no_weights", "no_biases"]
)
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_ekfac_closer_to_exact_than_kfac_weight_sharing(
    cnn_case,
    shuffle: bool,
    exclude: str,
    separate_weight_and_bias: bool,
    kfac_approx: KFACType,
    fisher_type: FisherType,
):
    """Test that EKFAC is closer in Frobenius norm to the exact quantity than KFAC.

    For models with weight sharing.
    """
    model, loss_func, params, data, batch_size_fn = cnn_case
    params = maybe_exclude_or_shuffle_parameters(params, model, exclude, shuffle)

    # Compute exact block-wise ground truth quantity.
    linop = (
        EFLinearOperator if fisher_type == FisherType.EMPIRICAL else GGNLinearOperator
    )
    exact = block_diagonal(
        linop,
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        return_numpy=False,
    )

    # Compute KFAC and EKFAC.
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=fisher_type,
        kfac_approx=kfac_approx,
    )
    kfac_mat = kfac @ eye(kfac.shape[1])
    ekfac = EKFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type=fisher_type,
        kfac_approx=kfac_approx,
    )
    ekfac_mat = ekfac @ eye(ekfac.shape[1])

    # Compute and compare (relative) distances to the exact quantity.
    exact_norm = linalg.matrix_norm(exact)
    exact_kfac_dist = linalg.matrix_norm(exact - kfac_mat) / exact_norm
    exact_ekfac_dist = linalg.matrix_norm(exact - ekfac_mat) / exact_norm
    assert exact_kfac_dist > exact_ekfac_dist or (
        allclose_report(exact_kfac_dist, exact_ekfac_dist, atol=1e-6)
        if exclude == "weight"
        else False
    )  # For no_weights the numerical error might dominate.
