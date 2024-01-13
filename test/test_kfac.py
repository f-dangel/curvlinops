"""Contains tests for ``curvlinops.kfac``."""

from test.cases import DEVICES, DEVICES_IDS
from test.utils import (
    Conv2dModel,
    WeightShareModel,
    classification_targets,
    ggn_block_diagonal,
    regression_targets,
)
from typing import Dict, Iterable, List, Tuple, Union

from einops.layers.torch import Rearrange
from numpy import eye
from pytest import mark
from scipy.linalg import block_diag
from torch import Tensor, device, manual_seed, rand, randperm
from torch.nn import (
    CrossEntropyLoss,
    Flatten,
    Linear,
    Module,
    MSELoss,
    Parameter,
    ReLU,
    Sequential,
)

from curvlinops.examples.utils import report_nonclose
from curvlinops.gradient_moments import EFLinearOperator
from curvlinops.kfac import KFACLinearOperator


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
    assert exclude in [None, "weight", "bias"]
    model, loss_func, params, data = kfac_exact_case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    if shuffle:
        permutation = randperm(len(params))
        params = [params[i] for i in permutation]

    ggn = ggn_block_diagonal(
        model,
        loss_func,
        params,
        data,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type="type-2",
        loss_average=loss_average,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac_mat = kfac @ eye(kfac.shape[1])

    report_nonclose(ggn, kfac_mat)

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(kfac._input_covariances) == 0


@mark.parametrize("setting", ["expand", "reduce"])
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
        setting: The weight-sharing setting to use. Can be ``'expand'`` or ``'reduce'``.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the KFAC matrix.
    """
    assert exclude in [None, "weight", "bias"]
    model, loss_func, params, data = kfac_weight_sharing_exact_case
    model.setting = setting
    if isinstance(model, Conv2dModel):
        # parameters are only initialized after the setting property is set
        params = [p for p in model.parameters() if p.requires_grad]
    data = data[setting]

    # set appropriate loss_average argument based on loss reduction and setting
    if loss_func.reduction == "mean":
        if setting == "expand":
            loss_average = "batch+sequence"
        else:
            loss_average = "batch"
    else:
        loss_average = None

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    if shuffle:
        permutation = randperm(len(params))
        params = [params[i] for i in permutation]

    ggn = ggn_block_diagonal(
        model,
        loss_func,
        params,
        data,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type="type-2",
        kfac_approx=setting,  # choose KFAC approximation consistent with setting
        loss_average=loss_average,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac_mat = kfac @ eye(kfac.shape[1])

    report_nonclose(ggn, kfac_mat, rtol=1e-4)

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(kfac._input_covariances) == 0


@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac_mc(
    kfac_exact_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ],
    shuffle: bool,
):
    """Test the KFAC implementation using MC samples against the exact GGN.

    Args:
        kfac_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
    """
    model, loss_func, params, data = kfac_exact_case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    if shuffle:
        permutation = randperm(len(params))
        params = [params[i] for i in permutation]

    ggn = ggn_block_diagonal(model, loss_func, params, data)
    kfac = KFACLinearOperator(
        model, loss_func, params, data, mc_samples=2_000, loss_average=loss_average
    )

    kfac_mat = kfac @ eye(kfac.shape[1])

    atol = {"sum": 5e-1, "mean": 5e-3}[loss_func.reduction]
    rtol = {"sum": 2e-2, "mean": 2e-2}[loss_func.reduction]

    report_nonclose(ggn, kfac_mat, rtol=rtol, atol=atol)


def test_kfac_one_datum(
    kfac_exact_one_datum_case: Tuple[
        Module, CrossEntropyLoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ]
):
    model, loss_func, params, data = kfac_exact_one_datum_case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    ggn = ggn_block_diagonal(model, loss_func, params, data)
    kfac = KFACLinearOperator(
        model, loss_func, params, data, fisher_type="type-2", loss_average=loss_average
    )
    kfac_mat = kfac @ eye(kfac.shape[1])

    report_nonclose(ggn, kfac_mat)


def test_kfac_mc_one_datum(
    kfac_exact_one_datum_case: Tuple[
        Module, CrossEntropyLoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ]
):
    model, loss_func, params, data = kfac_exact_one_datum_case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    ggn = ggn_block_diagonal(model, loss_func, params, data)
    kfac = KFACLinearOperator(
        model, loss_func, params, data, mc_samples=10_000, loss_average=loss_average
    )
    kfac_mat = kfac @ eye(kfac.shape[1])

    atol = {"sum": 1e-3, "mean": 1e-3}[loss_func.reduction]
    rtol = {"sum": 3e-2, "mean": 3e-2}[loss_func.reduction]

    report_nonclose(ggn, kfac_mat, rtol=rtol, atol=atol)


def test_kfac_ef_one_datum(
    kfac_exact_one_datum_case: Tuple[
        Module, CrossEntropyLoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ]
):
    model, loss_func, params, data = kfac_exact_one_datum_case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    ef_blocks = []  # list of per-parameter EFs
    for param in params:
        ef = EFLinearOperator(model, loss_func, [param], data)
        ef_blocks.append(ef @ eye(ef.shape[1]))
    ef = block_diag(*ef_blocks)

    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        fisher_type="empirical",
        loss_average=loss_average,
    )
    kfac_mat = kfac @ eye(kfac.shape[1])

    report_nonclose(ef, kfac_mat)


@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_kfac_inplace_activations(dev: device):
    """Test that KFAC works if the network has in-place activations.

    We use a test case with a single datum as KFAC becomes exact as the number of
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

    # 1) compare KFAC and GGN
    ggn = ggn_block_diagonal(model, loss_func, params, data)

    kfac = KFACLinearOperator(model, loss_func, params, data, mc_samples=2_000)
    kfac_mat = kfac @ eye(kfac.shape[1])

    atol = {"sum": 5e-1, "mean": 2e-3}[loss_func.reduction]
    rtol = {"sum": 2e-2, "mean": 2e-2}[loss_func.reduction]

    report_nonclose(ggn, kfac_mat, rtol=rtol, atol=atol)

    # 2) Compare GGN (inplace=True) and GGN (inplace=False)
    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False
    ggn_no_inplace = ggn_block_diagonal(model, loss_func, params, data)

    report_nonclose(ggn, ggn_no_inplace)


@mark.parametrize("fisher_type", ["type-2", "mc", "empirical"])
@mark.parametrize("loss", [MSELoss, CrossEntropyLoss], ids=["mse", "ce"])
@mark.parametrize("reduction", ["mean", "sum"])
@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_multi_dim_output(
    fisher_type: str,
    loss: Union[MSELoss, CrossEntropyLoss],
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
    loss_average = None if reduction == "sum" else "batch+sequence"
    if isinstance(loss_func, MSELoss):
        data = [
            (rand(2, 7, 5, 5), regression_targets((2, 7, 5, 3))),
            (rand(4, 7, 5, 5), regression_targets((4, 7, 5, 3))),
        ]
        manual_seed(711)
        model = Sequential(Linear(5, 4), Linear(4, 3)).to(dev)
    else:
        data = [
            (rand(2, 7, 5, 5), classification_targets((2, 7, 5), 3)),
            (rand(4, 7, 5, 5), classification_targets((4, 7, 5), 3)),
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
        model,
        loss_func,
        params,
        data,
        fisher_type=fisher_type,
        loss_average=loss_average,
    )
    kfac_mat = kfac @ eye(kfac.shape[1])

    # KFAC for deep linear network with 4d input and equivalent 2d output
    manual_seed(711)
    model_flat = Sequential(
        Linear(5, 4),
        Linear(4, 3),
        Flatten(start_dim=0, end_dim=-2),
    ).to(dev)
    params_flat = list(model_flat.parameters())
    data_flat = [
        (x, y.flatten(start_dim=0, end_dim=-2))
        if isinstance(loss_func, MSELoss)
        else (x, y.flatten(start_dim=0))
        for x, y in data
    ]
    kfac_flat = KFACLinearOperator(
        model_flat,
        loss_func,
        params_flat,
        data_flat,
        fisher_type=fisher_type,
        loss_average=loss_average,
    )
    kfac_flat_mat = kfac_flat @ eye(kfac_flat.shape[1])

    report_nonclose(kfac_mat, kfac_flat_mat)
