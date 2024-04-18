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

from einops import rearrange
from einops.layers.torch import Rearrange
from numpy import eye
from numpy.linalg import det, norm, slogdet
from pytest import mark, raises, skip
from scipy.linalg import block_diag
from torch import Tensor, cat, cuda, device
from torch import eye as torch_eye
from torch import isinf, isnan, manual_seed, rand, randperm
from torch.nn import (
    BCEWithLogitsLoss,
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
    model, loss_func, params, data, batch_size_fn = kfac_exact_case
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
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
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
    model, loss_func, params, data, batch_size_fn = kfac_weight_sharing_exact_case
    model.setting = setting
    if isinstance(model, Conv2dModel):
        # parameters are only initialized after the setting property is set
        params = [p for p in model.parameters() if p.requires_grad]
    data = data[setting]

    # set appropriate loss_average argument based on loss reduction and setting
    loss_average = (
        ("batch+sequence" if setting == "expand" else "batch")
        if loss_func.reduction == "mean"
        else None
    )

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
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
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
    model, loss_func, params, data, batch_size_fn = kfac_exact_case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    if shuffle:
        permutation = randperm(len(params))
        params = [params[i] for i in permutation]

    ggn = ggn_block_diagonal(
        model, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        mc_samples=2_000,
        loss_average=loss_average,
    )

    kfac_mat = kfac @ eye(kfac.shape[1])

    atol = {"sum": 5e-1, "mean": 5e-3}[loss_func.reduction]
    rtol = {"sum": 2e-2, "mean": 2e-2}[loss_func.reduction]

    report_nonclose(ggn, kfac_mat, rtol=rtol, atol=atol)


def test_kfac_one_datum(
    kfac_exact_one_datum_case: Tuple[
        Module,
        Union[BCEWithLogitsLoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[Tensor, Tensor]],
    ]
):
    model, loss_func, params, data, batch_size_fn = kfac_exact_one_datum_case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    ggn = ggn_block_diagonal(
        model, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type="type-2",
        loss_average=loss_average,
    )
    kfac_mat = kfac @ eye(kfac.shape[1])

    report_nonclose(ggn, kfac_mat)


def test_kfac_mc_one_datum(
    kfac_exact_one_datum_case: Tuple[
        Module,
        Union[BCEWithLogitsLoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[Tensor, Tensor]],
    ]
):
    model, loss_func, params, data, batch_size_fn = kfac_exact_one_datum_case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    ggn = ggn_block_diagonal(
        model, loss_func, params, data, batch_size_fn=batch_size_fn
    )
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        mc_samples=11_000,
        loss_average=loss_average,
    )
    kfac_mat = kfac @ eye(kfac.shape[1])

    atol = {"sum": 1e-3, "mean": 1e-3}[loss_func.reduction]
    rtol = {"sum": 3e-2, "mean": 3e-2}[loss_func.reduction]

    report_nonclose(ggn, kfac_mat, rtol=rtol, atol=atol)


def test_kfac_ef_one_datum(
    kfac_exact_one_datum_case: Tuple[
        Module,
        Union[BCEWithLogitsLoss, CrossEntropyLoss],
        List[Parameter],
        Iterable[Tuple[Tensor, Tensor]],
    ]
):
    model, loss_func, params, data, batch_size_fn = kfac_exact_one_datum_case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    ef_blocks = []  # list of per-parameter EFs
    for param in params:
        ef = EFLinearOperator(
            model, loss_func, [param], data, batch_size_fn=batch_size_fn
        )
        ef_blocks.append(ef @ eye(ef.shape[1]))
    ef = block_diag(*ef_blocks)

    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
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


@mark.parametrize("fisher_type", KFACLinearOperator._SUPPORTED_FISHER_TYPE)
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
        (
            (x, y.flatten(start_dim=0, end_dim=-2))
            if isinstance(loss_func, MSELoss)
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
        loss_average=loss_average,
    )
    kfac_flat_mat = kfac_flat @ eye(kfac_flat.shape[1])

    report_nonclose(kfac_mat, kfac_flat_mat)


def test_bug_device_change_invalidates_parameter_mapping():
    """Reproduce #77: Loading KFAC from GPU to CPU invalidates the internal mapping.

    This leads to some parameter blocks not being updated inside ``.matmat``.
    """
    if not cuda.is_available():
        skip("This test requires a GPU.")
    gpu, cpu = device("cuda"), device("cpu")

    manual_seed(0)

    model = Sequential(Linear(5, 4), ReLU(), Linear(4, 4)).to(gpu)
    data = [(rand(2, 5), regression_targets((2, 4)))]
    loss_func = MSELoss().to(gpu)

    kfac = KFACLinearOperator(
        model,
        loss_func,
        list(model.parameters()),
        data,
        fisher_type="empirical",
        check_deterministic=False,  # turn off to avoid implicit device changes
        progressbar=True,
    )
    x = rand(kfac.shape[1]).numpy()
    kfac_x_gpu = kfac @ x

    kfac.to_device(cpu)
    kfac_x_cpu = kfac @ x

    report_nonclose(kfac_x_gpu, kfac_x_cpu)


def test_torch_matmat(case):
    """Test that the torch_matmat method of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = case

    loss_average = None if loss_func.reduction == "sum" else "batch"
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        loss_average=loss_average,
    )
    device = kfac._device
    # KFAC.dtype is a numpy data type
    dtype = next(kfac._model_func.parameters()).dtype

    num_vectors = 16
    x = rand(kfac.shape[1], num_vectors, dtype=dtype, device=device)
    kfac_x = kfac.torch_matmat(x)
    assert x.device == kfac_x.device
    assert x.dtype == kfac_x.dtype
    assert kfac_x.shape == (kfac.shape[0], x.shape[1])
    kfac_x = kfac_x.cpu().numpy()

    # Test list input format
    x_list = kfac._torch_preprocess(x)
    kfac_x_list = kfac.torch_matmat(x_list)
    kfac_x_list = cat([rearrange(M, "k ... -> (...) k") for M in kfac_x_list])
    report_nonclose(kfac_x, kfac_x_list.cpu().numpy())

    # Test against multiplication with dense matrix
    identity = torch_eye(kfac.shape[1], dtype=dtype, device=device)
    kfac_mat = kfac.torch_matmat(identity)
    kfac_mat_x = kfac_mat @ x
    report_nonclose(kfac_x, kfac_mat_x.cpu().numpy())

    # Test against _matmat
    kfac_x_numpy = kfac @ x.cpu().numpy()
    report_nonclose(kfac_x, kfac_x_numpy)


def test_torch_matvec(case):
    """Test that the torch_matvec method of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = case

    loss_average = None if loss_func.reduction == "sum" else "batch"
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        loss_average=loss_average,
    )
    device = kfac._device
    # KFAC.dtype is a numpy data type
    dtype = next(kfac._model_func.parameters()).dtype

    with raises(ValueError):
        # Test that torch_matvec does not accept matrix input
        kfac.torch_matvec(rand(3, 5, dtype=dtype, device=device))

    x = rand(kfac.shape[1], dtype=dtype, device=device)
    kfac_x = kfac.torch_matvec(x)
    assert x.device == kfac_x.device
    assert x.dtype == kfac_x.dtype
    assert kfac_x.shape == x.shape
    kfac_x = kfac_x.cpu().numpy()

    # Test list input format
    # split parameter blocks
    dims = [p.numel() for p in kfac._params]
    split_x = x.split(dims)
    # unflatten parameter dimension
    assert len(split_x) == len(kfac._params)
    x_list = [res.reshape(p.shape) for res, p in zip(split_x, kfac._params)]
    kfac_x_list = kfac.torch_matvec(x_list)
    kfac_x_list = cat([rearrange(M, "... -> (...)") for M in kfac_x_list])
    report_nonclose(kfac_x, kfac_x_list.cpu().numpy())

    # Test against multiplication with dense matrix
    identity = torch_eye(kfac.shape[1], dtype=dtype, device=device)
    kfac_mat = kfac.torch_matmat(identity)
    kfac_mat_x = kfac_mat @ x
    report_nonclose(kfac_x, kfac_mat_x.cpu().numpy())

    # Test against _matmat
    kfac_x_numpy = kfac @ x.cpu().numpy()
    report_nonclose(kfac_x, kfac_x_numpy)


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
def test_trace(case, exclude, separate_weight_and_bias, check_deterministic):
    """Test that the trace property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = case

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    loss_average = None if loss_func.reduction == "sum" else "batch"
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        loss_average=loss_average,
        check_deterministic=check_deterministic,
    )

    # Check for equivalence of trace property and naive trace computation
    trace = kfac.trace
    trace_naive = (kfac @ eye(kfac.shape[1])).trace()
    report_nonclose(trace.cpu().numpy(), trace_naive)

    # Check that the trace property is properly cached and reset
    assert kfac._trace == trace
    kfac._compute_kfac()
    assert kfac._trace is None


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
def test_frobenius_norm(case, exclude, separate_weight_and_bias, check_deterministic):
    """Test that the Frobenius norm property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = case

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    loss_average = None if loss_func.reduction == "sum" else "batch"
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        loss_average=loss_average,
        check_deterministic=check_deterministic,
    )

    # Check for equivalence of frobenius_norm property and the naive computation
    frobenius_norm = kfac.frobenius_norm
    frobenius_norm_naive = norm(kfac @ eye(kfac.shape[1]))
    report_nonclose(frobenius_norm.cpu().numpy(), frobenius_norm_naive)

    # Check that the frobenius_norm property is properly cached and reset
    assert kfac._frobenius_norm == frobenius_norm
    kfac._compute_kfac()
    assert kfac._frobenius_norm is None


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
def test_det(case, exclude, separate_weight_and_bias, check_deterministic):
    """Test that the determinant property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = case

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    loss_average = None if loss_func.reduction == "sum" else "batch"
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        loss_average=loss_average,
        check_deterministic=check_deterministic,
    )

    # add damping manually to avoid singular matrices
    if not check_deterministic:
        kfac._compute_kfac()
    assert kfac._input_covariances or kfac._gradient_covariances
    delta = 1.0  # requires much larger damping value compared to ``logdet``
    for aaT in kfac._input_covariances.values():
        aaT.add_(
            torch_eye(aaT.shape[0], dtype=aaT.dtype, device=aaT.device), alpha=delta
        )
    for ggT in kfac._gradient_covariances.values():
        ggT.add_(
            torch_eye(ggT.shape[0], dtype=ggT.dtype, device=ggT.device), alpha=delta
        )

    # Check for equivalence of the det property and naive determinant computation
    determinant = kfac.det
    # verify that the determinant is not trivial as this would make the test useless
    assert determinant != 0.0 and determinant != 1.0
    det_naive = det(kfac @ eye(kfac.shape[1]))
    report_nonclose(determinant.cpu().numpy(), det_naive)

    # Check that the det property is properly cached and reset
    assert kfac._det == determinant
    kfac._compute_kfac()
    assert kfac._det is None


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
def test_logdet(case, exclude, separate_weight_and_bias, check_deterministic):
    """Test that the log determinant property of KFACLinearOperator works."""
    model, loss_func, params, data, batch_size_fn = case

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    loss_average = None if loss_func.reduction == "sum" else "batch"
    kfac = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
        loss_average=loss_average,
        check_deterministic=check_deterministic,
    )

    # add damping manually to avoid singular matrices
    if not check_deterministic:
        kfac._compute_kfac()
    assert kfac._input_covariances or kfac._gradient_covariances
    delta = 1e-3  # only requires much smaller damping value compared to ``det``
    for aaT in kfac._input_covariances.values():
        aaT.add_(
            torch_eye(aaT.shape[0], dtype=aaT.dtype, device=aaT.device), alpha=delta
        )
    for ggT in kfac._gradient_covariances.values():
        ggT.add_(
            torch_eye(ggT.shape[0], dtype=ggT.dtype, device=ggT.device), alpha=delta
        )

    # Check for equivalence of the logdet property and naive log determinant computation
    log_det = kfac.logdet
    # verify that the log determinant is finite and not nan
    assert not isinf(log_det) and not isnan(log_det)
    sign, logabsdet = slogdet(kfac @ eye(kfac.shape[1]))
    log_det_naive = sign * logabsdet
    report_nonclose(log_det.cpu().numpy(), log_det_naive, rtol=2e-5)

    # Check that the logdet property is properly cached and reset
    assert kfac._logdet == log_det
    kfac._compute_kfac()
    assert kfac._logdet is None


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
    assert exclude in [None, "weight", "bias"]
    model, loss_func, params, data, batch_size_fn = case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    if shuffle:
        permutation = randperm(len(params))
        params = [params[i] for i in permutation]

    # Compute KFAC with `fisher_type="empirical"` (could be any but "forward-only")
    foof_simulated = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        loss_average=loss_average,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type="empirical",
    )
    # Manually set all gradient covariances to the identity to simulate FOOF
    for name, block in foof_simulated._gradient_covariances.items():
        foof_simulated._gradient_covariances[name] = torch_eye(
            block.shape[0], dtype=block.dtype, device=block.device
        )
    simulated_foof_mat = foof_simulated @ eye(foof_simulated.shape[1])

    # Compute KFAC with `fisher_type="forward-only"`
    foof = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        loss_average=loss_average,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type="forward-only",
    )
    foof_mat = foof @ eye(foof.shape[1])

    # Check for equivalence
    assert len(foof_simulated._input_covariances) == len(foof._input_covariances)
    assert len(foof_simulated._gradient_covariances) == len(foof._gradient_covariances)
    report_nonclose(simulated_foof_mat, foof_mat)

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
    assert exclude in [None, "weight", "bias"]
    model, loss_func, params, data, batch_size_fn = single_layer_case
    loss_average = None if loss_func.reduction == "sum" else "batch"

    if exclude is not None:
        names = {p.data_ptr(): name for name, p in model.named_parameters()}
        params = [p for p in params if exclude not in names[p.data_ptr()]]

    if shuffle:
        permutation = randperm(len(params))
        params = [params[i] for i in permutation]

    # Compute exact block-diagonal GGN
    ggn = ggn_block_diagonal(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
    )

    # Compute KFAC with `fisher_type="forward-only"`
    foof = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        loss_average=loss_average,
        separate_weight_and_bias=separate_weight_and_bias,
        fisher_type="forward-only",
    )
    foof_mat = foof @ eye(foof.shape[1])

    # Check for equivalence
    num_data = sum(X.shape[0] for X, _ in data)
    y: Tensor = data[0][1]
    out_dim = y.shape[1]
    # See the docstring for the explanation of the scale
    scale = num_data if loss_average is None else 1 / out_dim
    report_nonclose(ggn, 2 * scale * foof_mat)

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(foof._input_covariances) == 0


@mark.parametrize("setting", ["expand", "reduce"])
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
        setting: The weight-sharing setting to use. Can be ``'expand'`` or ``'reduce'``.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
        exclude: Which parameters to exclude. Can be ``'weight'``, ``'bias'``,
            or ``None``.
        separate_weight_and_bias: Whether to treat weight and bias as separate blocks in
            the KFAC matrix.
    """
    assert exclude in [None, "weight", "bias"]
    model, loss_func, params, data, batch_size_fn = single_layer_weight_sharing_case
    model.setting = setting
    if isinstance(model, Conv2dModel):
        # parameters are only initialized after the setting property is set
        params = [p for p in model.parameters() if p.requires_grad]
    data = data[setting]

    # set appropriate loss_average argument based on loss reduction and setting
    loss_average = (
        ("batch+sequence" if setting == "expand" else "batch")
        if loss_func.reduction == "mean"
        else None
    )

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
        batch_size_fn=batch_size_fn,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    foof = KFACLinearOperator(
        model,
        loss_func,
        params,
        data,
        batch_size_fn=batch_size_fn,
        fisher_type="forward-only",
        kfac_approx=setting,  # choose KFAC approximation consistent with setting
        loss_average=loss_average,
        separate_weight_and_bias=separate_weight_and_bias,
    )
    foof_mat = foof @ eye(foof.shape[1])

    # Check for equivalence
    num_data = sum(X.shape[0] for X, _ in data)
    X, y = next(iter(data))
    out_dim = y.shape[-1]
    # See the docstring for the explanation of the scale
    scale = num_data if loss_average is None else 1 / out_dim
    if loss_average is None and setting == "expand":
        sequence_length = (
            (X.shape[-2] + 1) * (X.shape[-1] + 1)
            if isinstance(model, Conv2dModel)
            else X.shape[1:-1].numel()
        )
        scale *= sequence_length
    report_nonclose(ggn, 2 * scale * foof_mat, rtol=1e-4)

    # Check that input covariances were not computed
    if exclude == "weight":
        assert len(foof._input_covariances) == 0
