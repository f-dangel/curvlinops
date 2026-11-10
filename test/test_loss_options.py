"""Test curvature operators against ground truth for loss-function options.

See https://github.com/f-dangel/curvlinops/issues/311.
"""

from pytest import FixtureRequest, mark
from torch import Tensor, allclose, cat, manual_seed
from torch.nn import Module

from curvlinops import (
    EFLinearOperator,
    EKFACLinearOperator,
    FisherType,
    GGNDiagonalLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
    KFOCLinearOperator,
)
from curvlinops.examples.functorch import (
    functorch_empirical_fisher,
    functorch_ggn,
    functorch_hessian,
)
from curvlinops.utils import allclose_report
from test.loss_options_cases import (
    DENSE_CASES,
    METAMORPHIC_CASES,
    ONE_DATUM_CASES,
    REDUCTIONS,
    case_params,
    maybe_xfail,
    reference_ef,
)
from test.test_kfac import MC_TOLS
from test.utils import (
    block_diagonal_of,
    compare_matmat,
    compare_matmat_expectation,
    eye_like,
)

# Fewer samples than in the other test files to keep this module fast
DENSE_MC_SAMPLES = 100
KFAC_MC_SAMPLES = 1_000


def _setup(case):
    """Build model, parameters and data of a case.

    Args:
        case: The loss-option case.

    Returns:
        Model, parameter dictionary, and data.
    """
    model = case["model_func"]()
    params = dict(model.named_parameters())
    manual_seed(case["seed"])
    return model, params, case["data"]()


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize(
    "case",
    case_params([c for c in DENSE_CASES if c["id"] in {"ce_default", "bce_default"}]),
)
def test_reference_ef_matches_functorch_default(case, reduction: str):
    """Check that ``reference_ef`` matches the functorch EF for default losses.

    Args:
        case: Loss-option case with a default criterion.
        reduction: Reduction of the criterion.
    """
    model, params, data = _setup(case)
    loss_func = case["loss_func"](reduction)
    ef = reference_ef(model, loss_func, params, data)
    ef_functorch = functorch_empirical_fisher(model, loss_func, params, data)
    assert allclose(ef, ef_functorch)


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("case", case_params(METAMORPHIC_CASES))
def test_metamorphic_reference_loss_matches(case, reduction: str):
    """Check that dropping ignored rows reproduces the criterion's semantics.

    Args:
        case: Metamorphic loss-option case.
        reduction: Reduction of the criterion.
    """
    model, params, data = _setup(case)
    data_ref = case["data_reference"]()

    X, y = cat([x for x, _ in data]), cat([t for _, t in data])
    X_ref, y_ref = cat([x for x, _ in data_ref]), cat([t for _, t in data_ref])

    loss = case["loss_func"](reduction)(model(X), y)
    loss_ref = case["loss_func_reference"](reduction)(model(X_ref), y_ref)
    assert allclose(loss, loss_ref)


DENSE_OPERATORS = ["hessian", "ggn", "ggn_diag", "ef"]


def _dense_reference(
    operator_id: str, model: Module, loss_func: Module, params: dict, data: list
) -> Tensor:
    """Compute the ground-truth matrix of a dense-family operator.

    Args:
        operator_id: Identifier of the operator.
        model: The neural network.
        loss_func: The criterion.
        params: Parameters.
        data: Mini-batches.

    Returns:
        The dense ground-truth matrix.
    """
    if operator_id == "hessian":
        return functorch_hessian(model, loss_func, params, data)
    if operator_id in {"ggn", "ggn_mc"}:
        return functorch_ggn(model, loss_func, params, data)
    if operator_id in {"ggn_diag", "ggn_diag_mc"}:
        return functorch_ggn(model, loss_func, params, data).diag().diag()
    return reference_ef(model, loss_func, params, data)


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("case", case_params(DENSE_CASES))
@mark.parametrize("operator_id", DENSE_OPERATORS)
def test_dense_matrix(
    request: FixtureRequest, operator_id: str, case: dict, reduction: str
):
    """Compare exact operators against autodiff through the real criterion.

    Args:
        request: The pytest request.
        operator_id: Identifier of the operator.
        case: Loss-option case.
        reduction: Reduction of the criterion.
    """
    maybe_xfail(request, operator_id, case["id"], reduction)
    model, params, data = _setup(case)
    loss_func = case["loss_func"](reduction)

    op_cls = {
        "hessian": HessianLinearOperator,
        "ggn": GGNLinearOperator,
        "ggn_diag": GGNDiagonalLinearOperator,
        "ef": EFLinearOperator,
    }[operator_id]
    op = op_cls(model, loss_func, params, data)
    mat = _dense_reference(operator_id, model, loss_func, params, data).detach()
    compare_matmat(op, mat)


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("case", case_params(DENSE_CASES))
@mark.parametrize("operator_id", ["ggn_mc", "ggn_diag_mc"])
def test_dense_mc(
    request: FixtureRequest, operator_id: str, case: dict, reduction: str
):
    """Compare MC-approximated GGN operators against the exact GGN.

    Args:
        request: The pytest request.
        operator_id: Identifier of the operator.
        case: Loss-option case.
        reduction: Reduction of the criterion.
    """
    maybe_xfail(request, operator_id, case["id"], reduction)
    model, params, data = _setup(case)
    loss_func = case["loss_func"](reduction)
    mat = _dense_reference(operator_id, model, loss_func, params, data).detach()

    if operator_id == "ggn_mc":
        op = GGNLinearOperator(
            model, loss_func, params, data, mc_samples=DENSE_MC_SAMPLES
        )
        compare_matmat_expectation(op, mat, max_repeats=40, check_every=100, **MC_TOLS)
    else:
        # the diagonal operator draws its samples once at construction
        op = GGNDiagonalLinearOperator(
            model, loss_func, params, data, mc_samples=DENSE_MC_SAMPLES * 40
        )
        scale = mat.abs().max()
        assert allclose_report(mat / scale, (op @ eye_like(op)) / scale, **MC_TOLS)


KFAC_BACKENDS = list(KFACLinearOperator._BACKENDS)
ONE_DATUM_OPERATORS = [
    f"{op}[{backend}]"
    for op in ["kfac_type2", "kfac_mc", "kfac_ef", "ekfac_type2", "ekfac_ef"]
    for backend in KFAC_BACKENDS
]
METAMORPHIC_OPERATORS = [
    f"{op}[{backend}]"
    for op in [
        "kfac_type2",
        "kfac_mc",
        "kfac_ef",
        "kfac_forward_only",
        "ekfac_type2",
        "ekfac_ef",
    ]
    for backend in KFAC_BACKENDS
] + ["kfoc_mc[make_fx]"]

_FISHER = {
    "type2": FisherType.TYPE2,
    "mc": FisherType.MC,
    "ef": FisherType.EMPIRICAL,
    "forward_only": FisherType.FORWARD_ONLY,
}


def _kfac_family_matrix(
    operator_id: str,
    model: Module,
    loss_func: Module,
    params: dict,
    data: list,
    mc: bool = True,
) -> Tensor:
    """Compute the dense matrix of a KFAC-family operator.

    Args:
        operator_id: Identifier like ``'kfac_type2[hooks]'``.
        model: The neural network.
        loss_func: The criterion.
        params: Parameters.
        data: Mini-batches.
        mc: Whether to use MC sampling for ``*_mc`` operators. If ``False``, uses
            the exact (type-2) Fisher instead.

    Returns:
        The dense matrix of the operator.
    """
    name, backend = operator_id[:-1].split("[")
    family, fisher = name.split("_", 1)
    fisher_type = _FISHER[fisher]
    kwargs = {"backend": backend}
    if fisher == "mc":
        if mc:
            kwargs["mc_samples"] = KFAC_MC_SAMPLES
        else:
            fisher_type = FisherType.TYPE2
    if family == "kfoc":
        kwargs.pop("backend")
    op_cls = {
        "kfac": KFACLinearOperator,
        "ekfac": EKFACLinearOperator,
        "kfoc": KFOCLinearOperator,
    }[family]
    if family == "kfoc" and fisher_type == FisherType.MC:
        kwargs["mc_samples"] = KFAC_MC_SAMPLES
    op = op_cls(model, loss_func, params, data, fisher_type=fisher_type, **kwargs)
    return op @ eye_like(op)


def _compare_kfac(operator_id: str, expected: Tensor, actual: Tensor):
    """Assert that two matrices agree, with MC tolerances for MC operators.

    Args:
        operator_id: Identifier of the operator.
        expected: Ground-truth matrix.
        actual: Matrix of the operator.
    """
    if "_mc" in operator_id:
        scale = expected.abs().max()
        assert allclose_report(expected / scale, actual / scale, **MC_TOLS)
    else:
        assert allclose_report(expected, actual)


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("case", case_params(ONE_DATUM_CASES))
@mark.parametrize("operator_id", ONE_DATUM_OPERATORS)
def test_kfac_one_datum(
    request: FixtureRequest, operator_id: str, case: dict, reduction: str
):
    """Compare KFAC-type operators against the block-diagonal truth for one datum.

    Args:
        request: The pytest request.
        operator_id: Identifier of the operator.
        case: Loss-option case.
        reduction: Reduction of the criterion.
    """
    maybe_xfail(request, operator_id, case["id"], reduction)
    model, params, data = _setup(case)
    loss_func = case["loss_func"](reduction)

    if "_ef" in operator_id:
        truth = reference_ef(model, loss_func, params, data)
    else:
        truth = functorch_ggn(model, loss_func, params, data).detach()
    truth = block_diagonal_of(truth, model, params)

    mat = _kfac_family_matrix(operator_id, model, loss_func, params, data)
    _compare_kfac(operator_id, truth, mat)


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("case", case_params(METAMORPHIC_CASES))
@mark.parametrize("operator_id", METAMORPHIC_OPERATORS)
def test_kfac_metamorphic(
    request: FixtureRequest, operator_id: str, case: dict, reduction: str
):
    """Check that ignored rows are absent from KFAC-type operators.

    The reference is the same operator (exact Fisher in place of MC) applied to the
    data without ignored rows, using a default criterion.

    Args:
        request: The pytest request.
        operator_id: Identifier of the operator.
        case: Metamorphic loss-option case.
        reduction: Reduction of the criterion.
    """
    maybe_xfail(request, operator_id, case["id"], reduction)
    model, params, data = _setup(case)
    loss_func = case["loss_func"](reduction)
    data_ref = case["data_reference"]()
    if operator_id.startswith("kfoc"):
        # KFOC only supports data with exactly one batch
        data, data_ref = (
            [(cat([x for x, _ in d]), cat([y for _, y in d]))] for d in (data, data_ref)
        )
    loss_func_ref = case["loss_func_reference"](reduction)

    truth = _kfac_family_matrix(
        operator_id, model, loss_func_ref, params, data_ref, mc=False
    )
    mat = _kfac_family_matrix(operator_id, model, loss_func, params, data)
    _compare_kfac(operator_id, truth.detach(), mat.detach())
