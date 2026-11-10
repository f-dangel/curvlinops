"""Test curvature operators against ground truth for loss-function options."""

from copy import deepcopy
from enum import Enum

from pytest import FixtureRequest, mark, param
from torch import Tensor, allclose, cat, stack
from torch.func import functional_call, jacrev
from torch.linalg import lstsq
from torch.nn import CrossEntropyLoss, Module

from curvlinops import FisherType
from curvlinops.examples.functorch import (
    functorch_empirical_fisher,
    functorch_ggn,
    functorch_hessian,
)
from curvlinops.utils import allclose_report
from test.loss_options_cases import (
    EXPECTED_FAILURES,
    REDUCTIONS,
    Operator,
    Option,
    Reason,
    XFail,
    _OperatorSpec,
    drop_ignored,
    maybe_xfail,
)
from test.test_kfac import MC_TOLS
from test.utils import (
    block_diagonal_of,
    compare_matmat,
    compare_matmat_expectation,
    eye_like,
)

DENSE_EXACT_OPERATORS = [Operator.HESSIAN, Operator.GGN, Operator.GGN_DIAG, Operator.EF]
DENSE_MC_OPERATORS = [Operator.GGN_MC, Operator.GGN_DIAG_MC]
KFAC_OPERATORS = [
    Operator.KFAC_TYPE2,
    Operator.KFAC_MC,
    Operator.KFAC_EF,
    Operator.EKFAC_TYPE2,
    Operator.EKFAC_EF,
]
KFAC_DROP_OPERATORS = [
    *KFAC_OPERATORS,
    Operator.KFAC_FORWARD_ONLY,
    Operator.KFOC_MC,
]

# KFAC is exact on one datum, which cannot have both ignored and observed targets
ONE_DATUM_OPTIONS = [o for o in Option if o.value.ignore_index is None]

# Dropping ignored data points needs one target per data point
DROP_OPTIONS = [
    o for o in Option if o.value.ignore_index is not None and not o.value.seq
]


def _name(member: Enum) -> str:
    """Name an enum member in test ids.

    Args:
        member: The enum member.

    Returns:
        The lower-case member name.
    """
    return member.name.lower()


def _with_backends(operators: list[Operator]) -> list:
    """Pair each operator with each of its backends.

    Args:
        operators: The operators.

    Returns:
        List of ``pytest.param(operator, backend)``.
    """
    return [
        param(
            op, backend, id=_name(op) if backend is None else f"{_name(op)}-{backend}"
        )
        for op in operators
        for backend in op.value.backends
    ]


def _merge(data: list[tuple[Tensor, Tensor]]) -> list[tuple[Tensor, Tensor]]:
    """Merge mini-batches into one.

    Args:
        data: Mini-batches.

    Returns:
        A list with a single mini-batch.
    """
    return [(cat([X for X, _ in data]), cat([y for _, y in data]))]


def reference_ef(
    model: Module,
    loss_func: Module,
    params: dict[str, Tensor],
    data: list[tuple[Tensor, Tensor]],
) -> Tensor:
    """Compute the empirical Fisher of a criterion with its options.

    The EF is ``c * sum_k grad(l_k) grad(l_k)^T`` over loss terms ``l_k``, one
    per position for ``CrossEntropyLoss`` and one per output row otherwise. The
    terms come from the criterion with ``reduction='none'``, so they carry its
    options. ``c = 1`` for ``'sum'``. For ``'mean'``, ``c`` is the ratio of the
    mean- to the sum-reduced loss on all data, i.e. PyTorch's denominator.

    Args:
        model: The neural network.
        loss_func: The criterion, including its reduction.
        params: Parameters w.r.t. which the EF is computed.
        data: Mini-batches.

    Returns:
        The EF as a dense matrix.
    """
    # NOTE(#311): with `weight`/`pos_weight` each term's gradient carries w once,
    # so the EF carries w^2. Whether it should carry w (data duplication) instead
    # is open.
    [(X, y)] = _merge(data)
    params = {name: p.detach() for name, p in params.items()}

    loss_terms = deepcopy(loss_func)
    loss_terms.reduction = "none"

    def terms(params: dict[str, Tensor]) -> Tensor:
        losses = loss_terms(functional_call(model, params, (X,)), y)

        match loss_func:
            case CrossEntropyLoss():
                return losses.flatten()
            case _:
                return losses.flatten(end_dim=-2).sum(-1)

    jac = cat([j.flatten(start_dim=1) for j in jacrev(terms)(params).values()], dim=1)
    ef = jac.T @ jac

    if loss_func.reduction == "sum":
        return ef

    output = functional_call(model, params, (X,))

    return loss_func(output, y) / loss_terms(output, y).sum() * ef


def _exact(
    operator: Operator,
    model: Module,
    loss_func: Module,
    params: dict[str, Tensor],
    data: list[tuple[Tensor, Tensor]],
) -> Tensor:
    """Compute the matrix a dense operator represents (or estimates) via autodiff.

    Args:
        operator: The operator under test.
        model: The neural network.
        loss_func: The criterion.
        params: Parameters.
        data: Mini-batches.

    Returns:
        The ground-truth matrix.

    Raises:
        ValueError: If the operator has no dense ground truth.
    """
    match operator:
        case Operator.HESSIAN:
            truth = functorch_hessian(model, loss_func, params, data)
        case Operator.GGN | Operator.GGN_MC:
            truth = functorch_ggn(model, loss_func, params, data)
        case Operator.GGN_DIAG | Operator.GGN_DIAG_MC:
            truth = functorch_ggn(model, loss_func, params, data).diag().diag()
        case Operator.EF:
            truth = reference_ef(model, loss_func, params, data)
        case _:
            raise ValueError(f"{operator} has no dense ground truth.")

    return truth.detach()


def _assert_close(expected: Tensor, actual: Tensor, mc: bool):
    """Assert that two matrices agree, with looser tolerances for MC estimates.

    MC comparisons divide by the largest entry of ``expected``, so the
    tolerances work for both reductions.

    Args:
        expected: Ground-truth matrix.
        actual: Matrix of the operator.
        mc: Whether ``actual`` is an MC estimate.
    """
    scale, tols = (expected.abs().max(), MC_TOLS) if mc else (1.0, {})
    assert allclose_report(expected / scale, actual / scale, **tols)


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("option", [Option.CE_DEFAULT, Option.BCE_DEFAULT], ids=_name)
def test_reference_ef_matches_functorch(option: Option, reduction: str):
    """Check ``reference_ef`` against the functorch EF for default criteria."""
    model, data, loss_func = option.model(), option.data(), option.criterion(reduction)
    params = dict(model.named_parameters())

    assert allclose(
        reference_ef(model, loss_func, params, data),
        functorch_empirical_fisher(model, loss_func, params, data),
    )


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("option", DROP_OPTIONS, ids=_name)
def test_drop_ignored_preserves_loss(option: Option, reduction: str):
    """Check that dropping ignored data points leaves the loss unchanged."""
    model, data = option.model(), option.data()

    [(X, y)] = _merge(data)
    [(X_ref, y_ref)] = _merge(drop_ignored(data, option.value.ignore_index))

    loss = option.criterion(reduction)(model(X), y)
    loss_ref = CrossEntropyLoss(reduction=reduction)(model(X_ref), y_ref)

    assert allclose(loss, loss_ref)


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("option", Option, ids=_name)
@mark.parametrize("operator", DENSE_EXACT_OPERATORS, ids=_name)
def test_dense_exact(
    request: FixtureRequest, operator: Operator, option: Option, reduction: str
):
    """Compare exact operators against autodiff through the real criterion."""
    maybe_xfail(request, operator, option, reduction)

    model, data, loss_func = option.model(), option.data(), option.criterion(reduction)
    params = dict(model.named_parameters())

    op = operator.value.build(model, loss_func, params, data)

    compare_matmat(op, _exact(operator, model, loss_func, params, data))


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("option", Option, ids=_name)
@mark.parametrize("operator", DENSE_MC_OPERATORS, ids=_name)
def test_dense_mc(
    request: FixtureRequest, operator: Operator, option: Option, reduction: str
):
    """Compare MC-approximated GGN operators against the exact GGN."""
    maybe_xfail(request, operator, option, reduction)

    model, data, loss_func = option.model(), option.data(), option.criterion(reduction)
    params = dict(model.named_parameters())

    op = operator.value.build(model, loss_func, params, data)
    truth = _exact(operator, model, loss_func, params, data)

    match operator:
        case Operator.GGN_MC:
            compare_matmat_expectation(
                op, truth, max_repeats=40, check_every=100, **MC_TOLS
            )
        case Operator.GGN_DIAG_MC:
            # draws its samples once at construction, so there is nothing to repeat
            _assert_close(truth, op @ eye_like(op), mc=True)


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("option", ONE_DATUM_OPTIONS, ids=_name)
@mark.parametrize("operator,backend", _with_backends(KFAC_OPERATORS))
def test_kfac_one_datum(
    request: FixtureRequest,
    operator: Operator,
    backend: str,
    option: Option,
    reduction: str,
):
    """Compare KFAC-type operators against the block-diagonal truth on one datum."""
    maybe_xfail(request, operator, option, reduction, backend)

    model = option.model(deep_linear=True)
    data, loss_func = option.data(one_datum=True), option.criterion(reduction)
    params = dict(model.named_parameters())

    match operator.value.fisher_type:
        case FisherType.EMPIRICAL:
            truth = reference_ef(model, loss_func, params, data)
        case _:
            truth = functorch_ggn(model, loss_func, params, data)

    truth = block_diagonal_of(truth.detach(), model, params)
    op = operator.value.build(model, loss_func, params, data, backend)

    _assert_close(truth, op @ eye_like(op), operator.is_mc)


@mark.parametrize("reduction", REDUCTIONS)
@mark.parametrize("option", DROP_OPTIONS, ids=_name)
@mark.parametrize("operator,backend", _with_backends(KFAC_DROP_OPERATORS))
def test_kfac_ignored_equals_dropped(
    request: FixtureRequest,
    operator: Operator,
    backend: str | None,
    option: Option,
    reduction: str,
):
    """Check that ignored data points are absent from KFAC-type operators.

    The reference is the same operator (exact Fisher in place of MC) on the data
    without the ignored data points, with a default criterion.
    """
    maybe_xfail(request, operator, option, reduction, backend)

    model = option.model()
    params = dict(model.named_parameters())

    data = option.data()
    data_ref = drop_ignored(data, option.value.ignore_index)

    if operator.value.single_batch:
        data, data_ref = _merge(data), _merge(data_ref)

    loss_func_ref = CrossEntropyLoss(reduction=reduction)
    truth = operator.value.exact().build(
        model, loss_func_ref, params, data_ref, backend
    )
    op = operator.value.build(model, option.criterion(reduction), params, data, backend)

    _assert_close(
        (truth @ eye_like(truth)).detach(), (op @ eye_like(op)).detach(), operator.is_mc
    )


def _inputs(operator: Operator, option: Option) -> tuple[Module, list]:
    """Create the model and data that the matrix tests use for a cell.

    KFAC-type operators run on one datum, except for ignored targets, which use
    the test that drops ignored data points.

    Returns:
        The model and the mini-batches.
    """
    one_datum = (
        operator.value.fisher_type is not None and option.value.ignore_index is None
    )

    return option.model(deep_linear=one_datum), option.data(one_datum=one_datum)


def _matrix(
    spec: _OperatorSpec,
    model: Module,
    loss_func: Module,
    data: list[tuple[Tensor, Tensor]],
    backend: str | None,
) -> Tensor:
    """Build an operator and return its dense matrix.

    Returns:
        The operator's matrix.
    """
    if spec.single_batch:
        data = _merge(data)

    op = spec.build(model, loss_func, dict(model.named_parameters()), data, backend)

    return (op @ eye_like(op)).detach()


def _without_option(
    option: Option, reduction: str, data: list[tuple[Tensor, Tensor]]
) -> tuple[Module, list[tuple[Tensor, Tensor]]]:
    """Switch the option off: drop it from the criterion, or un-ignore the targets.

    Returns:
        The criterion and the mini-batches without the option.
    """
    match option.value.ignore_index:
        case None:
            return option.plain_criterion(reduction), data
        case ignore_index:
            # class 0 is valid and differs from either ignore index
            unignored = [(X, y.where(y != ignore_index, 0)) for X, y in data]
            return option.criterion(reduction), unignored


def _reason_cells() -> list:
    """List every expected-failure cell whose reason can be checked.

    Crashes are left out: ``raises`` in their xfail already pins the exception.

    Returns:
        List of ``pytest.param(xfail, reduction, backend)``.
    """
    return [
        param(
            xfail,
            reduction,
            backend,
            id="-".join(
                [_name(xfail.operator), _name(xfail.option), reduction]
                + ([] if backend is None else [backend])
            ),
        )
        for xfail in EXPECTED_FAILURES
        if xfail.reason is not Reason.CRASHES
        for reduction, backend in xfail.cells()
    ]


@mark.parametrize("xfail,reduction,backend", _reason_cells())
def test_xfail_reason(xfail: XFail, reduction: str, backend: str | None):
    """Check that a cell fails for the reason the table gives."""
    operator, option = xfail.operator, xfail.option
    model, data = _inputs(operator, option)

    def matrix(loss_func: Module, data: list, spec: _OperatorSpec = operator.value):
        return _matrix(spec, model, loss_func, data, backend)

    with_option = (option.criterion(reduction), data)
    without_option = _without_option(option, reduction, data)

    match xfail.reason:
        case Reason.OPTION_IGNORED:
            assert allclose(matrix(*with_option), matrix(*without_option))

        case Reason.MEAN_NORMALIZATION:
            # 'sum' passes, and 'mean' is a rescaling of the per-batch 'sum' results
            assert reduction == "mean"
            assert not any(
                x.matches(operator, option, "sum", backend) for x in EXPECTED_FAILURES
            )

            sums = stack(
                [matrix(option.criterion("sum"), [batch]).flatten() for batch in data],
                dim=1,
            )
            mean = matrix(*with_option).flatten().unsqueeze(-1)

            assert allclose(sums @ lstsq(sums, mean).solution, mean)

        case Reason.IGNORED_INPUTS_COUNTED:
            # Un-ignoring the targets changes the result, so the gradients respect them
            assert not allclose(matrix(*with_option), matrix(*without_option))

            # Forward-only KFAC computes only the input factor, which all Fisher types
            # share. It does not change, so the ignored data points are in it
            forward_only = Operator.KFAC_FORWARD_ONLY.value
            assert allclose(
                matrix(*with_option, spec=forward_only),
                matrix(*without_option, spec=forward_only),
            )
