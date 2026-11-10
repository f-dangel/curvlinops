"""Test curvature operators against ground truth for loss-function options.

See https://github.com/f-dangel/curvlinops/issues/311.
"""

from pytest import mark
from torch import allclose, cat, manual_seed

from curvlinops.examples.functorch import functorch_empirical_fisher
from test.loss_options_cases import (
    DENSE_CASES,
    METAMORPHIC_CASES,
    REDUCTIONS,
    case_params,
    reference_ef,
)


def _setup(case):
    """Build model, parameters and data of a case.

    Args:
        case: The loss-option case.

    Returns:
        Model, parameter dictionary, and data.
    """
    model = case["model_func"]()
    params = {n: p.detach() for n, p in model.named_parameters()}
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
