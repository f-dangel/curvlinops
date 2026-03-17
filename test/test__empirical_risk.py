"""Tests for ``curvlinops._empirical_risk._EmpiricalRiskMixin``."""

from pytest import raises
from torch import Tensor, rand
from torch.nn import Linear, MSELoss

from curvlinops._empirical_risk import _EmpiricalRiskMixin


class _DummyMixin(_EmpiricalRiskMixin):
    """Minimal subclass for testing _EmpiricalRiskMixin validation."""


class _DummyFunctionalMixin(_EmpiricalRiskMixin):
    """Minimal subclass with SUPPORTS_FUNCTIONAL = True."""

    SUPPORTS_FUNCTIONAL: bool = True


def test_model_func_and_params_validation():
    """Test that invalid model_func/params combinations raise ValueErrors."""
    model = Linear(2, 1)
    loss = MSELoss()
    data = [(rand(3, 2), rand(3, 1))]
    params_dict = dict(model.named_parameters())
    params_list = list(model.parameters())
    no_check = {"check_deterministic": False}

    def f(params: dict[str, Tensor], x: Tensor) -> Tensor:
        return x @ params["w"]

    # Module + dict params
    with raises(ValueError, match="Module model_func requires params as list"):
        _DummyMixin(model, loss, params_dict, data, **no_check)

    # Callable + SUPPORTS_FUNCTIONAL=False
    with raises(ValueError, match="does not support callable model_func"):
        _DummyMixin(f, loss, {"w": rand(2, 1)}, data, **no_check)

    # Callable + list params
    with raises(ValueError, match="Callable model_func requires params as dict"):
        _DummyFunctionalMixin(f, loss, params_list, data, **no_check)

    # Non-callable, non-Module
    with raises(ValueError, match="model_func must be an nn.Module or a callable"):
        _DummyFunctionalMixin("not_a_model", loss, {"w": rand(2, 1)}, data, **no_check)

    # Module + list of tensors that are not the model's parameters
    fake_params = [rand(p.shape) for p in params_list]
    with raises(ValueError, match="not found in model"):
        _DummyMixin(model, loss, fake_params, data, **no_check)
