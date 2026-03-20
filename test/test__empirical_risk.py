"""Tests for ``curvlinops._empirical_risk._EmpiricalRiskMixin``."""

from pytest import raises
from torch import Tensor, rand
from torch.nn import Linear, MSELoss

from curvlinops._empirical_risk import _EmpiricalRiskMixin


class _DummyMixin(_EmpiricalRiskMixin):
    """Minimal subclass for testing _EmpiricalRiskMixin validation."""


def test_model_func_and_params_validation():
    """Test that invalid model_func/params combinations raise ValueErrors."""
    model = Linear(2, 1)
    loss = MSELoss()
    data = [(rand(3, 2), rand(3, 1))]
    params_dict = dict(model.named_parameters())
    no_check = {"check_deterministic": False}

    def f(params: dict[str, Tensor], x: Tensor) -> Tensor:
        return x @ params["w"]

    # Non-callable, non-Module
    with raises(ValueError, match="model_func must be an nn.Module or a callable"):
        _DummyMixin("not_a_model", loss, {"w": rand(2, 1)}, data, **no_check)

    # Module + dict params works
    _DummyMixin(model, loss, params_dict, data, **no_check)

    # Callable + dict params works
    _DummyMixin(f, loss, {"w": rand(2, 1)}, data, **no_check)
