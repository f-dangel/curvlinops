"""Contains pytest fixtures that are visible by other files."""

from test.cases import ADJOINT_CASES, CASES, NON_DETERMINISTIC_CASES
from typing import Callable, Dict, Iterable, List, Tuple

from numpy import random
from pytest import fixture
from torch import Tensor, manual_seed


def initialize_case(
    case: Dict,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor],
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
]:
    random.seed(case["seed"])
    manual_seed(case["seed"])

    model_func = case["model_func"]().to(case["device"])
    loss_func = case["loss_func"]().to(case["device"])
    params = [p for p in model_func.parameters() if p.requires_grad]
    data = case["data"]()

    return model_func, loss_func, params, data


@fixture(params=CASES)
def case(
    request,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor],
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
]:
    case = request.param
    yield initialize_case(case)


@fixture(params=NON_DETERMINISTIC_CASES)
def non_deterministic_case(
    request,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor],
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
]:
    case = request.param
    yield initialize_case(case)


@fixture(params=ADJOINT_CASES)
def adjoint(request) -> bool:
    return request.param
