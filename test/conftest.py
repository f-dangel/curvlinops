"""Contains pytest fixtures that are visible by other files."""

import test.utils
from collections.abc import MutableMapping
from test.cases import (
    ADJOINT_CASES,
    CASES,
    CNN_CASES,
    INV_CASES,
    NON_DETERMINISTIC_CASES,
)
from test.kfac_cases import (
    KFAC_EXACT_CASES,
    KFAC_EXACT_ONE_DATUM_CASES,
    KFAC_WEIGHT_SHARING_EXACT_CASES,
    SINGLE_LAYER_CASES,
    SINGLE_LAYER_WEIGHT_SHARING_CASES,
)
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from numpy import random
from pytest import fixture
from torch import Tensor, manual_seed
from torch.nn import Module, MSELoss


def initialize_case(
    case: Dict,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor],
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
    Optional[Callable[[MutableMapping], int]],
]:
    random.seed(case["seed"])
    manual_seed(case["seed"])

    model_func = case["model_func"]().to(case["device"])
    loss_func = case["loss_func"]().to(case["device"])
    params = [p for p in model_func.parameters() if p.requires_grad]
    data = case["data"]()

    # In some KFAC cases,
    # ``data = {KFACType.EXPAND: [(X, y), ...], KFACType.REDUCE: [(X, y), ...]}``
    # unlike the standard ``data = [(X: Tensor | MutableMapping, y), ...]``.
    # We ignore the former since the latter is included in KFAC cases, and thus the
    # feature of ``MutableMapping`` inputs is sufficiently covered already.
    if not isinstance(data, dict) and isinstance(next(iter(data))[0], MutableMapping):
        batch_size_fn = test.utils.batch_size_fn
    else:
        batch_size_fn = None

    return model_func, loss_func, params, data, batch_size_fn


@fixture(params=CASES)
def case(
    request,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor],
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
    Optional[Callable[[MutableMapping], int]],
]:
    case = request.param
    yield initialize_case(case)


@fixture(params=INV_CASES)
def inv_case(
    request,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor],
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
    Optional[Callable[[MutableMapping], int]],
]:
    case = request.param
    yield initialize_case(case)


@fixture(params=CNN_CASES)
def cnn_case(
    request,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor],
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
    Optional[Callable[[MutableMapping], int]],
]:
    cnn_case = request.param
    yield initialize_case(cnn_case)


@fixture(params=NON_DETERMINISTIC_CASES)
def non_deterministic_case(
    request,
) -> Tuple[
    Callable[[Tensor], Tensor],
    Callable[[Tensor, Tensor], Tensor],
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
    Optional[Callable[[MutableMapping], int]],
]:
    case = request.param
    yield initialize_case(case)


@fixture(params=ADJOINT_CASES)
def adjoint(request) -> bool:
    return request.param


@fixture(params=KFAC_EXACT_CASES)
def kfac_exact_case(
    request,
) -> Tuple[
    Module,
    MSELoss,
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
    Optional[Callable[[MutableMapping], int]],
]:
    """Prepare a test case for which KFAC equals the GGN.

    Yields:
        A neural network, the mean-squared error function, a list of parameters, and
        a data set.
    """
    case = request.param
    yield initialize_case(case)


@fixture(params=KFAC_WEIGHT_SHARING_EXACT_CASES)
def kfac_weight_sharing_exact_case(
    request,
) -> Tuple[
    Module,
    MSELoss,
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
    Optional[Callable[[MutableMapping], int]],
]:
    """Prepare a test case with weight-sharing for which KFAC equals the GGN.

    Yields:
        A neural network, the mean-squared error function, a list of parameters, and
        a data set.
    """
    case = request.param
    yield initialize_case(case)


@fixture(params=KFAC_EXACT_ONE_DATUM_CASES)
def kfac_exact_one_datum_case(
    request,
) -> Tuple[
    Module,
    Module,
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
    Optional[Callable[[MutableMapping], int]],
]:
    """Prepare a test case for which KFAC equals the GGN and one datum is used.

    Yields:
        A neural network, loss function, a list of parameters, and
        a data set with a single datum.
    """
    case = request.param
    yield initialize_case(case)


@fixture(params=SINGLE_LAYER_CASES)
def single_layer_case(
    request,
) -> Tuple[
    Module,
    Module,
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
    Optional[Callable[[MutableMapping], int]],
]:
    """Prepare a test case with a single-layer model for which FOOF is exact.

    Yields:
        A neural network, loss function, a list of parameters, and
        a data set with a single datum.
    """
    case = request.param
    yield initialize_case(case)


@fixture(params=SINGLE_LAYER_WEIGHT_SHARING_CASES)
def single_layer_weight_sharing_case(
    request,
) -> Tuple[
    Module,
    Module,
    List[Tensor],
    Iterable[Tuple[Tensor, Tensor]],
    Optional[Callable[[MutableMapping], int]],
]:
    """Test case with a single-layer model with weight-sharing for which FOOF is exact.

    Yields:
        A neural network, loss function, a list of parameters, and
        a data set with a single datum.
    """
    case = request.param
    yield initialize_case(case)
