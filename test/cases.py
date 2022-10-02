"""Contains test cases for linear operators."""

from test.utils import classification_targets, get_available_devices
from typing import Callable, Dict, Iterable, List, Tuple

from numpy import random
from torch import Tensor, manual_seed, rand
from torch.nn import CrossEntropyLoss, Linear, ReLU, Sequential

DEVICES = get_available_devices()
DEVICES_IDS = [f"dev={d}" for d in DEVICES]

LINOPS = []

# Add test cases here
CASES_NO_DEVICE = [
    {
        "model_func": lambda: Sequential(Linear(10, 5), ReLU(), Linear(5, 2)),
        "loss_func": lambda: CrossEntropyLoss(reduction="mean"),
        "data": lambda: [
            (rand(3, 10), classification_targets((3,), 2)),
            (rand(4, 10), classification_targets((4,), 2)),
        ],
        "seed": 0,
    }
]


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


CASES = []
for case in CASES_NO_DEVICE:
    for device in DEVICES:
        case_with_device = {**case, "device": device}
        CASES.append(case_with_device)
