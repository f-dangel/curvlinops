"""Contains test cases for linear operators."""

from test.utils import classification_targets, get_available_devices

from torch import rand
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

CASES = []
for case in CASES_NO_DEVICE:
    for device in DEVICES:
        case_with_device = {**case, "device": device}
        CASES.append(case_with_device)
