"""Contains test cases for the KFAC linear operator."""

from functools import partial
from test.utils import classification_targets, get_available_devices, regression_targets

from torch import rand
from torch.nn import CrossEntropyLoss, Linear, MSELoss, Sequential

# Add test cases here, devices and loss function with different reductions will be
# added automatically below
KFAC_EXPAND_EXACT_CASES_NO_DEVICE_NO_LOSS_FUNC = [
    ###############################################################################
    #                                  REGRESSION                                 #
    ###############################################################################
    # deep linear network with scalar output
    {
        "model_func": lambda: Sequential(Linear(6, 3), Linear(3, 1)),
        "data": lambda: [
            (rand(2, 6), regression_targets((2, 1))),
            (rand(5, 6), regression_targets((5, 1))),
        ],
        "seed": 0,
    },
    # deep linear network with vector output
    {
        "model_func": lambda: Sequential(Linear(5, 4), Linear(4, 3)),
        "data": lambda: [
            (rand(1, 5), regression_targets((1, 3))),
            (rand(7, 5), regression_targets((7, 3))),
        ],
        "seed": 0,
    },
]

KFAC_EXPAND_EXACT_CASES = []
for case in KFAC_EXPAND_EXACT_CASES_NO_DEVICE_NO_LOSS_FUNC:
    for device in get_available_devices():
        for reduction in ["mean", "sum"]:
            case_with_device_and_loss_func = {
                **case,
                "device": device,
                "loss_func": partial(MSELoss, reduction=reduction),
            }
            KFAC_EXPAND_EXACT_CASES.append(case_with_device_and_loss_func)


# Add test cases here, devices will be added automatically below
KFAC_EXPAND_EXACT_ONE_DATUM_CASES_NO_DEVICE = [
    ###############################################################################
    #                                CLASSIFICATION                               #
    ###############################################################################
    # deep linear network with vector output (both reductions)
    {
        "model_func": lambda: Sequential(Linear(5, 4), Linear(4, 3)),
        "loss_func": lambda: CrossEntropyLoss(reduction="mean"),
        "data": lambda: [(rand(1, 5), classification_targets((1,), 3))],
        "seed": 0,
    },
    {
        "model_func": lambda: Sequential(Linear(5, 4), Linear(4, 3)),
        "loss_func": lambda: CrossEntropyLoss(reduction="sum"),
        "data": lambda: [(rand(1, 5), classification_targets((1,), 3))],
        "seed": 0,
    },
]

KFAC_EXPAND_EXACT_ONE_DATUM_CASES = []
for case in KFAC_EXPAND_EXACT_ONE_DATUM_CASES_NO_DEVICE:
    for device in get_available_devices():
        case_with_device = {
            **case,
            "device": device,
        }
        KFAC_EXPAND_EXACT_ONE_DATUM_CASES.append(case_with_device)
