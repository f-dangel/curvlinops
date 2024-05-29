"""Contains test cases for the KFAC linear operator."""

from collections import UserDict
from functools import partial
from test.cases import ModelWithDictInput
from test.utils import (
    Conv2dModel,
    WeightShareModel,
    binary_classification_targets,
    classification_targets,
    get_available_devices,
    regression_targets,
)

from torch import rand
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Identity,
    Linear,
    MSELoss,
    Sequential,
)

from curvlinops.kfac import KFACType

# Add test cases here, devices and loss function with different reductions will be
# added automatically below
KFAC_EXACT_CASES_NO_DEVICE_NO_LOSS_FUNC = [
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
    # deep linear network with dict-like input
    {
        "model_func": lambda: ModelWithDictInput(num_classes=3, nonlin=Identity),
        "data": lambda: [
            (UserDict({"x": rand(1, 10)}), regression_targets((1, 3))),
            ({"x": rand(7, 10)}, regression_targets((7, 3))),
        ],
        "seed": 0,
    },
]

KFAC_EXACT_CASES = []
for case in KFAC_EXACT_CASES_NO_DEVICE_NO_LOSS_FUNC:
    for device in get_available_devices():
        for reduction in ["mean", "sum"]:
            case_with_device_and_loss_func = {
                **case,
                "device": device,
                "loss_func": partial(MSELoss, reduction=reduction),
            }
            KFAC_EXACT_CASES.append(case_with_device_and_loss_func)


# Add test cases here, devices and loss function with different reductions will be
# added automatically below
KFAC_WEIGHT_SHARING_EXACT_CASES_NO_DEVICE_NO_LOSS_FUNC = [
    ###############################################################################
    #                                  REGRESSION                                 #
    ###############################################################################
    # deep linear network with vector output and weight-sharing dimensions
    {
        "model_func": lambda: WeightShareModel(Linear(5, 4), Linear(4, 3)),
        "data": lambda: {
            KFACType.EXPAND: [
                (rand(2, 4, 8, 5), regression_targets((2, 4, 8, 3))),
                (rand(7, 4, 8, 5), regression_targets((7, 4, 8, 3))),
            ],
            KFACType.REDUCE: [
                (rand(1, 4, 8, 5), regression_targets((1, 3))),
                (rand(7, 4, 8, 5), regression_targets((7, 3))),
            ],
        },
        "seed": 0,
    },
    # Conv2d module with vector output (uses average pooling for reduce setting)
    {
        "model_func": lambda: Conv2dModel(),
        "data": lambda: {
            KFACType.EXPAND: [
                (rand(2, 3, 32, 32), regression_targets((2, 33, 33, 2))),
                (rand(7, 3, 32, 32), regression_targets((7, 33, 33, 2))),
            ],
            KFACType.REDUCE: [
                (rand(1, 3, 32, 32), regression_targets((1, 2))),
                (rand(8, 3, 32, 32), regression_targets((8, 2))),
            ],
        },
        "seed": 0,
    },
]

KFAC_WEIGHT_SHARING_EXACT_CASES = []
for case in KFAC_WEIGHT_SHARING_EXACT_CASES_NO_DEVICE_NO_LOSS_FUNC:
    for device in get_available_devices():
        for reduction in ["mean", "sum"]:
            case_with_device_and_loss_func = {
                **case,
                "device": device,
                "loss_func": partial(MSELoss, reduction=reduction),
            }
            KFAC_WEIGHT_SHARING_EXACT_CASES.append(case_with_device_and_loss_func)


# Add test cases here, devices will be added automatically below
KFAC_EXACT_ONE_DATUM_CASES_NO_DEVICE = [
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
    # deep linear network with vector output and BCEWithLogitsLoss (both reductions)
    {
        "model_func": lambda: Sequential(Linear(4, 3), Linear(3, 2)),
        "loss_func": lambda: BCEWithLogitsLoss(reduction="mean"),
        "data": lambda: [(rand(1, 4), binary_classification_targets((1, 2)))],
        "seed": 0,
    },
    {
        "model_func": lambda: Sequential(Linear(4, 3), Linear(3, 2)),
        "loss_func": lambda: BCEWithLogitsLoss(reduction="sum"),
        "data": lambda: [(rand(1, 4), binary_classification_targets((1, 2)))],
        "seed": 0,
    },
]

KFAC_EXACT_ONE_DATUM_CASES = []
for case in KFAC_EXACT_ONE_DATUM_CASES_NO_DEVICE:
    for device in get_available_devices():
        case_with_device = {
            **case,
            "device": device,
        }
        KFAC_EXACT_ONE_DATUM_CASES.append(case_with_device)


# Add test cases here, devices and loss function with different reductions will be
# added automatically below
SINGLE_LAYER_CASES_NO_DEVICE_NO_LOSS_FUNC = [
    ###############################################################################
    #                                  REGRESSION                                 #
    ###############################################################################
    # single-layer linear network with scalar output
    {
        "model_func": lambda: Linear(6, 1),
        "data": lambda: [
            (rand(5, 6), regression_targets((5, 1))),
            (rand(5, 6), regression_targets((5, 1))),
        ],
        "seed": 0,
    },
    # single-layer linear network with vector output
    {
        "model_func": lambda: Linear(5, 3),
        "data": lambda: [
            (rand(7, 5), regression_targets((7, 3))),
            (rand(7, 5), regression_targets((7, 3))),
        ],
        "seed": 0,
    },
]

SINGLE_LAYER_CASES = []
for case in SINGLE_LAYER_CASES_NO_DEVICE_NO_LOSS_FUNC:
    for device in get_available_devices():
        for reduction in ["mean", "sum"]:
            case_with_device_and_loss_func = {
                **case,
                "device": device,
                "loss_func": partial(MSELoss, reduction=reduction),
            }
            SINGLE_LAYER_CASES.append(case_with_device_and_loss_func)


# Add test cases here, devices and loss function with different reductions will be
# added automatically below
SINGLE_LAYER_WEIGHT_SHARING_CASES_NO_DEVICE_NO_LOSS_FUNC = [
    ###############################################################################
    #                                  REGRESSION                                 #
    ###############################################################################
    # single linear layer with vector output and weight-sharing dimensions
    {
        "model_func": lambda: WeightShareModel(Linear(5, 3)),
        "data": lambda: {
            KFACType.EXPAND: [
                (rand(7, 4, 8, 5), regression_targets((7, 4, 8, 3))),
                (rand(7, 4, 8, 5), regression_targets((7, 4, 8, 3))),
            ],
            KFACType.REDUCE: [
                (rand(8, 4, 8, 5), regression_targets((8, 3))),
                (rand(8, 4, 8, 5), regression_targets((8, 3))),
            ],
        },
        "seed": 0,
    },
    # Conv2d module with vector output (uses average pooling for reduce setting)
    {
        "model_func": lambda: Conv2dModel(),
        "data": lambda: {
            KFACType.EXPAND: [
                (rand(7, 3, 32, 32), regression_targets((7, 33, 33, 2))),
                (rand(7, 3, 32, 32), regression_targets((7, 33, 33, 2))),
            ],
            KFACType.REDUCE: [
                (rand(8, 3, 32, 32), regression_targets((8, 2))),
                (rand(8, 3, 32, 32), regression_targets((8, 2))),
            ],
        },
        "seed": 0,
    },
]

SINGLE_LAYER_WEIGHT_SHARING_CASES = []
for case in SINGLE_LAYER_WEIGHT_SHARING_CASES_NO_DEVICE_NO_LOSS_FUNC:
    for device in get_available_devices():
        for reduction in ["mean", "sum"]:
            case_with_device_and_loss_func = {
                **case,
                "device": device,
                "loss_func": partial(MSELoss, reduction=reduction),
            }
            SINGLE_LAYER_WEIGHT_SHARING_CASES.append(case_with_device_and_loss_func)
