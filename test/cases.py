"""Contains test cases for linear operators."""

from collections import UserDict
from collections.abc import MutableMapping
from test.utils import (
    WeightShareModel,
    binary_classification_targets,
    classification_targets,
    get_available_devices,
    regression_targets,
)

from torch import rand, rand_like
from torch.nn import (
    BatchNorm1d,
    BCEWithLogitsLoss,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    MSELoss,
    ReLU,
    Sequential,
)
from torch.utils.data import DataLoader, TensorDataset

from curvlinops.kfac import KFACType

DEVICES = get_available_devices()
DEVICES_IDS = [f"dev={d}" for d in DEVICES]


class ModelWithDictInput(Module):
    def __init__(self, num_classes=2, nonlin=ReLU):
        super().__init__()
        self.net = Sequential(Linear(10, 5), nonlin(), Linear(5, num_classes))

    def forward(self, data: MutableMapping):
        device = next(self.parameters()).device
        x = data["x"].to(device)
        return self.net(x)


# Add test cases for numerically unstable inverses here
INV_CASES_NO_DEVICE = [
    ###############################################################################
    #                                CLASSIFICATION                               #
    ###############################################################################
    # softmax cross-entropy loss
    {
        "model_func": lambda: Sequential(Linear(10, 5), ReLU(), Linear(5, 3)),
        "loss_func": lambda: CrossEntropyLoss(reduction="mean"),
        "data": lambda: [
            (rand(3, 10), classification_targets((3,), 3)),
            (rand(4, 10), classification_targets((4,), 3)),
        ],
        "seed": 0,
    },
    # same as above, but uses reduction='sum'
    {
        "model_func": lambda: Sequential(Linear(10, 5), ReLU(), Linear(5, 3)),
        "loss_func": lambda: CrossEntropyLoss(reduction="sum"),
        "data": lambda: [
            (rand(3, 10), classification_targets((3,), 3)),
            (rand(4, 10), classification_targets((4,), 3)),
        ],
        "seed": 0,
    },
    # binary softmax cross-entropy loss, one output
    {
        "model_func": lambda: Sequential(Linear(10, 5), ReLU(), Linear(5, 1)),
        "loss_func": lambda: BCEWithLogitsLoss(reduction="mean"),
        "data": lambda: [
            (rand(3, 10), binary_classification_targets((3, 1))),
            (rand(4, 10), binary_classification_targets((4, 1))),
        ],
        "seed": 0,
    },
    # binary softmax cross-entropy loss, multiple outputs (tests the reduction factor)
    {
        "model_func": lambda: Sequential(Linear(10, 5), ReLU(), Linear(5, 2)),
        "loss_func": lambda: BCEWithLogitsLoss(reduction="mean"),
        "data": lambda: [
            (rand(3, 10), binary_classification_targets((3, 2))),
            (rand(4, 10), binary_classification_targets((4, 2))),
        ],
        "seed": 0,
    },
    # binary softmax cross-entropy loss, multiple outputs and sum reduction
    {
        "model_func": lambda: Sequential(Linear(10, 5), ReLU(), Linear(5, 2)),
        "loss_func": lambda: BCEWithLogitsLoss(reduction="sum"),
        "data": lambda: [
            (rand(3, 10), binary_classification_targets((3, 2))),
            (rand(4, 10), binary_classification_targets((4, 2))),
        ],
        "seed": 0,
    },
    ###############################################################################
    #                                  REGRESSION                                 #
    ###############################################################################
    {
        "model_func": lambda: Sequential(Linear(8, 5), ReLU(), Linear(5, 3)),
        "loss_func": lambda: MSELoss(reduction="mean"),
        "data": lambda: [
            (rand(2, 8), regression_targets((2, 3))),
            (rand(6, 8), regression_targets((6, 3))),
        ],
        "seed": 0,
    },
    # same as above, but uses reduction='sum'
    {
        "model_func": lambda: Sequential(Linear(8, 5), ReLU(), Linear(5, 3)),
        "loss_func": lambda: MSELoss(reduction="sum"),
        "data": lambda: [
            (rand(2, 8), regression_targets((2, 3))),
            (rand(6, 8), regression_targets((6, 3))),
        ],
        "seed": 0,
    },
    ###############################################################################
    #                               DICT-LIKE X                                   #
    ###############################################################################
    # Cross entropy
    {
        "model_func": lambda: ModelWithDictInput(num_classes=2),
        "loss_func": lambda: CrossEntropyLoss(reduction="mean"),
        "data": lambda: [
            (UserDict({"x": rand(3, 10)}), classification_targets((3,), 2)),
            ({"x": rand(4, 10)}, classification_targets((4,), 2)),
        ],
        "seed": 0,
    },
    # BCE
    {
        "model_func": lambda: ModelWithDictInput(num_classes=1),
        "loss_func": lambda: BCEWithLogitsLoss(reduction="mean"),
        "data": lambda: [
            (UserDict({"x": rand(3, 10)}), binary_classification_targets((3, 1))),
            ({"x": rand(4, 10)}, binary_classification_targets((4, 1))),
        ],
        "seed": 0,
    },
    # MSE
    {
        "model_func": lambda: ModelWithDictInput(num_classes=2),
        "loss_func": lambda: MSELoss(reduction="mean"),
        "data": lambda: [
            (UserDict({"x": rand(3, 10)}), regression_targets((3, 2))),
            ({"x": rand(4, 10)}, regression_targets((4, 2))),
        ],
        "seed": 0,
    },
]

INV_CASES = []
for case in INV_CASES_NO_DEVICE:
    for device in DEVICES:
        case_with_device = {**case, "device": device}
        INV_CASES.append(case_with_device)


# add test cases here
CASES_NO_DEVICE = INV_CASES_NO_DEVICE + [
    # softmax cross-entropy loss with additional input/output dimension
    {
        "model_func": lambda: WeightShareModel(
            Sequential(Linear(10, 5), ReLU(), Linear(5, 3)),
            setting=KFACType.EXPAND,
            loss="CE",
        ),
        "loss_func": lambda: CrossEntropyLoss(reduction="mean"),
        "data": lambda: [
            (rand(3, 5, 10), classification_targets((3, 5), 3)),
            (rand(4, 5, 10), classification_targets((4, 5), 3)),
        ],
        "seed": 0,
    },
    # same as above, but uses reduction='sum'
    {
        "model_func": lambda: WeightShareModel(
            Sequential(Linear(10, 5), ReLU(), Linear(5, 3)),
            setting=KFACType.EXPAND,
            loss="CE",
        ),
        "loss_func": lambda: CrossEntropyLoss(reduction="sum"),
        "data": lambda: [
            (rand(3, 5, 10), classification_targets((3, 5), 3)),
            (rand(4, 5, 10), classification_targets((4, 5), 3)),
        ],
        "seed": 0,
    },
    # binary softmax cross-entropy loss, multiple outputs, additional input/output
    # dimension, and mean reduction (tests the reduction factor)
    {
        "model_func": lambda: Sequential(Linear(10, 5), ReLU(), Linear(5, 2)),
        "loss_func": lambda: BCEWithLogitsLoss(reduction="mean"),
        "data": lambda: [
            (rand(3, 5, 10), binary_classification_targets((3, 5, 2))),
            (rand(4, 5, 10), binary_classification_targets((4, 5, 2))),
        ],
        "seed": 0,
    },
    # binary softmax cross-entropy loss, multiple outputs, additional input/output
    # dimension, and sum reduction
    {
        "model_func": lambda: Sequential(Linear(10, 5), ReLU(), Linear(5, 2)),
        "loss_func": lambda: BCEWithLogitsLoss(reduction="sum"),
        "data": lambda: [
            (rand(3, 5, 10), binary_classification_targets((3, 5, 2))),
            (rand(4, 5, 10), binary_classification_targets((4, 5, 2))),
        ],
        "seed": 0,
    },
    # MSE loss, inputs and targets with additional dimension
    {
        "model_func": lambda: Sequential(Linear(8, 5), ReLU(), Linear(5, 3)),
        "loss_func": lambda: MSELoss(reduction="mean"),
        "data": lambda: [
            (rand(2, 5, 8), regression_targets((2, 5, 3))),
            (rand(6, 5, 8), regression_targets((6, 5, 3))),
        ],
        "seed": 0,
    },
    # same as above, but uses reduction='sum'
    {
        "model_func": lambda: Sequential(Linear(8, 5), ReLU(), Linear(5, 3)),
        "loss_func": lambda: MSELoss(reduction="sum"),
        "data": lambda: [
            (rand(2, 5, 8), regression_targets((2, 5, 3))),
            (rand(6, 5, 8), regression_targets((6, 5, 3))),
        ],
        "seed": 0,
    },
]

CASES = []
for case in CASES_NO_DEVICE:
    for device in DEVICES:
        case_with_device = {**case, "device": device}
        CASES.append(case_with_device)


# CNN model for classification task
CNN_CASES_NO_DEVICE = [
    {
        "model_func": lambda: Sequential(
            Conv2d(1, 6, 5),
            ReLU(),
            MaxPool2d(2),
            Conv2d(6, 16, 5),
            ReLU(),
            MaxPool2d(2),
            Flatten(),
            Linear(16 * 4 * 4, 10),
        ),
        "loss_func": lambda: CrossEntropyLoss(),
        "data": lambda: [
            (rand(5, 1, 28, 28), classification_targets((5,), 10)),
            (rand(5, 1, 28, 28), classification_targets((5,), 10)),
        ],
        "seed": 0,
    },
]
CNN_CASES = []
for cnn_case in CNN_CASES_NO_DEVICE:
    for device in DEVICES:
        case_with_device = {**cnn_case, "device": device}
        CNN_CASES.append(case_with_device)


NON_DETERMINISTIC_CASES_NO_DEVICE = []


def _non_deterministic_case_dropout():
    """Non-deterministic case due to dropout in the model."""
    N, D_in, H, C = 3, 10, 5, 2
    num_batches = 4
    num_samples = N * num_batches

    def data():
        X = rand(num_samples, D_in)
        y = classification_targets((num_samples,), C)
        dataset = TensorDataset(X, y)
        deterministic_data = DataLoader(
            dataset, batch_size=N, shuffle=False, drop_last=False
        )

        return deterministic_data

    return {
        "model_func": lambda: Sequential(
            Linear(D_in, H), Dropout(), ReLU(), Linear(H, C)
        ),
        "loss_func": lambda: CrossEntropyLoss(reduction="mean"),
        "data": data,
        "seed": 0,
    }


NON_DETERMINISTIC_CASES_NO_DEVICE += [_non_deterministic_case_dropout()]


def _non_deterministic_case_batchnorm():
    """Non-deterministic case due to Batchnorm in the model and shuffled batches."""
    N, D_in, H, C = 3, 10, 5, 2
    num_batches = 4
    num_samples = N * num_batches

    def model_func():
        bn = BatchNorm1d(H)
        bn.weight.data = rand_like(bn.weight.data)
        bn.bias.data = rand_like(bn.bias.data)

        return Sequential(Linear(D_in, H), bn, ReLU(), Linear(H, C))

    def data():
        X = rand(num_samples, D_in)
        y = classification_targets((num_samples,), C)
        dataset = TensorDataset(X, y)
        shuffled_data = DataLoader(dataset, batch_size=N, shuffle=True, drop_last=False)

        return shuffled_data

    return {
        "model_func": model_func,
        "loss_func": lambda: CrossEntropyLoss(reduction="mean"),
        "data": data,
        "seed": 0,
    }


NON_DETERMINISTIC_CASES_NO_DEVICE += [_non_deterministic_case_batchnorm()]


def _non_deterministic_case_drop_last():
    """Non-deterministic case due discarded samples if last batch is smaller than N."""
    N, D_in, H, C = 3, 10, 5, 2
    num_batches = 4
    num_samples = N * num_batches

    def data():
        X = rand(num_samples, D_in)
        y = classification_targets((num_samples,), C)
        dataset = TensorDataset(X, y)

        N_nondivisible = 5
        nondeterministic_data = DataLoader(
            dataset, batch_size=N_nondivisible, shuffle=True, drop_last=True
        )

        return nondeterministic_data

    return {
        "model_func": lambda: Sequential(Linear(D_in, H), ReLU(), Linear(H, C)),
        "loss_func": lambda: CrossEntropyLoss(reduction="mean"),
        "data": data,
        "seed": 0,
    }


NON_DETERMINISTIC_CASES_NO_DEVICE += [_non_deterministic_case_drop_last()]

NON_DETERMINISTIC_CASES = []
for case in NON_DETERMINISTIC_CASES_NO_DEVICE:
    for device in DEVICES:
        case_with_device = {**case, "device": device}
        NON_DETERMINISTIC_CASES.append(case_with_device)

ADJOINT_CASES = [False, True]
