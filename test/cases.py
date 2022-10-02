"""Contains test cases for linear operators."""

from test.utils import classification_targets, get_available_devices

from torch import rand, rand_like
from torch.nn import BatchNorm1d, CrossEntropyLoss, Dropout, Linear, ReLU, Sequential
from torch.utils.data import DataLoader, TensorDataset

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
    },
    # same as above, but uses reduction='sum'
    {
        "model_func": lambda: Sequential(Linear(10, 5), ReLU(), Linear(5, 2)),
        "loss_func": lambda: CrossEntropyLoss(reduction="sum"),
        "data": lambda: [
            (rand(3, 10), classification_targets((3,), 2)),
            (rand(4, 10), classification_targets((4,), 2)),
        ],
        "seed": 0,
    },
]

CASES = []
for case in CASES_NO_DEVICE:
    for device in DEVICES:
        case_with_device = {**case, "device": device}
        CASES.append(case_with_device)


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
        _model = Sequential(Linear(D_in, H), BatchNorm1d(H), ReLU(), Linear(H, C))
        # by default, BN weight=1, bias=0
        bn = _model[1]
        bn.weight.data = rand_like(bn.weight.data)
        bn.bias.data = rand_like(bn.bias.data)

        return _model

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
