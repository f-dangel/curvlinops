"""Utility functions to test `curvlinops`."""

from torch import cuda, device, randint


def get_available_devices():
    """Return CPU and, if present, GPU device.

    Returns:
        [device]: Available devices for `torch`.
    """
    devices = [device("cpu")]

    if cuda.is_available():
        devices.append(device("cuda"))

    return devices


def classification_targets(size, num_classes):
    """Create random targets for classes 0, ..., `num_classes - 1`."""
    return randint(size=size, low=0, high=num_classes)
