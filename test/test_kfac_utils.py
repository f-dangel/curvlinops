"""Test KFAC's utility functions."""

from pytest import raises
from torch import as_tensor, ones, zeros
from torch.nn import BCEWithLogitsLoss

from curvlinops.kfac_utils import _check_binary_if_BCEWithLogitsLoss


def test_check_binary_if_BCEWithLogitsLoss():
    """Test checking for binary entries of a tensor."""
    bce_loss = BCEWithLogitsLoss()

    # This should not raise an error
    # tensor containing zeros and ones
    binary_tensor = as_tensor([1.0, 0.0, 0.0, 0.0])
    _check_binary_if_BCEWithLogitsLoss(binary_tensor, bce_loss)
    # only zeros
    _check_binary_if_BCEWithLogitsLoss(zeros(3), bce_loss)
    # only ones
    _check_binary_if_BCEWithLogitsLoss(ones(3), bce_loss)

    # This should raise a NotImplementedError
    nonbinary_tensor = as_tensor([0.5, 0.0, 1.0])
    with raises(NotImplementedError):
        _check_binary_if_BCEWithLogitsLoss(nonbinary_tensor, bce_loss)
