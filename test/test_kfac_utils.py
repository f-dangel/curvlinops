"""Test KFAC's utility functions."""

from typing import Tuple

from pytest import mark, raises
from torch import Tensor, as_tensor, manual_seed, ones, randint, randn, zeros
from torch.func import hessian
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from curvlinops.kfac_utils import (
    _check_binary_if_BCEWithLogitsLoss,
    loss_hessian_matrix_sqrt,
)
from curvlinops.utils import allclose_report


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


@mark.parametrize(
    "output_shape", [(3,), (3, 4), (3, 2, 2)], ids=["no_seq", "1d_seq", "2d_seq"]
)
def test_loss_hessian_matrix_sqrt_cross_entropy_sequence(output_shape: Tuple[int, ...]):
    """Test loss_hessian_matrix_sqrt for CrossEntropyLoss with sequence-valued predictions.

    Compares the Hessian square root S with the true Hessian H by verifying S @ S.T == H.

    Args:
        output_shape: Shape of the model output (without batch dimension).
            First element is the number of classes C, remaining elements are
            sequence dimensions.
    """
    manual_seed(0)

    loss_func = CrossEntropyLoss(reduction="sum")

    C = output_shape[0]
    seq_shape = output_shape[1:] if len(output_shape) > 1 else ()

    # Create prediction with batch dimension: [1, C, *seq_shape]
    output_one_datum = randn(1, *output_shape)

    # Target shape depends on sequence: [1, *seq_shape] with class indices
    # Create random class indices for targets
    target_one_datum = randint(0, C, (1, *seq_shape))

    # Compute Hessian square root using the function under test
    hess_sqrt = loss_hessian_matrix_sqrt(output_one_datum, target_one_datum, loss_func)

    # hess_sqrt has shape [*output_shape[1:], *output_shape[1:]] = [C, *seq, C, *seq]
    # Flatten to [C * prod(seq), C * prod(seq)] for matrix multiplication
    flat_dim = output_one_datum.numel()
    hess_sqrt_flat = hess_sqrt.reshape(flat_dim, flat_dim)

    # Compute S @ S.T
    hess_from_sqrt = hess_sqrt_flat @ hess_sqrt_flat.T

    # Compute true Hessian using torch.func.hessian
    def loss_fn(pred_flat: Tensor) -> Tensor:
        """Loss as function of flattened prediction."""
        pred = pred_flat.reshape(1, *output_shape)
        return loss_func(pred, target_one_datum)

    pred_flat = output_one_datum.flatten()
    true_hess = hessian(loss_fn)(pred_flat)

    assert allclose_report(hess_from_sqrt, true_hess)
