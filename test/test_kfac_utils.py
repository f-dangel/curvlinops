"""Test KFAC's utility functions."""

from typing import Tuple, Union

from pytest import mark, raises
from torch import Tensor, as_tensor, manual_seed, ones, randint, randn, zeros
from torch.func import hessian
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from curvlinops.kfac_utils import (
    _check_binary_if_BCEWithLogitsLoss,
    loss_hessian_matrix_sqrt,
)
from curvlinops.utils import allclose_report


def test_check_binary_if_BCEWithLogitsLoss():
    """Test checking for binary entries of a tensor for BCEWithLogitsLoss."""
    bce_loss, mse_loss = BCEWithLogitsLoss(), MSELoss()

    # This should not raise an error
    # tensor containing zeros and ones
    binary_tensor = as_tensor([1.0, 0.0, 0.0, 0.0])
    _check_binary_if_BCEWithLogitsLoss(binary_tensor, bce_loss)
    # only zeros
    _check_binary_if_BCEWithLogitsLoss(zeros(3), bce_loss)
    # only ones
    _check_binary_if_BCEWithLogitsLoss(ones(3), bce_loss)

    # A non-binary tensor should raise a NotImplementedError only if the loss if BCE
    nonbinary_tensor = as_tensor([0.5, 0.0, 1.0])
    _check_binary_if_BCEWithLogitsLoss(nonbinary_tensor, mse_loss)
    with raises(NotImplementedError):
        _check_binary_if_BCEWithLogitsLoss(nonbinary_tensor, bce_loss)


OUTPUT_SHAPES = [(3,), (3, 4), (3, 2, 5)]
OUTPUT_SHAPE_IDS = [f"{len(s)}d_sequence" for s in OUTPUT_SHAPES]

LOSS_FUNC_CLASSES = [CrossEntropyLoss, MSELoss, BCEWithLogitsLoss]
LOSS_FUNC_CLASS_IDS = [cls.__name__ for cls in LOSS_FUNC_CLASSES]


@mark.parametrize("output_shape", OUTPUT_SHAPES, ids=OUTPUT_SHAPE_IDS)
@mark.parametrize("loss_func_cls", LOSS_FUNC_CLASSES, ids=LOSS_FUNC_CLASS_IDS)
@mark.parametrize("reduction", ["mean", "sum"])
def test_loss_hessian_matrix_sqrt(
    output_shape: Tuple[int, ...],
    loss_func_cls: Union[CrossEntropyLoss, MSELoss, BCEWithLogitsLoss],
    reduction: str,
):
    """Test loss_hessian_matrix_sqrt for various loss functions.

    Compares the Hessian square root S with the true Hessian H by verifying
    S @ S.T == H.

    Args:
        output_shape: Shape of the model output (without batch dimension).
            For CrossEntropyLoss, the first element is the number of classes C,
            and remaining elements are sequence dimensions. For MSELoss and
            BCEWithLogitsLoss, all elements are treated as feature axes.
        loss_func_cls: The loss function class to test.
        reduction: The reduction method for the loss function.
    """
    assert loss_func_cls in {CrossEntropyLoss, MSELoss, BCEWithLogitsLoss}
    manual_seed(0)

    loss_func = loss_func_cls(reduction=reduction)

    # Create prediction with batch dimension: [1, *output_shape]
    output_one_datum = randn(1, *output_shape)

    # Create targets based on loss function type
    if loss_func_cls == CrossEntropyLoss:
        # For CrossEntropyLoss, first dim is class dimension C, rest are sequence dims
        C = output_shape[0]
        seq_shape = output_shape[1:] if len(output_shape) > 1 else ()
        # Target shape: [1, *seq_shape] with class indices in [0, C)
        target_one_datum = randint(0, C, (1, *seq_shape))
    elif loss_func_cls == MSELoss:
        # For MSELoss, all dims are feature axes; target shape matches output
        target_one_datum = randn(1, *output_shape)
    elif loss_func_cls == BCEWithLogitsLoss:
        # For BCEWithLogitsLoss, all dims are feature axes; target has binary values
        target_one_datum = randint(0, 2, (1, *output_shape)).float()

    # Compute Hessian square root using the function under test
    hess_sqrt = loss_hessian_matrix_sqrt(output_one_datum, target_one_datum, loss_func)

    # hess_sqrt has shape [*output_shape[1:], *output_shape[1:]] = [C, *seq, C, *seq]
    # Flatten to [C * prod(seq), C * prod(seq)] for matrix multiplication
    flat_dim = output_one_datum.numel()
    hess_sqrt_flat = hess_sqrt.reshape(flat_dim, flat_dim)

    # Compute S @ S.T
    hess_from_sqrt = hess_sqrt_flat @ hess_sqrt_flat.T

    # Compute true Hessian using torch.func.hessian
    def _loss_fn(pred_flat: Tensor) -> Tensor:
        """Loss as function of flattened prediction."""
        pred = pred_flat.reshape(1, *output_shape)
        return loss_func(pred, target_one_datum)

    pred_flat = output_one_datum.flatten()
    true_hess = hessian(_loss_fn)(pred_flat)

    assert allclose_report(hess_from_sqrt, true_hess)
