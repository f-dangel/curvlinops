"""Test KFAC's utility functions."""

from contextlib import nullcontext
from math import sqrt
from typing import Tuple, Union

from pytest import mark, raises, warns
from torch import Generator, Tensor, as_tensor, manual_seed, ones, randint, randn, zeros
from torch.func import hessian
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from curvlinops.kfac_utils import (
    _check_binary_if_BCEWithLogitsLoss,
    loss_hessian_matrix_sqrt,
    make_grad_output_sampler,
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

    # Create prediction with batch dimension: [*output_shape]
    output_one_datum = randn(*output_shape)

    # Create targets based on loss function type
    if loss_func_cls == CrossEntropyLoss:
        # For CrossEntropyLoss, first dim is class dimension C, rest are sequence dims
        C = output_shape[0]
        seq_shape = output_shape[1:] if len(output_shape) > 1 else ()
        # Target shape: [*seq_shape] with class indices in [0, C)
        target_one_datum = randint(0, C, seq_shape)
    elif loss_func_cls == MSELoss:
        # For MSELoss, all dims are feature axes; target shape matches output
        target_one_datum = randn(*output_shape)
    elif loss_func_cls == BCEWithLogitsLoss:
        # For BCEWithLogitsLoss, all dims are feature axes; target has binary values
        target_one_datum = randint(0, 2, output_shape).float()

    # Compute Hessian square root using the function under test
    # Check that the warning is raised for BCEWithLogitsLoss
    _check_binary_if_BCEWithLogitsLoss(target_one_datum, loss_func)
    with (
        warns(
            UserWarning,
            match="BCEWithLogitsLoss only supports binary targets.*not being verified",
        )
        if loss_func_cls == BCEWithLogitsLoss
        else nullcontext()
    ):
        hess_sqrt = loss_hessian_matrix_sqrt(
            output_one_datum, target_one_datum, loss_func
        )

    # hess_sqrt has shape [*output_shape, *output_shape] = [C, *seq, C, *seq]
    # Flatten to [C * prod(seq), C * prod(seq)] for matrix multiplication
    flat_dim = output_one_datum.numel()
    hess_sqrt_flat = hess_sqrt.reshape(flat_dim, flat_dim)

    # Compute S @ S.T
    hess_from_sqrt = hess_sqrt_flat @ hess_sqrt_flat.T

    # Compute true Hessian using torch.func.hessian
    def _loss_fn(pred_flat: Tensor) -> Tensor:
        """Loss as function of flattened prediction."""
        # NOTE We have to make the batch axis explicit for nn.CrossEntropyLoss
        pred_with_batch_axis = pred_flat.reshape(1, *output_shape)
        target_with_batch_axis = target_one_datum.unsqueeze(0)
        return loss_func(pred_with_batch_axis, target_with_batch_axis)

    pred_flat = output_one_datum.flatten()
    true_hess = hessian(_loss_fn)(pred_flat)

    assert allclose_report(hess_from_sqrt, true_hess)


@mark.parametrize("output_shape", OUTPUT_SHAPES, ids=OUTPUT_SHAPE_IDS)
@mark.parametrize("loss_func_cls", LOSS_FUNC_CLASSES, ids=LOSS_FUNC_CLASS_IDS)
@mark.parametrize("reduction", ["mean", "sum"])
def test_grad_output_sampler_convergence(
    output_shape: Tuple[int, ...],
    loss_func_cls: Union[CrossEntropyLoss, MSELoss, BCEWithLogitsLoss],
    reduction: str,
) -> None:
    """Test that sampled gradient outer products converge to the true Hessian.

    Verifies that E[g g^T] ≈ H where g are sampled gradients and H is the true
    Hessian.

    Args:
        output_shape: Shape of the model output (without batch dimension).
        loss_func_cls: The loss function class to test.
        reduction: The reduction method for the loss function.
    """
    manual_seed(0)

    loss_func = loss_func_cls(reduction=reduction)

    # Create prediction with batch size 1: [1, *output_shape]
    output_one_datum = randn(1, *output_shape).double()

    # Create targets based on loss function type
    if loss_func_cls == CrossEntropyLoss:
        C = output_shape[0]
        seq_shape = output_shape[1:] if len(output_shape) > 1 else ()
        target_one_datum = randint(0, C, (1, *seq_shape))
    elif loss_func_cls == MSELoss:
        target_one_datum = randn(1, *output_shape).double()
    elif loss_func_cls == BCEWithLogitsLoss:
        target_one_datum = randint(0, 2, (1, *output_shape)).float()
    else:
        raise NotImplementedError(f"Unsupported loss function: {loss_func_cls}")

    # Create gradient sampler
    sampler = make_grad_output_sampler(loss_func)

    # Sample many gradients with fixed generator for reproducibility
    generator = Generator().manual_seed(42)
    mc_samples = 600_000

    # Check that the warning is raised for BCEWithLogitsLoss
    _check_binary_if_BCEWithLogitsLoss(target_one_datum, loss_func)
    with (
        warns(
            UserWarning,
            match="BCEWithLogitsLoss only supports binary targets.*not being verified",
        )
        if loss_func_cls == BCEWithLogitsLoss
        else nullcontext()
    ):
        grad_samples = sampler(
            output_one_datum, mc_samples, target_one_datum, generator
        )

    # Compute empirical covariance: E[g g^T]
    # grad_samples has shape [num_samples, 1, *output_shape]
    # Flatten the non-sample dimensions
    flat_dim = output_one_datum.numel()
    grad_samples = grad_samples.reshape(mc_samples, flat_dim).div_(sqrt(mc_samples))
    empirical_cov = grad_samples.T @ grad_samples

    # Compute true Hessian using torch.func.hessian
    def loss_fn(pred_flat: Tensor) -> Tensor:
        """Loss as function of flattened prediction."""
        pred = pred_flat.reshape(1, *output_shape)
        return loss_func(pred, target_one_datum)

    pred_flat = output_one_datum.flatten()
    true_hess = hessian(loss_fn)(pred_flat)

    # Check convergence with looser tolerances due to Monte Carlo sampling
    # Scale by Hessian's abs max to make tolerances more transferable
    scale = true_hess.abs().max()
    assert allclose_report(
        empirical_cov / scale, true_hess / scale, rtol=1e-3, atol=5e-3
    )
