"""Test utility functions related to KFAC."""

from contextlib import nullcontext
from math import sqrt

from pytest import mark, raises, warns
from torch import (
    Generator,
    Tensor,
    as_tensor,
    cat,
    manual_seed,
    ones,
    rand_like,
    randint,
    randn,
    zeros,
)
from torch.func import hessian, vmap
from torch.nn import (
    BCEWithLogitsLoss,
    Conv2d,
    CrossEntropyLoss,
    Linear,
    MSELoss,
    Sequential,
)

from curvlinops.kfac_utils import (
    FromCanonicalLinearOperator,
    ToCanonicalLinearOperator,
    _check_binary_if_BCEWithLogitsLoss,
    _make_single_datum_sampler,
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
    output_shape: tuple[int, ...],
    loss_func_cls: CrossEntropyLoss | MSELoss | BCEWithLogitsLoss,
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
        """Loss as function of flattened prediction.

        Args:
            pred_flat: Flattened prediction 1d tensor of shape [C * prod(seq)].

        Returns:
            Loss value.
        """
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
    output_shape: tuple[int, ...],
    loss_func_cls: CrossEntropyLoss | MSELoss | BCEWithLogitsLoss,
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

    # Create gradient sampler (vmapped over batch)
    sampler = vmap(
        _make_single_datum_sampler(loss_func),
        in_dims=(0, None, 0, None),
        out_dims=1,
        randomness="different",
    )

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
        """Loss as function of flattened prediction.

        Args:
            pred_flat: Flattened prediction 1d tensor of shape [C * prod(seq)].

        Returns:
            Loss value.
        """
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


@mark.parametrize("separate_weight_and_bias", [True, False], ids=["separate", "joint"])
def test_CanonicalLinearOperator(separate_weight_and_bias: bool):
    """Test that canonicalization properly reorders, groups, and ungroups parameters."""
    manual_seed(0)

    # Define an unconventional order of the parameter space, mixing weights and biases.
    # NOTE This network is non-functional, we only care about its parameters
    net = Sequential(Conv2d(6, 5, 4), Linear(5, 3), Linear(3, 2, bias=False))

    # Natural order would be: w1, b1, w2, b2, w3
    # Create unconventional order: w1, b2, b1, w3, w2
    natural_params = list(net.parameters())
    new_order = [0, 3, 1, 4, 2]
    params = [natural_params[idx] for idx in new_order]

    # Define param_positions to map back to layers
    param_positions = [
        {"weight": 0, "bias": 2},  # layer1: weight at pos 0, bias at pos 2
        {"weight": 4, "bias": 1},  # layer2: weight at pos 4, bias at pos 1
        {"weight": 3},  # layer3: weight at pos 3
    ]

    # Extract param shapes, device, and dtype
    param_shapes = [p.shape for p in params]  # p.shape is already a Size object
    device = params[0].device
    dtype = params[0].dtype

    # Verify correct behavior of canonicalization for this case
    x = [rand_like(p) for p in params]
    x_w1, x_b2, x_b1, x_w3, x_w2 = x

    x_canonical = (
        [x_i.flatten() for x_i in [x_w1, x_b1, x_w2, x_b2, x_w3]]
        if separate_weight_and_bias
        else [
            # Conv kernel is 4d, we flatten it to 2d first before appending the bias
            cat([x_w1.flatten(start_dim=1), x_b1.unsqueeze(-1)], dim=-1).flatten(),
            cat([x_w2, x_b2.unsqueeze(-1)], dim=-1).flatten(),
            x_w3.flatten(),
        ]
    )

    # Multiplication with canonicalization operator should produce x_canonical
    to_canonical = ToCanonicalLinearOperator(
        param_shapes, param_positions, separate_weight_and_bias, device, dtype
    )
    to_canonical_x = to_canonical @ x
    assert len(to_canonical_x) == len(x_canonical)
    assert all(allclose_report(x1, x2) for x1, x2 in zip(to_canonical_x, x_canonical))

    # Multiplication of x_canonical with from_canonical operator should produce x
    from_canonical = FromCanonicalLinearOperator(
        param_shapes, param_positions, separate_weight_and_bias, device, dtype
    )
    from_canonical_x = from_canonical @ x_canonical
    assert len(from_canonical_x) == len(x)
    assert all(allclose_report(x1, x2) for x1, x2 in zip(from_canonical_x, x))

    # Check that the transpose operator is the inverse
    for P, v in zip([to_canonical, from_canonical], [x, x_canonical]):
        PTP_v = P.adjoint() @ (P @ v)
        assert len(PTP_v) == len(v)
        assert all(allclose_report(v1, v2) for v1, v2 in zip(PTP_v, v))
