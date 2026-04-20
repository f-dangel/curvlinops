"""Test utility functions related to the GGN (loss Hessian, sampling)."""

from math import sqrt

from pytest import mark, raises
from torch import (
    Generator,
    Tensor,
    float64,
    manual_seed,
    rand,
    randint,
    randn,
    tensor,
)
from torch.func import hessian, vmap
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    HuberLoss,
    Module,
    MSELoss,
    MultiLabelSoftMarginLoss,
    NLLLoss,
    PoissonNLLLoss,
    SmoothL1Loss,
)

from curvlinops.ggn_utils import (
    _make_single_datum_sampler,
    loss_hessian_matrix_sqrt,
)
from curvlinops.utils import allclose_report

OUTPUT_SHAPES = [(3,), (3, 4), (3, 2, 5)]
OUTPUT_SHAPE_IDS = [f"{len(s)}d_sequence" for s in OUTPUT_SHAPES]

LOSS_FUNC_CLASSES = [CrossEntropyLoss, MSELoss, BCEWithLogitsLoss]
LOSS_FUNC_CLASS_IDS = [cls.__name__ for cls in LOSS_FUNC_CLASSES]

AUTODIFF_LOSS_CASES = [
    (
        "PoissonNLLLoss",
        lambda reduction: PoissonNLLLoss(reduction=reduction),
        (2, 3),
        lambda output: rand(*output.shape, dtype=output.dtype),
    ),
    (
        "MultiLabelSoftMarginLoss",
        lambda reduction: MultiLabelSoftMarginLoss(reduction=reduction),
        (4,),
        lambda output: rand(*output.shape, dtype=output.dtype),
    ),
    (
        "HuberLoss",
        lambda reduction: HuberLoss(reduction=reduction, delta=1.0),
        (2, 3),
        lambda output: output + 0.1 * randn(*output.shape, dtype=output.dtype),
    ),
    (
        "SmoothL1Loss",
        lambda reduction: SmoothL1Loss(reduction=reduction, beta=1.0),
        (4,),
        lambda output: output + 0.1 * randn(*output.shape, dtype=output.dtype),
    ),
]


class RegularizedMSELoss(Module):
    """MSE loss with an additional output-space quadratic regularizer."""

    def __init__(self, reduction: str = "mean", reg_strength: float = 0.3):
        """Initialize the regularized loss.

        Args:
            reduction: Reduction applied by the base MSE loss.
            reg_strength: Coefficient of the output-space quadratic penalty.
        """
        super().__init__()
        self.reduction = reduction
        self.base_loss = MSELoss(reduction=reduction)
        self.register_buffer("reg_strength", tensor(reg_strength))

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute the regularized loss.

        Returns:
            Scalar regularized loss value.
        """
        return self.base_loss(prediction, target) + self.reg_strength * (
            prediction.square().mean()
        )


def _single_datum_loss_hessian(
    output_one_datum: Tensor, target_one_datum: Tensor, loss_func: Module
) -> Tensor:
    """Compute the single-datum output Hessian via autodiff.

    Returns:
        Hessian with respect to the flattened single-datum output.
    """
    output_shape = output_one_datum.shape

    def _loss_fn(pred_flat: Tensor) -> Tensor:
        pred_with_batch_axis = pred_flat.reshape(1, *output_shape)
        target_with_batch_axis = target_one_datum.unsqueeze(0)
        return loss_func(pred_with_batch_axis, target_with_batch_axis)

    return hessian(_loss_fn)(output_one_datum.flatten())


def _hessian_from_loss_hessian_matrix_sqrt(
    output_one_datum: Tensor, target_one_datum: Tensor, loss_func: Module
) -> Tensor:
    """Reconstruct the Hessian from the returned matrix square root.

    Returns:
        Hessian with respect to the flattened single-datum output.
    """
    hess_sqrt = loss_hessian_matrix_sqrt(output_one_datum, target_one_datum, loss_func)
    flat_dim = output_one_datum.numel()
    hess_sqrt_flat = hess_sqrt.reshape(flat_dim, flat_dim)
    return hess_sqrt_flat @ hess_sqrt_flat.T


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
        # For BCEWithLogitsLoss, all dims are feature axes; targets in [0, 1]
        target_one_datum = rand(*output_shape)

    # Compute Hessian square root using the function under test
    hess_sqrt = loss_hessian_matrix_sqrt(output_one_datum, target_one_datum, loss_func)

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


@mark.parametrize(
    ("_loss_name", "loss_factory", "output_shape", "target_fn"),
    AUTODIFF_LOSS_CASES,
    ids=[case[0] for case in AUTODIFF_LOSS_CASES],
)
@mark.parametrize("reduction", ["mean", "sum"])
def test_loss_hessian_matrix_sqrt_autodiff_fallback(
    _loss_name: str,
    loss_factory,
    output_shape: tuple[int, ...],
    target_fn,
    reduction: str,
):
    """Test autodiff-based Hessian square roots for supported non-analytic losses."""
    manual_seed(0)

    output_one_datum = randn(*output_shape, dtype=float64)
    target_one_datum = target_fn(output_one_datum)
    loss_func = loss_factory(reduction)

    hess_from_sqrt = _hessian_from_loss_hessian_matrix_sqrt(
        output_one_datum, target_one_datum, loss_func
    )
    true_hess = _single_datum_loss_hessian(output_one_datum, target_one_datum, loss_func)

    assert allclose_report(hess_from_sqrt, true_hess)


@mark.parametrize("reduction", ["mean", "sum"])
def test_loss_hessian_matrix_sqrt_autodiff_fallback_regularized_loss(
    reduction: str,
):
    """Test autodiff fallback for a custom loss module with registered state."""
    manual_seed(0)

    output_one_datum = randn(2, 3).double()
    target_one_datum = randn(2, 3).double()
    loss_func = RegularizedMSELoss(reduction=reduction)

    hess_from_sqrt = _hessian_from_loss_hessian_matrix_sqrt(
        output_one_datum, target_one_datum, loss_func
    )
    true_hess = _single_datum_loss_hessian(output_one_datum, target_one_datum, loss_func)

    assert allclose_report(hess_from_sqrt, true_hess)


def test_loss_hessian_matrix_sqrt_autodiff_fallback_requires_positive_definite():
    """Test that the autodiff fallback rejects singular Hessians."""
    manual_seed(0)

    output_one_datum = randn(4).log_softmax(dim=0).double()
    target_one_datum = randint(0, 4, ()).long()
    loss_func = NLLLoss(reduction="mean")

    with raises(ValueError, match="strictly positive definite"):
        loss_hessian_matrix_sqrt(output_one_datum, target_one_datum, loss_func)


@mark.parametrize("output_shape", OUTPUT_SHAPES, ids=OUTPUT_SHAPE_IDS)
@mark.parametrize("loss_func_cls", LOSS_FUNC_CLASSES, ids=LOSS_FUNC_CLASS_IDS)
@mark.parametrize("reduction", ["mean", "sum"])
def test_grad_output_sampler_convergence(
    output_shape: tuple[int, ...],
    loss_func_cls: CrossEntropyLoss | MSELoss | BCEWithLogitsLoss,
    reduction: str,
) -> None:
    """Test that sampled gradient outer products converge to the true Hessian.

    Verifies that E[g g^T] ~ H where g are sampled gradients and H is the true
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
        target_one_datum = rand(1, *output_shape).double()
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

    grad_samples = sampler(output_one_datum, mc_samples, target_one_datum, generator)

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
