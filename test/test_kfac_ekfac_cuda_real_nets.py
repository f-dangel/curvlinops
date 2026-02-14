"""CUDA integration tests for KFAC/EKFAC on realistic Conv/Linear networks."""

import torch
from pytest import mark
from torch import Tensor, device, float64, isfinite, manual_seed, rand, randint
from torch.nn import CrossEntropyLoss, Flatten, Linear, Module, ReLU, Sequential, Conv2d

from curvlinops.ekfac import EKFACLinearOperator
from curvlinops.kfac import FisherType, KFACLinearOperator
from test.utils import eye_like


def _make_mlp(dev: device) -> Module:
    return Sequential(
        Linear(128, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10),
    ).to(dev, float64)


def _make_convnet(dev: device) -> Module:
    return Sequential(
        Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
        ReLU(),
        Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        ReLU(),
        Flatten(),
        Linear(16 * 16 * 16, 10),
    ).to(dev, float64)


def _classification_data_mlp(dev: device):
    x1 = rand(8, 128, dtype=float64, device=dev)
    y1 = randint(0, 10, (8,), device=dev)
    x2 = rand(12, 128, dtype=float64, device=dev)
    y2 = randint(0, 10, (12,), device=dev)
    return [(x1, y1), (x2, y2)]


def _classification_data_conv(dev: device):
    x1 = rand(4, 3, 32, 32, dtype=float64, device=dev)
    y1 = randint(0, 10, (4,), device=dev)
    x2 = rand(6, 3, 32, 32, dtype=float64, device=dev)
    y2 = randint(0, 10, (6,), device=dev)
    return [(x1, y1), (x2, y2)]


@mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@mark.parametrize("linop_cls", [KFACLinearOperator, EKFACLinearOperator])
@mark.parametrize("fisher_type", [FisherType.TYPE2, FisherType.EMPIRICAL])
@mark.parametrize(
    "model_and_data_builder",
    [
        (_make_mlp, _classification_data_mlp),
        (_make_convnet, _classification_data_conv),
    ],
    ids=["mlp-linear", "convnet"],
)
def test_cuda_real_networks_smoke(linop_cls, fisher_type, model_and_data_builder):
    """Smoke test KFAC/EKFAC on CUDA for common MLP and ConvNet workloads."""
    manual_seed(0)
    dev = device("cuda")

    model_builder, data_builder = model_and_data_builder
    model = model_builder(dev)
    data = data_builder(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    loss_func = CrossEntropyLoss().to(dev)

    linop = linop_cls(
        model,
        loss_func,
        params,
        data,
        fisher_type=fisher_type,
        check_deterministic=False,
    )

    mat = linop @ eye_like(linop)
    assert isfinite(mat).all()

    # Also verify damped inverse application is numerically finite on random vectors.
    inv_linop = linop.inverse(damping=1e-2)
    v = rand(linop.shape[1], 3, dtype=float64, device=dev)
    out: Tensor = inv_linop @ v
    assert out.shape == v.shape
    assert isfinite(out).all()
