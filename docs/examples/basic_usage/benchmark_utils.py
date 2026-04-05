"""Utility functions for benchmark problem setup and configuration."""

import inspect
import json
from collections.abc import Iterable
from contextlib import nullcontext
from os import makedirs, path

import requests
import torch
from torch import Tensor, cuda, rand, randint, stack, zeros_like
from torch.nn import (
    CrossEntropyLoss,
    Linear,
    Module,
    Parameter,
    ReLU,
    Sequential,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchvision.models import ResNet50_Weights, resnet18, resnet50

from curvlinops import (
    EFLinearOperator,
    EKFACLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
)
from curvlinops._torch_base import PyTorchLinearOperator

# In the execution with sphinx-gallery, __file__ is not defined and we need
# to set it manually using the trick from https://stackoverflow.com/a/53293924
if "__file__" not in globals():
    __file__ = inspect.getfile(lambda: None)

HEREDIR = path.dirname(path.abspath(__file__))
RESULTDIR = path.join(HEREDIR, "benchmark")
makedirs(RESULTDIR, exist_ok=True)
REFERENCE_OP = "gradient_and_loss"

# -- Problem/linop constants --

PROBLEM_STRS = [
    "synthetic_mnist_mlp",
    "synthetic_cifar10_resnet18",
    "synthetic_imagenet_resnet50",
    "synthetic_shakespeare_nanogpt",
]

LINOP_STRS = [
    "Hessian",
    "Generalized Gauss-Newton",
    "Empirical Fisher",
    "Monte-Carlo Fisher",
    "EKFAC (hooks)",
    "EKFAC inverse (hooks)",
    "EKFAC (fx)",
    "EKFAC inverse (fx)",
    "KFAC (hooks)",
    "KFAC inverse (hooks)",
    "KFAC (fx)",
    "KFAC inverse (fx)",
]

# For matvec, backend doesn't matter — use hooks as the representative
MATVEC_LINOP_STRS = [
    "Hessian",
    "Generalized Gauss-Newton",
    "Empirical Fisher",
    "Monte-Carlo Fisher",
    "EKFAC (hooks)",
    "EKFAC inverse (hooks)",
    "KFAC (hooks)",
    "KFAC inverse (hooks)",
]

# Names that use KFAC-style parameter selection (only supported layers)
_KFAC_LIKE = {
    "KFAC (hooks)",
    "KFAC inverse (hooks)",
    "KFAC (fx)",
    "KFAC inverse (fx)",
    "EKFAC (hooks)",
    "EKFAC inverse (hooks)",
    "EKFAC (fx)",
    "EKFAC inverse (fx)",
}

# Linear operators that use JVPs need to handle attention differently because
# PyTorch's efficient attention does not implement double-backward yet.
# See https://github.com/pytorch/pytorch/issues/116350
HAS_JVP = (
    HessianLinearOperator,
    GGNLinearOperator,
    EFLinearOperator,
)

# Linop category sets for precompute sub-phase dispatch
_IS_EKFAC = {
    "EKFAC (hooks)",
    "EKFAC inverse (hooks)",
    "EKFAC (fx)",
    "EKFAC inverse (fx)",
}
_IS_KFAC_INVERSE_HOOKS = {"KFAC inverse (hooks)"}
_IS_FX = {
    "KFAC (fx)",
    "KFAC inverse (fx)",
    "EKFAC (fx)",
    "EKFAC inverse (fx)",
}
# Operators whose matvec supports torch.compile (0 graph breaks).
# run_operator will also measure compiled matvec and memory for these.
_IS_COMPILABLE = {
    "Hessian",
    "Generalized Gauss-Newton",
    "Monte-Carlo Fisher",
    "KFAC (hooks)",
    "KFAC inverse (hooks)",
    "KFAC (fx)",
    "KFAC inverse (fx)",
}

# Sub-phase operation names for precompute breakdown
EKFAC_PRECOMPUTE_OPS = ["kfac_factors", "eigenvalue_correction", "eigh"]
KFAC_INVERSE_PRECOMPUTE_OPS = ["kfac_factors", "cholesky_inverse"]
FX_PRECOMPUTE_OPS = ["kfac_factors", "tracing"]


# -- Path helpers --


def _problem_dir(problem_str: str) -> str:
    """Get the problem-specific subdirectory, creating it if needed.

    Args:
        problem_str: The problem.

    Returns:
        Absolute path to the problem subdirectory.
    """
    d = path.join(RESULTDIR, problem_str)
    makedirs(d, exist_ok=True)
    return d


def reference_benchpath(problem_str: str, device_str: str) -> str:
    """Get the path to save the reference gradient_and_loss benchmark.

    This is measured once per problem (not per linop).

    Args:
        problem_str: The problem.
        device_str: The device.

    Returns:
        The path to save the reference benchmark results.
    """
    return path.join(_problem_dir(problem_str), f"{REFERENCE_OP}_{device_str}.json")


def benchpath(
    linop_str: str,
    problem_str: str,
    device_str: str,
    op_str: str | None = None,
) -> str:
    """Get the path to save benchmark results.

    Results are stored under ``benchmark/{problem}/{linop}_{device}.json``.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        op_str: If given, appended before the extension (e.g. ``"peakmem"``
            produces a temporary file ``{linop}_{device}_peakmem.json``).

    Returns:
        The path to save the benchmark results.
    """
    name = linop_str.replace(" ", "-")
    suffix = f"_{op_str}" if op_str is not None else ""
    return path.join(_problem_dir(problem_str), f"{name}_{device_str}{suffix}.json")


def figpath(problem_str: str, device_str: str, metric: str = "time") -> str:
    """Get the path to save the figure.

    Args:
        problem_str: The problem.
        device_str: The device.
        metric: The metric to save. Default is ``'time'``.

    Returns:
        The path to save the figure.
    """
    return path.join(_problem_dir(problem_str), f"{metric}_{device_str}.pdf")


# -- Misc helpers --


def save_environment_info(result_dir: str):
    """Save PyTorch version and GPU info to a metadata file.

    Args:
        result_dir: Directory where ``environment.json`` is written.
    """
    info = {"pytorch_version": torch.__version__}
    if cuda.is_available():
        info["gpu"] = cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda

    info_path = path.join(result_dir, "environment.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    for key, value in info.items():
        print(f"  {key}: {value}")


def add_gradient_reference(ax, reference: float):
    """Add a dashed reference line and a top axis showing multiples of gradient time.

    Args:
        ax: The matplotlib axes.
        reference: The reference value (e.g. gradient computation time or memory).
    """
    ax.axvline(reference, color="black", linestyle="--")
    ax.secondary_xaxis(
        "top",
        functions=(lambda x: x / reference, lambda x: x * reference),
    ).set_xlabel("Relative to gradient computation")


def _get_precompute_ops(linop_str: str) -> list[str]:
    """Return the sub-phase operation names for a given linop.

    Args:
        linop_str: The linear operator name.

    Returns:
        List of sub-phase operation names.
    """
    if linop_str in _IS_EKFAC and linop_str in _IS_FX:
        return EKFAC_PRECOMPUTE_OPS + ["tracing"]
    elif linop_str in _IS_EKFAC:
        return EKFAC_PRECOMPUTE_OPS
    elif linop_str in _IS_KFAC_INVERSE_HOOKS:
        return KFAC_INVERSE_PRECOMPUTE_OPS
    elif linop_str in _IS_FX:
        return FX_PRECOMPUTE_OPS
    else:
        return ["kfac_factors"]


def attention_context(linop_or_cls, model: Module):
    """Context manager for the attention double-backward workaround.

    Efficient attention does not support double-backward. Returns
    ``sdpa_kernel(SDPBackend.MATH)`` when needed, otherwise ``nullcontext()``.

    Args:
        linop_or_cls: A linear operator instance or class.
        model: The neural net (checked for :class:`GPTWrapper`).

    Returns:
        A context manager.
    """
    if isinstance(linop_or_cls, type):
        has_jvp = issubclass(linop_or_cls, HAS_JVP)
    else:
        has_jvp = isinstance(linop_or_cls, HAS_JVP)
    if has_jvp and isinstance(model, GPTWrapper):
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


# -- Model helpers --


def maybe_download_nanogpt():
    """Download the nanoGPT model definition."""
    commit = "f08abb45bd2285627d17da16daea14dda7e7253e"
    repo = "https://raw.githubusercontent.com/karpathy/nanoGPT/"

    # download the model definition as 'nanogpt_model.py'
    model_url = f"{repo}{commit}/model.py"
    model_path = path.join(HEREDIR, "nanogpt_model.py")
    if not path.exists(model_path):
        url = requests.get(model_url)
        with open(model_path, "w") as f:
            f.write(url.content.decode("utf-8"))


class GPTWrapper(Module):
    """Wraps Karpathy's nanoGPT model repo so that it produces the flattened logits."""

    def __init__(self, gpt: Module):
        """Store the wrapped nanoGPT model.

        Args:
            gpt: The nanoGPT model.
        """
        super().__init__()
        self.gpt = gpt

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the nanoGPT model.

        Args:
            x: The input tensor. Has shape ``(batch_size, sequence_length)``.

        Returns:
            The flattened logits.
            Has shape ``(batch_size * sequence_length, vocab_size)``.
        """
        y_dummy = zeros_like(x)
        logits, _ = self.gpt(x, y_dummy)
        return logits.view(-1, logits.size(-1))


def setup_synthetic_shakespeare_nanogpt(
    batch_size: int = 4,
) -> tuple[GPTWrapper, CrossEntropyLoss, list[tuple[Tensor, Tensor]]]:
    """Set up the nanoGPT model and synthetic Shakespeare dataset for the benchmark.

    Args:
        batch_size: The batch size to use. Default is ``4``.

    Returns:
        A tuple containing the nanoGPT model, the loss function, and the data.
    """
    # download nanogpt_model and import GPT and GPTConfig from it
    maybe_download_nanogpt()
    from nanogpt_model import GPT, GPTConfig

    config = GPTConfig()
    block_size = config.block_size

    base = GPT(config)
    # Remove weight tying as this will break the parameter-to-layer detection
    base.transformer.wte.weight = Parameter(
        data=base.transformer.wte.weight.data.detach().clone()
    )

    model = GPTWrapper(base).eval()
    loss_function = CrossEntropyLoss(ignore_index=-1)

    # generate a synthetic Shakespeare and load one batch
    vocab_size = config.vocab_size
    train_data = randint(0, vocab_size, (5 * block_size,)).long()
    ix = randint(train_data.numel() - block_size, (batch_size,))
    X = stack([train_data[i : i + block_size] for i in ix])
    y = stack([train_data[i + 1 : i + 1 + block_size] for i in ix])
    # flatten the target because the GPT wrapper flattens the logits
    data = [(X, y.flatten())]

    return model, loss_function, data


def setup_synthetic_imagenet_resnet50(
    batch_size: int = 64,
) -> tuple[Module, CrossEntropyLoss, list[tuple[Tensor, Tensor]]]:
    """Set up ResNet50 on synthetic ImageNet for the benchmark.

    Args:
        batch_size: The batch size to use. Default is ``64``.

    Returns:
        A tuple containing the ResNet50 model, the loss function
        and the data.
    """
    X = rand(batch_size, 3, 224, 224)
    y = randint(0, 1000, (batch_size,))
    data = [(X, y)]
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    loss_function = CrossEntropyLoss()

    return model, loss_function, data


def setup_synthetic_cifar10_resnet18(
    batch_size: int = 512,
) -> tuple[Module, CrossEntropyLoss, list[tuple[Tensor, Tensor]]]:
    """Set up ResNet18 on synthetic CIFAR10 for the benchmark.

    Args:
        batch_size: The batch size to use. Default is ``512``.

    Returns:
        A tuple containing the ResNet18 model, the loss function
        and the data.
    """
    X = rand(batch_size, 3, 32, 32)
    num_classes = 10
    y = randint(0, num_classes, (batch_size,))
    data = [(X, y)]
    model = resnet18(num_classes=num_classes)
    loss_function = CrossEntropyLoss()

    return model, loss_function, data


def setup_synthetic_mnist_mlp(
    batch_size: int = 512,
) -> tuple[Sequential, CrossEntropyLoss, list[tuple[Tensor, Tensor]]]:
    """Set up a synthetic MNIST MLP problem for the benchmark.

    Args:
        batch_size: The batch size to use. Default is ``512``.

    Returns:
        The neural net, loss function, and data.
    """
    X = rand(batch_size, 784)
    y = randint(0, 10, (batch_size,))
    data = [(X, y)]
    model = Sequential(
        Linear(784, 1024),
        ReLU(),
        Linear(1024, 512),
        ReLU(),
        Linear(512, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10),
    )
    loss_function = CrossEntropyLoss()

    return model, loss_function, data


# -- Problem/linop setup --


def setup_linop(
    linop_str: str,
    model: Module,
    loss_function: Module,
    params: dict[str, Tensor],
    data: Iterable[tuple[Tensor, Tensor]],
    check_deterministic: bool = True,
) -> PyTorchLinearOperator:
    """Set up the linear operator.

    Args:
        linop_str: The linear operator to set up.
        model: The neural net.
        loss_function: The loss function.
        params: The parameters.
        data: The data.
        check_deterministic: Whether to check for determinism. Default is ``True``.

    Returns:
        The linear operator.
    """
    num_data = sum(X.shape[0] for (X, _) in data)
    args = (model, loss_function, params, data)
    kwargs = {"check_deterministic": check_deterministic, "num_data": num_data}

    if linop_str == "Monte-Carlo Fisher":
        kwargs["mc_samples"] = 1

    linop_cls = {
        "Hessian": HessianLinearOperator,
        "Generalized Gauss-Newton": GGNLinearOperator,
        "Empirical Fisher": EFLinearOperator,
        "Monte-Carlo Fisher": GGNLinearOperator,
        "KFAC (hooks)": KFACLinearOperator,
        "KFAC inverse (hooks)": KFACLinearOperator,
        "KFAC (fx)": KFACLinearOperator,
        "KFAC inverse (fx)": KFACLinearOperator,
        "EKFAC (hooks)": EKFACLinearOperator,
        "EKFAC inverse (hooks)": EKFACLinearOperator,
        "EKFAC (fx)": EKFACLinearOperator,
        "EKFAC inverse (fx)": EKFACLinearOperator,
    }[linop_str]

    # Select backend for KFAC/EKFAC and pass num_per_example_loss_terms
    if "(fx)" in linop_str:
        kwargs["backend"] = "make_fx"
    elif "(hooks)" in linop_str:
        kwargs["backend"] = "hooks"
    if linop_str in _KFAC_LIKE:
        X0, y0 = next(iter(data))
        kwargs["num_per_example_loss_terms"] = y0.numel() // X0.shape[0]
        kwargs["separate_weight_and_bias"] = False

    with attention_context(linop_cls, model):
        linop = linop_cls(*args, **kwargs)

    is_inverse = "inverse" in linop_str
    if is_inverse:
        linop = linop.inverse(damping=1e-3)

    return linop
