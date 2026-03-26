"""Utility functions for benchmark setup and timing."""

import inspect
import json
from collections.abc import Callable
from os import path
from time import perf_counter
from typing import Any

import requests
from torch import Tensor, cuda, rand, randint, stack, zeros_like
from torch.nn import CrossEntropyLoss, Module, Parameter
from torchvision.models import ResNet50_Weights, resnet18, resnet50


class TimeBenchmark:
    """Utility for timing functions and saving results to JSON.

    Handles skip-if-exists, CUDA synchronization, multi-repeat timing,
    and JSON persistence.

    Args:
        num_repeats: Number of repeats per measurement. Uses the minimum.
        skip_existing: Whether to skip measurements whose result file exists.
    """

    def __init__(
        self,
        num_repeats: int = 10,
        skip_existing: bool = True,
        compile: bool = False,
    ):
        """Set up the benchmark timer.

        Args:
            num_repeats: Number of repeats per measurement. Uses the minimum.
            skip_existing: Whether to skip measurements whose result file exists.
            compile: Whether to wrap functions with ``torch.compile`` before
                timing. The first repeat serves as compilation warmup.
        """
        self.num_repeats = num_repeats
        self.skip_existing = skip_existing
        self.compile = compile

    def time(
        self,
        func: Callable,
        is_cuda: bool,
        num_repeats: int | None = None,
        compile: bool | None = None,
    ) -> tuple[float, Any]:
        """Time a function and return (min_time, last_result).

        If compilation is enabled, the function is wrapped with
        ``torch.compile`` before timing. The first repeat serves as
        compilation warmup and is included in the timing (so the minimum
        reflects compiled performance after warmup).

        Args:
            func: The function to time.
            is_cuda: Whether to synchronize CUDA before/after.
            num_repeats: Override for the number of repeats.
            compile: Override for whether to compile. If ``None``, uses
                ``self.compile``.

        Returns:
            Tuple of (minimum time across repeats, last return value).
        """
        import torch

        do_compile = compile if compile is not None else self.compile
        if do_compile:
            func = torch.compile(func)

        n = num_repeats if num_repeats is not None else self.num_repeats
        times = []
        for _ in range(n):
            if is_cuda:
                cuda.synchronize()
            start = perf_counter()
            result = func()
            if is_cuda:
                cuda.synchronize()
            times.append(perf_counter() - start)
        return min(times), result

    def run(
        self, save_path: str, label: str, func: Callable, is_cuda: bool
    ) -> float | None:
        """Skip-if-exists, time, save to JSON, and print.

        Saves ``{"time": best}`` to ``save_path``.

        Args:
            save_path: Path to the JSON result file.
            label: Description for printing.
            func: The function to time.
            is_cuda: Whether to synchronize CUDA.

        Returns:
            The best time, or ``None`` if skipped.
        """
        if self.skip_existing and path.exists(save_path):
            print(f"[Time] Skipping {label}")
            return None

        best, _ = self.time(func, is_cuda)
        print(f"[Time] {label}: {best:.4f} s")
        with open(save_path, "w") as f:
            json.dump({"time": best}, f)
        return best

    def run_phases(
        self,
        save_path: str,
        label: str,
        phase_fns: dict[str, Callable],
        is_cuda: bool,
        no_compile: set[str] | None = None,
    ) -> dict[str, float] | None:
        """Time multiple phases and save all to a single JSON file.

        Each phase is timed independently. Phases run in order, so later
        phases can depend on side effects of earlier ones (e.g. via shared
        mutable state).

        Args:
            save_path: Path to the JSON result file.
            label: Description for printing.
            phase_fns: Ordered dict mapping phase names to callables.
            is_cuda: Whether to synchronize CUDA.
            no_compile: Phase names to exclude from ``torch.compile``
                (e.g. phases that use ``make_fx`` internally).

        Returns:
            Dict of ``{phase_name: best_time}``, or ``None`` if skipped.
        """
        if self.skip_existing and path.exists(save_path):
            print(f"[Time] Skipping {label}")
            return None

        skip_compile = no_compile or set()
        results = {}
        for phase_name, func in phase_fns.items():
            phase_compile = None if phase_name not in skip_compile else False
            best, _ = self.time(func, is_cuda, compile=phase_compile)
            results[phase_name] = best
            print(f"[Time] {label} / {phase_name}: {best:.4f} s")

        with open(save_path, "w") as f:
            json.dump(results, f)
        print(f"[Time] Saved {label}")
        return results


# In the execution with sphinx-gallery, __file__ is not defined and we need
# to set it manually using the trick from https://stackoverflow.com/a/53293924
if "__file__" not in globals():
    __file__ = inspect.getfile(lambda: None)

HEREDIR = path.dirname(path.abspath(__file__))


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
