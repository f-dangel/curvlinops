"""Utility functions for benchmark setup, timing, and memory measurement."""

import inspect
import json
from collections.abc import Callable, Iterable
from os import makedirs, path
from subprocess import CalledProcessError, CompletedProcess, run
from time import perf_counter
from typing import Any

import requests
import torch
from torch import Tensor, cuda, rand, randint, stack, zeros_like
from torch.nn import CrossEntropyLoss, Linear, Module, Parameter, ReLU, Sequential
from torchvision.models import ResNet50_Weights, resnet18, resnet50

from curvlinops import KFACLinearOperator
from curvlinops.computers._base import _EKFACMixin
from curvlinops.computers.ekfac_hooks import HooksEKFACComputer
from curvlinops.computers.ekfac_make_fx import MakeFxEKFACComputer
from curvlinops.computers.kfac_hooks import HooksKFACComputer, _use_params
from curvlinops.computers.kfac_make_fx import MakeFxKFACComputer


class Benchmark:
    """Utility for timing and memory-profiling functions, with JSON persistence.

    Handles skip-if-exists, CUDA synchronization, multi-repeat timing,
    peak memory measurement, and JSON persistence.

    Args:
        is_cuda: Whether to synchronize CUDA before/after each measurement.
        num_repeats: Number of repeats per timing measurement. Uses the minimum.
        skip_existing: Whether to skip measurements whose result file exists.
    """

    def __init__(
        self,
        is_cuda: bool,
        num_repeats: int = 10,
        skip_existing: bool = True,
    ):
        """Set up the benchmark.

        Args:
            is_cuda: Whether to synchronize CUDA before/after each measurement.
            num_repeats: Number of repeats per timing measurement. Uses the minimum.
            skip_existing: Whether to skip measurements whose result file exists.
        """
        self.is_cuda = is_cuda
        self.num_repeats = num_repeats
        self.skip_existing = skip_existing

    def time(self, func: Callable) -> tuple[float, Any]:
        """Time a function and return (min_time, last_result).

        Args:
            func: The function to time.

        Returns:
            Tuple of (minimum time across repeats, last return value).
        """
        times = []
        for _ in range(self.num_repeats):
            if self.is_cuda:
                cuda.synchronize()
            start = perf_counter()
            result = func()
            if self.is_cuda:
                cuda.synchronize()
            times.append(perf_counter() - start)
        return min(times), result

    def memory(self, func: Callable) -> float:
        """Measure peak memory of a function call in GiB.

        On CUDA, uses :func:`torch.cuda.max_memory_allocated`.
        On CPU, uses the ``memory_profiler`` package.

        Args:
            func: The function to measure.

        Returns:
            Peak memory in GiB.
        """
        if self.is_cuda:
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
            func()
            cuda.synchronize()
            return cuda.max_memory_allocated() / 2**30
        else:
            from memory_profiler import memory_usage

            # memory_usage with max_usage=True returns peak MiB
            return memory_usage(func, interval=1e-4, max_usage=True) / 2**10

    def run(self, save_path: str, label: str, func: Callable) -> float | None:
        """Skip-if-exists, time, save to JSON, and print.

        Saves ``{"time": best}`` to ``save_path``.

        Args:
            save_path: Path to the JSON result file.
            label: Description for printing.
            func: The function to time.

        Returns:
            The best time, or ``None`` if skipped.
        """
        if self.skip_existing and path.exists(save_path):
            print(f"[Time] Skipping {label}")
            return None

        best, _ = self.time(func)
        print(f"[Time] {label}: {best:.4f} s")
        with open(save_path, "w") as f:
            json.dump({"time": best}, f)
        return best

    def run_phases(
        self,
        save_path: str,
        label: str,
        phase_fns: dict[str, Callable],
    ) -> dict[str, float] | None:
        """Time multiple phases and save all to a single JSON file.

        Each phase is timed independently. Phases run in order, so later
        phases can depend on side effects of earlier ones (e.g. via shared
        mutable state).

        Args:
            save_path: Path to the JSON result file.
            label: Description for printing.
            phase_fns: Ordered dict mapping phase names to callables.

        Returns:
            Dict of ``{phase_name: best_time}``, or ``None`` if skipped.
        """
        if self.skip_existing and path.exists(save_path):
            print(f"[Time] Skipping {label}")
            return None

        results = {}
        for phase_name, func in phase_fns.items():
            best, _ = self.time(func)
            results[phase_name] = best
            print(f"[Time] {label} / {phase_name}: {best:.4f} s")

        with open(save_path, "w") as f:
            json.dump(results, f)
        print(f"[Time] Saved {label}")
        return results

    def run_memory(self, save_path: str, label: str, func: Callable) -> float | None:
        """Skip-if-exists, measure peak memory, save to JSON, and print.

        Saves ``{"peakmem": gib}`` to ``save_path``.

        Args:
            save_path: Path to the JSON result file.
            label: Description for printing.
            func: The function to measure.

        Returns:
            Peak memory in GiB, or ``None`` if skipped.
        """
        if self.skip_existing and path.exists(save_path):
            print(f"[Memory] Skipping {label}")
            return None

        peakmem_gib = self.memory(func)
        print(f"[Memory] {label}: {peakmem_gib:.2f} GiB")
        with open(save_path, "w") as f:
            json.dump({"peakmem": peakmem_gib}, f)
        return peakmem_gib


# In the execution with sphinx-gallery, __file__ is not defined and we need
# to set it manually using the trick from https://stackoverflow.com/a/53293924
if "__file__" not in globals():
    __file__ = inspect.getfile(lambda: None)

HEREDIR = path.dirname(path.abspath(__file__))
RESULTDIR = path.join(HEREDIR, "benchmark")
makedirs(RESULTDIR, exist_ok=True)
REFERENCE_OP = "gradient_and_loss"

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

# Sub-phase operation names for precompute breakdown
EKFAC_PRECOMPUTE_OPS = ["kfac_factors", "eigenvalue_correction", "eigh"]
KFAC_INVERSE_PRECOMPUTE_OPS = ["kfac_factors", "cholesky_inverse"]
FX_PRECOMPUTE_OPS = ["kfac_factors", "tracing"]


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


def run_verbose(cmd: list[str]) -> CompletedProcess:
    """Run a command and print stdout & stderr if it fails.

    Args:
        cmd: The command to run.

    Returns:
        CompletedProcess: The result of the command.

    Raises:
        CalledProcessError: If the command fails.
    """
    try:
        job = run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:", job.stdout)
        print("STDERR:", job.stderr)
        return job
    except CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise e


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


def setup_computer(
    linop_str: str,
    model: Module,
    loss_function: Module,
    params: dict[str, Tensor],
    data: Iterable[tuple[Tensor, Tensor]],
):
    """Set up the KFAC/EKFAC computer for sub-phase benchmarking.

    Args:
        linop_str: The linear operator.
        model: The neural net.
        loss_function: The loss function.
        params: The parameters.
        data: The data.

    Returns:
        The computer instance.
    """
    num_data = sum(X.shape[0] for (X, _) in data)
    X0, y0 = next(iter(data))
    num_per_example_loss_terms = y0.numel() // X0.shape[0]
    kwargs = dict(
        check_deterministic=False,
        num_data=num_data,
        num_per_example_loss_terms=num_per_example_loss_terms,
        separate_weight_and_bias=False,
    )
    computer_cls = {
        "EKFAC (hooks)": HooksEKFACComputer,
        "EKFAC inverse (hooks)": HooksEKFACComputer,
        "EKFAC (fx)": MakeFxEKFACComputer,
        "EKFAC inverse (fx)": MakeFxEKFACComputer,
        "KFAC (hooks)": HooksKFACComputer,
        "KFAC inverse (hooks)": HooksKFACComputer,
        "KFAC (fx)": MakeFxKFACComputer,
        "KFAC inverse (fx)": MakeFxKFACComputer,
    }[linop_str]
    return computer_cls(model, loss_function, params, data, **kwargs)


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


def make_precompute_phases(  # noqa: C901, PLR0915
    linop_str: str,
    model: Module,
    loss_function: Module,
    params: dict[str, Tensor],
    data,
) -> dict[str, callable]:
    """Build an ordered dict of phase_name -> callable for precompute sub-phases.

    Phases run in order. Later phases can depend on results of earlier ones
    via the shared ``state`` dict captured by closures.

    Args:
        linop_str: The linear operator name.
        model: The neural net.
        loss_function: The loss function.
        params: The parameters.
        data: The data.

    Returns:
        Ordered dict mapping sub-phase names to timing callables.
    """
    num_data = sum(X.shape[0] for (X, _) in data)
    X0, y0 = next(iter(data))
    num_per_example_loss_terms = y0.numel() // X0.shape[0]
    common_kwargs = dict(
        check_deterministic=False,
        num_data=num_data,
        num_per_example_loss_terms=num_per_example_loss_terms,
        separate_weight_and_bias=False,
    )
    state = {}  # shared mutable state between phases
    phases = {}

    if linop_str in _IS_EKFAC and linop_str not in _IS_FX:
        # EKFAC hooks: factors → correction → eigh
        computer = setup_computer(linop_str, model, loss_function, params, data)

        def ekfac_factors():
            with _use_params(computer._model_module, computer._params):
                state["ic"], state["gc"], state["m"] = (
                    computer._compute_kronecker_factors()
                )

        def ekfac_correction():
            with _use_params(computer._model_module, computer._params):
                computer.compute_eigenvalue_correction(
                    state["ic"], state["gc"], state["m"]
                )

        def ekfac_eigh():
            ic = {k: v.clone() for k, v in state["ic"].items()}
            gc = {k: v.clone() for k, v in state["gc"].items()}
            state["ic"] = _EKFACMixin._eigenvectors_(ic)
            state["gc"] = _EKFACMixin._eigenvectors_(gc)

        phases["kfac_factors"] = ekfac_factors
        phases["eigenvalue_correction"] = ekfac_correction
        phases["eigh"] = ekfac_eigh

    elif linop_str in _IS_KFAC_INVERSE_HOOKS:
        # KFAC inverse hooks: factors → Cholesky inverse
        def kfac_inv_factors():
            state["linop"] = KFACLinearOperator(
                model, loss_function, params, data, **common_kwargs
            )

        def cholesky_inverse():
            state["linop"].inverse(damping=1e-3)

        phases["kfac_factors"] = kfac_inv_factors
        phases["cholesky_inverse"] = cholesky_inverse

    elif linop_str in _IS_FX and linop_str in _IS_EKFAC:
        # EKFAC FX: tracing → factors → correction → eigh
        computer = setup_computer(linop_str, model, loss_function, params, data)

        def ekfac_fx_tracing():
            state["traced_io"] = computer._trace_io_functions()

        def ekfac_fx_factors():
            state["ic"], state["gc"], state["m"] = computer._compute_kronecker_factors(
                state["traced_io"]
            )

        def ekfac_fx_correction():
            computer.compute_eigenvalue_correction(
                state["ic"], state["gc"], state["m"], state["traced_io"]
            )

        def ekfac_fx_eigh():
            ic = {k: v.clone() for k, v in state["ic"].items()}
            gc = {k: v.clone() for k, v in state["gc"].items()}
            state["ic"] = _EKFACMixin._eigenvectors_(ic)
            state["gc"] = _EKFACMixin._eigenvectors_(gc)

        phases["tracing"] = ekfac_fx_tracing
        phases["kfac_factors"] = ekfac_fx_factors
        phases["eigenvalue_correction"] = ekfac_fx_correction
        phases["eigh"] = ekfac_fx_eigh

    elif linop_str in _IS_FX:
        # KFAC FX: tracing → factors
        computer = setup_computer(linop_str, model, loss_function, params, data)

        def kfac_fx_tracing():
            state["traced_io"] = computer._trace_io_functions()

        def kfac_fx_factors():
            computer._compute_kronecker_factors(state["traced_io"])

        phases["tracing"] = kfac_fx_tracing
        phases["kfac_factors"] = kfac_fx_factors

    else:
        # Plain KFAC hooks: single phase
        def kfac_factors():
            KFACLinearOperator(model, loss_function, params, data, **common_kwargs)

        phases["kfac_factors"] = kfac_factors

    return phases
