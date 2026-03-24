r"""Benchmarking linear operators
=============================

In this tutorial, we demonstrate how to evaluate the run time and memory performance
of linear operators. This allows to get a feeling for how expensive each operator is,
compared to a gradient computation.

.. warning::
    For pedagogical reasons, this example considers a small synthetic problem which may
    not reflect the relative cost of linear operators on larger problems. However, the
    following example can easily be applied to larger problems that are not executed
    when building the documentation.

Let's get the imports out of the way.
"""

import inspect
import json
import sys
from collections.abc import Iterable
from contextlib import nullcontext
from itertools import product
from math import floor
from os import environ, makedirs, path
from shutil import which
from subprocess import CalledProcessError, CompletedProcess, run
from time import perf_counter

import matplotlib.pyplot as plt
from benchmark_utils import (
    GPTWrapper,
    setup_synthetic_cifar10_resnet18,
    setup_synthetic_imagenet_resnet50,
    setup_synthetic_shakespeare_nanogpt,
)
from torch import Tensor, arange, cuda, device, manual_seed, rand, randint
from torch.nn import (
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
from tueplots import bundles

from curvlinops import (
    EFLinearOperator,
    EKFACLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
)
from curvlinops._torch_base import PyTorchLinearOperator
from curvlinops.computers._base import _EKFACMixin
from curvlinops.computers.ekfac_hooks import HooksEKFACComputer
from curvlinops.computers.ekfac_make_fx import MakeFxEKFACComputer
from curvlinops.computers.kfac_hooks import HooksKFACComputer
from curvlinops.computers.kfac_make_fx import MakeFxKFACComputer
from curvlinops.examples import gradient_and_loss

# %%
#
# Let's also set up some variables that will be useful to generate and store results.

# In the execution with sphinx-gallery, __file__ is not defined and we need
# to set it manually using the trick from https://stackoverflow.com/a/53293924
if "__file__" not in globals():
    __file__ = inspect.getfile(lambda: None)

# Define paths where results are stored and paths we must parse for in the
# output of PyTorch's profiler.
SCRIPTNAME = path.basename(__file__)
HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
RESULTDIR = path.join(HEREDIR, "benchmark")
makedirs(RESULTDIR, exist_ok=True)

# Linear operators that use JVPs need to handle attention differently because PyTorch's
# efficient attention does not implement double-backward yet. See
# https://github.com/pytorch/pytorch/issues/116350
HAS_JVP = (
    HessianLinearOperator,
    GGNLinearOperator,
    EFLinearOperator,
)

# When running on RTD, we only want to execute the small example
ON_RTD = environ.get("READTHEDOCS", "False") == "True"
# Use LaTeX if available
USETEX = which("latex") is not None

# Devices to run the benchmark on
DEVICE_STRS = ["cuda"] if cuda.is_available() else ["cpu"]

# Whether to skip runs for which measurements already exists
SKIP_EXISTING = True


def save_environment_info():
    """Save PyTorch version and GPU info to a metadata file in the results directory."""
    import torch

    info = {"pytorch_version": torch.__version__}
    if cuda.is_available():
        info["gpu"] = cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda

    info_path = path.join(RESULTDIR, "environment.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    for key, value in info.items():
        print(f"  {key}: {value}")


save_environment_info()

# %%
#
# Setup
# -----
#
# Before we can write our benchmarking code, we need to define a couple of setup
# functions, which will make it easier to add new problems to this benchmark.
#
# First, we define a setup function that generates the neural network, loss
# function, data and parameters from which the linear operator is later
# created:

# Supported problems
if ON_RTD:
    PROBLEM_STRS = ["synthetic_mnist_cnn"]
else:
    PROBLEM_STRS = [
        "synthetic_mnist_cnn",
        "synthetic_cifar10_resnet18",
        "synthetic_imagenet_resnet50",
        "synthetic_shakespeare_nanogpt",
    ]


def setup_synthetic_mnist_cnn(
    batch_size: int = 512,
) -> tuple[Sequential, CrossEntropyLoss, list[tuple[Tensor, Tensor]]]:
    """Set up a synthetic MNIST CNN problem for the benchmark.

    Args:
        batch_size: The batch size to use. Default is ``512``.

    Returns:
        The neural net, loss function, and data.
    """
    X = rand(batch_size, 1, 28, 28)
    y = randint(0, 10, (batch_size,))
    data = [(X, y)]
    model = Sequential(
        Conv2d(1, 16, 3, padding=1),
        MaxPool2d(2),
        ReLU(),
        Conv2d(16, 16, 3, padding=1),
        MaxPool2d(2),
        ReLU(),
        Conv2d(16, 16, 3, padding=1),
        MaxPool2d(2),
        ReLU(),
        Flatten(),
        Linear(144, 64),
        ReLU(),
        Linear(64, 10),
    )
    loss_function = CrossEntropyLoss()

    return model, loss_function, data


def setup_problem(
    problem_str: str, linop_str: str, dev: device
) -> tuple[Module, Module, dict[str, Tensor], Iterable[tuple[Tensor, Tensor]]]:
    """Set up the neural net, loss function, parameters, and data.

    Args:
        problem_str: The problem to set up.
        linop_str: The linear operator that is investigated.
        dev: The device to use.

    Returns:
        The neural net, loss function, parameters, and data.
    """
    setup_func = {
        "synthetic_mnist_cnn": setup_synthetic_mnist_cnn,
        "synthetic_cifar10_resnet18": setup_synthetic_cifar10_resnet18,
        "synthetic_imagenet_resnet50": setup_synthetic_imagenet_resnet50,
        "synthetic_shakespeare_nanogpt": setup_synthetic_shakespeare_nanogpt,
    }[problem_str]
    model, loss_function, data = setup_func()

    # Put model in evaluation mode so curvature matrices are well-defined
    # even on data sets
    model = model.eval().to(dev)
    loss_function = loss_function.to(dev)

    # Only use parameters of supported layers for KFAC
    if linop_str in _KFAC_LIKE:
        params = {}
        for mod_name, mod in model.named_modules():
            if not isinstance(mod, (Linear, Conv2d)):
                continue
            # ignore the last layer of GPT because it has 50k outputs, which
            # will yield an extremely large Kronecker factor
            if all(d <= 50_000 for d in mod.weight.shape):
                for p_name, p in mod.named_parameters(recurse=False):
                    full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                    if p.requires_grad:
                        params[full_name] = p
    else:
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}

    return model, loss_function, params, data


# %%
#
# We will compare the following linear operators:

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

# Names that show precompute + matvec bars in visualization
_HAS_PRECOMPUTE = _KFAC_LIKE

# Names that are EKFAC (precompute is split into factors + eigh + correction)
_IS_EKFAC = {
    "EKFAC (hooks)",
    "EKFAC inverse (hooks)",
    "EKFAC (fx)",
    "EKFAC inverse (fx)",
}

# Names that are KFAC inverse (precompute includes inverse computation)
_IS_INVERSE = {
    "KFAC inverse (hooks)",
    "KFAC inverse (fx)",
    "EKFAC inverse (hooks)",
    "EKFAC inverse (fx)",
}

# %%
#
# And we are interested in the following sub-routines:

# Operations we are interested in (per linop)
OP_STRS = ["precompute", "matvec"]

# EKFAC precompute is split into sub-phases for detailed analysis
# Order: factors first, then correction, then decomposition (eigh dominates, shown last)
EKFAC_PRECOMPUTE_OPS = ["kfac_factors", "eigenvalue_correction", "eigh"]

# KFAC inverse precompute is split into factors + Cholesky inverse
KFAC_INVERSE_PRECOMPUTE_OPS = ["kfac_factors", "cholesky_inverse"]

# FX backend precompute is split into tracing + factor computation
FX_PRECOMPUTE_OPS = ["kfac_factors", "tracing"]

# Names that are KFAC inverse hooks (precompute split into factors + Cholesky)
_IS_KFAC_INVERSE_HOOKS = {"KFAC inverse (hooks)"}

# Names that use the FX backend
_IS_FX = {
    "KFAC (fx)",
    "KFAC inverse (fx)",
    "EKFAC (fx)",
    "EKFAC inverse (fx)",
}

# The gradient_and_loss reference is measured once per problem (not per linop)
REFERENCE_OP = "gradient_and_loss"

# %%
#
# The next setup function creates linear operators:


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

    # Double-backward through efficient attention is unsupported, disable fused kernels
    # (https://github.com/pytorch/pytorch/issues/116350#issuecomment-1954667011)
    attention_double_backward = linop_cls in HAS_JVP and isinstance(model, GPTWrapper)
    with sdpa_kernel(SDPBackend.MATH) if attention_double_backward else nullcontext():
        linop = linop_cls(*args, **kwargs)

    is_inverse = "inverse" in linop_str
    if is_inverse:
        linop = linop.inverse(damping=1e-3)

    return linop


# %%
#
# For EKFAC operators, we also want to measure the sub-phases of precomputation
# (Kronecker factors, eigendecomposition, eigenvalue correction) separately.
# To do so, we construct the computer object directly.


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


# %%
#
# Last, we define a convenience function that generates the output files where
# the benchmark results will be stored and later plotted.


def benchpath(
    linop_str: str,
    problem_str,
    device_str: str,
    op_str: str,
    metric: str = "time",
) -> str:
    """Get the path to save the benchmark results.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        op_str: Operation name that is benchmarked.
        metric: The metric to save. Default is ``'time'``.

    Returns:
        The path to save the benchmark results.
    """
    return path.join(
        RESULTDIR,
        f"{metric}_{linop_str.replace(' ', '-')}_{problem_str}_{device_str}"
        + f"_{op_str}.json",
    )


def reference_benchpath(
    problem_str: str, device_str: str, metric: str = "time"
) -> str:
    """Get the path to save the reference gradient_and_loss benchmark.

    This is measured once per problem (not per linop).

    Args:
        problem_str: The problem.
        device_str: The device.
        metric: The metric to save. Default is ``'time'``.

    Returns:
        The path to save the reference benchmark results.
    """
    return path.join(
        RESULTDIR,
        f"{metric}_{REFERENCE_OP}_{problem_str}_{device_str}.json",
    )


# %%
#
# Run time benchmark
# ------------------
#
# We split the time benchmark into two parts:
#
# 1. **Matvec benchmark**: measures the time of a single matrix-vector product for
#    all linear operators. This includes KFAC/EKFAC variants whose matvec is cheap
#    after pre-computation.
#
# 2. **Precompute benchmark**: measures the pre-computation cost for KFAC/EKFAC
#    operators, broken down into sub-phases (Kronecker factors, eigendecomposition,
#    eigenvalue correction, Cholesky inverse, FX tracing).


def _time_function(func, is_cuda: bool, num_repeats: int) -> float:
    """Time a function and return the best (minimum) time.

    Args:
        func: The function to time.
        is_cuda: Whether to synchronize CUDA before/after.
        num_repeats: Number of repeats.

    Returns:
        Tuple of (minimum time, last result).
    """
    times = []
    for _ in range(num_repeats):
        if is_cuda:
            cuda.synchronize()
        start = perf_counter()
        result = func()
        if is_cuda:
            cuda.synchronize()
        times.append(perf_counter() - start)
    return min(times), result


def _save_and_print(savepath: str, metric_name: str, key: str, value, label: str):
    """Save a benchmark result to JSON and print it.

    Args:
        savepath: Path to the JSON file.
        metric_name: Name for printing (e.g. 'Time', 'Memory').
        key: JSON key (e.g. 'time', 'peakmem').
        value: The measured value.
        label: Description for the print message.
    """
    print(f"[{metric_name}] {label}:\n\tBest: {value:.4f}")
    with open(savepath, "w") as f:
        json.dump({key: value}, f)


def _skip_if_exists(savepath: str, label: str) -> bool:
    """Check if a result file exists and print skip message if so.

    Args:
        savepath: Path to check.
        label: Description for the skip message.

    Returns:
        True if the file exists and should be skipped.
    """
    if SKIP_EXISTING and path.exists(savepath):
        print(f"[Time] Skipping {label}")
        return True
    return False


# %%
#
# Reference baseline
# ^^^^^^^^^^^^^^^^^^
#
# The gradient_and_loss reference is measured once per problem (not per linop).


def run_reference_benchmark(
    problem_str: str, device_str: str, num_repeats: int = 1
):
    """Measure gradient_and_loss once per problem as a reference baseline.

    Uses all model parameters (not a KFAC subset) so the reference is
    linop-independent.

    Args:
        problem_str: The problem.
        device_str: The device.
        num_repeats: Number of repeats. Uses the minimum time.
    """
    savepath = reference_benchpath(problem_str, device_str)
    if _skip_if_exists(savepath, f"reference on {problem_str} and {device_str}"):
        return

    manual_seed(0)
    dev = device(device_str)
    is_cuda = "cuda" in str(dev)
    model, loss_function, params, data = setup_problem(problem_str, "Hessian", dev)

    def func():
        _ = gradient_and_loss(model, loss_function, params, data)

    best, _ = _time_function(func, is_cuda, num_repeats)
    _save_and_print(
        savepath, "Time", "time", best,
        f"Reference gradient_and_loss on {problem_str} and {device_str}",
    )


# %%
#
# Matvec benchmark
# ^^^^^^^^^^^^^^^^
#
# Measures the time of a single matrix-vector product for each linear operator.


def run_matvec_benchmark(
    linop_str: str, problem_str: str, device_str: str, num_repeats: int = 1
):
    """Benchmark the matvec time for a linear operator.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        num_repeats: Number of repeats.
    """
    savepath = benchpath(linop_str, problem_str, device_str, "matvec")
    label = f"{linop_str}'s matvec on {problem_str} and {device_str}"
    if _skip_if_exists(savepath, label):
        return

    manual_seed(0)
    dev = device(device_str)
    is_cuda = "cuda" in str(dev)
    model, loss_function, params, data = setup_problem(problem_str, linop_str, dev)

    linop = setup_linop(
        linop_str, model, loss_function, params, data, check_deterministic=False
    )
    v = rand(linop.shape[1], device=dev)

    def func():
        attention_double_backward = isinstance(linop, HAS_JVP) and isinstance(
            model, GPTWrapper
        )
        with (
            sdpa_kernel(SDPBackend.MATH) if attention_double_backward else nullcontext()
        ):
            _ = linop @ v

    best, _ = _time_function(func, is_cuda, num_repeats)
    _save_and_print(savepath, "Time", "time", best, label)


# %%
#
# Precompute benchmark
# ^^^^^^^^^^^^^^^^^^^^
#
# Measures the pre-computation cost for KFAC/EKFAC operators, broken down into
# sub-phases. Each operator type has different sub-phases:
#
# - **KFAC (hooks)**: Kronecker factors
# - **KFAC inverse (hooks)**: Kronecker factors + Cholesky inverse
# - **EKFAC (hooks)**: Kronecker factors + eigenvalue correction + eigendecomposition
# - **EKFAC inverse (hooks)**: same as EKFAC (inverse is trivial)
# - **KFAC (fx)**: FX tracing + Kronecker factors
# - **KFAC inverse (fx)**: FX tracing + Kronecker factors (+ Cholesky, not yet split)
# - **EKFAC (fx)**: FX tracing + Kronecker factors + correction + decomposition


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


def run_precompute_benchmark(  # noqa: C901
    linop_str: str, problem_str: str, device_str: str, num_repeats: int = 1
):
    """Benchmark precomputation sub-phases for a KFAC/EKFAC operator.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        num_repeats: Number of repeats per sub-phase.
    """
    all_ops = _get_precompute_ops(linop_str)
    ops_to_run = []
    for op_str in all_ops:
        savepath = benchpath(linop_str, problem_str, device_str, op_str)
        label = f"{linop_str} on {problem_str} and {device_str} for {op_str}"
        if _skip_if_exists(savepath, label):
            continue
        ops_to_run.append(op_str)
    if not ops_to_run:
        return

    manual_seed(0)
    dev = device(device_str)
    is_cuda = "cuda" in str(dev)
    model, loss_function, params, data = setup_problem(problem_str, linop_str, dev)

    num_data = sum(X.shape[0] for (X, _) in data)
    X0, y0 = next(iter(data))
    num_per_example_loss_terms = y0.numel() // X0.shape[0]
    common_kwargs = dict(
        check_deterministic=False,
        num_data=num_data,
        num_per_example_loss_terms=num_per_example_loss_terms,
        separate_weight_and_bias=False,
    )

    for op_str in ops_to_run:
        if op_str == "kfac_factors" and linop_str not in _IS_FX:
            if linop_str in _IS_EKFAC:
                # Use computer to measure just factor computation
                computer = setup_computer(
                    linop_str, model, loss_function, params, data
                )

                def f_factors():
                    with computer._computation_context():
                        return computer._compute_kronecker_factors()

                best, (input_cov, grad_cov, mapping) = _time_function(
                    f_factors, is_cuda, num_repeats
                )
            else:
                # KFAC or KFAC inverse: measure full KFAC construction
                def f_kfac():
                    return KFACLinearOperator(
                        model, loss_function, params, data, **common_kwargs
                    )

                best, kfac_linop = _time_function(f_kfac, is_cuda, num_repeats)

        elif op_str == "eigh":
            # Ensure factors are available
            if "input_cov" not in dir():
                computer = setup_computer(
                    linop_str, model, loss_function, params, data
                )
                with computer._computation_context():
                    input_cov, grad_cov, mapping = (
                        computer._compute_kronecker_factors()
                    )
            input_cov_copy = {k: v.clone() for k, v in input_cov.items()}
            grad_cov_copy = {k: v.clone() for k, v in grad_cov.items()}

            def f_eigh():
                ic = {k: v.clone() for k, v in input_cov_copy.items()}
                gc = {k: v.clone() for k, v in grad_cov_copy.items()}
                return _EKFACMixin._eigenvectors_(ic), _EKFACMixin._eigenvectors_(gc)

            best, (input_cov, grad_cov) = _time_function(
                f_eigh, is_cuda, num_repeats
            )

        elif op_str == "eigenvalue_correction":
            # Ensure eigenvectors are available
            if "mapping" not in dir():
                computer = setup_computer(
                    linop_str, model, loss_function, params, data
                )
                with computer._computation_context():
                    input_cov, grad_cov, mapping = (
                        computer._compute_kronecker_factors()
                    )
                input_cov = _EKFACMixin._eigenvectors_(input_cov)
                grad_cov = _EKFACMixin._eigenvectors_(grad_cov)

            def f_correction():
                with computer._computation_context():
                    return computer.compute_eigenvalue_correction(
                        input_cov, grad_cov, mapping
                    )

            best, _ = _time_function(f_correction, is_cuda, num_repeats)

        elif op_str == "cholesky_inverse":
            # Ensure we have a KFAC linop
            if "kfac_linop" not in dir():
                kfac_linop = KFACLinearOperator(
                    model, loss_function, params, data, **common_kwargs
                )

            def f_inverse():
                return kfac_linop.inverse(damping=1e-3)

            best, _ = _time_function(f_inverse, is_cuda, num_repeats)

        elif op_str == "tracing":
            from curvlinops.computers.io_collector import with_kfac_io
            from curvlinops.kfac_utils import FisherType
            from curvlinops.utils import make_functional_call

            model_func = make_functional_call(model)
            X_example = next(iter(data))[0].to(dev)

            def f_tracing():
                return with_kfac_io(model_func, X_example, params, FisherType.MC)

            best, _ = _time_function(f_tracing, is_cuda, num_repeats)

        elif op_str == "kfac_factors" and linop_str in _IS_FX:
            # FX factors = total precompute - tracing
            tracing_path = benchpath(linop_str, problem_str, device_str, "tracing")
            precompute_path = benchpath(
                linop_str, problem_str, device_str, "precompute"
            )
            if path.exists(tracing_path) and path.exists(precompute_path):
                with open(tracing_path) as f:
                    tracing_time = json.load(f)["time"]
                with open(precompute_path) as f:
                    precompute_time = json.load(f)["time"]
                best = max(precompute_time - tracing_time, 0.0)
            else:
                # Need precompute measurement first; run it
                run_matvec_benchmark(linop_str, problem_str, device_str, num_repeats)
                continue

        label = f"{linop_str}'s {op_str} on {problem_str} and {device_str}"
        savepath = benchpath(linop_str, problem_str, device_str, op_str)
        _save_and_print(savepath, "Time", "time", best, label)


# %%
#
# We also need a total precompute measurement for operators where sub-phases are
# measured (so FX kfac_factors can be derived as total - tracing).


def run_total_precompute_benchmark(
    linop_str: str, problem_str: str, device_str: str, num_repeats: int = 1
):
    """Benchmark total precompute time (used by FX factors derivation).

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        num_repeats: Number of repeats.
    """
    savepath = benchpath(linop_str, problem_str, device_str, "precompute")
    label = f"{linop_str}'s precompute on {problem_str} and {device_str}"
    if _skip_if_exists(savepath, label):
        return

    manual_seed(0)
    dev = device(device_str)
    is_cuda = "cuda" in str(dev)
    model, loss_function, params, data = setup_problem(problem_str, linop_str, dev)

    def func():
        _ = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )

    best, _ = _time_function(func, is_cuda, num_repeats)
    _save_and_print(savepath, "Time", "time", best, label)


# %%
#
# Main benchmark execution
# ^^^^^^^^^^^^^^^^^^^^^^^^

if __name__ == "__main__":
    # Reference baselines (once per problem)
    for device_str, problem_str in product(DEVICE_STRS, PROBLEM_STRS):
        run_reference_benchmark(problem_str, device_str, num_repeats=10)

    # Matvec benchmark (all linops)
    for device_str, problem_str, linop_str in product(
        DEVICE_STRS, PROBLEM_STRS, MATVEC_LINOP_STRS
    ):
        run_matvec_benchmark(linop_str, problem_str, device_str, num_repeats=10)

    # Total precompute (needed for FX factors derivation)
    for device_str, problem_str, linop_str in product(
        DEVICE_STRS, PROBLEM_STRS, [l for l in LINOP_STRS if l in _KFAC_LIKE]
    ):
        run_total_precompute_benchmark(
            linop_str, problem_str, device_str, num_repeats=10
        )

    # Precompute sub-phase breakdown (KFAC-like only)
    for device_str, problem_str, linop_str in product(
        DEVICE_STRS, PROBLEM_STRS, [l for l in LINOP_STRS if l in _KFAC_LIKE]
    ):
        run_precompute_benchmark(
            linop_str, problem_str, device_str, num_repeats=10
        )

# %%
#
# Visualization
# ^^^^^^^^^^^^^


def _add_gradient_reference_axis(ax, reference: float):
    """Add a dashed reference line and a top axis showing multiples of gradient time.

    Args:
        ax: The matplotlib axes.
        reference: The gradient computation time in seconds.
    """
    ax.axvline(reference, color="black", linestyle="--")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Relative to gradient computation")
    _, x_max = ax.get_xlim()
    num_gradients = x_max / reference
    spacing = 1 / 4
    num_ticks = 1 + floor(num_gradients / spacing)
    while num_ticks > 8:
        spacing *= 2
        num_ticks = 1 + floor(num_gradients / spacing)
    ax2.set_xticks(arange(0, num_ticks) * spacing * reference)
    ax2.set_xticklabels(arange(0, num_ticks * spacing, spacing).tolist())


def visualize_matvec_benchmark(
    linop_strs: list[str], problem_str: str, device_str: str
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize matvec times for all operators.

    Args:
        linop_strs: The linear operators.
        problem_str: The problem.
        device_str: The device.

    Returns:
        The figure and axes.
    """
    fig, ax = plt.subplots()

    for idx, name in enumerate(linop_strs):
        with open(benchpath(name, problem_str, device_str, "matvec"), "r") as f:
            matvec_time = json.load(f)["time"]
        ax.barh(idx, width=matvec_time, color="tab:blue")

    ax.set_yticks(list(range(len(linop_strs))))
    # Strip backend suffix — matvec is backend-independent
    ax.set_yticklabels([n.replace(" (hooks)", "") for n in linop_strs])
    ax.set_xlabel("Time [s]")

    with open(reference_benchpath(problem_str, device_str), "r") as f:
        reference = json.load(f)["time"]
    _add_gradient_reference_axis(ax, reference)

    return fig, ax


def visualize_precompute_benchmark(
    linop_strs: list[str], problem_str: str, device_str: str
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize precompute sub-phase breakdown for KFAC/EKFAC operators.

    Args:
        linop_strs: The KFAC/EKFAC linear operators to plot.
        problem_str: The problem.
        device_str: The device.

    Returns:
        The figure and axes.
    """
    kfac = [l for l in linop_strs if l in _KFAC_LIKE]
    fig, ax = plt.subplots()

    precompute_colors = {
        "kfac_factors": "tab:green",
        "eigenvalue_correction": "tab:red",
        "eigh": "tab:orange",
        "cholesky_inverse": "tab:purple",
        "tracing": "tab:brown",
    }
    precompute_labels = {
        "kfac_factors": "Kronecker factors",
        "eigenvalue_correction": "Eigen-correction",
        "eigh": "Eigen-decomposition",
        "cholesky_inverse": "Cholesky inverse",
        "tracing": "FX tracing",
    }
    labels_shown = set()

    for idx, name in enumerate(kfac):
        sub_ops = _get_precompute_ops(name)

        left = 0.0
        for op in sub_ops:
            # For plain KFAC hooks, the file is "precompute"
            is_plain_kfac_hooks = (
                name not in _IS_EKFAC
                and name not in _IS_FX
                and name not in _IS_KFAC_INVERSE_HOOKS
            )
            file_op = (
                "precompute"
                if op == "kfac_factors" and is_plain_kfac_hooks
                else op
            )
            fpath = benchpath(name, problem_str, device_str, file_op)
            if not path.exists(fpath):
                continue
            with open(fpath, "r") as f:
                t = json.load(f)["time"]
            label = precompute_labels[op] if op not in labels_shown else None
            ax.barh(
                idx, width=t, left=left,
                color=precompute_colors[op], label=label, height=0.6,
            )
            labels_shown.add(op)
            left += t

    ax.set_yticks(list(range(len(kfac))))
    ax.set_yticklabels(kfac)
    ax.set_xlabel("Time [s]")

    with open(reference_benchpath(problem_str, device_str), "r") as f:
        reference = json.load(f)["time"]
    _add_gradient_reference_axis(ax, reference)

    ax.legend()
    return fig, ax


# %%
#
# Figure paths and plotting


def figpath(problem_str: str, device_str: str, metric: str = "time") -> str:
    """Get the path to save the figure.

    Args:
        problem_str: The problem.
        device_str: The device.
        metric: The metric to save. Default is ``'time'``.

    Returns:
        The path to save the figure.
    """
    return path.join(RESULTDIR, f"{metric}_{problem_str}_{device_str}.pdf")


if __name__ == "__main__":
    plot_config = bundles.icml2024(column="full" if ON_RTD else "half", usetex=USETEX)

    for problem_str, device_str in product(PROBLEM_STRS, DEVICE_STRS):
        with plt.rc_context(plot_config):
            fig, ax = visualize_matvec_benchmark(
                MATVEC_LINOP_STRS, problem_str, device_str
            )
            plt.savefig(
                figpath(problem_str, device_str, metric="time_matvec"),
                bbox_inches="tight",
            )
            plt.close()

        kfac_linops = [l for l in LINOP_STRS if l in _KFAC_LIKE]
        with plt.rc_context(plot_config):
            fig, ax = visualize_precompute_benchmark(
                kfac_linops, problem_str, device_str
            )
            plt.savefig(
                figpath(problem_str, device_str, metric="time_precompute"),
                bbox_inches="tight",
            )
            plt.close()

# %%
#
# As hinted at in the introduction, the numbers we observe in this pedagogical example
# may not reflect the relative cost of linear operators on larger problems and GPUs.
# However, we should see a rough tendency that Hessian-vector products are more costly
# than GGN-vector products, and that KFAC costs only a few gradients to pre-compute,
# while being very cheap to multiply with. Also, inverting KFAC adds some additional
# run time.
#
# Memory benchmark
# ================
#
# Measuring the memory consumption of some routines comes with some additional
# challenges. We use the
# `memory_profiler <https://github.com/pythonprofilers/memory_profiler>`_
# library on CPU, whereas we rely on :func:`torch.cuda.max_memory_allocated` on GPU.
#
# To avoid memory allocations from previous operations to impact the currently
# benchmarked function, we run each benchmark in a separate Python session by executing
# a separate script (`memory_benchmark.py`). This script
# re-uses most of the functionality developed in this tutorial, and the function that
# is profiled looks very similar to the one we used for the run time benchmark.
# Also, since memory consumption is more deterministic, we don't have to repeat each
# measurement.
#
# Collection
# ^^^^^^^^^^
#
# To generate memory measurements, let's execute the memory benchmarking script for
# each linear operator and operation we are interested in.
#
# We define the following helper function which simplifies inspecting the output of
# the call to ``memory_benchmark.py``, and troubleshoot if the call fails:


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


# %%
#
# Let's run the benchmark:

# Release all allocated GPU memory so it can be used by the memory benchmark script
if cuda.is_available():
    cuda.empty_cache()

if __name__ == "__main__":
    # measure memory consumption in individual Python sessions to avoid memory
    # allocations from previous operations into the currently benchmarked operation.

    # Reference gradient_and_loss (once per problem)
    for device_str, problem_str in product(DEVICE_STRS, PROBLEM_STRS):
        cmd = [
            sys.executable,
            path.join(path.dirname(__file__), "memory_benchmark.py"),
            f"--problem={problem_str}",
            f"--device={device_str}",
            "--reference",
        ]
        print(f"Running command: {' '.join(cmd)}")
        run_verbose(cmd)

    # Per-linop measurements (precompute and matvec only)
    for device_str, problem_str, linop_str, op_str in product(
        DEVICE_STRS, PROBLEM_STRS, LINOP_STRS, OP_STRS
    ):
        cmd = [
            sys.executable,
            path.join(path.dirname(__file__), "memory_benchmark.py"),
            f"--linop={linop_str}",
            f"--problem={problem_str}",
            f"--device={device_str}",
            f"--op={op_str}",
        ]
        print(f"Running command: {' '.join(cmd)}")
        run_verbose(cmd)

# %%
#
# Visualization
# ^^^^^^^^^^^^^


def visualize_peakmem_benchmark(
    linop_strs: list[str], problem_str: str, device_str: str
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize the peak memory benchmark results.

    Args:
        linop_strs: The linear operators.
        problem_str: The problem.
        device_str: The device.

    Returns:
        The figure and axes of the plot.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("Peak memory [GiB]")

    # Visualize the peak memory consumption of each linear operator's matvec
    for name in linop_strs:
        with open(
            benchpath(name, problem_str, device_str, "matvec", metric="peakmem"), "r"
        ) as f:
            mem = json.load(f)["peakmem"]
        ax.barh(name, mem, color="blue")

    # Get memory consumption of gradient computation
    with open(reference_benchpath(problem_str, device_str, metric="peakmem"), "r") as f:
        reference = json.load(f)["peakmem"]

    # Add an additional axis that shows memory in multiples of gradients
    ax.axvline(reference, color="black", linestyle="--")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Relative to gradient computation")

    # Choose a reasonable number of ticks
    _, x_max = ax.get_xlim()
    num_gradients = x_max / reference
    spacing = 1 / 4
    num_ticks = 1 + floor(num_gradients / spacing)
    while num_ticks > 8:
        spacing *= 2
        num_ticks = 1 + floor(num_gradients / spacing)

    ax2.set_xticks(arange(0, num_ticks) * spacing * reference)
    ax2.set_xticklabels(arange(0, num_ticks * spacing, spacing).tolist())

    return fig, ax


if __name__ == "__main__":
    plot_config = bundles.icml2024(column="full" if ON_RTD else "half", usetex=USETEX)

    for problem_str, device_str in product(PROBLEM_STRS, DEVICE_STRS):
        with plt.rc_context(plot_config):
            fig, ax = visualize_peakmem_benchmark(LINOP_STRS, problem_str, device_str)
            plt.savefig(
                figpath(problem_str, device_str, metric="peakmem"), bbox_inches="tight"
            )

# %%
#
# As hinted at in the introduction, the numbers we observe in this pedagogical example
# may not reflect the relative memory consumption on larger problems and GPUs.
#
# Conclusion
# ==========
#
# In this tutorial, we have demonstrated how to evaluate the run time and memory
# performance of linear operators. This allows to get a feeling for how expensive each
# operator is, compared to a gradient computation.
#
# While we only looked at a small synthetic problem, the same methodology can be applied
# to larger problems.
