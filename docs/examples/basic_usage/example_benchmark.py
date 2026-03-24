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

# FX backend precompute is split into tracing + factor computation
FX_PRECOMPUTE_OPS = ["tracing", "kfac_factors"]

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
# We are interested in comparing a gradient computation with a linear operator's
# matrix-vector multiplication. For operators with an internal pre-computed
# representation (e.g. KFAC), we also want to disentangle the cost of pre-computation
# versus applying a matrix-vector product.
#
# For each linear operator, we measure the execution times of different routines and
# store them for later visualization. To account for warm-up, we repeat each measurement
# multiple times, then use the minimum value as proxy for run time.
#
# Collection
# ^^^^^^^^^^
#
# The function that executes the profiling and extracts the run times looks as follows:


def _time_function(func, is_cuda: bool, num_repeats: int) -> float:
    """Time a function and return the best (minimum) time.

    Args:
        func: The function to time.
        is_cuda: Whether to synchronize CUDA before/after.
        num_repeats: Number of repeats.

    Returns:
        The minimum time across repeats.
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
    if SKIP_EXISTING and path.exists(savepath):
        print(f"[Time] Skipping reference on {problem_str} and {device_str}")
        return

    manual_seed(0)
    dev = device(device_str)
    is_cuda = "cuda" in str(dev)
    # Use "Hessian" to get all parameters (not the KFAC subset)
    model, loss_function, params, data = setup_problem(problem_str, "Hessian", dev)

    def func():
        _ = gradient_and_loss(model, loss_function, params, data)

    best, _ = _time_function(func, is_cuda, num_repeats)
    print(
        f"[Time] Reference gradient_and_loss on {problem_str} and {device_str}:"
        + f"\n\tBest: {best:.4f} s"
    )

    with open(savepath, "w") as f:
        json.dump({"time": best}, f)


def run_time_benchmark(  # noqa: C901
    linop_str: str, problem_str: str, device_str: str, op_str: str, num_repeats: int = 1
):
    """Execute the benchmark for a given linear operator class and save results.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        op_str: The operation to benchmark (``"precompute"`` or ``"matvec"``).
        num_repeats: The number of repeats. Default is ``1``. Will use the smallest
            run time of all repeats as proxy for run time.
    """
    savepath = benchpath(linop_str, problem_str, device_str, op_str)
    if SKIP_EXISTING and path.exists(savepath):
        print(
            f"[Time] Skipping {linop_str} on {problem_str} and {device_str} for "
            + f"{op_str}"
        )
        return

    manual_seed(0)  # make deterministic

    # Set up the problem
    dev = device(device_str)
    is_cuda = "cuda" in str(dev)
    model, loss_function, params, data = setup_problem(problem_str, linop_str, dev)

    def f_precompute():
        _ = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )

    # Generate one linear operator and vector for multiplication
    linop = setup_linop(
        linop_str, model, loss_function, params, data, check_deterministic=False
    )
    v = rand(linop.shape[1], device=dev)

    def f_matvec():
        # Double-backward through efficient attention is unsupported, disable fused kernels
        # (https://github.com/pytorch/pytorch/issues/116350#issuecomment-1954667011)
        attention_double_backward = isinstance(linop, HAS_JVP) and isinstance(
            model, GPTWrapper
        )
        with (
            sdpa_kernel(SDPBackend.MATH) if attention_double_backward else nullcontext()
        ):
            _ = linop @ v

    # carry out pre-computation if we want to perform matvecs, because we don't
    # want them to be included in the measurement.
    if op_str == "matvec":
        f_precompute()

    func = {"precompute": f_precompute, "matvec": f_matvec}[op_str]

    best, _ = _time_function(func, is_cuda, num_repeats)
    print(
        f"[Time] {linop_str}'s {op_str} on {problem_str} and {device_str}:"
        + f"\n\tBest: {best:.4f} s"
    )

    with open(savepath, "w") as f:
        json.dump({"time": best}, f)


def run_ekfac_precompute_benchmark(  # noqa: C901
    linop_str: str, problem_str: str, device_str: str, num_repeats: int = 1
):
    """Benchmark EKFAC precomputation sub-phases separately.

    Measures kfac_factors, eigh, and eigenvalue_correction individually.

    Args:
        linop_str: The EKFAC linear operator variant.
        problem_str: The problem.
        device_str: The device.
        num_repeats: Number of repeats per sub-phase.
    """
    # Check which sub-phases still need to be measured
    ops_to_run = []
    for op_str in EKFAC_PRECOMPUTE_OPS:
        savepath = benchpath(linop_str, problem_str, device_str, op_str)
        if SKIP_EXISTING and path.exists(savepath):
            print(
                f"[Time] Skipping {linop_str} on {problem_str} and {device_str} for "
                + f"{op_str}"
            )
        else:
            ops_to_run.append(op_str)
    if not ops_to_run:
        return

    manual_seed(0)
    dev = device(device_str)
    is_cuda = "cuda" in str(dev)
    model, loss_function, params, data = setup_problem(problem_str, linop_str, dev)
    computer = setup_computer(linop_str, model, loss_function, params, data)

    for op_str in ops_to_run:
        if op_str == "kfac_factors":

            def f_factors():
                with computer._computation_context():
                    return computer._compute_kronecker_factors()

            best, (input_cov, grad_cov, mapping) = _time_function(
                f_factors, is_cuda, num_repeats
            )

        elif op_str == "eigh":
            # Ensure factors are available
            if "input_cov" not in dir():
                with computer._computation_context():
                    input_cov, grad_cov, mapping = (
                        computer._compute_kronecker_factors()
                    )

            # Make copies so each repeat starts from un-decomposed factors
            input_cov_copy = {k: v.clone() for k, v in input_cov.items()}
            grad_cov_copy = {k: v.clone() for k, v in grad_cov.items()}

            def f_eigh():
                ic = {k: v.clone() for k, v in input_cov_copy.items()}
                gc = {k: v.clone() for k, v in grad_cov_copy.items()}
                return _EKFACMixin._eigenvectors_(ic), _EKFACMixin._eigenvectors_(gc)

            best, (input_cov_eig, grad_cov_eig) = _time_function(
                f_eigh, is_cuda, num_repeats
            )
            # Store eigenvectors for the correction phase
            input_cov = input_cov_eig
            grad_cov = grad_cov_eig

        elif op_str == "eigenvalue_correction":
            # Ensure eigenvectors are available
            if "mapping" not in dir():
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

        print(
            f"[Time] {linop_str}'s {op_str} on {problem_str} and {device_str}:"
            + f"\n\tBest: {best:.4f} s"
        )
        savepath = benchpath(linop_str, problem_str, device_str, op_str)
        with open(savepath, "w") as f:
            json.dump({"time": best}, f)


def run_fx_precompute_benchmark(
    linop_str: str, problem_str: str, device_str: str, num_repeats: int = 1
):
    """Benchmark FX backend precomputation sub-phases separately.

    Measures tracing (``with_kfac_io``) and factor computation individually.

    Args:
        linop_str: The FX-backend linear operator variant.
        problem_str: The problem.
        device_str: The device.
        num_repeats: Number of repeats per sub-phase.
    """
    ops_to_run = []
    for op_str in FX_PRECOMPUTE_OPS:
        savepath = benchpath(linop_str, problem_str, device_str, op_str)
        if SKIP_EXISTING and path.exists(savepath):
            print(
                f"[Time] Skipping {linop_str} on {problem_str} and {device_str} for "
                + f"{op_str}"
            )
        else:
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

    computer_cls = {
        "KFAC (fx)": MakeFxKFACComputer,
        "KFAC inverse (fx)": MakeFxKFACComputer,
        "EKFAC (fx)": MakeFxEKFACComputer,
        "EKFAC inverse (fx)": MakeFxEKFACComputer,
    }[linop_str]

    for op_str in ops_to_run:
        if op_str == "tracing":
            # Time just the with_kfac_io tracing call
            from curvlinops.computers.io_collector import with_kfac_io
            from curvlinops.utils import make_functional_call
            from curvlinops.kfac_utils import FisherType

            model_func = make_functional_call(model)
            X_example = next(iter(data))[0].to(dev)

            def f_tracing():
                return with_kfac_io(model_func, X_example, params, FisherType.MC)

            best, _ = _time_function(f_tracing, is_cuda, num_repeats)

        elif op_str == "kfac_factors":
            # Time factor computation with pre-traced function (no tracing cost)
            # First, trace once to warm up
            comp = computer_cls(
                model, loss_function, params, data,
                check_deterministic=False, num_data=num_data,
                num_per_example_loss_terms=num_per_example_loss_terms,
                separate_weight_and_bias=False,
            )

            # Time full compute() minus the tracing overhead:
            # We measure a second compute() call where the data loop encounters
            # the same batch size (already traced and cached in traced_io_fns)
            # Unfortunately the per-instance cache is lost, so we time the
            # full compute and subtract the tracing time.
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
                # Fallback: cannot compute without both measurements
                continue

        print(
            f"[Time] {linop_str}'s {op_str} on {problem_str} and {device_str}:"
            + f"\n\tBest: {best:.4f} s"
        )
        savepath = benchpath(linop_str, problem_str, device_str, op_str)
        with open(savepath, "w") as f:
            json.dump({"time": best}, f)


# %%
#
# Now we can run the benchmark for each linear operator and visualize the results.

if __name__ == "__main__":
    # Reference baselines (once per problem, linop-independent)
    for device_str, problem_str in product(DEVICE_STRS, PROBLEM_STRS):
        run_reference_benchmark(problem_str, device_str, num_repeats=10)

    # Linop benchmarks (precompute and matvec)
    for device_str, problem_str, linop_str, op_str in product(
        DEVICE_STRS, PROBLEM_STRS, LINOP_STRS, OP_STRS
    ):
        run_time_benchmark(linop_str, problem_str, device_str, op_str, num_repeats=10)

    # EKFAC precompute sub-phases
    for device_str, problem_str, linop_str in product(
        DEVICE_STRS, PROBLEM_STRS, [l for l in LINOP_STRS if l in _IS_EKFAC]
    ):
        run_ekfac_precompute_benchmark(
            linop_str, problem_str, device_str, num_repeats=10
        )

    # FX backend precompute sub-phases (tracing + factor computation)
    for device_str, problem_str, linop_str in product(
        DEVICE_STRS, PROBLEM_STRS, [l for l in LINOP_STRS if l in _IS_FX]
    ):
        run_fx_precompute_benchmark(
            linop_str, problem_str, device_str, num_repeats=10
        )

# %%
#
# Visualization
# ^^^^^^^^^^^^^
#
# At this point, we have collected the run time data for each linear operator and can
# visualize the results. We will plot the run time of the gradient computation and the
# matrix-vector product for each linear operator. For KFAC, we will also show the
# pre-computation time.
#
# Here is the plotting function:


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


def visualize_time_benchmark(
    linop_strs: list[str], problem_str: str, device_str: str
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Visualize run time benchmarks as a 2x1 plot.

    Top panel: non-KFAC operators (matvec only).
    Bottom panel: KFAC/EKFAC operators with precompute breakdown and matvec.

    Args:
        linop_strs: The linear operators.
        problem_str: The problem.
        device_str: The device.

    Returns:
        The figure and list of axes.
    """
    non_kfac = [l for l in linop_strs if l not in _KFAC_LIKE]
    kfac = [l for l in linop_strs if l in _KFAC_LIKE]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        gridspec_kw={"height_ratios": [len(non_kfac), len(kfac)]},
    )

    # --- Top panel: non-KFAC operators (matvec bar only) ---
    for idx, name in enumerate(non_kfac):
        with open(benchpath(name, problem_str, device_str, "matvec"), "r") as f:
            matvec_time = json.load(f)["time"]
        ax_top.barh(
            idx, width=matvec_time, color="tab:blue",
            label="matvec" if idx == 0 else None,
        )

    ax_top.set_yticks(list(range(len(non_kfac))))
    ax_top.set_yticklabels(non_kfac)
    ax_top.legend()

    with open(reference_benchpath(problem_str, device_str), "r") as f:
        reference = json.load(f)["time"]
    _add_gradient_reference_axis(ax_top, reference)

    # --- Bottom panel: KFAC/EKFAC with precompute breakdown (log scale) ---
    # Shades of green for precompute sub-phases, orange for tracing, blue for matvec
    precompute_colors = {
        "kfac_factors": "#2ca02c",           # green
        "eigenvalue_correction": "#66c266",  # medium green
        "eigh": "#a6d9a6",                   # light green
        "tracing": "tab:orange",             # orange for FX tracing
        "precompute": "#2ca02c",             # green (same as factors)
    }
    precompute_labels = {
        "kfac_factors": "Kronecker factors",
        "eigenvalue_correction": "correction",
        "eigh": "decomposition",
        "tracing": "tracing",
        "precompute": "Kronecker factors",
    }
    labels_shown = set()

    for idx, name in enumerate(kfac):
        # Read matvec time
        with open(benchpath(name, problem_str, device_str, "matvec"), "r") as f:
            matvec_time = json.load(f)["time"]

        if name in _IS_EKFAC:
            # Stacked bar for EKFAC sub-phases
            sub_ops = EKFAC_PRECOMPUTE_OPS
        elif name in _IS_FX:
            # Stacked bar for FX sub-phases (tracing + factors)
            sub_ops = FX_PRECOMPUTE_OPS
        else:
            sub_ops = None

        if sub_ops is not None:
            left = 0.0
            for op in sub_ops:
                fpath = benchpath(name, problem_str, device_str, op)
                if not path.exists(fpath):
                    continue
                with open(fpath, "r") as f:
                    t = json.load(f)["time"]
                label = precompute_labels[op] if op not in labels_shown else None
                ax_bot.barh(
                    idx - 0.2, width=t, left=left,
                    color=precompute_colors[op], label=label, height=0.4,
                )
                labels_shown.add(op)
                left += t
        else:
            # Single bar for hooks KFAC precompute
            with open(
                benchpath(name, problem_str, device_str, "precompute"), "r"
            ) as f:
                precompute_time = json.load(f)["time"]
            label = (
                precompute_labels["precompute"]
                if "kfac_factors" not in labels_shown
                else None
            )
            ax_bot.barh(
                idx - 0.2, width=precompute_time,
                color=precompute_colors["precompute"], label=label, height=0.4,
            )
            labels_shown.add("kfac_factors")

        # Matvec bar
        label = "matvec" if "matvec" not in labels_shown else None
        ax_bot.barh(
            idx + 0.2, width=matvec_time,
            color="tab:blue", label=label, height=0.4,
        )
        labels_shown.add("matvec")

    ax_bot.set_yticks(list(range(len(kfac))))
    ax_bot.set_yticklabels(kfac)
    ax_bot.set_xlabel("Time [s]")
    ax_bot.set_xscale("log")

    with open(reference_benchpath(problem_str, device_str), "r") as f:
        reference_kfac = json.load(f)["time"]
    ax_bot.axvline(reference_kfac, color="black", linestyle="--")

    ax_bot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)

    return fig, [ax_top, ax_bot]


# %%
#
# And a convenience function to produce save paths for figures.


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


# %%
#
# Let's take a look at the results:

if __name__ == "__main__":
    plot_config = bundles.icml2024(column="full" if ON_RTD else "half", usetex=USETEX)

    for problem_str, device_str in product(PROBLEM_STRS, DEVICE_STRS):
        with plt.rc_context(plot_config):
            fig, axes = visualize_time_benchmark(LINOP_STRS, problem_str, device_str)
            plt.savefig(figpath(problem_str, device_str), bbox_inches="tight")

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
