r"""
Benchmarking linear operators
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
from contextlib import nullcontext
from itertools import product
from math import floor
from os import environ, makedirs, path
from subprocess import CalledProcessError, CompletedProcess, run
from time import perf_counter
from typing import Iterable, List, Tuple

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
    Parameter,
    ReLU,
    Sequential,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
from tueplots import bundles

from curvlinops import (
    EFLinearOperator,
    EKFACLinearOperator,
    FisherMCLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACInverseLinearOperator,
    KFACLinearOperator,
)
from curvlinops._torch_base import PyTorchLinearOperator

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
    FisherMCLinearOperator,
    EFLinearOperator,
)

# When running on RTD, we only want to execute the small example and also
# take into account that there is no LaTeX installation
ON_RTD = environ.get("READTHEDOCS", "False") == "True"
USETEX = not ON_RTD

# Devices to run the benchmark on
DEVICE_STRS = ["cuda"] if cuda.is_available() else ["cpu"]

# Whether to skip runs for which measurements already exists
SKIP_EXISTING = True

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
) -> Tuple[Sequential, CrossEntropyLoss, List[Tuple[Tensor, Tensor]]]:
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
) -> Tuple[Module, Module, List[Parameter], Iterable[Tuple[Tensor, Tensor]]]:
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
    if linop_str in {"KFAC", "KFAC inverse", "EKFAC", "EKFAC inverse"}:
        params = []
        supported_layers = [
            m for m in model.modules() if isinstance(m, (Linear, Conv2d))
        ]
        for m in supported_layers:
            # ignore the last layer of GPT because it has 50k outputs, which
            # will yield an extremely large Kronecker factor
            if all(d <= 50_000 for d in m.weight.shape):
                params.extend([p for p in m.parameters() if p.requires_grad])
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    return model, loss_function, params, data


# %%
#
# We will compare the following linear operators:

LINOP_STRS = [
    "Hessian",
    # NOTE Hessian block-diagonal takes much longer because we have to loop
    # the HVPs over blocks; therefore we exclude it from the benchmark
    # "Block-diagonal Hessian",
    "Generalized Gauss-Newton",
    "Empirical Fisher",
    "Monte-Carlo Fisher",
    "EKFAC",
    "EKFAC inverse",
    "KFAC",
    "KFAC inverse",
]

# %%
#
# And we are interested in the following sub-routines:

# Operations we are interested in
OP_STRS = ["gradient_and_loss", "precompute", "matvec"]

# %%
#
# The next setup function creates linear operators:


def setup_linop(
    linop_str: str,
    model: Module,
    loss_function: Module,
    params: List[Parameter],
    data: Iterable[Tuple[Tensor, Tensor]],
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

    if linop_str == "Block-diagonal Hessian":
        # Figure out parameters that are in a layer (defined by an nn.Module) that
        # forms a block, and get the parameter ids for each block
        ids = [p.data_ptr() for p in params]
        blocks = []

        for mod in model.modules():
            total = [p.data_ptr() for p in mod.parameters() if p.data_ptr() in ids]
            children = [
                p.data_ptr()
                for child in mod.children()
                for p in child.parameters()
                if p.data_ptr() in total
            ]
            if block := [ptr for ptr in total if ptr not in children]:
                blocks.append(block)

        # re-order parameters so that parameters of blocks are consecutive
        blocks_flat = sum(blocks, [])
        new_order = [ids.index(ptr) for ptr in blocks_flat]
        params = [params[i] for i in new_order]

        kwargs["block_sizes"] = [len(block) for block in blocks]

    linop_cls = {
        "Hessian": HessianLinearOperator,
        "Block-diagonal Hessian": HessianLinearOperator,
        "Generalized Gauss-Newton": GGNLinearOperator,
        "Empirical Fisher": EFLinearOperator,
        "Monte-Carlo Fisher": FisherMCLinearOperator,
        "KFAC": KFACLinearOperator,
        "KFAC inverse": KFACLinearOperator,
        "EKFAC": EKFACLinearOperator,
        "EKFAC inverse": EKFACLinearOperator,
    }[linop_str]

    # Double-backward through efficient attention is unsupported, disable fused kernels
    # (https://github.com/pytorch/pytorch/issues/116350#issuecomment-1954667011)
    attention_double_backward = linop_cls in HAS_JVP and isinstance(model, GPTWrapper)
    with sdpa_kernel(SDPBackend.MATH) if attention_double_backward else nullcontext():
        linop = linop_cls(*args, **kwargs)

    if linop_str in {"KFAC inverse", "EKFAC inverse"}:
        linop = KFACInverseLinearOperator(
            linop,
            damping=1e-3,
            cache=True,
            use_exact_damping=linop_str == "EKFAC inverse",
        )

    return linop


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


def run_time_benchmark(  # noqa: C901
    linop_str: str, problem_str: str, device_str: str, op_str: str, num_repeats: int = 1
):
    """Execute the benchmark for a given linear operator class and save results.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        op_str: The operation to benchmark.
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
    linop = setup_linop(
        linop_str, model, loss_function, params, data, check_deterministic=False
    )
    v = rand(linop.shape[1], device=dev)

    # Select function that will be profiled
    def f_gradient_and_loss():
        if isinstance(linop, KFACInverseLinearOperator):
            _ = linop._A.gradient_and_loss()
        else:
            _ = linop.gradient_and_loss()

    def f_precompute():
        if isinstance(linop, (KFACLinearOperator, EKFACLinearOperator)):
            linop.compute_kronecker_factors()
        if isinstance(linop, EKFACLinearOperator):
            linop.compute_eigenvalue_correction()
        if isinstance(linop, KFACInverseLinearOperator):
            linop._A.compute_kronecker_factors()
            if isinstance(linop._A, EKFACLinearOperator):
                linop._A.compute_eigenvalue_correction()
            # damp and invert the Kronecker matrices
            for mod_name in linop._A._mapping:
                linop._compute_or_get_cached_inverse(mod_name)

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

    func = {
        "gradient_and_loss": f_gradient_and_loss,
        "precompute": f_precompute,
        "matvec": f_matvec,
    }[op_str]

    times = []
    for _ in range(num_repeats):
        # Wait until all operations on the GPU are done
        if is_cuda:
            cuda.synchronize()

        # Measure execution time and write to file
        start = perf_counter()
        _ = func()
        if is_cuda:
            cuda.synchronize()
        end = perf_counter()
        times.append(end - start)

    best = min(times)
    print(
        f"[Time] {linop_str}'s {op_str} on {problem_str} and {device_str}:"
        + f"\n\tBest: {best:.4f} s\n\tRepeats: {[round(t, 5) for t in times]}"
    )

    with open(savepath, "w") as f:
        json.dump({"time": best}, f)


# %%
#
# Now we can run the benchmark for each linear operator and visualize the results.

if __name__ == "__main__":
    for device_str, problem_str, linop_str, op_str in product(
        DEVICE_STRS, PROBLEM_STRS, LINOP_STRS, OP_STRS
    ):
        run_time_benchmark(linop_str, problem_str, device_str, op_str, num_repeats=10)

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


def visualize_time_benchmark(
    linop_strs: List[str], problem_str: str, device_str: str
) -> Tuple[plt.Figure, plt.Axes]:
    """Visualize the run time benchmark results.

    Args:
        linop_strs: The linear operators.
        problem_str: The problem.
        device_str: The device.

    Returns:
        The figure and axes of the plot.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("Time [s]")

    # Visualize the run time of each linear operator
    for idx, name in enumerate(linop_strs):
        # gather results:
        results = {}
        for op_str in OP_STRS:
            with open(benchpath(name, problem_str, device_str, op_str), "r") as f:
                results[op_str] = json.load(f)["time"]

        if name in {"KFAC", "KFAC inverse", "EKFAC", "EKFAC inverse"}:
            ax.barh(
                idx - 0.2,
                width=results["precompute"],
                color="green",
                label="precompute" if name == "KFAC" else None,
                height=0.4,
            )
            ax.barh(
                idx + 0.2,
                width=results["matvec"],
                color="blue",
                label="matvec" if name == "KFAC" else None,
                height=0.4,
            )
        else:
            ax.barh(idx, width=results["matvec"], color="blue")

    ax.set_yticks(list(range(len(linop_strs))))
    ax.set_yticklabels(linop_strs)

    # Add an additional axis that shows run time in multiples of gradients
    with open(
        benchpath(linop_strs[0], problem_str, device_str, "gradient_and_loss"), "r"
    ) as f:
        reference = json.load(f)["time"]
    ax.axvline(reference, color="black", linestyle="--")

    # Make the top x axis show multiples of the reference time
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

    ax.legend()

    return fig, ax


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
            fig, ax = visualize_time_benchmark(LINOP_STRS, problem_str, device_str)
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


def run_verbose(cmd: List[str]) -> CompletedProcess:
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
    for device_str, problem_str, linop_str, op_str in product(
        DEVICE_STRS, PROBLEM_STRS, LINOP_STRS, OP_STRS
    ):
        cmd = [
            "python",
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
    linop_strs: List[str], problem_str: str, device_str: str
) -> Tuple[plt.Figure, plt.Axes]:
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
    reference_savepath = benchpath(
        linop_strs[0],
        problem_str,
        device_str,
        "gradient_and_loss",
        metric="peakmem",
    )
    with open(reference_savepath, "r") as f:
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
