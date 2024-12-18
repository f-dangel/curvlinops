r"""
Benchmarking linear operators
=============================

In this tutorial, we demonstrate how to evaluate the run time and memory performance
of linear operators. This allows to get a feeling for how expensive each operator is,
compared to a gradient computation.

.. warning::
    For pedagogical reasons, this example considers a small synthetic problem which may
    not reflect the relative cost of linear operators on larger problems. However, the
    following example can easily be applied to such larger problems.

Let's get the imports out of the way.
"""

import inspect
import json
import re
from itertools import product
from math import floor
from os import getenv, makedirs, path
from subprocess import CalledProcessError, CompletedProcess, run
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.profiler
from torch import Tensor, arange, cuda, device, manual_seed, profiler, rand, randint
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
from tueplots import bundles

from curvlinops import (
    EFLinearOperator,
    FisherMCLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
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

# Available devices
DEVICE_STRS = ["cpu"] + (["cuda"] if cuda.is_available() else [])

# LaTeX is not available in Github actions.
# Therefore, we are turning it off if the script executes on GHA.
CI = bool(getenv("CI"))
USETEX = not CI

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
PROBLEM_STRS = ["synthetic_mnist_cnn"]


def setup_problem(
    problem_str: str, dev: device
) -> Tuple[Module, Module, List[Parameter], Iterable[Tuple[Tensor, Tensor]]]:
    """Set up the neural net, loss function, parameters, and data.

    Args:
        problem_str: The problem to set up.
        dev: The device to use.

    Returns:
        The neural net, loss function, parameters, and data.

    Raises:
        ValueError: If the problem is unknown.
    """
    if problem_str != "synthetic_mnist_cnn":
        raise ValueError(f"Unknown problem: {problem_str}")

    batch_size = 64
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
    ).to(dev)
    loss_function = CrossEntropyLoss().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]

    return model, loss_function, params, data


# %%
#
# We will compare the following linear operators:

LINOP_STRS = [
    "Hessian",
    "Block-diagonal Hessian",
    "Generalized Gauss-Newton",
    "Empirical Fisher",
    "Monte-Carlo Fisher",
    "KFAC",
]

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
        num_tensors_layer = [
            len(list(child.parameters())) for child in model.children()
        ]
        kwargs["block_sizes"] = [s for s in num_tensors_layer if s != 0]

    linop_cls = {
        "Hessian": HessianLinearOperator,
        "Block-diagonal Hessian": HessianLinearOperator,
        "Generalized Gauss-Newton": GGNLinearOperator,
        "Empirical Fisher": EFLinearOperator,
        "Monte-Carlo Fisher": FisherMCLinearOperator,
        "KFAC": KFACLinearOperator,
    }[linop_str]

    return linop_cls(*args, **kwargs)


# %%
#
# Last, we define a convenience function that generates the output files where
# the benchmark results will be stored and later plotted.


def benchpath(
    linop_str: str,
    problem_str,
    device_str: str,
    metric: str = "time",
    op_str: Optional[str] = None,
) -> str:
    """Get the path to save the benchmark results.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        metric: The metric to save. Default is ``'time'``.
        op_str: Operation name that is benchmarked. Only required for memory benchmark
            where each function is benchmarked separately.

    Returns:
        The path to save the benchmark results.
    """
    op_str = "" if op_str is None else f"_{op_str}"
    return path.join(
        RESULTDIR,
        f"{metric}_{linop_str}_{problem_str}_{device_str}_{op_str}.json",
    )


# %%
#
# Run time benchmark
# ------------------
#
# To inspect the run time of specific sub-routines, we use the PyTorch profiler.
# Specifically, we are interested in comparing a gradient computation with a
# linear operator's matrix-vector multiplication. For operators with an internal
# pre-computed representation (e.g. KFAC), we also want to disentangle the cost of
# pre-computation versus applying a matrix-vector product.
#
# Our approach for a single linear operator is roughly as follows: We compute the
# gradient and a matrix-vector product and execute the operation stack with the PyTorch
# profiler, then export this information to a JSON file. This JSON file is then parsed
# to extract only the run time of the functions we are interested in. These are then
# stored in another JSON file. Finally, we will visualize the results.
#
# Collection
# ^^^^^^^^^^
#
# The function that executes the profiling and extracts the run times looks as follows:


def run_time_benchmark(linop_str: str, problem_str: str, device_str: str):
    """Execute the benchmark for a given linear operator class and save results.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
    """
    manual_seed(0)  # make deterministic

    # Set up the problem
    dev = device(device_str)
    is_cuda = "cuda" in str(dev)
    model, loss_function, params, data = setup_problem(problem_str, dev)
    linop = setup_linop(linop_str, model, loss_function, params, data)
    v = rand(linop.shape[1], device=dev)

    # Wait until all operations on the GPU are done
    if is_cuda:
        cuda.synchronize()

    # Define the function that will be profiled
    def f():
        _ = linop.gradient_and_loss()
        # for KFAC, we want to disentangle how long it takes to compute the Kronecker
        # factors versus how long a matrix-vector product with KFAC takes
        if isinstance(linop, KFACLinearOperator):
            linop._compute_kfac()
        _ = linop @ v

        # Wait until all operations on the GPU are done
        if is_cuda:
            cuda.synchronize()

    # Profile the function with stack tracing enabled so we can parse the output and
    # extract the time spent in each function
    with profiler.profile(
        # TODO make GPU profiling work
        activities=[profiler.ProfilerActivity.CPU],
        with_stack=True,
        # NOTE This may likely break or not be necessary in future versions of PyTorch
        # https://github.com/pytorch/pytorch/issues/100253#issuecomment-1579804477
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        f()

    print(f"[Time] {linop_str} on {problem_str} and {device_str}")
    # Extracting the relevant run times is a bit tedious, see the function below
    results = extract_times(prof, linop_str, problem_str, device_str)

    result_file = benchpath(linop_str, problem_str, device_str)
    with open(result_file, "w") as f_trace:
        print("Writing results to", result_file)
        json.dump(results, f_trace)


# %%
#
# The function that processes the raw profiler output is a bit tedious.
# Feel free to skip the details:


def extract_times(
    prof: torch.profiler.profile, linop_str: str, problem_str: str, device_str: str
) -> Dict[str, float]:
    """Export and process the profiler's record into the relevant run times.

    Args:
        prof: The profiler.
        linop_str: The linear operator that was recorded.
        problem_str: The problem that was recorded.
        device_str: The device on which the computation was carried out.

    Raises:
        ValueError: If the regexp matching fails, either due to no or multiple matches.

    Returns:
        The relevant run times as dictionary. Run times are in seconds.
    """
    # Export the profiler's trace to a JSON file
    trace_file = benchpath(linop_str, problem_str, device_str)
    prof.export_chrome_trace(trace_file)

    # Load the profiling JSON file, collect function timings
    with open(trace_file, "r") as f_trace:
        trace_data = json.load(f_trace)
    events = trace_data.get("traceEvents", [])

    timings = {}
    for event in events:
        if event.get("ph") == "X":  # "X" phase indicates a complete event with duration
            function_name = event["name"]
            duration = event.get("dur", 0)  # Duration is in microseconds
            timings[function_name] = timings.get(function_name, 0) + duration

    # Extract timings of the functions we care about by regexp matching
    patterns = {
        "total": rf"{SCRIPTNAME}\(\d+\): f",
        "matvec": r".*curvlinops/_torch_base\.py\(\d+\): __matmul__",
        "gradient_and_loss": r".*curvlinops/_torch_base\.py\(\d+\): gradient_and_loss",
    }
    if linop_str == "KFAC":
        patterns["precompute"] = r".*curvlinops/kfac\.py\(\d+\): _compute_kfac"

    # Print timings
    results = {}
    for name, pattern in patterns.items():
        matches = 0
        for func, time in timings.items():
            if re.match(pattern, func):
                print(f"\t{name}: {time:.2f} μs")
                matches += 1
                results[name] = time * 1e-6  # convert to seconds
        if matches != 1:
            raise ValueError(f"Expected 1 match for {name}, found {matches} matches.")

    return results


# %%
#
# Now we can run the benchmark for each linear operator and visualize the results.

if __name__ == "__main__":
    for device_str, problem_str, linop_str in product(
        DEVICE_STRS, PROBLEM_STRS, LINOP_STRS
    ):
        run_time_benchmark(linop_str, problem_str, device_str)

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
        with open(benchpath(name, problem_str, device_str), "r") as f:
            results = json.load(f)

        if name == "KFAC":
            ax.barh(name, results["precompute"], color="green", label="precompute")
        ax.barh(
            name, results["matvec"], color="blue", label="matvec" if idx == 0 else None
        )

    # Add an additional axis that shows run time in multiples of gradients
    with open(benchpath(linop_strs[0], problem_str, device_str), "r") as f:
        results = json.load(f)
    reference = results["gradient_and_loss"]
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
    plot_config = bundles.icml2024(column="full" if CI else "half", usetex=USETEX)

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
# while being very cheap to multiply with.
#
# Memory benchmark
# ================
#
# Measuring the memory consumption of some routines comes with some additional
# challenges. In fact, we do not use the PyTorch profiler for this task, and instead
# use the :ref:`memory_profiler <https://github.com/pythonprofilers/memory_profiler>`
# library on CPU, whereas we rely on :func:`torch.cuda.max_memory_allocated` on GPU.
#
# To avoid memory allocations from previous operations to impact the currently
# benchmarked function, we run each benchmark in a separate Python session by executing
# a Python script :download:`py <memory_benchmark.py>`. This script
# re-uses most of the functionality developed in this tutorial, and the function that
# is profiled looks very similar to the one we used for the run time benchmark.
#
# .. note::
#    If you know a better way to measure memory consumption, please let us know and/or
#    submit a pull request.
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

# Operations whose memory consumption is benchmarked
OP_STRS = ["gradient_and_loss", "matvec"]

if __name__ == "__main__":
    # measure memory consumption in individual Python sessions to avoid memory
    # allocations from previous operations into the currently benchmarked operation.
    for device_str, problem_str, linop_str, op_str in product(
        DEVICE_STRS, PROBLEM_STRS, LINOP_STRS, OP_STRS
    ):
        cmd = [
            "python",
            "memory_benchmark.py",
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
        savepath = benchpath(
            name, problem_str, device_str, metric="peakmem", op_str="matvec"
        )
        with open(savepath, "r") as f:
            peakmem = json.load(f)["peakmem"]
        ax.barh(name, peakmem, color="blue")

    # Get memory consumption of gradient computation
    reference_savepath = benchpath(
        linop_strs[0],
        problem_str,
        device_str,
        metric="peakmem",
        op_str="gradient_and_loss",
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
    plot_config = bundles.icml2024(column="full" if CI else "half", usetex=True)

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
