import json
import re
from math import floor
from os import makedirs, path
from subprocess import CalledProcessError, CompletedProcess, run
from typing import Iterable, List, Tuple

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

from curvlinops import GGNLinearOperator, HessianLinearOperator
from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.fisher import FisherMCLinearOperator
from curvlinops.gradient_moments import EFLinearOperator
from curvlinops.kfac import KFACLinearOperator

SCRIPTNAME = path.basename(__file__)
HERE = path.abspath(__file__)
HEREDIR = path.dirname(HERE)
RESULTDIR = path.join(HEREDIR, "benchmark")
makedirs(RESULTDIR, exist_ok=True)

LINOP_STRS = [
    "Hessian",
    "Block-diagonal Hessian",
    "Generalized Gauss-Newton",
    "Empirical Fisher",
    "Monte-Carlo Fisher",
    "KFAC",
]
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

    batch_size = 128
    X = rand(batch_size, 1, 28, 28)
    y = randint(0, 10, (batch_size,))
    data = [(X, y)]

    model = Sequential(
        Conv2d(1, 32, 3, padding=1),
        MaxPool2d(2),
        ReLU(),
        Conv2d(32, 64, 3, padding=1),
        MaxPool2d(2),
        ReLU(),
        Conv2d(64, 32, 3, padding=1),
        MaxPool2d(2),
        ReLU(),
        Flatten(),
        Linear(288, 128),
        ReLU(),
        Linear(128, 10),
    ).to(dev)
    loss_function = CrossEntropyLoss().to(dev)

    params = [p for p in model.parameters() if p.requires_grad]

    return model, loss_function, params, data


def setup_linop(
    linop_str: str,
    model: Module,
    loss_function: Module,
    params: List[Parameter],
    data: Iterable[Tuple[Tensor, Tensor]],
    check_deterministic: bool = True,
) -> CurvatureLinearOperator:
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


def benchpath(
    linop_str: str,
    problem_str,
    device_str: str,
    processed: bool = False,
    metric: str = "time",
) -> str:
    """Get the path to save the benchmark results.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        processed: Whether the file is processed. Default is ``False``.
        metric: The metric to save. Default is ``'time'``.

    Returns:
        The path to save the benchmark results.
    """
    processed_str = "_processed" if processed else ""
    return path.join(
        RESULTDIR,
        f"{metric}_{linop_str}_{problem_str}_{device_str}{processed_str}.json",
    )


def run_time_benchmark(linop_str: str, problem_str: str, device_str: str):
    """Execute the benchmark for a given linear operator class and save results.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
    """
    manual_seed(0)  # make deterministic

    dev = device(device_str)
    is_cuda = "cuda" in str(dev)
    model, loss_function, params, data = setup_problem(problem_str, dev)
    linop = setup_linop(linop_str, model, loss_function, params, data)
    v = rand(linop.shape[1], device=dev)

    if is_cuda:
        cuda.synchronize()

    def f():
        _ = linop.gradient_and_loss()
        if isinstance(linop, KFACLinearOperator):
            linop._compute_kfac()
        _ = linop @ v

        if is_cuda:
            cuda.synchronize()

    # Profiling with stack tracing enabled
    with profiler.profile(
        # TODO make GPU profiling work
        activities=[profiler.ProfilerActivity.CPU],
        with_stack=True,
        # NOTE This may likely break or not be necessary in future versions of PyTorch
        # https://github.com/pytorch/pytorch/issues/100253#issuecomment-1579804477
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        f()

    # Save the trace
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

    patterns = {
        "total": rf"{SCRIPTNAME}\(\d+\): f",
        "@": r".*curvlinops/_torch_base\.py\(\d+\): __matmul__",
        "gradient_and_loss": r".*curvlinops/_torch_base\.py\(\d+\): gradient_and_loss",
    }
    if linop_str == "KFAC":
        patterns["precompute"] = r".*curvlinops/kfac\.py\(\d+\): _compute_kfac"

    # Print timings and store them in a json file
    print(f"[Time] {linop_str} on {problem_str} and {device_str}")
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

    processed_trace_file = benchpath(linop_str, problem_str, device_str, processed=True)
    with open(processed_trace_file, "w") as f_trace:
        json.dump(results, f_trace)


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
    for idx, name in enumerate(linop_strs):
        with open(benchpath(name, problem_str, device_str, processed=True), "r") as f:
            results = json.load(f)

        if name == "KFAC":
            ax.barh(name, results["precompute"], color="green", label="precompute")
        ax.barh(name, results["@"], color="blue", label="matvec" if idx == 0 else None)

        if idx == 0:
            reference = results["gradient_and_loss"]
            ax.axvline(reference, color="black", linestyle="--")
            # make the top x axis show multiples of the reference time
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())

            _, x_max = ax.get_xlim()
            num_gradients = x_max / reference
            spacing = 1 / 4
            num_ticks = 1 + floor(num_gradients / spacing)
            while num_ticks > 10:
                spacing *= 2
                num_ticks = 1 + floor(num_gradients / spacing)

            ax2.set_xticks(arange(0, num_ticks) * spacing * reference)
            ax2.set_xticklabels(arange(0, num_ticks * spacing, spacing).tolist())
            ax2.set_xlabel("Relative to gradient computation")

    ax.set_xlabel("Time [s]")
    ax.legend()

    return fig, ax


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
    for idx, name in enumerate(linop_strs):
        savepath = benchpath(name, problem_str, device_str, metric="peakmem").replace(
            ".json", "_matvec.json"
        )
        with open(savepath, "r") as f:
            peakmem = json.load(f)["peakmem"]

        ax.barh(name, peakmem, color="blue")

        if idx == 0:
            reference_savepath = savepath.replace("matvec", "gradient_and_loss")
            with open(reference_savepath, "r") as f:
                reference = json.load(f)["peakmem"]
            ax.axvline(reference, color="black", linestyle="--")
            # make the top x axis show multiples of the reference time
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())

            _, x_max = ax.get_xlim()
            num_gradients = x_max / reference
            spacing = 1 / 4
            num_ticks = 1 + floor(num_gradients / spacing)
            while num_ticks > 10:
                spacing *= 2
                num_ticks = 1 + floor(num_gradients / spacing)

            ax2.set_xticks(arange(0, num_ticks) * spacing * reference)
            ax2.set_xticklabels(arange(0, num_ticks * spacing, spacing).tolist())
            ax2.set_xlabel("Relative to gradient computation")

    ax.set_xlabel("Peak memory [GiB]")

    return fig, ax


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


if __name__ == "__main__":
    DEVICE_STR = "cuda" if cuda.is_available() else "cpu"
    PROBLEM_STR = "synthetic_mnist_cnn"

    for linop_str in LINOP_STRS:
        run_time_benchmark(linop_str, PROBLEM_STR, DEVICE_STR)

    plot_config = bundles.icml2024(column="half", usetex=True)
    with plt.rc_context(plot_config):
        fig, ax = visualize_time_benchmark(LINOP_STRS, PROBLEM_STR, DEVICE_STR)
        figpath = path.join(RESULTDIR, f"time_{PROBLEM_STR}_{DEVICE_STR}.pdf")
        plt.savefig(figpath, bbox_inches="tight")

    # measure memory consumption in individual Python sessions to avoid memory
    # allocations from previous operations into the currently benchmarked operation.
    for linop_str in LINOP_STRS:
        for op in ["gradient_and_loss", "matvec"]:
            cmd = [
                "python",
                "memory_benchmark.py",
                f"--linop={linop_str}",
                f"--problem={PROBLEM_STR}",
                f"--device={DEVICE_STR}",
                f"--op={op}",
            ]
            print(f"Running command: {' '.join(cmd)}")
            run_verbose(cmd)

    plot_config = bundles.icml2024(column="half", usetex=True)
    with plt.rc_context(plot_config):
        fig, ax = visualize_peakmem_benchmark(LINOP_STRS, PROBLEM_STR, DEVICE_STR)
        figpath = path.join(RESULTDIR, f"peakmem_{PROBLEM_STR}_{DEVICE_STR}.pdf")
        plt.savefig(figpath, bbox_inches="tight")
