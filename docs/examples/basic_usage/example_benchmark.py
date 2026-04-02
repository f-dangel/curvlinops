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
from os import environ, path, remove
from shutil import which

import matplotlib.pyplot as plt
from benchmark_utils import (
    RESULTDIR,
    Benchmark,
    GPTWrapper,
    _get_precompute_ops,
    add_gradient_reference,
    benchpath,
    figpath,
    make_precompute_phases,
    reference_benchpath,
    run_verbose,
    save_environment_info,
    setup_synthetic_cifar10_resnet18,
    setup_synthetic_imagenet_resnet50,
    setup_synthetic_mnist_mlp,
    setup_synthetic_shakespeare_nanogpt,
)
from torch import Tensor, cuda, device, manual_seed, rand
from torch.nn import Conv2d, Linear, Module
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
from curvlinops.examples import gradient_and_loss

# %%
#
# Let's also set up some variables that will be useful to generate and store results.

# In the execution with sphinx-gallery, __file__ is not defined and we need
# to set it manually using the trick from https://stackoverflow.com/a/53293924
if "__file__" not in globals():
    __file__ = inspect.getfile(lambda: None)

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


save_environment_info(RESULTDIR)

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
    PROBLEM_STRS = ["synthetic_mnist_mlp"]
else:
    PROBLEM_STRS = [
        "synthetic_mnist_mlp",
        "synthetic_cifar10_resnet18",
        "synthetic_imagenet_resnet50",
        "synthetic_shakespeare_nanogpt",
    ]


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
        "synthetic_mnist_mlp": setup_synthetic_mnist_mlp,
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
# Main benchmark execution
# ^^^^^^^^^^^^^^^^^^^^^^^^

if __name__ == "__main__":
    # Reference baselines (once per problem)
    for device_str, problem_str in product(DEVICE_STRS, PROBLEM_STRS):
        bench = Benchmark(
            is_cuda="cuda" in device_str,
            num_repeats=10,
            skip_existing=SKIP_EXISTING,
        )
        manual_seed(0)
        dev = device(device_str)
        model, loss_function, params, data = setup_problem(problem_str, "Hessian", dev)
        bench.run(
            reference_benchpath(problem_str, device_str),
            f"Reference on {problem_str} and {device_str}",
            lambda: gradient_and_loss(model, loss_function, params, data),  # noqa: B023
        )

    # Per-linop benchmarks: matvec time (+ precompute sub-phases for KFAC-like)
    # All results for one linop on one problem go into a single JSON file.
    for device_str, problem_str, linop_str in product(
        DEVICE_STRS, PROBLEM_STRS, MATVEC_LINOP_STRS
    ):
        bench = Benchmark(
            is_cuda="cuda" in device_str,
            num_repeats=10,
            skip_existing=SKIP_EXISTING,
        )
        save_path = benchpath(linop_str, problem_str, device_str)
        label = f"{linop_str} on {problem_str} and {device_str}"
        if bench.skip_existing and path.exists(save_path):
            print(f"[Time] Skipping {label}")
            continue

        manual_seed(0)
        dev = device(device_str)
        model, loss_function, params, data = setup_problem(problem_str, linop_str, dev)

        # NOTE Disable deterministic check as it will otherwise compute matvecs
        linop = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )
        v = rand(linop.shape[1], device=dev)

        # Double-backward through efficient attention is unsupported, disable
        # fused kernels (pytorch/pytorch#116350)
        def _matvec():  # noqa: B023
            attention_db = isinstance(linop, HAS_JVP) and isinstance(  # noqa: B023
                model,  # noqa: B023
                GPTWrapper,
            )
            with sdpa_kernel(SDPBackend.MATH) if attention_db else nullcontext():
                _ = linop @ v  # noqa: B023

        matvec_time, _ = bench.time(_matvec)
        results = {"matvec": matvec_time}
        print(f"[Time] {label} / matvec: {matvec_time:.4f} s")

        # Precompute sub-phases (KFAC-like only)
        if linop_str in _KFAC_LIKE:
            phases = make_precompute_phases(
                linop_str, model, loss_function, params, data
            )
            for phase_name, func in phases.items():
                t, _ = bench.time(func)
                results[phase_name] = t
                print(f"[Time] {label} / {phase_name}: {t:.4f} s")

        with open(save_path, "w") as f:
            json.dump(results, f)
        print(f"[Time] Saved {label}")

# %%
#
# Visualization
# ^^^^^^^^^^^^^


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
        with open(benchpath(name, problem_str, device_str), "r") as f:
            matvec_time = json.load(f)["matvec"]
        ax.barh(idx, width=matvec_time, color="tab:blue")

    ax.set_yticks(list(range(len(linop_strs))))
    # Strip backend suffix — matvec is backend-independent
    ax.set_yticklabels([n.replace(" (hooks)", "") for n in linop_strs])
    ax.set_xlabel("Time [s]")

    with open(reference_benchpath(problem_str, device_str), "r") as f:
        reference = json.load(f)["time"]
    add_gradient_reference(ax, reference)

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
    kfac = [linop for linop in linop_strs if linop in _KFAC_LIKE]
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

        # Read sub-phases from the operator's benchmark file
        fpath = benchpath(name, problem_str, device_str)
        if not path.exists(fpath):
            continue
        with open(fpath, "r") as f:
            precompute_data = json.load(f)

        left = 0.0
        for op in sub_ops:
            if op not in precompute_data:
                continue
            t = precompute_data[op]
            label = precompute_labels[op] if op not in labels_shown else None
            ax.barh(
                idx,
                width=t,
                left=left,
                color=precompute_colors[op],
                label=label,
                height=0.6,
            )
            labels_shown.add(op)
            left += t

    ax.set_yticks(list(range(len(kfac))))
    ax.set_yticklabels(kfac)
    ax.set_xlabel("Time [s]")

    with open(reference_benchpath(problem_str, device_str), "r") as f:
        reference = json.load(f)["time"]
    add_gradient_reference(ax, reference)

    ax.legend()
    return fig, ax


# %%
#
# Figure paths and plotting


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

        kfac_linops = [linop for linop in LINOP_STRS if linop in _KFAC_LIKE]
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

    # Per-linop: measure peak memory and merge into the operator's JSON file
    for device_str, problem_str, linop_str in product(
        DEVICE_STRS, PROBLEM_STRS, MATVEC_LINOP_STRS
    ):
        operator_path = benchpath(linop_str, problem_str, device_str)
        # Skip if the file already has peakmem
        if path.exists(operator_path):
            with open(operator_path) as f:
                existing = json.load(f)
            if "peakmem" in existing:
                print(f"[Memory] Skipping {linop_str} on {problem_str}")
                continue

        # Run memory measurement in subprocess (writes to a temp file)
        mem_path = benchpath(linop_str, problem_str, device_str, op_str="peakmem")
        cmd = [
            sys.executable,
            path.join(path.dirname(__file__), "memory_benchmark.py"),
            f"--linop={linop_str}",
            f"--problem={problem_str}",
            f"--device={device_str}",
        ]
        print(f"Running command: {' '.join(cmd)}")
        run_verbose(cmd)

        # Merge peakmem into the operator's single file
        if path.exists(mem_path) and path.exists(operator_path):
            with open(mem_path) as f:
                peakmem = json.load(f)["peakmem"]
            with open(operator_path) as f:
                data = json.load(f)
            data["peakmem"] = peakmem
            with open(operator_path, "w") as f:
                json.dump(data, f)
            # Remove the temp file
            remove(mem_path)

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

    # Visualize the peak memory consumption of each linear operator
    labels = [n.replace(" (hooks)", "") for n in linop_strs]
    for idx, name in enumerate(linop_strs):
        fpath = benchpath(name, problem_str, device_str)
        with open(fpath, "r") as f:
            mem = json.load(f)["peakmem"]
        ax.barh(idx, mem, color="blue")
    ax.set_yticks(list(range(len(linop_strs))))
    ax.set_yticklabels(labels)

    # Get memory consumption of gradient computation
    with open(reference_benchpath(problem_str, device_str), "r") as f:
        reference = json.load(f)["peakmem"]
    add_gradient_reference(ax, reference)

    return fig, ax


if __name__ == "__main__":
    plot_config = bundles.icml2024(column="full" if ON_RTD else "half", usetex=USETEX)

    for problem_str, device_str in product(PROBLEM_STRS, DEVICE_STRS):
        with plt.rc_context(plot_config):
            fig, ax = visualize_peakmem_benchmark(
                MATVEC_LINOP_STRS, problem_str, device_str
            )
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
