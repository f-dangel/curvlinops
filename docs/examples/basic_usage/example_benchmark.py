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
from itertools import product
from os import environ
from shutil import which

import matplotlib.pyplot as plt
from benchmark_execute import Benchmark
from benchmark_utils import (
    _IS_COMPILABLE,
    _KFAC_LIKE,
    LINOP_STRS,
    MATVEC_LINOP_STRS,
    RESULTDIR,
    _get_precompute_ops,
    add_gradient_reference,
    figpath,
    save_environment_info,
)
from benchmark_utils import (
    PROBLEM_STRS as ALL_PROBLEM_STRS,
)
from torch import cuda
from tueplots import bundles

# %%
#
# Let's also set up some variables that will be useful to generate and store results.

# In the execution with sphinx-gallery, __file__ is not defined and we need
# to set it manually using the trick from https://stackoverflow.com/a/53293924
if "__file__" not in globals():
    __file__ = inspect.getfile(lambda: None)

# When running on RTD, we only want to execute the small example
ON_RTD = environ.get("READTHEDOCS", "False") == "True"
# Use LaTeX if available
USETEX = which("latex") is not None

# Devices to run the benchmark on
DEVICE_STRS = ["cuda"] if cuda.is_available() else ["cpu"]

# Whether to skip runs for which measurements already exist
SKIP_EXISTING = True

# Supported problems (use only the small MLP on RTD)
PROBLEM_STRS = ["synthetic_mnist_mlp"] if ON_RTD else ALL_PROBLEM_STRS


save_environment_info(RESULTDIR)

# %%
#
# Benchmark execution
# -------------------
#
# The :class:`~benchmark_execute.Benchmark` class handles all measurements.
# For each problem and device, we measure a reference gradient computation
# and then each linear operator. Run time is measured in-process (minimum over
# multiple repeats), while peak memory is measured in isolated subprocesses to
# avoid allocation artifacts.

for problem_str, device_str in product(PROBLEM_STRS, DEVICE_STRS):
    bench = Benchmark(problem_str, device_str, skip_existing=SKIP_EXISTING)
    bench.run_reference()
    for linop_str in LINOP_STRS:
        bench.run_operator(linop_str)

# %%
#
# Run time visualization
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We first visualize the matrix-vector product times for all operators, and then
# the precompute sub-phase breakdown for KFAC-like operators.


def visualize_matvec_benchmark(
    bench: Benchmark, linop_strs: list[str]
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize matvec times for all operators.

    Args:
        bench: The benchmark instance (for loading results).
        linop_strs: The linear operators.

    Returns:
        The figure and axes.
    """
    fig, ax = plt.subplots()

    for idx, name in enumerate(linop_strs):
        data = bench.load_operator(name)
        ax.barh(idx, width=data["matvec"], color="tab:blue")

    ax.set_yticks(list(range(len(linop_strs))))
    # Strip backend suffix — matvec is backend-independent
    ax.set_yticklabels([n.replace(" (hooks)", "") for n in linop_strs])
    ax.set_xlabel("Time [s]")

    reference = bench.load_reference()["time"]
    add_gradient_reference(ax, reference)

    return fig, ax


def visualize_precompute_benchmark(
    bench: Benchmark, linop_strs: list[str]
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize precompute sub-phase breakdown for KFAC/EKFAC operators.

    Args:
        bench: The benchmark instance (for loading results).
        linop_strs: The KFAC/EKFAC linear operators to plot.

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

        precompute_data = bench.load_operator(name)

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
    ax.set_xscale("log")

    reference = bench.load_reference()["time"]
    add_gradient_reference(ax, reference)

    ax.legend(bbox_to_anchor=(0.5, -0.45), loc="upper center", borderaxespad=0, ncol=2)
    return fig, ax


# %%
#
# Let's now visualize the results. We first show the matrix-vector product times.

plot_config = bundles.icml2024(column="full" if ON_RTD else "half", usetex=USETEX)
plt.rcParams["savefig.bbox"] = "tight"
kfac_linops = [linop for linop in LINOP_STRS if linop in _KFAC_LIKE]

for problem_str, device_str in product(PROBLEM_STRS, DEVICE_STRS):
    bench = Benchmark(problem_str, device_str)
    with plt.rc_context(plot_config):
        fig, ax = visualize_matvec_benchmark(bench, MATVEC_LINOP_STRS)
        plt.savefig(
            figpath(problem_str, device_str, metric="time_matvec"),
            bbox_inches="tight",
        )

# %%
#
# And the precompute sub-phase breakdown for KFAC-like operators.

for problem_str, device_str in product(PROBLEM_STRS, DEVICE_STRS):
    bench = Benchmark(problem_str, device_str)
    with plt.rc_context(plot_config):
        fig, ax = visualize_precompute_benchmark(bench, kfac_linops)
        plt.savefig(
            figpath(problem_str, device_str, metric="time_precompute"),
            bbox_inches="tight",
        )

# %%
#
# As hinted at in the introduction, the numbers we observe in this pedagogical example
# may not reflect the relative cost of linear operators on larger problems and GPUs.
# However, we should see a rough tendency that Hessian-vector products are more costly
# than GGN-vector products, and that KFAC costs only a few gradients to pre-compute,
# while being very cheap to multiply with. Also, inverting KFAC adds some additional
# run time.
#
# Memory visualization
# ^^^^^^^^^^^^^^^^^^^^
#
# The peak memory benchmark results are collected alongside the run time measurements
# by the :class:`~benchmark_execute.Benchmark` class. Memory measurements are run
# in separate Python sessions to avoid allocation artifacts.


def visualize_peakmem_benchmark(
    bench: Benchmark, linop_strs: list[str]
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize the peak memory benchmark results.

    Args:
        bench: The benchmark instance (for loading results).
        linop_strs: The linear operators.

    Returns:
        The figure and axes of the plot.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("Peak memory [GiB]")

    # Visualize the peak memory consumption of each linear operator
    for idx, name in enumerate(linop_strs):
        data = bench.load_operator(name)
        ax.barh(idx, data["peakmem"], color="blue")
    ax.set_yticks(list(range(len(linop_strs))))
    ax.set_yticklabels(linop_strs)

    # Get memory consumption of gradient computation
    reference = bench.load_reference()["peakmem"]
    add_gradient_reference(ax, reference)

    return fig, ax


# %%
#
# Let's visualize the peak memory consumption.

for problem_str, device_str in product(PROBLEM_STRS, DEVICE_STRS):
    bench = Benchmark(problem_str, device_str)
    with plt.rc_context(plot_config):
        fig, ax = visualize_peakmem_benchmark(bench, LINOP_STRS)
        plt.savefig(
            figpath(problem_str, device_str, metric="peakmem"), bbox_inches="tight"
        )

# %%
#
# As hinted at in the introduction, the numbers we observe in this pedagogical example
# may not reflect the relative memory consumption on larger problems and GPUs.
#
# Compiled operator visualization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For operators that support ``torch.compile``, we compare eager and compiled
# performance. The reference gradient computation is also shown in both modes
# (traced with ``make_fx``, then compiled).

# Compilable operators for matvec (backend-independent, hooks only)
COMPILABLE_MATVEC_LINOPS = [
    name for name in MATVEC_LINOP_STRS if name in _IS_COMPILABLE
]
# Compilable operators for peakmem (backend matters, show all)
COMPILABLE_LINOPS = [name for name in LINOP_STRS if name in _IS_COMPILABLE]


def _visualize_compiled(
    bench: Benchmark,
    linop_strs: list[str],
    eager_key: str,
    compiled_key: str,
    ref_eager_key: str,
    ref_compiled_key: str,
    xlabel: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualize eager vs compiled measurements for compilable operators.

    Args:
        bench: The benchmark instance (for loading results).
        linop_strs: The linear operators to plot.
        eager_key: JSON key for the eager measurement (e.g. ``"matvec"``).
        compiled_key: JSON key for the compiled measurement.
        ref_eager_key: JSON key for the eager reference.
        ref_compiled_key: JSON key for the compiled reference.
        xlabel: X-axis label.

    Returns:
        The figure and axes.
    """
    reference = bench.load_reference()

    fig, ax = plt.subplots()

    for idx, name in enumerate(linop_strs):
        data = bench.load_operator(name)
        # Eager bar behind (full opacity), compiled bar on top (semi-transparent)
        ax.barh(
            idx,
            data[eager_key],
            color="tab:blue",
            label="eager" if idx == 0 else None,
        )
        ax.barh(
            idx,
            data[compiled_key],
            color="tab:cyan",
            alpha=0.7,
            label="compiled" if idx == 0 else None,
        )

    ax.set_yticks(list(range(len(linop_strs))))
    ax.set_yticklabels(linop_strs)
    ax.set_xlabel(xlabel)

    add_gradient_reference(ax, reference[ref_eager_key])
    ax.axvline(
        reference[ref_compiled_key],
        color="gray",
        linestyle=":",
        label="gradient (compiled)",
    )

    ax.legend(fontsize="small")
    return fig, ax


def visualize_compiled_matvec(bench):
    """Visualize eager vs compiled matvec times.

    Returns:
        The figure and axes.
    """
    return _visualize_compiled(
        bench,
        COMPILABLE_MATVEC_LINOPS,
        "matvec",
        "matvec_compiled",
        "time",
        "time_compiled",
        "Time [s]",
    )


def visualize_compiled_peakmem(bench):
    """Visualize eager vs compiled peak memory.

    Returns:
        The figure and axes.
    """
    return _visualize_compiled(
        bench,
        COMPILABLE_LINOPS,
        "peakmem",
        "peakmem_compiled",
        "peakmem",
        "peakmem_compiled",
        "Peak memory [GiB]",
    )


# %%
#
# Compiled matvec comparison.

for problem_str, device_str in product(PROBLEM_STRS, DEVICE_STRS):
    bench = Benchmark(problem_str, device_str)
    with plt.rc_context(plot_config):
        fig, ax = visualize_compiled_matvec(bench)
        plt.savefig(
            figpath(problem_str, device_str, metric="time_compiled"),
            bbox_inches="tight",
        )

# %%
#
# Compiled peak memory comparison.

for problem_str, device_str in product(PROBLEM_STRS, DEVICE_STRS):
    bench = Benchmark(problem_str, device_str)
    with plt.rc_context(plot_config):
        fig, ax = visualize_compiled_peakmem(bench)
        plt.savefig(
            figpath(problem_str, device_str, metric="peakmem_compiled"),
            bbox_inches="tight",
        )

# %%
#
# Conclusion
# ==========
#
# In this tutorial, we have demonstrated how to evaluate the run time and memory
# performance of linear operators. This allows to get a feeling for how expensive each
# operator is, compared to a gradient computation.
#
# While we only looked at a small synthetic problem, the same methodology can be applied
# to larger problems, as shown below.

# %%
#
# GPU benchmark results
# =====================
#
# The plots above were generated on CPU for a small MLP on synthetic MNIST. Below, we
# show benchmark results that were pre-computed on a GPU for all supported problems.

PROBLEM_TITLES = {
    "synthetic_mnist_mlp": "MNIST MLP",
    "synthetic_cifar10_resnet18": "CIFAR-10 ResNet-18",
    "synthetic_imagenet_resnet50": "ImageNet ResNet-50",
    "synthetic_shakespeare_nanogpt": "Shakespeare nanoGPT",
}
# %%
#
# Matvec times (GPU)
# ------------------

for problem_str in ALL_PROBLEM_STRS:
    gpu_bench = Benchmark(problem_str, "cuda")
    with plt.rc_context(plot_config):
        fig, ax = visualize_matvec_benchmark(gpu_bench, MATVEC_LINOP_STRS)
        ax.set_title(PROBLEM_TITLES[problem_str])

# %%
#
# Precompute breakdown (GPU)
# --------------------------

for problem_str in ALL_PROBLEM_STRS:
    gpu_bench = Benchmark(problem_str, "cuda")
    with plt.rc_context(plot_config):
        fig, ax = visualize_precompute_benchmark(gpu_bench, kfac_linops)
        ax.set_title(PROBLEM_TITLES[problem_str])

# %%
#
# Peak memory (GPU)
# -----------------

for problem_str in ALL_PROBLEM_STRS:
    gpu_bench = Benchmark(problem_str, "cuda")
    with plt.rc_context(plot_config):
        fig, ax = visualize_peakmem_benchmark(gpu_bench, LINOP_STRS)
        ax.set_title(PROBLEM_TITLES[problem_str])

# %%
#
# Compiled matvec comparison (GPU)
# --------------------------------

for problem_str in ALL_PROBLEM_STRS:
    gpu_bench = Benchmark(problem_str, "cuda")
    with plt.rc_context(plot_config):
        fig, ax = visualize_compiled_matvec(gpu_bench)
        ax.set_title(PROBLEM_TITLES[problem_str])

# %%
#
# Compiled peak memory comparison (GPU)
# --------------------------------------

for problem_str in ALL_PROBLEM_STRS:
    gpu_bench = Benchmark(problem_str, "cuda")
    with plt.rc_context(plot_config):
        fig, ax = visualize_compiled_peakmem(gpu_bench)
        ax.set_title(PROBLEM_TITLES[problem_str])
