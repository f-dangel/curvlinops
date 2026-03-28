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
from os import environ, makedirs, path, remove
from shutil import which
from subprocess import CalledProcessError, CompletedProcess, run

import matplotlib.pyplot as plt
from benchmark_utils import (
    GPTWrapper,
    TimeBenchmark,
    save_environment_info,
    setup_synthetic_cifar10_resnet18,
    setup_synthetic_imagenet_resnet50,
    setup_synthetic_shakespeare_nanogpt,
)
from torch import Tensor, arange, cuda, device, manual_seed, rand, randint
from torch.nn import (
    Conv2d,
    CrossEntropyLoss,
    Linear,
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
from curvlinops.computers.kfac_hooks import HooksKFACComputer, _use_params
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
HEREDIR = path.dirname(path.abspath(__file__))
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

# Names that are EKFAC (precompute is split into factors + eigh + correction)
_IS_EKFAC = {
    "EKFAC (hooks)",
    "EKFAC inverse (hooks)",
    "EKFAC (fx)",
    "EKFAC inverse (fx)",
}

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
#
# Both use the :class:`TimeBenchmark` utility from ``benchmark_utils`` for timing,
# skip-if-exists, and JSON persistence.


def _get_precompute_ops(linop_str: str) -> list[str]:
    """Return the sub-phase operation names for a given linop.

    Args:
        linop_str: The linear operator name.

    Returns:
        List of sub-phase operation names.
    """
    if linop_str in _IS_EKFAC and linop_str in _IS_FX:
        return ["kfac_factors", "eigenvalue_correction", "eigh", "tracing"]
    elif linop_str in _IS_EKFAC:
        return ["kfac_factors", "eigenvalue_correction", "eigh"]
    elif linop_str in _IS_KFAC_INVERSE_HOOKS:
        return ["kfac_factors", "cholesky_inverse"]
    elif linop_str in _IS_FX:
        return ["kfac_factors", "tracing"]
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
    from curvlinops.computers.io_collector import with_kfac_io
    from curvlinops.kfac_utils import FisherType
    from curvlinops.utils import make_functional_call

    num_data = sum(X.shape[0] for (X, _) in data)
    X0, y0 = next(iter(data))
    num_per_example_loss_terms = y0.numel() // X0.shape[0]
    common_kwargs = dict(
        check_deterministic=False,
        num_data=num_data,
        num_per_example_loss_terms=num_per_example_loss_terms,
        separate_weight_and_bias=False,
    )
    dev = next(iter(params.values())).device
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
        # EKFAC FX: factors → correction → eigh → tracing
        computer = setup_computer(linop_str, model, loss_function, params, data)

        def ekfac_fx_factors():
            # Measure total precompute, subtract other phases later
            total, _ = TimeBenchmark(is_cuda="cuda" in str(dev), num_repeats=1).time(
                lambda: setup_linop(
                    linop_str,
                    model,
                    loss_function,
                    params,
                    data,
                    check_deterministic=False,
                ),
            )
            state["total"] = total
            # Also compute factors for eigh/correction dependencies
            state["traced_io"] = computer._trace_io_functions()
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

        def ekfac_fx_tracing():
            model_func = make_functional_call(model)
            X_example = next(iter(data))[0].to(dev)
            with_kfac_io(model_func, X_example, params, FisherType.MC)

        phases["kfac_factors"] = ekfac_fx_factors
        phases["eigenvalue_correction"] = ekfac_fx_correction
        phases["eigh"] = ekfac_fx_eigh
        phases["tracing"] = ekfac_fx_tracing

    elif linop_str in _IS_FX:
        # KFAC FX: factors → tracing
        def kfac_fx_tracing():
            model_func = make_functional_call(model)
            X_example = next(iter(data))[0].to(dev)
            with_kfac_io(model_func, X_example, params, FisherType.MC)

        def kfac_fx_factors():
            # Measure total precompute, subtract tracing later
            total, _ = TimeBenchmark(is_cuda="cuda" in str(dev), num_repeats=1).time(
                lambda: setup_linop(
                    linop_str,
                    model,
                    loss_function,
                    params,
                    data,
                    check_deterministic=False,
                ),
            )
            state["total"] = total

        phases["kfac_factors"] = kfac_fx_factors
        phases["tracing"] = kfac_fx_tracing

    else:
        # Plain KFAC hooks: single phase
        def kfac_factors():
            KFACLinearOperator(model, loss_function, params, data, **common_kwargs)

        phases["kfac_factors"] = kfac_factors

    return phases


def _postprocess_fx_factors(results: dict[str, float]) -> dict[str, float]:
    """For FX operators, derive kfac_factors = total - tracing - other phases.

    Args:
        results: Raw timing results from run_phases.

    Returns:
        Adjusted results with kfac_factors derived by subtraction.
    """
    if "tracing" in results and results.get("kfac_factors", 0) > 0:
        # kfac_factors was measured as total; subtract all other timed phases
        other = sum(v for k, v in results.items() if k != "kfac_factors")
        results["kfac_factors"] = max(results["kfac_factors"] - other, 0.0)
    return results


# %%
#
# Main benchmark execution
# ^^^^^^^^^^^^^^^^^^^^^^^^

if __name__ == "__main__":
    # Reference baselines (once per problem)
    for device_str, problem_str in product(DEVICE_STRS, PROBLEM_STRS):
        bench = TimeBenchmark(
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
        bench = TimeBenchmark(
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

            # For FX operators, derive kfac_factors by subtraction
            if linop_str in _IS_FX:
                results = _postprocess_fx_factors(results)

        with open(save_path, "w") as f:
            json.dump(results, f)
        print(f"[Time] Saved {label}")

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
        with open(benchpath(name, problem_str, device_str), "r") as f:
            matvec_time = json.load(f)["matvec"]
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
    return path.join(_problem_dir(problem_str), f"{metric}_{device_str}.pdf")


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

    with open(reference_benchpath(problem_str, device_str), "r") as f:
        reference = json.load(f)["peakmem"]
    _add_gradient_reference_axis(ax, reference)

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
