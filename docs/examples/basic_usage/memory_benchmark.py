"""Peak memory benchmark for linear operators.

This script measures peak memory usage for selected linear operators on synthetic
problems and stores results alongside the runtime benchmarks.

Each measurement runs in a separate Python session to avoid memory allocation
artifacts from previous operations. The measured peak memory covers the full
pipeline: model setup → operator construction → matrix-vector product.
"""

import json
from argparse import ArgumentParser
from contextlib import nullcontext
from os import path

from benchmark_utils import GPTWrapper
from example_benchmark import (
    HAS_JVP,
    MATVEC_LINOP_STRS,
    PROBLEM_STRS,
    SKIP_EXISTING,
    benchpath,
    reference_benchpath,
    setup_linop,
    setup_problem,
)
from memory_profiler import memory_usage
from torch import cuda, device, manual_seed, rand
from torch.nn.attention import SDPBackend, sdpa_kernel

from curvlinops.examples import gradient_and_loss


def _measure_peak_memory(func, is_cuda: bool) -> float:
    """Run func and return peak memory in bytes.

    Args:
        func: The function to measure.
        is_cuda: Whether to use CUDA memory tracking.

    Returns:
        Peak memory in bytes.
    """
    if is_cuda:
        func()
        cuda.synchronize()
        peakmem_bytes = cuda.max_memory_allocated()
        cuda.reset_peak_memory_stats()
    else:
        peakmem_bytes = memory_usage(func, interval=1e-4, max_usage=True) * 2**20
    return peakmem_bytes


def run_reference_peakmem_benchmark(problem_str: str, device_str: str):
    """Measure peak memory for gradient_and_loss (once per problem).

    Uses all model parameters (not a KFAC subset).

    Args:
        problem_str: The problem.
        device_str: The device.
    """
    savepath = reference_benchpath(problem_str, device_str)

    # Skip if peakmem already in the reference file
    if SKIP_EXISTING and path.exists(savepath):
        with open(savepath) as f:
            existing = json.load(f)
        if "peakmem" in existing:
            print(f"[Memory] Skipping reference on {problem_str} and {device_str}")
            return

    dev = device(device_str)
    is_cuda = "cuda" in str(dev)

    def func():
        manual_seed(0)
        model, loss_function, params, data = setup_problem(problem_str, "Hessian", dev)
        _ = gradient_and_loss(model, loss_function, params, data)
        if is_cuda:
            cuda.synchronize()

    peakmem_gib = _measure_peak_memory(func, is_cuda) / 2**30
    print(
        f"[Memory] Reference gradient_and_loss on {problem_str} and {device_str}:"
        + f" {peakmem_gib:.2f} GiB"
    )

    # Merge into reference file (which may already have time data)
    existing = {}
    if path.exists(savepath):
        with open(savepath) as f:
            existing = json.load(f)
    existing["peakmem"] = peakmem_gib
    with open(savepath, "w") as f:
        json.dump(existing, f)


def run_peakmem_benchmark(linop_str: str, problem_str: str, device_str: str):
    """Measure peak memory for the full pipeline: setup → precompute → matvec.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
    """
    savepath = benchpath(linop_str, problem_str, device_str, op_str="peakmem")
    if SKIP_EXISTING and path.exists(savepath):
        print(f"[Memory] Skipping {linop_str} on {problem_str} and {device_str}")
        return

    dev = device(device_str)
    is_cuda = "cuda" in str(dev)

    def func():
        manual_seed(0)
        model, loss_function, params, data = setup_problem(problem_str, linop_str, dev)
        linop = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )
        v = rand(linop.shape[1], device=dev)

        attention_double_backward = isinstance(linop, HAS_JVP) and isinstance(
            model, GPTWrapper
        )
        with (
            sdpa_kernel(SDPBackend.MATH) if attention_double_backward else nullcontext()
        ):
            _ = linop @ v

        if is_cuda:
            cuda.synchronize()

    peakmem_gib = _measure_peak_memory(func, is_cuda) / 2**30
    print(
        f"[Memory] {linop_str} on {problem_str} and {device_str}:"
        + f" {peakmem_gib:.2f} GiB"
    )

    with open(savepath, "w") as f:
        json.dump({"peakmem": peakmem_gib}, f)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run memory benchmark for a given linear operator."
    )
    parser.add_argument(
        "--linop",
        type=str,
        help="The linear operator class to benchmark.",
        choices=MATVEC_LINOP_STRS,
    )
    parser.add_argument(
        "--problem",
        type=str,
        help="The problem to benchmark.",
        choices=PROBLEM_STRS,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="The device to benchmark.",
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Measure reference gradient_and_loss (ignores --linop).",
    )

    args = parser.parse_args()
    if args.reference:
        run_reference_peakmem_benchmark(args.problem, args.device)
    else:
        run_peakmem_benchmark(args.linop, args.problem, args.device)
