"""Peak memory benchmark for linear operators.

This script measures peak memory usage for selected linear operators on synthetic
problems and stores results alongside the runtime benchmarks.
"""

import json
from argparse import ArgumentParser
from contextlib import nullcontext
from os import path

from benchmark_utils import GPTWrapper
from example_benchmark import (
    HAS_JVP,
    LINOP_STRS,
    OP_STRS,
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
    savepath = reference_benchpath(problem_str, device_str, metric="peakmem")
    if SKIP_EXISTING and path.exists(savepath):
        print(
            f"[Memory] Skipping reference on {problem_str} and {device_str}"
        )
        return

    dev = device(device_str)
    is_cuda = "cuda" in str(dev)

    def func():
        manual_seed(0)
        # Use "Hessian" to get all parameters (not the KFAC subset)
        model, loss_function, params, data = setup_problem(
            problem_str, "Hessian", dev
        )
        _ = gradient_and_loss(model, loss_function, params, data)
        if is_cuda:
            cuda.synchronize()

    peakmem_gib = _measure_peak_memory(func, is_cuda) / 2**30
    print(
        f"[Memory] Reference gradient_and_loss on {problem_str} and {device_str}:"
        + f" {peakmem_gib:.2f} GiB"
    )

    with open(savepath, "w") as f:
        json.dump({"peakmem": peakmem_gib}, f)


def run_peakmem_benchmark(  # noqa: C901
    linop_str: str, problem_str: str, device_str: str, op_str: str
):
    """Execute the memory benchmark for a given linear operator class and save results.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        op_str: The operation that is benchmarked (``"precompute"`` or ``"matvec"``).
    """
    savepath = benchpath(linop_str, problem_str, device_str, op_str, metric="peakmem")
    if SKIP_EXISTING and path.exists(savepath):
        print(
            f"[Memory] Skipping {linop_str} on {problem_str} and {device_str} for "
            + f"{op_str}"
        )
        return

    dev = device(device_str)
    is_cuda = "cuda" in str(dev)

    def f_precompute():
        manual_seed(0)  # make deterministic

        model, loss_function, params, data = setup_problem(problem_str, linop_str, dev)
        # NOTE Disable deterministic check as it will otherwise compute matvecs
        _ = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )

        if is_cuda:
            cuda.synchronize()

    def f_matvec():
        manual_seed(0)  # make deterministic

        model, loss_function, params, data = setup_problem(problem_str, linop_str, dev)
        # NOTE Disable deterministic check as it will otherwise compute matvecs
        linop = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )
        v = rand(linop.shape[1], device=dev)

        # Double-backward through efficient attention is unsupported, disable fused kernels
        # (https://github.com/pytorch/pytorch/issues/116350#issuecomment-1954667011)
        attention_double_backward = isinstance(linop, HAS_JVP) and isinstance(
            model, GPTWrapper
        )
        with (
            sdpa_kernel(SDPBackend.MATH) if attention_double_backward else nullcontext()
        ):
            _ = linop @ v

        if is_cuda:
            cuda.synchronize()

    func = {"precompute": f_precompute, "matvec": f_matvec}[op_str]

    peakmem_gib = _measure_peak_memory(func, is_cuda) / 2**30
    print(
        f"[Memory] {linop_str}'s {op_str} on {problem_str} and {device_str}:"
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
        choices=LINOP_STRS,
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
        "--op",
        type=str,
        help="The operation to benchmark.",
        choices=OP_STRS,
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Measure reference gradient_and_loss (ignores --linop and --op).",
    )

    args = parser.parse_args()
    if args.reference:
        run_reference_peakmem_benchmark(args.problem, args.device)
    else:
        run_peakmem_benchmark(args.linop, args.problem, args.device, args.op)
