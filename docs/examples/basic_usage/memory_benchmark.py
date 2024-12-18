import json
from argparse import ArgumentParser

from example_benchmark import (
    LINOP_STRS,
    OP_STRS,
    PROBLEM_STRS,
    benchpath,
    setup_linop,
    setup_problem,
)
from memory_profiler import memory_usage
from torch import cuda, device, manual_seed, rand

from curvlinops import KFACInverseLinearOperator, KFACLinearOperator


def run_peakmem_benchmark(linop_str: str, problem_str: str, device_str: str, op: str):
    """Execute the memory benchmark for a given linear operator class and save results.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        write_results: Whether to write the results to a file. Default is ``True``.
    """
    dev = device(device_str)
    is_cuda = "cuda" in str(dev)

    def f_gradient_and_loss():
        manual_seed(0)  # make deterministic

        model, loss_function, params, data = setup_problem(problem_str, dev)
        # NOTE Disable deterministic check as it will otherwise compute matvecs
        linop = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )

        if isinstance(linop, KFACInverseLinearOperator):
            _ = linop._A.gradient_and_loss()
        else:
            _ = linop.gradient_and_loss()

        if is_cuda:
            cuda.synchronize()

    def f_matvec():
        manual_seed(0)  # make deterministic

        model, loss_function, params, data = setup_problem(problem_str, dev)
        linop = setup_linop(linop_str, model, loss_function, params, data)
        v = rand(linop.shape[1], device=dev)

        if isinstance(linop, KFACLinearOperator):
            linop._compute_kfac()

        if isinstance(linop, KFACInverseLinearOperator):
            linop._A._compute_kfac()
            # damp and invert the Kronecker matrices
            for mod_name in linop._A._mapping:
                linop._compute_or_get_cached_inverse(mod_name)

        _ = linop @ v

        if is_cuda:
            cuda.synchronize()

    f = {"gradient_and_loss": f_gradient_and_loss, "matvec": f_matvec}[op]

    if is_cuda:
        f()
        cuda.synchronize()
        peakmem_bytes = cuda.max_memory_allocated()
        cuda.reset_peak_memory_stats()
    else:
        peakmem_bytes = memory_usage(f, interval=1e-4, max_usage=True) * 2**20

    peakmem_gib = peakmem_bytes / 2**30
    print(
        f"[Memory, {op}] {linop_str} on {problem_str} and {device_str}: {peakmem_gib:.2f} GiB"
    )

    savepath = benchpath(linop_str, problem_str, device_str, metric="peakmem").replace(
        ".json", f"_{op}.json"
    )
    with open(savepath, "w") as f_result:
        json.dump({"peakmem": peakmem_gib}, f_result)


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

    args = parser.parse_args()
    run_peakmem_benchmark(args.linop, args.problem, args.device, args.op)
