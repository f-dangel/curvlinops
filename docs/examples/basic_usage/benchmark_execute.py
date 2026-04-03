"""Benchmark execution engine for linear operator measurements.

Contains the :class:`Benchmark` class that handles time and memory measurements.
The ``__main__`` block serves as the CLI entry point for subprocess-based memory
measurements launched by :meth:`Benchmark.run_reference` and
:meth:`Benchmark.run_operator`.
"""

import json
import sys
from argparse import ArgumentParser
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from os import path, remove
from subprocess import CalledProcessError, CompletedProcess, run
from time import perf_counter

from benchmark_utils import (
    _IS_EKFAC,
    _IS_FX,
    _IS_KFAC_INVERSE_HOOKS,
    _KFAC_LIKE,
    LINOP_STRS,
    PROBLEM_STRS,
    attention_context,
    benchpath,
    reference_benchpath,
    setup_linop,
    setup_synthetic_cifar10_resnet18,
    setup_synthetic_imagenet_resnet50,
    setup_synthetic_mnist_mlp,
    setup_synthetic_shakespeare_nanogpt,
)
from memory_profiler import memory_usage
from torch import Tensor, cuda, device, manual_seed, rand
from torch.nn import Conv2d, Linear, Module

from curvlinops import KFACLinearOperator
from curvlinops.computers._base import _EKFACMixin
from curvlinops.computers.ekfac_hooks import HooksEKFACComputer
from curvlinops.computers.ekfac_make_fx import MakeFxEKFACComputer
from curvlinops.computers.kfac_hooks import HooksKFACComputer, _use_params
from curvlinops.computers.kfac_make_fx import MakeFxKFACComputer
from curvlinops.examples import gradient_and_loss


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


def _merge_json(filepath: str, key: str, value):
    """Read a JSON file, set ``key`` to ``value``, and write back.

    Creates the file if it does not exist. Preserves existing keys.

    Args:
        filepath: Path to the JSON file.
        key: The key to set.
        value: The value to assign.
    """
    existing = {}
    if path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)
    existing[key] = value
    with open(filepath, "w") as f:
        json.dump(existing, f)


class Benchmark:
    """High-level benchmark runner for linear operator measurements.

    Handles time and memory profiling for benchmark problems. Time measurements
    run in-process; memory measurements run in isolated subprocesses to avoid
    allocation artifacts from previous operations.

    The tutorial interacts with this class via :meth:`run_reference`,
    :meth:`run_operator`, :meth:`load_reference`, and :meth:`load_operator`.

    Args:
        problem_str: The problem name (e.g. ``"synthetic_mnist_mlp"``).
        device_str: The device string (e.g. ``"cpu"``, ``"cuda"``).
        skip_existing: Whether to skip measurements whose results already exist.
        num_repeats: Number of repeats for timing. Uses the minimum.
    """

    def __init__(
        self,
        problem_str: str,
        device_str: str,
        skip_existing: bool = True,
        num_repeats: int = 10,
    ):
        """Set up the benchmark.

        Args:
            problem_str: The problem name.
            device_str: The device string.
            skip_existing: Whether to skip measurements whose results exist.
            num_repeats: Number of repeats for timing. Uses the minimum.
        """
        self.problem_str = problem_str
        self.device_str = device_str
        self.is_cuda = "cuda" in device_str
        self.skip_existing = skip_existing
        self.num_repeats = num_repeats

    def setup_problem(self, linop_str: str):
        """Seed RNG, create device, and set up the problem.

        Args:
            linop_str: The linear operator that is investigated.

        Returns:
            The neural net, loss function, parameters, and data.
        """
        manual_seed(0)
        dev = device(self.device_str)

        setup_func = {
            "synthetic_mnist_mlp": setup_synthetic_mnist_mlp,
            "synthetic_cifar10_resnet18": setup_synthetic_cifar10_resnet18,
            "synthetic_imagenet_resnet50": setup_synthetic_imagenet_resnet50,
            "synthetic_shakespeare_nanogpt": setup_synthetic_shakespeare_nanogpt,
        }[self.problem_str]
        model, loss_function, data = setup_func()

        model = model.eval().to(dev)
        loss_function = loss_function.to(dev)

        if linop_str in _KFAC_LIKE:
            params = {}
            for mod_name, mod in model.named_modules():
                if not isinstance(mod, (Linear, Conv2d)):
                    continue
                if all(d <= 50_000 for d in mod.weight.shape):
                    for p_name, p in mod.named_parameters(recurse=False):
                        full_name = f"{mod_name}.{p_name}" if mod_name else p_name
                        if p.requires_grad:
                            params[full_name] = p
        else:
            params = {n: p for n, p in model.named_parameters() if p.requires_grad}

        return model, loss_function, params, data

    # -- High-level API --

    def run_reference(self):
        """Measure time and peak memory for ``gradient_and_loss``.

        Time is measured in-process. Memory is measured in an isolated subprocess.
        Results are saved to the reference benchmark JSON file.
        """
        self._run_reference_time()
        self._run_reference_memory()

    def run_operator(self, linop_str: str):
        """Measure matvec time, precompute sub-phases, and peak memory.

        Time is measured in-process. Memory is measured in an isolated subprocess.
        All results are saved to the operator's benchmark JSON file.

        Args:
            linop_str: The linear operator name.
        """
        self._run_operator_time(linop_str)
        self._run_operator_memory(linop_str)

    def load_reference(self) -> dict:
        """Load reference benchmark results.

        Returns:
            Dict with keys ``"time"`` and/or ``"peakmem"``.
        """
        with open(reference_benchpath(self.problem_str, self.device_str)) as f:
            return json.load(f)

    def load_operator(self, linop_str: str) -> dict:
        """Load operator benchmark results.

        Args:
            linop_str: The linear operator name.

        Returns:
            Dict with keys like ``"matvec"``, ``"kfac_factors"``, ``"peakmem"``, etc.
        """
        with open(benchpath(linop_str, self.problem_str, self.device_str)) as f:
            return json.load(f)

    # -- Low-level measurement --

    def time(self, func, context=None):
        """Time a function or pipeline of functions.

        For a single callable, returns ``(min_time, last_result)``.

        For a pipeline (list of tuples), the first callable takes no arguments;
        each subsequent callable receives the return value of the previous one.
        The repeat loop is the **outer** loop so that each repeat runs the full
        pipeline from scratch, avoiding issues with in-place operations on
        intermediate state. Returns ``(dict_of_min_times, last_result)``.

        Pipeline tuples are ``(name, callable)`` pairs. An optional ``context``
        callable can be passed that returns a context manager; the entire pipeline
        (all phases) runs inside it each repeat. A fresh context manager is
        created per repeat to support generator-based context managers.

        Args:
            func: A callable, or a list of ``(name, callable)`` pairs.
            context: Optional callable returning a context manager that wraps
                the full pipeline per repeat. Defaults to ``nullcontext``.

        Returns:
            ``(float, Any)`` for a single callable, or
            ``(dict[str, float], Any)`` for a pipeline.
        """
        single = callable(func)
        if single:
            func = [("_", func)]
        if context is None:
            context = nullcontext

        phase_times = {name: [] for name, _ in func}
        result = None
        for _ in range(self.num_repeats):
            state = None
            with context():
                for name, phase_fn in func:
                    if self.is_cuda:
                        cuda.synchronize()
                    start = perf_counter()
                    state = phase_fn() if state is None else phase_fn(state)
                    if self.is_cuda:
                        cuda.synchronize()
                    phase_times[name].append(perf_counter() - start)
            result = state

        min_times = {name: min(t) for name, t in phase_times.items()}
        return min_times["_"] if single else min_times, result

    def memory(self, func: Callable) -> float:
        """Measure peak memory of a function call in GiB.

        On CUDA, uses :func:`torch.cuda.max_memory_allocated`.
        On CPU, uses the ``memory_profiler`` package.

        Args:
            func: The function to measure.

        Returns:
            Peak memory in GiB.
        """
        if self.is_cuda:
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
            func()
            cuda.synchronize()
            return cuda.max_memory_allocated() / 2**30
        else:
            # memory_usage with max_usage=True returns peak MiB
            return memory_usage(func, interval=1e-4, max_usage=True) / 2**10

    # -- Internal: time measurements (in-process) --

    def _run_reference_time(self):
        """Time gradient_and_loss and save to the reference JSON file."""
        savepath = reference_benchpath(self.problem_str, self.device_str)
        label = f"Reference on {self.problem_str} and {self.device_str}"

        if self.skip_existing and path.exists(savepath):
            with open(savepath) as f:
                existing = json.load(f)
            if "time" in existing:
                print(f"[Time] Skipping {label}")
                return

        model, loss_function, params, data = self.setup_problem("Hessian")
        best, _ = self.time(
            lambda: gradient_and_loss(model, loss_function, params, data)
        )
        print(f"[Time] {label}: {best:.4f} s")

        _merge_json(savepath, "time", best)

    def _run_operator_time(self, linop_str: str):
        """Time matvec and precompute sub-phases, save to operator JSON."""
        savepath = benchpath(linop_str, self.problem_str, self.device_str)
        label = f"{linop_str} on {self.problem_str} and {self.device_str}"

        if self.skip_existing and path.exists(savepath):
            print(f"[Time] Skipping {label}")
            return

        model, loss_function, params, data = self.setup_problem(linop_str)
        # NOTE Disable deterministic check as it will otherwise compute matvecs
        linop = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )
        v = rand(linop.shape[1], device=linop.device)

        def _matvec():
            with attention_context(linop, model):
                _ = linop @ v

        matvec_time, _ = self.time(_matvec)
        results = {"matvec": matvec_time}
        print(f"[Time] {label} / matvec: {matvec_time:.4f} s")

        if linop_str in _KFAC_LIKE:
            phases, ctx = make_precompute_phases(
                linop_str, model, loss_function, params, data
            )
            phase_times, _ = self.time(phases, context=ctx)
            for phase_name, t in phase_times.items():
                results[phase_name] = t
                print(f"[Time] {label} / {phase_name}: {t:.4f} s")

        with open(savepath, "w") as f:
            json.dump(results, f)
        print(f"[Time] Saved {label}")

    # -- Internal: memory measurements (subprocess) --

    def _run_reference_memory(self):
        """Spawn subprocess to measure reference peak memory."""
        savepath = reference_benchpath(self.problem_str, self.device_str)
        label = f"reference on {self.problem_str} and {self.device_str}"

        if self.skip_existing and path.exists(savepath):
            with open(savepath) as f:
                existing = json.load(f)
            if "peakmem" in existing:
                print(f"[Memory] Skipping {label}")
                return

        self._run_subprocess("--reference")

    def _run_operator_memory(self, linop_str: str):
        """Spawn subprocess for operator peak memory, merge into operator JSON."""
        savepath = benchpath(linop_str, self.problem_str, self.device_str)
        label = f"{linop_str} on {self.problem_str}"

        if self.skip_existing and path.exists(savepath):
            with open(savepath) as f:
                existing = json.load(f)
            if "peakmem" in existing:
                print(f"[Memory] Skipping {label}")
                return

        mem_path = benchpath(
            linop_str, self.problem_str, self.device_str, op_str="peakmem"
        )
        self._run_subprocess(f"--linop={linop_str}")

        if path.exists(mem_path) and path.exists(savepath):
            with open(mem_path) as f:
                peakmem = json.load(f)["peakmem"]
            _merge_json(savepath, "peakmem", peakmem)
            remove(mem_path)

    def _run_subprocess(self, *extra_args: str):
        """Run benchmark_execute.py as a subprocess with given extra arguments."""
        cmd = [
            sys.executable,
            path.join(path.dirname(__file__), "benchmark_execute.py"),
            f"--problem={self.problem_str}",
            f"--device={self.device_str}",
            *extra_args,
        ]
        print(f"Running command: {' '.join(cmd)}")
        run_verbose(cmd)


# -- Helpers for precompute sub-phase timing --


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


def make_precompute_phases(  # noqa: C901
    linop_str: str,
    model: Module,
    loss_function: Module,
    params: dict[str, Tensor],
    data,
) -> tuple[list[tuple[str, callable]], callable | None]:
    """Build a pipeline of precompute sub-phases for timing.

    Returns a list of ``(name, callable)`` pairs. The first callable takes no
    arguments; each subsequent callable receives the return value of the
    previous one. This pipeline is passed to :meth:`Benchmark.time`.

    Args:
        linop_str: The linear operator name.
        model: The neural net.
        loss_function: The loss function.
        params: The parameters.
        data: The data.

    Returns:
        Tuple of ``(phases, context)`` where ``phases`` is a list of
        ``(name, callable)`` pairs forming a pipeline, and ``context`` is
        an optional callable returning a context manager that wraps the
        full pipeline (or ``None``).
    """
    num_data = sum(X.shape[0] for (X, _) in data)
    X0, y0 = next(iter(data))
    num_per_example_loss_terms = y0.numel() // X0.shape[0]
    common_kwargs = dict(
        check_deterministic=False,
        num_data=num_data,
        num_per_example_loss_terms=num_per_example_loss_terms,
        separate_weight_and_bias=False,
    )

    if linop_str in _IS_EKFAC and linop_str not in _IS_FX:
        # EKFAC hooks: factors → eigh → correction
        computer = setup_computer(linop_str, model, loss_function, params, data)

        def ekfac_factors():
            return computer._compute_kronecker_factors()

        def ekfac_eigh(state):
            input_cov, grad_cov, mapping = state
            input_cov = _EKFACMixin._eigenvectors_(input_cov)
            grad_cov = _EKFACMixin._eigenvectors_(grad_cov)
            return (input_cov, grad_cov, mapping)

        def ekfac_correction(state):
            input_cov, grad_cov, mapping = state
            computer.compute_eigenvalue_correction(input_cov, grad_cov, mapping)
            return (input_cov, grad_cov, mapping)

        phases = [
            ("kfac_factors", ekfac_factors),
            ("eigh", ekfac_eigh),
            ("eigenvalue_correction", ekfac_correction),
        ]

        def context():
            return _use_params(computer._model_module, computer._params)

        return phases, context

    if linop_str in _IS_KFAC_INVERSE_HOOKS:
        # KFAC inverse hooks: factors → Cholesky inverse
        def kfac_inv_factors():
            return KFACLinearOperator(
                model, loss_function, params, data, **common_kwargs
            )

        def cholesky_inverse(linop):
            return linop.inverse(damping=1e-3)

        return [
            ("kfac_factors", kfac_inv_factors),
            ("cholesky_inverse", cholesky_inverse),
        ], None

    if linop_str in _IS_FX and linop_str in _IS_EKFAC:
        # EKFAC FX: tracing → factors → eigh → correction
        computer = setup_computer(linop_str, model, loss_function, params, data)

        def ekfac_fx_tracing():
            return computer._trace_io_functions()

        def ekfac_fx_factors(traced_io):
            input_cov, grad_cov, mapping = computer._compute_kronecker_factors(
                traced_io
            )
            return (input_cov, grad_cov, mapping, traced_io)

        def ekfac_fx_eigh(state):
            input_cov, grad_cov, mapping, traced_io = state
            input_cov = _EKFACMixin._eigenvectors_(input_cov)
            grad_cov = _EKFACMixin._eigenvectors_(grad_cov)
            return (input_cov, grad_cov, mapping, traced_io)

        def ekfac_fx_correction(state):
            input_cov, grad_cov, mapping, traced_io = state
            computer.compute_eigenvalue_correction(
                input_cov, grad_cov, mapping, traced_io
            )
            return (input_cov, grad_cov, mapping)

        return [
            ("tracing", ekfac_fx_tracing),
            ("kfac_factors", ekfac_fx_factors),
            ("eigh", ekfac_fx_eigh),
            ("eigenvalue_correction", ekfac_fx_correction),
        ], None

    if linop_str in _IS_FX:
        # KFAC FX: tracing → factors
        computer = setup_computer(linop_str, model, loss_function, params, data)

        def kfac_fx_tracing():
            return computer._trace_io_functions()

        def kfac_fx_factors(traced_io):
            return computer._compute_kronecker_factors(traced_io)

        return [
            ("tracing", kfac_fx_tracing),
            ("kfac_factors", kfac_fx_factors),
        ], None

    # Plain KFAC hooks: single phase
    def kfac_factors():
        return KFACLinearOperator(model, loss_function, params, data, **common_kwargs)

    return [("kfac_factors", kfac_factors)], None


# -- Subprocess entry points for memory measurement --


def _run_reference_peakmem(problem_str: str, device_str: str):
    """Measure peak memory for gradient_and_loss (called in subprocess).

    Merges the result into the reference benchmark JSON file.

    Args:
        problem_str: The problem.
        device_str: The device.
    """
    bench = Benchmark(problem_str, device_str)

    def func():
        model, loss_function, params, data = bench.setup_problem("Hessian")
        _ = gradient_and_loss(model, loss_function, params, data)
        if bench.is_cuda:
            cuda.synchronize()

    peakmem_gib = bench.memory(func)
    print(
        f"[Memory] Reference gradient_and_loss on {problem_str} and {device_str}:"
        f" {peakmem_gib:.2f} GiB"
    )

    _merge_json(reference_benchpath(problem_str, device_str), "peakmem", peakmem_gib)


def _run_operator_peakmem(linop_str: str, problem_str: str, device_str: str):
    """Measure peak memory for the full pipeline (called in subprocess).

    Writes the result to a temporary peakmem JSON file that will be merged
    by the parent process.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
    """
    savepath = benchpath(linop_str, problem_str, device_str, op_str="peakmem")
    bench = Benchmark(problem_str, device_str)

    def func():
        model, loss_function, params, data = bench.setup_problem(linop_str)
        linop = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )
        v = rand(linop.shape[1], device=linop.device)
        with attention_context(linop, model):
            _ = linop @ v
        if bench.is_cuda:
            cuda.synchronize()

    peakmem_gib = bench.memory(func)
    print(
        f"[Memory] {linop_str} on {problem_str} and {device_str}: {peakmem_gib:.2f} GiB"
    )
    with open(savepath, "w") as f:
        json.dump({"peakmem": peakmem_gib}, f)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Run memory benchmark measurement in an isolated subprocess."
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
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="The device to benchmark.",
        required=True,
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Measure reference gradient_and_loss (ignores --linop).",
    )

    args = parser.parse_args()
    if args.reference:
        _run_reference_peakmem(args.problem, args.device)
    else:
        _run_operator_peakmem(args.linop, args.problem, args.device)
