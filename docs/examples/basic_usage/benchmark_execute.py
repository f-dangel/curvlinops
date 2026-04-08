"""Benchmark execution engine for linear operator measurements.

Contains the :class:`Benchmark` class that handles time and memory measurements.
The ``__main__`` block serves as the CLI entry point for subprocess-based memory
measurements launched by :meth:`Benchmark.run_reference` and
:meth:`Benchmark.run_operator`.

**Time** is measured in-process: minimum over ``num_repeats`` calls, with
``cuda.synchronize()`` before and after each call. Compiled measurements use
:meth:`_time_or_nan` which catches ``RuntimeError`` from ``torch.compile``
failures (e.g. FX-traced operators containing ``autograd.grad``).

**Peak memory** is measured in isolated subprocesses to avoid allocation
artifacts. For compiled measurements, a warmup call runs first (the 1st call
of a ``torch.compiled`` function traces in eager mode, not compiled), followed
by ``gc.collect()`` to free compilation reference cycles. The actual
measurement captures steady-state compiled memory.
"""

from __future__ import annotations

import gc
import json
import sys
from argparse import ArgumentParser
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from functools import partial
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
    _get_precompute_ops,
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
from torch import Tensor, cuda, device, manual_seed, nan, rand
from torch import compile as torch_compile
from torch.compiler import reset as reset_compiler
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


def _merge_json(filepath: str, category: str, key: str, value):
    """Read a JSON file, set ``data[category][key]`` to ``value``, and write back.

    Creates the file if it does not exist. Preserves existing keys.

    Args:
        filepath: Path to the JSON file.
        category: Top-level category (``"eager"`` or ``"compiled"``).
        key: The key to set inside the category.
        value: The value to assign.
    """
    existing = {}
    if path.exists(filepath):
        with open(filepath) as f:
            existing = json.load(f)
    existing.setdefault(category, {})[key] = value
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)


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
            Dict with keys ``"eager"`` and ``"compiled"``, each
            containing ``"time"`` and/or ``"peakmem"``.
        """
        with open(reference_benchpath(self.problem_str, self.device_str)) as f:
            return json.load(f)

    def load_operator(self, linop_str: str) -> dict:
        """Load operator benchmark results.

        Args:
            linop_str: The linear operator name.

        Returns:
            Dict with keys ``"eager"`` and optionally ``"compiled"``,
            each containing keys like ``"matvec"``, ``"peakmem"``, etc.
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

    # -- Internal: skip logic --

    @staticmethod
    def _read_result(savepath: str, category: str, key: str):
        """Read a specific result from a JSON file.

        Args:
            savepath: Path to the JSON results file.
            category: The category (``"eager"`` or ``"compiled"``).
            key: The measurement key (e.g. ``"matvec"``, ``"peakmem"``).

        Returns:
            The value if found, ``None`` otherwise.
        """
        if not path.exists(savepath):
            return None
        with open(savepath) as f:
            return json.load(f).get(category, {}).get(key)

    def _has_result(self, savepath: str, category: str, key: str):
        """Check if a result exists and should be skipped.

        Args:
            savepath: Path to the JSON results file.
            category: The category (``"eager"`` or ``"compiled"``).
            key: The measurement key (e.g. ``"matvec"``, ``"peakmem"``).

        Returns:
            The existing value if found and ``skip_existing`` is enabled,
            ``None`` otherwise.
        """
        if not self.skip_existing:
            return None
        return self._read_result(savepath, category, key)

    # -- Internal: time measurements (in-process) --

    def _time_or_nan(self, func, **kwargs):
        """Like :meth:`time`, but return NaN on ``RuntimeError``.

        Args:
            func: Passed to :meth:`time`.
            **kwargs: Passed to :meth:`time`.

        Returns:
            Same as :meth:`time`, but with NaN values on failure.
        """
        try:
            return self.time(func, **kwargs)
        except RuntimeError as e:
            print(f"  torch.compile RuntimeError: {e}")
            if callable(func):
                return float(nan), None
            return {name: float(nan) for name, _ in func}, None

    def _get_timer(self, compiled: bool):
        """Return :meth:`_time_or_nan` for compiled, :meth:`time` for eager.

        Args:
            compiled: Whether this is a compiled measurement.

        Returns:
            The timing callable.
        """
        return self._time_or_nan if compiled else self.time

    def _run_reference_time(self):
        """Time gradient_and_loss (eager + compiled) and save to reference JSON."""
        # Memory subprocesses between operators pollute the inductor filesystem
        # cache, causing torch.compile to silently fall back to eager speed.
        reset_compiler()
        savepath = reference_benchpath(self.problem_str, self.device_str)
        label = f"Reference on {self.problem_str} and {self.device_str}"
        model, loss_function, params, data = self.setup_problem("Hessian")

        for category, compiled in [("eager", False), ("compiled", True)]:
            existing = self._has_result(savepath, category, "time")
            if existing is not None:
                print(f"[Time] Skipping {label} ({category}): {existing:.4f} s")
                continue
            fn = torch_compile(gradient_and_loss) if compiled else gradient_and_loss
            timer = self._get_timer(compiled)
            best, _ = timer(partial(fn, model, loss_function, params, data))
            print(f"[Time] {label} ({category}): {best:.4f} s")
            _merge_json(savepath, category, "time", best)

    def _run_operator_time(self, linop_str: str):
        """Time matvec and precompute phases (eager + compiled), save to JSON."""
        # Memory subprocesses between operators pollute the inductor filesystem
        # cache, causing torch.compile to silently fall back to eager speed.
        reset_compiler()
        savepath = benchpath(linop_str, self.problem_str, self.device_str)
        label = f"{linop_str} on {self.problem_str} and {self.device_str}"

        model, loss_function, params, data = self.setup_problem(linop_str)
        linop = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )
        v = rand(linop.shape[1], device=linop.device)
        ctx = partial(attention_context, linop, model)

        for category, compiled in [("eager", False), ("compiled", True)]:
            existing = self._has_result(savepath, category, "matvec")
            if existing is not None:
                print(
                    f"[Time] Skipping {label} / matvec ({category}): {existing:.4f} s"
                )
            else:
                matvec_fn = (
                    torch_compile(lambda: linop @ v) if compiled else lambda: linop @ v
                )
                timer = self._get_timer(compiled)
                matvec_time, _ = timer(matvec_fn, context=ctx)
                _merge_json(savepath, category, "matvec", matvec_time)
                print(f"[Time] {label} / matvec ({category}): {matvec_time:.4f} s")

            if linop_str in _KFAC_LIKE:
                expected_ops = _get_precompute_ops(linop_str)
                if self.skip_existing and path.exists(savepath):
                    with open(savepath) as f:
                        existing_cat = json.load(f).get(category, {})
                    all_exist = all(op in existing_cat for op in expected_ops)
                else:
                    all_exist = False
                if all_exist:
                    print(f"[Time] Skipping {label} / precompute ({category})")
                else:
                    phases, phase_ctx = make_precompute_phases(
                        linop_str, model, loss_function, params, data
                    )
                    if compiled:
                        phases = [(name, torch_compile(fn)) for name, fn in phases]
                    timer = self._get_timer(compiled)
                    phase_times, _ = timer(phases, context=phase_ctx)
                    for phase_name, t in phase_times.items():
                        _merge_json(savepath, category, phase_name, t)
                        print(f"[Time] {label} / {phase_name} ({category}): {t:.4f} s")

    # -- Internal: memory measurements (subprocess) --

    def _run_reference_memory(self):
        """Spawn subprocess to measure reference peak memory."""
        savepath = reference_benchpath(self.problem_str, self.device_str)
        label = f"reference on {self.problem_str} and {self.device_str}"

        for category, compiled in [("eager", False), ("compiled", True)]:
            existing = self._has_result(savepath, category, "peakmem")
            if existing is not None:
                print(f"[Memory] Skipping {label} ({category}): {existing:.2f} GiB")
                continue
            self._try_subprocess("--reference", *self._compiled_flag(compiled))
            if self._read_result(savepath, category, "peakmem") is None:
                print(f"[Memory] FAILED {label} ({category}), storing NaN")
                _merge_json(savepath, category, "peakmem", float(nan))

    def _run_operator_memory(self, linop_str: str):
        """Spawn subprocess for operator peak memory, merge into operator JSON."""
        savepath = benchpath(linop_str, self.problem_str, self.device_str)
        label = f"{linop_str} on {self.problem_str}"

        for category, compiled in [("eager", False), ("compiled", True)]:
            existing = self._has_result(savepath, category, "peakmem")
            if existing is not None:
                print(f"[Memory] Skipping {label} ({category}): {existing:.2f} GiB")
                continue
            mem_path = benchpath(
                linop_str, self.problem_str, self.device_str, op_str="peakmem"
            )
            self._try_subprocess(f"--linop={linop_str}", *self._compiled_flag(compiled))
            if path.exists(mem_path) and path.exists(savepath):
                with open(mem_path) as f:
                    mem_data = json.load(f)
                for cat, sub in mem_data.items():
                    for key, value in sub.items():
                        _merge_json(savepath, cat, key, value)
                remove(mem_path)
            if self._read_result(savepath, category, "peakmem") is None:
                print(f"[Memory] FAILED {label} ({category}), storing NaN")
                _merge_json(savepath, category, "peakmem", float(nan))

    @staticmethod
    def _compiled_flag(compiled: bool) -> tuple[str, ...]:
        """Return CLI flags for the subprocess compiled mode.

        Returns:
            ``("--compiled",)`` if compiled, else ``()``.
        """
        return ("--compiled",) if compiled else ()

    def _try_subprocess(self, *extra_args: str):
        """Run benchmark_execute.py as a subprocess, suppressing failures.

        Args:
            *extra_args: Extra CLI arguments.
        """
        cmd = [
            sys.executable,
            path.join(path.dirname(__file__), "benchmark_execute.py"),
            f"--problem={self.problem_str}",
            f"--device={self.device_str}",
            *extra_args,
        ]
        print(f"Running command: {' '.join(cmd)}")
        try:
            run_verbose(cmd)
        except CalledProcessError:
            pass


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

        def ekfac_eigh(state):
            input_cov, grad_cov, mapping = state
            input_cov = _EKFACMixin._eigenvectors_(input_cov)
            grad_cov = _EKFACMixin._eigenvectors_(grad_cov)
            return (input_cov, grad_cov, mapping)

        phases = [
            ("kfac_factors", computer._compute_kronecker_factors),
            ("eigh", ekfac_eigh),
            (
                "eigenvalue_correction",
                lambda state: computer.compute_eigenvalue_correction(*state),
            ),
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
            traced_batch = computer._trace_batch_functions()
            (
                inputs_and_grad_outputs_batch_fns,
                _,
                io_groups,
                io_param_names,
                layer_hparams,
            ) = computer._trace_io_batch_functions()
            return (
                traced_batch,
                inputs_and_grad_outputs_batch_fns,
                io_groups,
                io_param_names,
                layer_hparams,
            )

        def ekfac_fx_factors(state):
            traced_batch, *io_state = state
            input_cov, grad_cov, mapping = computer._compute_kronecker_factors(
                traced_batch
            )
            return (input_cov, grad_cov, mapping, *io_state)

        def ekfac_fx_eigh(state):
            input_cov, grad_cov, mapping, *io_state = state
            input_cov = _EKFACMixin._eigenvectors_(input_cov)
            grad_cov = _EKFACMixin._eigenvectors_(grad_cov)
            return (input_cov, grad_cov, mapping, *io_state)

        def ekfac_fx_correction(state):
            (
                input_cov,
                grad_cov,
                mapping,
                inputs_and_grad_outputs_batch_fns,
                io_groups,
                io_pnames,
                lhp,
            ) = state
            return computer.compute_eigenvalue_correction(
                input_cov,
                grad_cov,
                mapping,
                inputs_and_grad_outputs_batch_fns,
                io_groups,
                io_pnames,
                lhp,
            )

        return [
            ("tracing", ekfac_fx_tracing),
            ("kfac_factors", ekfac_fx_factors),
            ("eigh", ekfac_fx_eigh),
            ("eigenvalue_correction", ekfac_fx_correction),
        ], None

    if linop_str in _IS_FX:
        # KFAC FX: tracing → factors
        computer = setup_computer(linop_str, model, loss_function, params, data)
        return [
            ("tracing", computer._trace_batch_functions),
            ("kfac_factors", computer._compute_kronecker_factors),
        ], None

    # Plain KFAC hooks: single phase
    def kfac_factors():
        return KFACLinearOperator(model, loss_function, params, data, **common_kwargs)

    return [("kfac_factors", kfac_factors)], None


# -- Subprocess entry points for memory measurement --


def _run_reference_peakmem(problem_str: str, device_str: str, compiled: bool):
    """Measure peak memory for gradient_and_loss (called in subprocess).

    Args:
        problem_str: The problem.
        device_str: The device.
        compiled: Whether to measure with ``torch.compile``.
    """
    bench = Benchmark(problem_str, device_str)
    savepath = reference_benchpath(problem_str, device_str)
    category = "compiled" if compiled else "eager"

    def func():
        model, loss_function, params, data = bench.setup_problem("Hessian")
        gradient_and_loss(model, loss_function, params, data)

    if compiled:
        func = torch_compile(func)
        # Warmup: the 1st call traces in eager (not compiled), and leaves
        # reference cycles from compilation. Run + gc to get steady state.
        func()
        gc.collect()

    def func_and_sync():
        func()
        if bench.is_cuda:
            cuda.synchronize()

    peakmem_gib = bench.memory(func_and_sync)
    print(
        f"[Memory] Reference gradient_and_loss ({category}) on {problem_str}"
        f" and {device_str}: {peakmem_gib:.2f} GiB"
    )
    _merge_json(savepath, category, "peakmem", peakmem_gib)


def _run_operator_peakmem(
    linop_str: str, problem_str: str, device_str: str, compiled: bool
):
    """Measure peak memory for a single matvec variant (called in subprocess).

    Writes results to a temporary peakmem JSON file that will be merged
    by the parent process.

    Args:
        linop_str: The linear operator.
        problem_str: The problem.
        device_str: The device.
        compiled: Whether to measure with ``torch.compile``.
    """
    savepath = benchpath(linop_str, problem_str, device_str, op_str="peakmem")
    bench = Benchmark(problem_str, device_str)
    category = "compiled" if compiled else "eager"

    def func():
        model, loss_function, params, data = bench.setup_problem(linop_str)
        linop = setup_linop(
            linop_str, model, loss_function, params, data, check_deterministic=False
        )
        v = rand(linop.shape[1], device=linop.device)
        with attention_context(linop, model):
            _ = linop @ v

    if compiled:
        func = torch_compile(func)
        # Warmup: the 1st call traces in eager (not compiled), and leaves
        # reference cycles from compilation. Run + gc to get steady state.
        func()
        gc.collect()

    def func_and_sync():
        func()
        if bench.is_cuda:
            cuda.synchronize()

    peakmem_gib = bench.memory(func_and_sync)
    print(
        f"[Memory] {linop_str} ({category}) on {problem_str}"
        f" and {device_str}: {peakmem_gib:.2f} GiB"
    )
    results = {category: {"peakmem": peakmem_gib}}
    with open(savepath, "w") as f:
        json.dump(results, f, indent=2)


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
    parser.add_argument(
        "--compiled",
        action="store_true",
        help="Measure with torch.compile (default: eager).",
    )

    args = parser.parse_args()
    if args.reference:
        _run_reference_peakmem(args.problem, args.device, args.compiled)
    else:
        _run_operator_peakmem(args.linop, args.problem, args.device, args.compiled)
