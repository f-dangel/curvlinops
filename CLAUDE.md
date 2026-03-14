# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**curvlinops** provides scipy-compatible linear operators for curvature matrices (GGN, Fisher, Hessian, Jacobian, KFAC) in PyTorch. The operators act on parameter space represented as `list[Tensor]` (one tensor per parameter), not flattened vectors. `model_func` must be a `Module` (not a generic `Callable`).

## Commands

```bash
# Install
make install          # pip install -e .
make install-test     # with test deps
make install-dev      # all dev deps

# Test
make test                              # full suite with coverage and doctests
pytest test/test_ggn.py -v             # single file
pytest test/test_ggn.py::test_name -v  # single test

# Lint
make lint             # format check + ruff check
make ruff-format      # auto-format
make ruff             # ruff fix (with --preview)
```

Ruff is the only linter (no black/isort/flake8). Config in `pyproject.toml`. Line length 88.

## Architecture

### Base Classes

- **`PyTorchLinearOperator`** (`_torch_base.py`): Core interface. Subclasses implement `_matmat()` and `_adjoint()`. Supports composition (`+`, `@`, `*`), export to scipy via `.to_scipy()`. Properties `device`/`dtype` are abstract.

- **`_EmpiricalRiskMixin`** (`_empirical_risk.py`): Mixin providing data iteration (`_loop_over_data`), normalization, deterministic checks, gradient computation. Stores `_model_func`, `_loss_func`, `_params` (a `dict[str, Parameter]` mapping full parameter names to tensors), `_data`. Concrete `device`/`dtype`.

- **`CurvatureLinearOperator(_EmpiricalRiskMixin, PyTorchLinearOperator)`**: Base for curvature operators (GGN, Hessian, Fisher). MRO: **mixin must come first** so its concrete `device`/`dtype` override the abstract stubs.

### Curvature Operators (inherit from CurvatureLinearOperator)

`GGNLinearOperator` (also supports MC-Fisher via `mc_samples > 0`), `HessianLinearOperator`, `EFLinearOperator`, `JacobianLinearOperator` — each implements `_matmat` for its specific matrix-vector product using `torch.func` (vjp, jvp, vmap, jacrev).

### Structural Operators (inherit from PyTorchLinearOperator)

`DiagonalLinearOperator`, `BlockDiagonalLinearOperator`, `KroneckerProductLinearOperator`, `EighDecomposedLinearOperator`, `SubmatrixLinearOperator`.

### Computer Pattern

Separates raw computation from the linear operator wrapper:
- `GGNDiagonalComputer` → computes `List[Tensor]` → wrapped by `GGNDiagonalLinearOperator(DiagonalLinearOperator)`
- `KFACComputer` → computes Kronecker factors → wrapped by `KFACLinearOperator`
- `EKFACComputer` → eigenvalue-corrected KFAC → wrapped by `EKFACLinearOperator`

Computers inherit from `_EmpiricalRiskMixin` (not `PyTorchLinearOperator`).

### Deterministic Check Chain

`_EmpiricalRiskMixin._check_deterministic()` → subclass overrides call `super()` first, then add checks. `CurvatureLinearOperator` additionally runs `_check_deterministic_matvec` after init. Helpers in `_checks.py`.

### KFAC Canonical Space

KFAC uses canonical space converters (`ToCanonicalLinearOperator` / `FromCanonicalLinearOperator`) to transform between parameter space and the block-diagonal Kronecker structure. These use `dict[str, Size]` for parameter shapes and `list[dict[str, str]]` for parameter groups (local name → full qualified name).

### Parameter Representation

Internally, `_params` is a `dict[str, Parameter]` keyed by full qualified name (e.g. `'0.weight'`). The public API accepts `list[Parameter]` (converted via `identify_free_parameters`). KFAC's `_mapping` is `dict[str, dict[str, str]]` mapping module names → {local param name → full param name}.

## Key Conventions

- **Public API** is defined by the `.rst` files under `docs/rtd/`. Only functions/classes documented there are considered public. All other functions are internal backend and can be changed freely.
- Tests use `float64` for numerical precision. GPU tests auto-detected via `test/utils.py::get_available_devices()`.
- Test cases defined in `test/cases.py` and `test/kfac_cases.py` as fixture classes.
- Docstrings follow numpy style (enforced by ruff D rules).
- `torch.func` (vmap, vjp, jvp, jacrev) is used throughout for automatic differentiation — not `torch.autograd` for new code.
