# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Migration guide: `params` is now `dict[str, Tensor]`

All operators now require `params` as a `dict[str, Tensor]` instead of
`list[Parameter]`. This affects every call site. Here is how to update your code:

```python
# Before
params = [p for p in model.parameters() if p.requires_grad]
H = HessianLinearOperator(model, loss, params, data)

# After
params = {n: p for n, p in model.named_parameters() if p.requires_grad}
H = HessianLinearOperator(model, loss, params, data)
```

See [PR #283](https://github.com/f-dangel/curvlinops/pull/283) for details.

### Added/New

- Add `KFOCLinearOperator` — Frobenius-optimal rank-one Kronecker
  approximation of each per-layer GGN block, obtained via the top singular
  pair of the block's Van Loan rearrangement (bias-only blocks store the
  exact bias GGN). Subclasses `KFACLinearOperator` and reuses the inherited
  matvec/inverse/eigh machinery; only the factor-computation step is
  replaced. Scope: single-batch data, `FisherType.TYPE2`.
  References: Schnaus, Lee, Triebel (BDL@NeurIPS 2021); Koroko et al. (arXiv:2201.10285)
  ([PR](https://github.com/f-dangel/curvlinops/pull/299))

- Add preconditioner support to `NeumannInverseLinearOperator` via a new
  `preconditioner` argument, enabling the preconditioned Neumann/Richardson
  iteration `A⁻¹ ≈ α Σₖ (I - α P A)ᵏ P` (inspired by Wang et al., NeurIPS 2025).
  Also document and add an example for the existing `preconditioner` option of
  `CGInverseLinearOperator`'s `cg_hyperparameters`

- Support plain callable `(params_dict, X) -> prediction` as `model_func`
  (with `params` as `dict[str, Tensor]`):
  - `HessianLinearOperator` ([PR](https://github.com/f-dangel/curvlinops/pull/275))
  - `GGNLinearOperator` ([PR](https://github.com/f-dangel/curvlinops/pull/277))
  - `EFLinearOperator` ([PR](https://github.com/f-dangel/curvlinops/pull/278))
  - `GGNDiagonalLinearOperator` ([PR](https://github.com/f-dangel/curvlinops/pull/279))
  - `JacobianLinearOperator` and `TransposedJacobianLinearOperator`
    ([PR](https://github.com/f-dangel/curvlinops/pull/280))
  - `KFACLinearOperator` and `EKFACLinearOperator` with `backend="make_fx"`
    ([PR](https://github.com/f-dangel/curvlinops/pull/271))

- **Backward-incompatible:** Add MC-sampling option (`mc_samples`) to
  `GGNLinearOperator` as replacement for the Fisher, remove `FisherMCLinearOperator`
  and the `mode` parameter from `GGNDiagonalLinearOperator`/`GGNDiagonalComputer`.
  Use `mc_samples=0` (default) for the exact GGN, positive values for MC approximation
  ([PR](https://github.com/f-dangel/curvlinops/pull/255))

- Add a linear operator for the exact or Monte-Carlo-approximated GGN diagonal
  ([PR](https://github.com/f-dangel/curvlinops/pull/241))

- Support left multiplication with linear operators (`X @ A` with `X` a tensor or tensor list)
  ([PR](https://github.com/f-dangel/curvlinops/pull/226))

- Support division of linear operators by scalars (i.e. `A_scaled = A / scale`)
  ([PR](https://github.com/f-dangel/curvlinops/pull/237))

- **Backward-incompatible:** (E)KFAC's `det, logdet, trace, frobenius_norm` properties are now functions
  ([PR](https://github.com/f-dangel/curvlinops/pull/232))

- **Backward-incompatible:** Reduce side effects in (E)KFAC's computation, modifying the entries of `state_dict`
  ([PR](https://github.com/f-dangel/curvlinops/pull/228))

- **Backward-incompatible:** Remove `KFACInverseLinearOperator`, replace with `(E)KFACLinearOperator.inverse()`
  ([PR](https://github.com/f-dangel/curvlinops/pull/244))

- Support non-binary (soft) labels in `[0, 1]` for `BCEWithLogitsLoss` across all
  operators (GGN, Fisher, KFAC, diagonal). The loss Hessian `diag(σ(f)·(1-σ(f)))` is
  independent of targets, so the exact and MC GGN are valid for any target in `[0, 1]`
  ([PR](https://github.com/f-dangel/curvlinops/pull/257))

### Fixed/Removed

- **Backward-incompatible:** Migrate `params` from `list[Parameter]` to
  `dict[str, Tensor]` across all operators. Pass
  ``dict(model.named_parameters())`` instead of ``list(model.parameters())``.
  Remove `SUPPORTS_FUNCTIONAL` flag and `identify_free_parameters` from init
  ([PR](https://github.com/f-dangel/curvlinops/pull/283))

- **Backward-incompatible:** Remove block-diagonal Hessian support (`block_sizes`
  parameter from `HessianLinearOperator` and `CurvatureLinearOperator`)
  ([PR](https://github.com/f-dangel/curvlinops/pull/267))

- **Backward-incompatible:** Remove `curvlinops.experimental` module
  (`ActivationHessianLinearOperator`)
  ([PR](https://github.com/f-dangel/curvlinops/pull/268))

- **Backward-incompatible:** Remove `(E)KFACLinearOperator`'s `state_dict` and `from_state_dict` methods, use `torch.save(K, path)` and `torch.load(path)` instead
  ([PR](https://github.com/f-dangel/curvlinops/pull/249))

- Remove `_check_binary_if_BCEWithLogitsLoss` safeguard; BCE targets are no longer
  restricted to binary values
  ([PR](https://github.com/f-dangel/curvlinops/pull/257))

### Internal

- Introduce `LayerIO` / `LayerIOSnapshot` orchestration layer in
  `curvlinops/computers/io_collector/layer_io.py`, a setup-once owner of
  shape-independent IO-collector metadata (parameter groups, IO-layer
  mappings) plus a per-shape cache of FX-traced `io_fn`s. Snapshots expose
  on-demand per-group accessors at three granularities (`raw`,
  `standardized_io`, `per_sample_grads`) so structural-GGN approximators
  can consume per-batch IO without re-deriving the plumbing. `trace_context`
  context manager wraps `_enable_requires_grad` so callers don't have to
  import the autograd-ownership helper directly.

  Migrate `MakeFxKFACComputer` to use `LayerIO`; it bootstraps one `LayerIO`
  per `compute()` call and reuses it across all batch sizes (today's
  `make_compute_kfac_batch` redoes IO-collector setup per shape). Move the
  pure post-processing helpers (`_build_param_groups_from_io`, `_bias_pad`,
  `make_group_gatherers`) from `kfac_make_fx.py` to a new
  `io_collector/groups.py`; re-exported from `kfac_make_fx.py` for
  backward compatibility with `EKFAC` and `KFOC` (migrated in follow-up PRs)

- Scope the FX backends' `requires_grad` mutation to tracing only.
  `MakeFxKFACComputer` / `MakeFxKFOCComputer` previously flipped
  `requires_grad=True` on every tensor in the user's `params` dict at
  ``__init__`` with no restore (a silent side effect on user-owned
  tensors, e.g., re-enabling gradient tracking on a frozen layer).
  Wrap the `make_fx` call sites with the existing save/restore
  `_enable_requires_grad` context manager (now in `utils.py`) so any
  prior `requires_grad` state is preserved after `compute()` returns.
  KFOC's `compute()` now traces its IO getter and replays under
  `no_grad()` to keep the autograd-using portion contained inside an
  FX graph

- Add `intermediate_as_batch` flag to the FX backend's
  `make_compute_kfac_io_batch` (opt-in unflattened IO — with
  `FisherType.TYPE2`, the collector output directly reconstructs the exact
  per-layer GGN block). Also fold the FX backend's reduction scaling into a
  single `1/sqrt(effective_batch)` pre-multiply on `grad_outputs`, removing
  the compensating `mul_(N)` calls in KFAC-fx's `compute_batch` and
  EKFAC-fx's eigcorr. `KFACLinearOperator`/`EKFACLinearOperator` behavior
  unchanged ([PR](https://github.com/f-dangel/curvlinops/pull/295))

- Benchmark tutorial: shorten plot labels (GGN, MC Fisher, ^{-1}), nest JSON
  data under `eager`/`compiled` keys, merge compiled overlay into main plots
  (5 plot types down to 3), remove stale compiled peakmem for (E)KFAC
  ([PR #291](https://github.com/f-dangel/curvlinops/pull/291))

- ``torch.compile`` support for matvecs:
  - `HessianLinearOperator`: replace `cached_property` with eager `_init_mp()`
    in `CurvatureLinearOperator` and all subclasses.
    ([PR #286](https://github.com/f-dangel/curvlinops/pull/286))
  - `GGNLinearOperator` (exact and MC): replace `torch.Generator` with global
    RNG seeded via `fork_rng` + `manual_seed` in `_matmat`.
    ([PR #287](https://github.com/f-dangel/curvlinops/pull/287))
  - `EFLinearOperator`: no code changes needed (already compile-friendly after
    PR #286). Add test and benchmark.
    ([PR #288](https://github.com/f-dangel/curvlinops/pull/288))
  - `KFACLinearOperator` and `EKFACLinearOperator`: replace `einops.einsum`
    with `torch.einsum` in `KroneckerProductLinearOperator`, call `_matmat`
    instead of `@` in `BlockDiagonalLinearOperator`, add `_adjoint_matmat`
    to avoid operator instantiation during tracing, remove numpy from
    `split_list`.
    ([PR #290](https://github.com/f-dangel/curvlinops/pull/290))
  - `KFACLinearOperator`/`EKFACLinearOperator` (`make_fx` backend) precompute:
    trace the entire per-batch computation (IO collection, backward pass,
    covariance einsums; for EKFAC also the eigenvalue correction) into single
    FX graphs via new `make_compute_kfac_batch` and
    `make_compute_ekfac_eigencorrection_batch` factories, splitting tracing
    from accumulation. Switch IO-collector tracing to fake-tensor mode to
    reduce tracing memory. Isolate global RNG state during factor accumulation
    via a new `fork_rng_with_seed` utility and only seed for `FisherType.MC`
    (previously the FX backends mutated the caller's global RNG as a side
    effect of `manual_seed`). Replace `einops` reduce/rearrange with native
    tensor ops for compile stability. Extend compile tests to cover KFAC/EKFAC
    precompute and multi-batch-size datasets.
    ([PR #292](https://github.com/f-dangel/curvlinops/pull/292))
  - Fix compiled peak memory benchmark measuring only the matvec slice instead
    of the full pipeline (setup + compilation + matvec)
    ([PR #289](https://github.com/f-dangel/curvlinops/pull/289))
  - Benchmark: always measure compiled performance (matvec, precompute phases,
    peak memory) for all operators. Remove `_IS_COMPILABLE` gate and
    `make_compiled_gradient_and_loss` (use `torch.compile(gradient_and_loss)`).
    ([PR #293](https://github.com/f-dangel/curvlinops/pull/293))

- Restructure benchmark tutorial: add both backends (hooks, `make_fx`) for
  KFAC/EKFAC, break down precompute into sub-phases (Kronecker factors,
  Eigen-decomposition, Eigen-correction, Cholesky inverse, FX tracing),
  show all plots in sphinx-gallery, and add GPU results gallery for all problems.
  Fix `gradient_and_loss` using ~2x peak GPU memory (now uses `torch.autograd.grad`
  instead of `torch.func.grad_and_value`)
  ([PR #284](https://github.com/f-dangel/curvlinops/pull/284))

- Add test documenting `torch.compile` graph breaks for `HessianLinearOperator`
  ([PR #285](https://github.com/f-dangel/curvlinops/pull/285))

- Add `_use_params` context manager to temporarily set module parameters
  during hooks-based KFAC/EKFAC computation, enabling correct behavior
  when `params` dict values differ from the module's own parameters
  ([PR](https://github.com/f-dangel/curvlinops/pull/282))

- Update `functorch_gradient_and_loss` to accept callable model_func.
  Add `to_functional` test helper and parametrized functional tests for
  all operators
  ([PR](https://github.com/f-dangel/curvlinops/pull/281))

- Refactor `_data_prediction_loss_gradient` to use `torch.func.grad_and_value`
  instead of `torch.autograd.grad`, removing the `requires_grad` requirement
  on params for callable model functions
  ([PR](https://github.com/f-dangel/curvlinops/pull/276))

- Simplify `make_grad_output_fn` to accept `FisherType` directly instead
  of mode strings. Remove mode-mapping dicts from callers
  ([PR](https://github.com/f-dangel/curvlinops/pull/273))

- Extract `_BaseKFACComputer` and `_EKFACMixin` to eliminate diamond
  inheritance. Rename `KFACComputer` → `HooksKFACComputer`,
  `EKFACComputer` → `HooksEKFACComputer`. Move base class to
  `computers/_base.py`, hooks to `kfac_hooks.py`/`ekfac_hooks.py`
  ([PR](https://github.com/f-dangel/curvlinops/pull/272))

- Pass `_model_func` callable to `make_batch_*` functions instead of
  `Module`, removing redundant `make_functional_call` wrapping. Align
  IO collector to `(params, x)` convention
  ([PR](https://github.com/f-dangel/curvlinops/pull/270))

- Rename internal `_model_func` to `_model_module`. Add functional
  `_model_func` with signature `(params_dict, X) -> prediction` via
  `make_functional_call`, used for predictions and shape inference
  ([PR](https://github.com/f-dangel/curvlinops/pull/269))

- Simplify `make_functional_call`: drop redundant frozen parameter capture
  (`functional_call` already falls back to module state). Split
  `make_functional_model_and_loss` into `make_functional_call` and
  `make_functional_loss`. Remove unused `param_names` arguments from
  `make_batch_*` functions
  ([PR](https://github.com/f-dangel/curvlinops/pull/266))

- Introduce parameter groups for KFAC/EKFAC. Covariance dicts keyed by
  tuples of parameter names instead of synthetic layer names. Support
  weight tying and mixed-bias configurations in the make_fx backend.
  Remove `ParameterUsage` dataclass in favor of plain `dict[str, str]`
  ([PR](https://github.com/f-dangel/curvlinops/pull/265),
  [PR](https://github.com/f-dangel/curvlinops/pull/264),
  [PR](https://github.com/f-dangel/curvlinops/pull/263))

- Migrate internal parameter representation from `list[Parameter]` to
  `dict[str, Parameter]` (keyed by fully-qualified name). Public API unchanged
  (`list[Parameter]` still accepted). Tighten `model_func` type from `Callable`
  to `Module`. KFAC `_mapping` values change from `int` positions to `str` names
  ([PR](https://github.com/f-dangel/curvlinops/pull/262))

- Reduce KFAC/EKFAC test suite from ~6,100 to ~1,100 tests by consolidating
  `exclude`/`shuffle`/`separate_weight_and_bias` parametrization into the four
  type-2 exactness tests (kfac/ekfac × standard/weight_sharing) and shrinking
  Conv2d spatial dimensions in weight-sharing test cases
  ([PR](https://github.com/f-dangel/curvlinops/pull/260))

- Move shared GGN utilities (`loss_hessian_matrix_sqrt`, `make_grad_output_fn`,
  `_make_single_datum_sampler`) from
  `kfac_utils.py` to new `ggn_utils.py`; `kfac_utils.py` now only contains
  KFAC-specific code (patch extraction, canonical space converters)
  ([PR](https://github.com/f-dangel/curvlinops/pull/255))

- Generalize IO collector to handle linear layers with >2D inputs
  ([PR](https://github.com/f-dangel/curvlinops/pull/259))

- Add a collector for in/outputs of linear weight sharing layers based on `make_fx`
  ([PR](https://github.com/f-dangel/curvlinops/pull/252))

- Add KFAC-specific IO collector (`with_kfac_io`) that wraps the generic collector
  with validation and structured output for KFAC computation
  ([PR](https://github.com/f-dangel/curvlinops/pull/253))

- Add `make_fx` backend for `KFACLinearOperator` via `MakeFxKFACComputer`, which
  computes Kronecker factors using FX graph tracing instead of hooks. Selectable
  via `backend="make_fx"` parameter
  ([PR](https://github.com/f-dangel/curvlinops/pull/258))

- Add `make_fx` backend for `EKFACLinearOperator` via `MakeFxEKFACComputer`,
  which computes eigenvalue-corrected Kronecker factors using FX graph tracing
  instead of hooks. Selectable via `backend="make_fx"` parameter
  ([PR](https://github.com/f-dangel/curvlinops/pull/261))

- Import `FisherType` and `KFACType` directly from `curvlinops.kfac_utils`
  instead of `curvlinops.kfac`, reducing coupling between the linear operator
  and utility modules
  ([PR](https://github.com/f-dangel/curvlinops/pull/253))

- Unify KFAC's gradient output computation for all Fisher types (`TYPE2`, `MC`,
  `EMPIRICAL`, `FORWARD_ONLY`) via `make_grad_output_fn` in `kfac_utils.py`,
  removing `_maybe_adjust_loss_scale`
  ([PR](https://github.com/f-dangel/curvlinops/pull/251))

- Modernize type annotations
  ([PR](https://github.com/f-dangel/curvlinops/pull/250))

- Introduce computer classes for KFAC and EKFAC in a `computer` submodule (move GGN diagonal computer, too).
  Computers compute Kronecker factors and eigencorrections.
  The linear operators handle assembling them into linear operators.
  ([PR](https://github.com/f-dangel/curvlinops/pull/249))

- Cache `.pytest_cache` in CI and run tests with `--ff` (failed-first) for faster feedback cycles
  ([PR](https://github.com/f-dangel/curvlinops/pull/248))

- Add sequence protocol (`__iter__`, `__len__`, `__getitem__`, `__setitem__`) to
  `BlockDiagonalLinearOperator`, `KroneckerProductLinearOperator`, and
  `_ChainPyTorchLinearOperator`; add `eigenvalues` property to
  `EighDecomposedLinearOperator`; extract reusable shape/device/dtype validation helpers
  ([PR](https://github.com/f-dangel/curvlinops/pull/247))

- Merge tests for testing matrix-matrix & matrix-vector products
  with the original and transposed operator, reducing number of tests
  ([PR](https://github.com/f-dangel/curvlinops/pull/222))

- Merge GitHub actions for linting and formatting with `ruff`
  ([PR](https://github.com/f-dangel/curvlinops/pull/225))

- Execute many tests in `float64`, allowing to lower many tolerances
  ([PR](https://github.com/f-dangel/curvlinops/pull/224))

- Add linear operators for basic mathematical structures
  - `BlockDiagonalLinearOperator` for matrices `block_diag(B_1, B_2, ...)`
    ([PR](https://github.com/f-dangel/curvlinops/pull/212))
  - `KroneckerProductLinearOperator` for matrices `S_1 ⊗ S_2 ⊗ ...`
    ([PR](https://github.com/f-dangel/curvlinops/pull/211))
  - `EighDecomposedLinearOperator` for `eigh`-decomposed matrices `Q diag(λ) Q^T` with orthogonal `Q`
    ([PR](https://github.com/f-dangel/curvlinops/pull/210))
  - `DiagonalLinearOperator` for diagonal matrices `diag(λ)`
    ([PR](https://github.com/f-dangel/curvlinops/pull/238))

- Major simplification of `KFACLinearOperator` and `EKFACLinearOperator`
  - Reduce side effects in (E)KFAC's computation and reduce caching logic
    ([PR1](https://github.com/f-dangel/curvlinops/pull/227) (**backward-incompatible**),
     [PR2](https://github.com/f-dangel/curvlinops/pull/228))
  - Introduce canonicalization operators that convert from parameter space (determined by order of parameters) to KFAC's canonical space (block-diagonal Kronecker-factored matrix) and back
    ([PR](https://github.com/f-dangel/curvlinops/pull/229))
  - Implement (E)KFAC using structured operators, i.e. `P @ K @ PT` with `P, PT` converters to and back from the canonical basis and `K` block-diagonal Kronecker-factored. Modifies the entries of `state_dict`.
    ([PR](https://github.com/f-dangel/curvlinops/pull/230)) (**backward-incompatible**)

- Generalize computing the loss function's Hessian square root for sequence-valued predictions
  ([PR](https://github.com/f-dangel/curvlinops/pull/231))

- Use `ruff` for docstring linting (remove `darglint` and `pydocstyle`)
  ([PR](https://github.com/f-dangel/curvlinops/pull/234))

- Introduce `_EmpiricalRiskMixin` interface to allow separating deterministic checks and state pre-computation from linear operators
  ([PR](https://github.com/f-dangel/curvlinops/pull/236))

- Generalize sampling output gradients for sequence-valued predictions
  ([PR](https://github.com/f-dangel/curvlinops/pull/235))

- Introduce a computer class for the GGN diagonal
  ([PR](https://github.com/f-dangel/curvlinops/pull/240))

- **Backward-incompatible:** Remove `CurvatureLinearOperator.gradient_and_loss()` and provide
  `curvlinops.examples.gradient_and_loss` as the replacement utility function
  ([PR](https://github.com/f-dangel/curvlinops/pull/245))

- Make `_ChainPyTorchLinearOperator` support more than two operators
  ([PR](https://github.com/f-dangel/curvlinops/pull/246))

## [3.0.1] - 2026-01-14

This patch provides major performance improvements for all curvature matrices (by using `torch.func`, and by improving EKFAC's eigenvalue correction) and deprecates Python 3.9 (we now require at least 3.10).

### Added/New

### Fixed/Removed

### Internal

- Add EKFAC and a new problem (ResNet18 on CIFAR10) to benchmark
  ([PR](https://github.com/f-dangel/curvlinops/pull/214)),
  expose a run time and memory inefficiency in EKFAC
  ([issue](https://github.com/f-dangel/curvlinops/issues/193)).
  This issue is successfully resolved with:

  - Improve performance of EKFAC by merging gradient computation and basis rotation into a single `einsum`
  ([PR](https://github.com/f-dangel/curvlinops/pull/215))

  - Improve performance of EKFAC by efficiently taking the square and sum over the batch dimension in the absence of weight sharing
  ([PR](https://github.com/f-dangel/curvlinops/pull/216))

  - Generalize trick for no weight sharing to mild weight sharing, further improve memory performance of EKFAC
  ([PR](https://github.com/f-dangel/curvlinops/pull/219))

- Minor fixes in docs, code and examples to reduce CI pipeline errors ([PR](https://github.com/f-dangel/curvlinops/pull/218))

- Updated supported Python version from 3.9 (deprecated) to 3.10 ([PR](https://github.com/f-dangel/curvlinops/pull/213))

- Use A6000 GPU instead of A40 to benchmark linear operators
  ([PR](https://github.com/f-dangel/curvlinops/pull/199))

- Use `torch.func` instead of BackPACK (`torch.autograd`) for:
  - GGN-vector products
    ([PR](https://github.com/f-dangel/curvlinops/pull/200))
  - Hessian-vector products
    ([PR](https://github.com/f-dangel/curvlinops/pull/201))
  - Empirical Fisher-vector products
    ([PR](https://github.com/f-dangel/curvlinops/pull/202))
  - MC-Fisher-vector products
    ([PR](https://github.com/f-dangel/curvlinops/pull/204))
  - (Transpose) Jacobian-vector products
    ([PR](https://github.com/f-dangel/curvlinops/pull/206))

- Refactor `torch.func` code: treat parameters as a single tuple argument,
  split data arguments into model input `X` and loss arguments `loss_args`,
  simplify `vmap` dims to `-1`
  ([PR](https://github.com/f-dangel/curvlinops/pull/220))

- Centralize `vmap` over matrix columns in `CurvatureLinearOperator._matmat_batch`;
  subclasses now implement `_matvec_batch` (single vector) instead
  ([PR](https://github.com/f-dangel/curvlinops/pull/256))

## [3.0.0] - 2025-10-16

This new major release realizes all features described in our [position paper](https://arxiv.org/abs/2501.19183).
Most importantly, **all linear operators are purely PyTorch by default now**.
If you prefer working with SciPy linear operators (as was the default in `2.x`), you can simply call `.to_scipy()` on a linear operator.

### Added/New

- Add warning about overwriting model parameters when loading state dict in (E)KFAC ([PR](https://github.com/f-dangel/curvlinops/pull/196))

- **Backward-incompatible:** Refactor class-based trace and diagonal estimators
  into functions ([PR](https://github.com/f-dangel/curvlinops/pull/168)) and assume
  PyTorch instead of SciPy linear operators ([PR](https://github.com/f-dangel/curvlinops/pull/188)):
  - If you used `HutchinsonTraceEstimator`, switch to `hutchinson_trace`
  - If you used `HutchPPTraceEstimator`, switch to `hutchpp_trace`
  - If you used `HutchinsonDiagonalEstimator`, switch to `hutchinson_diag`
  - If you used `HutchinsonSquaredFrobeniusNormEstimator`, switch to
    `hutchinson_squared_fro`

- Add diagonal estimation with the XDiag algorithm
  ([paper](https://arxiv.org/pdf/2301.07825),
   [PR](https://github.com/f-dangel/curvlinops/pull/167))

- Add trace estimation with the XTrace algorithm
  ([paper](https://arxiv.org/pdf/2301.07825),
   [PR](https://github.com/f-dangel/curvlinops/pull/166))

- Add a [use case
  example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_benchmark.html)
  that benchmarks the linear operators
  ([PR](https://github.com/f-dangel/curvlinops/pull/162))

- Add a [use case
  example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_trace_diagonal_estimation.html)
  on trace and diagonal estimation
  ([PR](https://github.com/f-dangel/curvlinops/pull/165))

- **Backward-incompatible:** Make linear operators purely PyTorch with a SciPy
  export option
  - `GGNLinearOperator` ([PR](https://github.com/f-dangel/curvlinops/pull/146))
  - `TransposedJacobianLinearOperator` and `JacobianLinearOperator`
    ([PR](https://github.com/f-dangel/curvlinops/pull/147))
  - `FisherMCLinearOperator`
    ([PR](https://github.com/f-dangel/curvlinops/pull/148))
  - `EFLinearOperator`
    ([PR](https://github.com/f-dangel/curvlinops/pull/149))
  - `ActivationHessianLinearOperator`
    ([PR](https://github.com/f-dangel/curvlinops/pull/157))
  - `KFACLinearOperator` and `KFACInverseLinearOperator`
    ([PR](https://github.com/f-dangel/curvlinops/pull/149)),
    deprecating their `.torch_matvec` and `.torch_matmat`,
    which are now available through `@`
  - `SubmatrixLinearOperator`
    ([PR](https://github.com/f-dangel/curvlinops/pull/177))
  - `CGInverseLinearOperator`, `NeumannInverseLinearOperator`, and `LSMRInverseLinearOperator`
    ([PR](https://github.com/f-dangel/curvlinops/pull/186)),
    also removing their public methods for specifying hyper-parameters and instead accepting them as keyword arguments in the constructor
    - Switch CG implementation from SciPy to GPyTorch `CGInverseLinearOperator`
      ([PR](https://github.com/f-dangel/curvlinops/pull/190))

- **Backward-incompatible** Refactor spectral density estimation methods to expect PyTorch linear operators and use PyTorch functions internally rather than SciPy/NumPy
  ([PR](https://github.com/f-dangel/curvlinops/pull/194))

- Add `EKFACLinearOperator` which implements EKFAC
  ([paper](https://arxiv.org/abs/1806.03884),
   [PR](https://github.com/f-dangel/curvlinops/pull/173))

- Implement composition rules for PyTorch linear operators
  ([PR](https://github.com/f-dangel/curvlinops/pull/185))
  For two linear operators `A`, `B` and a scalar `s`, this
  allows writing `A + B`, `A - B`, `A @ B`, `s * A`, `A * s`.

### Fixed/Removed

- Bug in `KFACInverseLinearOperator` where the damping was repeatedly (for every
  matrix-vector product) added to the eigenvalues corresponding to the bias
  parameters when `use_exact_damping=True` and
  `KFACLinearOperator._separate_weight_and_bias=True`
  ([PR](https://github.com/f-dangel/curvlinops/pull/156))
- More test cases for `KFACInverseLinearOperator` and bug fix in
    `.load_state_dict` ([PR](https://github.com/f-dangel/curvlinops/pull/158))
- **Backward-incompatible:** Remove `.to_device` function of linear operators,
  always carry out deterministic checks on the linear operator's device
  (previously always on CPU)
  ([PR](https://github.com/f-dangel/curvlinops/pull/160))
- Fixed the default value for `conlim` in `LSMRInverseLinearOperator` from `1e-8` to `1e8` ([PR](https://github.com/f-dangel/curvlinops/pull/180))
- Bug in XDiag implementation that would only work with dense matrices, but not with
  linear operators ([PR](https://github.com/f-dangel/curvlinops/pull/188))

### Internal

- Always run full tests
  ([PR](https://github.com/f-dangel/curvlinops/pull/161))
- Migrate linting, formatting, and import sorting from `flake8`, `black`, and
  `isort` to `ruff` ([PR](https://github.com/f-dangel/curvlinops/pull/164))
- Modify logo
  ([PR](https://github.com/f-dangel/curvlinops/pull/169))
- Test that two consecutive matrix-vector products of a linear operator match
  ([issue](https://github.com/f-dangel/curvlinops/issues/159),
   [PR](https://github.com/f-dangel/curvlinops/pull/175))
- Reduce usage of SciPy in tests and examples
  ([PR](https://github.com/f-dangel/curvlinops/pull/187))
- Introduce `property`s for a linear operator's data type (`.dtype`) and device (`.device`)
  ([PR](https://github.com/f-dangel/curvlinops/pull/189))
- Move utility linear operators for examples into a single file
  ([PR](https://github.com/f-dangel/curvlinops/pull/195))

## [2.0.1] - 2024-10-25

Minor bug fixes and documentation polishing.

### Added/New

- Comparison of `eigsh` with power iteration in [eigenvalue
  example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_eigenvalues.html#sphx-glr-basic-usage-example-eigenvalues-py)
  ([PR](https://github.com/f-dangel/curvlinops/pull/140))

### Fixed/Removed

- Deprecate Python 3.8 as it will reach its end of life in October 2024
  ([PR](https://github.com/f-dangel/curvlinops/pull/128))

- Improve `intersphinx` mapping to `curvlinops` objects
  ([issue](https://github.com/f-dangel/curvlinops/issues/138),
  [PR](https://github.com/f-dangel/curvlinops/pull/141))

### Internal

- Update Github action versions and cache `pip`
  ([PR](https://github.com/f-dangel/curvlinops/pull/129))

- Re-activate Monte-Carlo tests, refactor, and reduce their run time
  ([PR](https://github.com/f-dangel/curvlinops/pull/131))

- Add more matrices in visual tour code example and prettify plots
  ([PR](https://github.com/f-dangel/curvlinops/pull/134))

- Prettify visualizations in [spectral density
  example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_verification_spectral_density.html)
  ([PR](https://github.com/f-dangel/curvlinops/pull/139))

## [2.0.0] - 2024-08-15

This major release is almost fully backward compatible with the `1.x.y` release
except for one API change in `KFACLinearOperator`. Most notably, it adds
**support for HuggingFace LLMs**, ships a linear operator for the inverse of
KFAC, and offers many performance improvements.

### Breaking changes to `1.x.y`

- Remove `loss_average` argument from `KFACLinearOperator`
  [PR](https://github.com/f-dangel/curvlinops/pull/117)

### Added/New

- Support HuggingFace LLMs and provide an
  [example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_huggingface.html)
  ([PR](https://github.com/f-dangel/curvlinops/pull/100))

- Add Linear operator for the inverse of KFAC (`KFACInverseLinearOperator`)
  ([PR](https://github.com/f-dangel/curvlinops/pull/69))

  - Support exact and heuristic damping of the Kronecker factors when inverting
    ([PR](https://github.com/f-dangel/curvlinops/pull/93))

  - Add option to fall back to double precision if inversion fails in single precision
    ([PR](https://github.com/f-dangel/curvlinops/pull/102))

  - Add functionality to checkpoint a linear operator
    ([PR](https://github.com/f-dangel/curvlinops/pull/114))

- Add Estimation method for the squared Frobenius norm of a linear operator
  ([PR](https://github.com/f-dangel/curvlinops/pull/80))

  - Improve efficiency
    ([PR](https://github.com/f-dangel/curvlinops/pull/120))

- Add support for `BCEWithLogitsLoss` in `FisherMCLinearOperator` and
  `KFACLinearOperator`
  ([PR](https://github.com/f-dangel/curvlinops/pull/99))

- Improvements to `KFACLinearOperator`

  - Add functionality to compute exact trace, determinant, log determinant, and
    frobenius norm of `KFACLinearOperator`
    ([PR](https://github.com/f-dangel/curvlinops/pull/95))

  - Add option to compute input-based curvature, known as
    [FOOF](https://arxiv.org/abs/2201.12250)/[ISAAC](https://arxiv.org/abs/2305.00604)
    ([PR](https://github.com/f-dangel/curvlinops/pull/98))

  - Compute KFAC matrices without overwriting values in `.grad`
    ([PR](https://github.com/f-dangel/curvlinops/pull/104))

  - Add functionality to checkpoint a linear operator
    ([PR](https://github.com/f-dangel/curvlinops/pull/114))

- Add inverse linear operator `LSMRInverseLinearOperator` to multiply by solving
  a least-squares system with [LSMR](https://arxiv.org/abs/1006.0758)
  ([PR](https://github.com/f-dangel/curvlinops/pull/106))

- Improve linear operator interface
  - Add `num_data` argument to manually specify number of data points in a data
    loader and avoid one pass through the data
    ([PR](https://github.com/f-dangel/curvlinops/pull/70))
  - Support block-diagonal approximations in `HessianLinearOperator` via a new
    `block_sizes` argument
    ([PR](https://github.com/f-dangel/curvlinops/pull/74))

- Add option to multiply with KFAC and its inverse purely in PyTorch
  ([PR](https://github.com/f-dangel/curvlinops/pull/91))

- Improve performance when multiplying linear operators onto a matrix
  ([PR](https://github.com/f-dangel/curvlinops/pull/73))

- Improve performance of `EFLinearOperator`
  ([PR1](https://github.com/f-dangel/curvlinops/pull/84)
  [PR2](https://github.com/f-dangel/curvlinops/pull/88))
  and `FisherMCLinearOperator`
  ([PR1](https://github.com/f-dangel/curvlinops/pull/85)
  [PR2](https://github.com/f-dangel/curvlinops/pull/89))

- Implement adjoint of `SubmatrixLinearOperator`
  ([PR](https://github.com/f-dangel/curvlinops/pull/115))

### Fixed/Removed

- Device error of random number generator for `MCFisherLinearOperator` and
  `KFACLinearOperator` when running on GPU
  ([PR](https://github.com/f-dangel/curvlinops/pull/76))

- Broken parameter mapping for KFAC when loading a linear operator to a
  different device
  ([PR](https://github.com/f-dangel/curvlinops/pull/78))

- Device errors in tests
  ([PR](https://github.com/f-dangel/curvlinops/pull/103))

- Scaling issue for Fisher matrices and KFAC for model outputs with more
  than two dimensions and mean reduction
  ([issue](https://github.com/f-dangel/curvlinops/issues/108),
  [PR1](https://github.com/f-dangel/curvlinops/pull/109),
  [PR2](https://github.com/f-dangel/curvlinops/pull/110),
  [PR3](https://github.com/f-dangel/curvlinops/pull/112))

- Fix from introducing `Enum`s
  ([PR](https://github.com/f-dangel/curvlinops/pull/119))

- Fix output shapes of KFAC's `matvec` for convolution weights
  ([PR](https://github.com/f-dangel/curvlinops/pull/125))

### Internal

- Use latest `black` (`black==24.1.1`)
  ([PR](https://github.com/f-dangel/curvlinops/pull/72))

- Use module names instead of tensor addresses to identify parameters in KFAC
  ([PR](https://github.com/f-dangel/curvlinops/pull/79))

- Include links to source code in the documentation
  ([PR](https://github.com/f-dangel/curvlinops/pull/81))

- Run Github actions for pull requests to any branch
  ([PR](https://github.com/f-dangel/curvlinops/pull/97))

- Deprecate `pkg_resources`
  ([PR](https://github.com/f-dangel/curvlinops/pull/121))

- Migrate from `setup.py` to `pyproject.toml`
  ([PR](https://github.com/f-dangel/curvlinops/pull/123))

## [1.2.0] - 2024-01-12

This release ships with many new features and requires PyTorch 2:

### Added/New

- Linear operator for KFAC (Kronecker-Factored Approximate Curvature)
  with support for a broad range of options

  - Prototype (`torch.nn.MSELoss` and `torch.nn.Linear`)
    ([PR](https://github.com/f-dangel/curvlinops/pull/43))

  - Support with `torch.nn.CrossEntropyLoss`
    ([PR](https://github.com/f-dangel/curvlinops/pull/52))

  - Support empirical Fisher (using gradients from data distribution)
    ([PR](https://github.com/f-dangel/curvlinops/pull/54))
    and type-2 estimation (using columns from the Hessian's matrix square root)
    ([PR](https://github.com/f-dangel/curvlinops/pull/56))

  - Support arbitrary parameter order
    ([PR](https://github.com/f-dangel/curvlinops/pull/51)),
    weight-only or bias-only layers
    ([PR](https://github.com/f-dangel/curvlinops/pull/55)),
    and support treating weight and bias jointly
    ([PR](https://github.com/f-dangel/curvlinops/pull/57))

  - Support networks with in-place activations
    ([PR](https://github.com/f-dangel/curvlinops/pull/59))

  - Support models with >2d output
    ([PR](https://github.com/f-dangel/curvlinops/pull/62))

  - Support KFAC `'expand'` and `'reduce'` approximations
    for general weight-sharing layers
    ([PR](https://github.com/f-dangel/curvlinops/pull/63),
     [paper](https://arxiv.org/abs/2311.00636))

  - Support `torch.nn.Conv2d`
    ([PR](https://github.com/f-dangel/curvlinops/pull/64))

- Linear operator for taking sub-matrices of another linear operator
  ([PR](https://github.com/f-dangel/curvlinops/pull/25),
  [example](https://curvlinops.readthedocs.io/en/main/basic_usage/example_submatrices.html)
  ([PR](https://github.com/f-dangel/curvlinops/pull/26)))

- Linear operator for approximate inversion via the Neumann series
  ([PR](https://github.com/f-dangel/curvlinops/pull/28),
  [example](https://curvlinops.readthedocs.io/en/main/basic_usage/example_inverses.html#neumann-inverse-cg-alternative)
  ([PR](https://github.com/f-dangel/curvlinops/pull/29)))

- Linear operator for a neural network's output-parameter Jacobian
  ([PR](https://github.com/f-dangel/curvlinops/pull/32)) and its transpose
  ([PR](https://github.com/f-dangel/curvlinops/pull/34))

- Implement `adjoint` from `scipy.sparse.linalg.LinearOperator` interface
  ([PR](https://github.com/f-dangel/curvlinops/pull/33/files))

- [Example](https://curvlinops.readthedocs.io/en/main/basic_usage/example_model_merging.html) for Fisher-weighted model averaging
  ([PR](https://github.com/f-dangel/curvlinops/pull/37))

- Trace estimation via vanilla Hutchinson
  ([PR](https://github.com/f-dangel/curvlinops/pull/38))

- Trace estimation via [Hutch++](https://arxiv.org/abs/2010.09649)
  ([PR](https://github.com/f-dangel/curvlinops/pull/39))

- Diagonal estimation via Hutchinson
  ([PR](https://github.com/f-dangel/curvlinops/pull/40))

- Experimental: Linear operator for the Hessian of the loss w.r.t. an
  intermediate feature
  ([PR](https://github.com/f-dangel/curvlinops/pull/65))

### Fixed/Removed

- Allow for partially specified boundaries of the spectrum inside the spectral
  density estimation methods and only estimate the missing boundary
  ([PR](https://github.com/f-dangel/curvlinops/pull/27))

- Deprecate python 3.7
  ([PR](https://github.com/f-dangel/curvlinops/pull/32))

- For future releases, we will abandon the `development` branch and switch to a
  workflow where new features are directly merged into `main`.

### Internal

- Switch from `functorch` to `torch.func` in reference implementation of tests
  ([PR](https://github.com/f-dangel/curvlinops/pull/36))

## [1.1.0] - 2023-02-19

Adds various new features:

### Added/New

- Inverses of linear operators with multiplication via conjugate gradients
  ([PR](https://github.com/f-dangel/curvlinops/pull/9),
  [example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_inverses.html))

- Spectral density estimation methods from [papyan2020traces](https://jmlr.org/beta/papers/v21/20-933.html)
  ([PR](https://github.com/f-dangel/curvlinops/pull/14/files),
  [basic example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_verification_spectral_density.html))

  - Add caching to recycle Lanczos iterations between densities with different hyperparameters
    ([PR](https://github.com/f-dangel/curvlinops/pull/15),
     [demo 1](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_verification_spectral_density.html#for-multiple-hyperparameters),
     [demo 2](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_verification_spectral_density.html#id1))

- Example visualizing different supported curvature matrices
  ([PR](https://github.com/f-dangel/curvlinops/pull/16),
   [example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_visual_tour.html))

- Linear operator for the uncentered gradient covariance matrix (aka 'empirical Fisher')
  ([PR](https://github.com/f-dangel/curvlinops/pull/17))

- Example for computing eigenvalues with `scipy.linalg.sparse.eigsh`
  ([PR](https://github.com/f-dangel/curvlinops/pull/18),
   [example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_eigenvalues.html))

- Linear operator for a Monte-Carlo approximation of the Fisher
  ([PR1](https://github.com/f-dangel/curvlinops/pull/20),
   [PR2](https://github.com/f-dangel/curvlinops/pull/21),
   [example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_fisher_monte_carlo.html))

### Fixed/Removed

### Internal

- Refactor examples, extracting common `functorch` and array comparison methods
  ([PR](https://github.com/f-dangel/curvlinops/pull/10))

- Add description of the library on the RTD landing page
  ([PR](https://github.com/f-dangel/curvlinops/pull/11))

- Set up a proper test suite with cases
  ([PR](https://github.com/f-dangel/curvlinops/pull/12))

  - Add regression test cases
    ([PR](https://github.com/f-dangel/curvlinops/pull/19))

- Update code to latest versions of linting CI
  ([PR](https://github.com/f-dangel/curvlinops/pull/22))

## [1.0.0] - 2022-09-30

Initial release

[Unreleased]: https://github.com/f-dangel/curvlinops/compare/3.0.1...HEAD
[3.0.1]: https://github.com/f-dangel/curvlinops/releases/tag/3.0.1
[3.0.0]: https://github.com/f-dangel/curvlinops/releases/tag/3.0.0
[2.0.1]: https://github.com/f-dangel/curvlinops/releases/tag/2.0.1
[2.0.0]: https://github.com/f-dangel/curvlinops/releases/tag/2.0.0
[1.2.0]: https://github.com/f-dangel/curvlinops/releases/tag/1.2.0
[1.1.0]: https://github.com/f-dangel/curvlinops/releases/tag/1.1.0
[1.0.0]: https://github.com/f-dangel/curvlinops/releases/tag/1.0.0
