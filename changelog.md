# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added/New

### Fixed/Removed

### Internal

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
