# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/f-dangel/curvlinops/compare/1.2.0...HEAD
[1.2.0]: https://github.com/f-dangel/curvlinops/releases/tag/1.2.0
[1.1.0]: https://github.com/f-dangel/curvlinops/releases/tag/1.1.0
[1.0.0]: https://github.com/f-dangel/curvlinops/releases/tag/1.0.0
