# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/f-dangel/curvlinops/compare/1.1.0...HEAD
[1.1.0]: https://github.com/f-dangel/curvlinops/releases/tag/1.1.0
[1.0.0]: https://github.com/f-dangel/curvlinops/releases/tag/1.0.0
