# <img alt="Logo" src="./docs/rtd/assets/logo.svg" height="90"> Linear Operators for Curvature Matrices in PyTorch

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)
![tests](https://github.com/f-dangel/curvature-linear-operators/actions/workflows/test.yaml/badge.svg)
[![Coveralls](https://coveralls.io/repos/github/f-dangel/curvlinops/badge.svg?branch=main)](https://coveralls.io/github/f-dangel/curvlinops)

This library provides **lin**ear **op**erator**s**---a unified interface for matrix-free computation---for deep learning **curv**ature matrices in PyTorch.
`curvlinops` is inspired by SciPy's [`sparse.linalg.LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html) interface and implements a PyTorch version.

You can read our [position paper](https://arxiv.org/abs/2501.19183) to know more about why combining linear operators with curvature matrices might be a good idea.

Main features:

- **Broad support of curvature matrices.** `curvlinops` supports many common curvature matrices and approximations thereof, such as the Hessian, Fisher, generalized Gauss-Newton, and K-FAC ([overview](https://curvlinops.readthedocs.io/en/latest/linops.html#linear-operators), [visual tour](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_visual_tour.html#visualization)).

- **Unified interface.** All linear operators share the same interface, making it easy to switch between curvature matrices.

- **Purely PyTorch.** All computations can run on a GPU.

- **SciPy export.** You can export a `curvlinops` linear operator to a SciPy `LinearOperator` with `.to_scipy()`.
  This allows plugging it into `scipy`, while carrying out the heavy lifting (matrix-vector multiplies) in PyTorch on GPU.
  My favorite example is
[`scipy.sparse.linalg.eigsh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html) that lets you compute a subset of eigen-pairs ([example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_eigenvalues.html)).

- **Randomized estimation algorithms.** `curvlinops` offers functionality to estimate properties the matrix represented by a linear operators, like its spectral density ([example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_verification_spectral_density.html)),
  inverse ([example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_inverses.html)),
  trace & diagonal ([example](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_trace_diagonal_estimation.html)).

## Installation

```bash
pip install curvlinops-for-pytorch
```

## Useful Links

- [Basic
  usage](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_matrix_vector_products.html)

- [Advanced
  examples](https://curvlinops.readthedocs.io/en/latest/basic_usage/index.html)

- **Documentation:** https://curvlinops.readthedocs.io/en/latest/

- **Bug reports & feature requests:**
  https://github.com/f-dangel/curvlinops/issues

## Citation

If you find `curvlinops` useful for your work, consider citing our [position paper](https://arxiv.org/abs/2501.19183)

```bibtex

@article{dangel2025position,
  title =        {Position: Curvature Matrices Should Be Democratized via Linear
                  Operators},
  author =       {Dangel, Felix and Eschenhagen, Runa and Ormaniec, Weronika and
                  Fernandez, Andres and Tatzel, Lukas and Kristiadi, Agustinus},
  journal =      {arXiv 2501.19183},
  year =         2025,
}

```

## Future ideas

- Refactor the back-end for curvature-matrix multiplication into pure functions
  to improve recycle-ability and ease the use of `torch.compile`.

- Multi-GPU support.

- Include more curvature matrices
  - E.g. the [GGN's hierarchical decomposition](https://arxiv.org/abs/2008.11865)

###### Logo mage credits
- PyTorch logo: https://github.com/soumith, [CC BY-SA
  4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons
