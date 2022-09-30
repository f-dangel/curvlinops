# <img alt="Logo" src="./docs/rtd/assets/logo.svg" height="90"> scipy linear operators of deep learning matrices in PyTorch

[![Python
3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
![tests](https://github.com/f-dangel/curvature-linear-operators/actions/workflows/test.yaml/badge.svg)
[![Coveralls](https://coveralls.io/repos/github/f-dangel/curvlinops/badge.svg?branch=master)](https://coveralls.io/github/f-dangel/curvlinops)

This library implements
[`scipy.sparse.linalg.LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)s
for deep learning matrices, such as

- the Hessian
- the Fisher/generalized Gauss-Newton (GGN)

Matrix-vector products are carried out in PyTorch, i.e. potentially on a GPU.
The library supports defining these matrices not only on a mini-batch, but
on data sets (looping over batches during a `matvec` operation).

You can plug these linear operators into `scipy`, while carrying out the heavy
lifting (matrix-vector multiplies) in PyTorch on GPU. My favorite example for
such a routine is
[`scipy.sparse.linalg.eigsh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
that lets you compute a subset of eigenpairs.

- **Documentation:** https://curvlinops.readthedocs.io/en/latest/

- **Bug reports & feature requests:**
  https://github.com/f-dangel/curvlinops/issues

## Installation

```bash
pip install curvlinops-for-pytorch
```

## Examples

- [Basic
  usage](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_matrix_vector_products.html#sphx-glr-basic-usage-example-matrix-vector-products-py)

## Future ideas

Other features that could be supported in the future include:

- Other matrices

  - the un-centered gradient covariance (aka empirical Fisher)
  - the centered gradient covariance
  - terms of the [hierarchical GGN
    decomposition](https://arxiv.org/abs/2008.11865)

- Block-diagonal approximations (via `param_groups`)

- Inverse matrix-vector products by solving a linear system via conjugate
  gradients

  - This could allow computing generalization metrics like the Takeuchi
    Information Criterion (TIC), using inverse matrix-vector products in
    combination with Hutchinson trace estimation

###### Logo mage credits
- SciPy logo: Unknown, [CC BY-SA
  4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons
- PyTorch logo: https://github.com/soumith, [CC BY-SA
  4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons
