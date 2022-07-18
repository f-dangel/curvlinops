[👷🏗👷🏗 **Under development. Stay tuned. Feel free to open feature requests or contribute!** 👷🏗👷🏗]

# <img alt="Logo" src="./docs/rtd/assets/vivit_logo.svg" height="90"> scipy linear operators of deep learning matrices in PyTorch

[![Python
3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
![tests](https://github.com/f-dangel/curvature-linear-operators/actions/workflows/test.yaml/badge.svg)

This library would implement
[`scipy.sparse.linalg.LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)'s
for deep learning matrices, such as

- the Hessian
- the Fisher/generalized Gauss-Newton (GGN)

Matrix-vector products are carried out in PyTorch, i.e. potentially on a GPU.
The library would support defining these matrices not only on a mini-batch, but
on data sets (looping over batches during a `mavec` operation).

You could plug these linear operators into `scipy`, while carrying out the heavy
lifting (matrix-vector multiplies) on GPU. My favorite example for such a
routine is
[`scipy.sparse.linalg.eigsh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
that lets you compute a subset of eigenpairs.

- **Documentation:** would be at
  https://curvature-linear-operators.readthedocs.io/en/latest/

- **Bug reports & feature requests:**
  https://github.com/f-dangel/curvature-linear-operators/issues

## Installation

Currently, there is no PyPI release. You would need to install from GitHub via

```bash
pip install curvlinops-for-pytorch@git+https://github.com/f-dangel/curvature-linear-operators.git#egg=curvlinops-for-pytorch
```

## TODO Examples

Basic and advanced demos would be in the
[documentation](https://curvature-linear-operators.readthedocs.io/en/latest/basic_usage/index.html).

## Additional ideas

Other features that could be supported in the future include:

- Other matrices

  - the un-centered gradient covariance (aka empirical Fisher)
  - the centered gradient covariance
  - terms of the [hierarchical GGN decomposition](https://arxiv.org/abs/2008.11865)

- Block-diagonal approximations

- Inverse matrix-vector products by solving a linear system via conjugate
  gradients

  - This could allow computing generalization metrics like the Takeuchi
    Information Criterion (TIC), using inverse matrix-vector products in
    combination with Hutchinson trace estimation
