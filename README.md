# <img alt="Logo" src="./docs/rtd/assets/logo.svg" height="90"> scipy linear operators of deep learning matrices in PyTorch

[![Python
3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
![tests](https://github.com/f-dangel/curvature-linear-operators/actions/workflows/test.yaml/badge.svg)
[![Coveralls](https://coveralls.io/repos/github/f-dangel/curvlinops/badge.svg?branch=master)](https://coveralls.io/github/f-dangel/curvlinops)

This library implements
[`scipy.sparse.linalg.LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html)s
for deep learning matrices, such as

- the Hessian
- the Fisher/generalized Gauss-Newton (GGN)
- the Monte-Carlo approximated Fisher
- the Fisher/GGN's KFAC approximation (Kronecker-Factored Approximate Curvature)
- the uncentered gradient covariance (aka empirical Fisher)
- the output-parameter Jacobian of a neural net and its transpose

Matrix-vector products are carried out in PyTorch, i.e. potentially on a GPU.
The library supports defining these matrices not only on a mini-batch, but
on data sets (looping over batches during a `matvec` operation).

You can plug these linear operators into `scipy`, while carrying out the heavy
lifting (matrix-vector multiplies) in PyTorch on GPU. My favorite example for
such a routine is
[`scipy.sparse.linalg.eigsh`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
that lets you compute a subset of eigen-pairs.

The library also provides linear operator transformations, like taking the
inverse (inverse matrix-vector product via conjugate gradients) or slicing out
sub-matrices.

Finally, it offers functionality to probe properties of the represented
matrices, like their spectral density, trace, or diagonal.

- **Documentation:** https://curvlinops.readthedocs.io/en/latest/

- **Bug reports & feature requests:**
  https://github.com/f-dangel/curvlinops/issues

## Installation

```bash
pip install curvlinops-for-pytorch
```

## Examples

- [Basic
  usage](https://curvlinops.readthedocs.io/en/latest/basic_usage/example_matrix_vector_products.html)
- [Advanced
  examples](https://curvlinops.readthedocs.io/en/latest/basic_usage/index.html)

## Future ideas

Other features that could be supported in the future include:

- Other matrices

  - the centered gradient covariance
  - terms of the [hierarchical GGN
    decomposition](https://arxiv.org/abs/2008.11865)

###### Logo mage credits
- SciPy logo: Unknown, [CC BY-SA
  4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons
- PyTorch logo: https://github.com/soumith, [CC BY-SA
  4.0](https://creativecommons.org/licenses/by-sa/4.0), via Wikimedia Commons
