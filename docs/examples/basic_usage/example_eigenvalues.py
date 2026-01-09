r"""Eigenvalues
===============

This example demonstrates how to compute a subset of eigenvalues of a linear
operator, using :func:`scipy.sparse.linalg.eigsh`. Concretely, we will compute
leading eigenvalues of the Hessian.

As always, imports go first.
"""

from contextlib import redirect_stderr
from io import StringIO
from typing import List, Tuple

import numpy
import scipy
import torch
from torch import nn

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from curvlinops.utils import allclose_report

# make deterministic
torch.manual_seed(0)
numpy.random.seed(0)

# %%
#
# Setup
# -----
#
# We will use synthetic data, consisting of two mini-batches, a small MLP, and
# mean-squared error as loss function.

N = 20
D_in = 7
D_hidden = 5
D_out = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X1, y1 = torch.rand(N, D_in).to(DEVICE), torch.rand(N, D_out).to(DEVICE)
X2, y2 = torch.rand(N, D_in).to(DEVICE), torch.rand(N, D_out).to(DEVICE)

model = nn.Sequential(
    nn.Linear(D_in, D_hidden),
    nn.ReLU(),
    nn.Linear(D_hidden, D_hidden),
    nn.Sigmoid(),
    nn.Linear(D_hidden, D_out),
).to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]

loss_function = nn.MSELoss(reduction="mean").to(DEVICE)


# %%
#
# Linear operator
# ------------------
#
# We are ready to setup the linear operator. In this example, we will use the Hessian.

data = [(X1, y1), (X2, y2)]
H = HessianLinearOperator(model, loss_function, params, data).to_scipy()

# %%
#
# Leading eigenvalues
# -------------------
#
# Through :func:`scipy.sparse.linalg.eigsh`, we can obtain the leading
# :math:`k=3` eigenvalues.

k = 3
which = "LM"  # largest magnitude
top_k_evals, _ = scipy.sparse.linalg.eigsh(H, k=k, which=which)

print(f"Leading {k} Hessian eigenvalues: {top_k_evals}")

# %%
#
# Verifying results
# -----------------
#
# To double-check this result, let's compute the Hessian with
# :code:`functorch`, compute all its eigenvalues with
# :func:`scipy.linalg.eigh`, then extract the top :math:`k`.

H_functorch = functorch_hessian(model, loss_function, params, data).detach()
evals_functorch, _ = torch.linalg.eigh(H_functorch)
top_k_evals_functorch = evals_functorch[-k:]

print(f"Leading {k} Hessian eigenvalues (functorch): {top_k_evals_functorch}")

# %%
#
#  Both results should match.

print(f"Comparing leading {k} Hessian eigenvalues (linear operator vs. functorch).")
assert allclose_report(top_k_evals, top_k_evals_functorch.double(), rtol=1e-4)

# %%
#
# :func:`scipy.sparse.linalg.eigsh` can also compute other subsets of
# eigenvalues, and also their associated eigenvectors. Check out its
# documentation for more!


# %%
#
# Power iteration versus ``eigsh``
# --------------------------------
#
# Here, we compare the query efficiency of :func:`scipy.sparse.linalg.eigsh` with the
# `power iteration <https://en.wikipedia.org/wiki/Power_iteration>`_ method, a simple
# method to compute the leading eigenvalues (in terms of magnitude). We re-use the im-
# plementation from the `PyHessian library <https://github.com/amirgholami/PyHessian>`_
# and adapt it to work with SciPy arrays rather than PyTorch tensors:


def power_method(
    A: scipy.sparse.linalg.LinearOperator,
    max_iterations: int = 100,
    tol: float = 1e-3,
    k: int = 1,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Compute the top-k eigenpairs of a linear operator using power iteration.

    Code modified from PyHessian, see
    https://github.com/amirgholami/PyHessian/blob/72e5f0a0d06142387fccdab2226b4c6bae088202/pyhessian/hessian.py#L111-L156

    Args:
        A: Linear operator of dimension ``D`` whose top eigenpairs will be computed.
        max_iterations: Maximum number of iterations. Defaults to ``100``.
        tol: Relative tolerance between two consecutive iterations that has to be
            reached for convergence. Defaults to ``1e-3``.
        k: Number of eigenpairs to compute. Defaults to ``1``.

    Returns:
        The eigenvalues as array of shape ``[k]`` in descending order, and their
        corresponding eigenvectors as array of shape ``[D, k]``.
    """
    eigenvalues = []
    eigenvectors = []

    def normalize(v: numpy.ndarray) -> numpy.ndarray:
        return v / numpy.linalg.norm(v)

    def orthonormalize(v: numpy.ndarray, basis: List[numpy.ndarray]) -> numpy.ndarray:
        for basis_vector in basis:
            v -= numpy.dot(v, basis_vector) * basis_vector
        return normalize(v)

    computed_dim = 0
    while computed_dim < k:
        eigenvalue = None
        v = normalize(numpy.random.randn(A.shape[0]))

        for _ in range(max_iterations):
            v = orthonormalize(v, eigenvectors)
            Av = A @ v

            tmp_eigenvalue = v.dot(Av)
            v = normalize(Av)

            if eigenvalue is None:
                eigenvalue = tmp_eigenvalue
            elif abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                break
            else:
                eigenvalue = tmp_eigenvalue

        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        computed_dim += 1

    # sort in ascending order and convert into arrays
    eigenvalues = numpy.array(eigenvalues[::-1])
    eigenvectors = numpy.array(eigenvectors[::-1])

    return eigenvalues, eigenvectors


# %%
#
# Let's compute the top-3 eigenvalues via power iteration and verify they roughly match.
# Note that we are using a smaller :code:`tol` value than the PyHessian default value
# here to get better convergence, and we have to use relatively large tolerances for the
# comparison (which we didn't do when comparing :code:`eigsh` with :code:`eigh`).

top_k_evals_power, _ = power_method(H, tol=1e-4, k=k)
print(f"Comparing leading {k} Hessian eigenvalues (eigsh vs. power).")
assert allclose_report(
    top_k_evals_functorch.double(), top_k_evals_power, rtol=2e-2, atol=1e-6
)

# %%
#
# This indicates that the power method achieves poorer accuracy than :code:`eigsh`. But
# does it therefore require fewer matrix-vector products? To answer this, let's turn on
# the linear operator's progress bar, which allows us to count the number of
# matrix-vector products invoked by both eigen-solvers:

H = HessianLinearOperator(
    model, loss_function, params, data, progressbar=True
).to_scipy()

# determine number of matrix-vector products used by `eigsh`
with StringIO() as buf, redirect_stderr(buf):
    top_k_evals, _ = scipy.sparse.linalg.eigsh(H, k=k, which=which)
    # The tqdm progressbar will print "matmat" for each batch in a matrix-vector
    # product. Therefore, we need to divide by the number of batches
    queries_eigsh = buf.getvalue().count("matmat") // len(data)
print(f"eigsh used {queries_eigsh} matrix-vector products.")

# determine number of matrix-vector products used by power iteration
with StringIO() as buf, redirect_stderr(buf):
    top_k_evals_power, _ = power_method(H, k=k, tol=1e-4)
    # The tqdm progressbar will print "matmat" for each batch in a matrix-vector
    # product. Therefore, we need to divide by the number of batches
    queries_power = buf.getvalue().count("matmat") // len(data)
print(f"Power iteration used {queries_power} matrix-vector products.")

assert queries_power > queries_eigsh

# %%
#
# Sadly, the power iteration also does not offer computational benefits, consuming
# more matrix-vector products than :code:`eigsh`. While it is elegant and simple,
# it cannot compete with :code:`eigsh`, at least in the comparison provided here.
#
# Therefore, we recommend using :code:`eigsh` for computing eigenvalues. This method
# becomes accessible because :code:`curvlinops` interfaces with SciPy's linear
# operators.
