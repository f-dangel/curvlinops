r"""Eigenvalues
===============

This example demonstrates how to compute a subset of eigenvalues of a linear
operator, using :func:`scipy.sparse.linalg.eigsh`. Concretely, we will compute
leading eigenvalues of the Hessian.

As always, imports go first.
"""

import numpy
import scipy
import torch
from torch import nn

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from curvlinops.examples.utils import report_nonclose

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
H = HessianLinearOperator(model, loss_function, params, data)

# %%
#
# Leading eigenvalues
# -------------------
#
# Through :func:`scipy.sparse.linalg.eigsh`, we can obtain the leading
# :math:`k=3` eigenvalues.

k = 3
which = "LA"  # largest algebraic
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

H_functorch = (
    functorch_hessian(model, loss_function, params, data).detach().cpu().numpy()
)
evals_functorch, _ = scipy.linalg.eigh(H_functorch)
top_k_evals_functorch = evals_functorch[-k:]

print(f"Leading {k} Hessian eigenvalues (functorch): {top_k_evals_functorch}")

# %%
#
#  Both results should match.

print(f"Comparing leading {k} Hessian eigenvalues (linear operator vs. functorch).")
report_nonclose(top_k_evals, top_k_evals_functorch)

# %%
#
# :func:`scipy.sparse.linalg.eigsh` can also compute other subsets of
# eigenvalues, and also their associated eigenvectors. Check out its
# documentation for more!
