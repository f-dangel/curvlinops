r"""
Matrix-vector products
======================

This tutorial contains a basic demonstration how to set up ``LinearOperators``
for the Hessian and the GGN and how to multiply them to a vector.

First, the imports.
"""

import math
from typing import Tuple

import functorch
import matplotlib.pyplot as plt
import numpy
import torch
from torch import nn

from curvlinops import HessianLinearOperator

# make deterministic
torch.manual_seed(0)
numpy.random.seed(0)

# %%
# Setup
# -----
# Let's create some toy data, a small MLP, and use mean-squared error as loss function.

N = 4
D_in = 7
D_hidden = 5
D_out = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.rand(N, D_in).to(DEVICE)
y = torch.rand(N, D_out).to(DEVICE)

model = nn.Sequential(
    nn.Linear(D_in, D_hidden),
    nn.ReLU(),
    nn.Linear(D_hidden, D_hidden),
    nn.ReLU(),
    nn.Linear(D_hidden, D_out),
).to(DEVICE)

loss_function = nn.MSELoss(reduction="mean").to(DEVICE)


# %%
# Hessian-vector products
# -----------------------
#
# Setting up a linear operator for the Hessian is straightforward.

data = [(X, y)]
H = HessianLinearOperator(model, loss_function, data, DEVICE)

# %%
#
# We can now multiply by the Hessian. This operation will be carried out in
# PyTorch under the hood, but the operator is compatible with ``scipy``, so we
# can just pass a ``numpy`` vector to the matrix-multiplication.

D = H.shape[0]
v = numpy.random.rand(D)

Hv = H @ v

# %%
#
# To verify the result, we compute the Hessian using ``functorch``. To do that,
# we first convert the PyTorch layers into functions, and use them to construct
# the function that produces the loss. Then, we can take the Hessian of that
# function w.r.t. to the neural network parameters:

# convert modules to functions
model_fn, model_params = functorch.make_functional(model)
loss_function_fn, loss_function_fn_params = functorch.make_functional(loss_function)


def loss(X: torch.Tensor, y: torch.Tensor, params: Tuple[torch.Tensor]) -> torch.Tensor:
    """Compute the loss given a mini-batch (X, y) and the neural network parameters."""
    output = model_fn(params, X)
    return loss_function_fn(loss_function_fn_params, output, y)


params_argnum = 2
H_functorch = functorch.hessian(loss, argnums=params_argnum)

# %%
#
# This yields a function that computes the Hessian:

H_mat = H_functorch(X, y, model_params)

# %%
#
# ``functorch``'s output is a nested tuple that contains the Hessian blocks of
# :code:`(params[i], params[j])` at position :code:`[i, j]`. The following
# function flattens and concatenates this block structure into a matrix:


def blocks_to_matrix(blocks: Tuple[Tuple[torch.Tensor]]) -> torch.Tensor:
    """Convert a block representation into a matrix.

    Assumes the diagonal blocks to be quadratic to automatically detect their dimension.

    Args:
        blocks: Nested tuple of tensors that contains the ``(i, j)``th matrix
            block at index ``[i, j]``.

    Returns:
        Two-dimensional matrix with concatenated and flattened blocks.
    """
    num_blocks = len(blocks)
    row_blocks = []

    for idx in range(num_blocks):
        block_num_rows = int(math.sqrt(blocks[idx][idx].numel()))
        col_blocks = [b.reshape(block_num_rows, -1) for b in blocks[idx]]
        row_blocks.append(torch.cat(col_blocks, dim=1))

    return torch.cat(row_blocks)


# %%
#
# Let's convert the block representation into a matrix, and check that the
# multiplication onto ``v`` leads to the same result:

H_mat = blocks_to_matrix(H_mat).detach().cpu().numpy()
Hv_functorch = H_mat @ v

if numpy.allclose(Hv, Hv_functorch):
    print("Hessian-vector products match.")
else:
    raise ValueError("Hessian-vector products don't match.")


# %%
# Hessian-matrix products
# -----------------------
#
# We can also compute the Hessian's matrix representation with the linear
# operator, simply by multiplying it onto the identity matrix. (Of course, this
# only works if the Hessian is small enough.)
H_mat_from_linop = H @ numpy.eye(D)

# %%
#
# This should yield the same matrix as with :code:`functorch`.

if numpy.allclose(H_mat, H_mat_from_linop):
    print("Hessians match.")
else:
    raise ValueError("Hessians don't match.")

# %%
#
# Last, here's a visualization of the Hessian.

plt.figure()
plt.title("Hessian")
plt.imshow(H_mat)
plt.colorbar()
