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

from curvlinops import GGNLinearOperator, HessianLinearOperator

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
    nn.Sigmoid(),
    nn.Linear(D_hidden, D_out),
).to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]

loss_function = nn.MSELoss(reduction="mean").to(DEVICE)


# %%
# Hessian-vector products
# -----------------------
#
# Setting up a linear operator for the Hessian is straightforward.

data = [(X, y)]
H = HessianLinearOperator(model, loss_function, params, data)

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

# %%
# GGN-vector products
# -------------------
#
# Setting up a linear operator for the Fisher/GGN is identical to the Hessian.

GGN = GGNLinearOperator(model, loss_function, params, data)

# %%
#
# Let's compute a GGN-vector product.

D = H.shape[0]
v = numpy.random.rand(D)

GGNv = GGN @ v

# %%
#
# To verify the result, we will use ``functorch`` to compute the GGN. For that,
# we use that the GGN corresponds to the Hessian if we replace the neural
# network by its linearization.
#
# As above, we first convert the PyTorch layers into functions, then linearize
# the model, and use it to construct the function that produces the loss. Then,
# we can take the Hessian of that function w.r.t. to the linearized neural
# network's parameters:

# convert modules to functions
model_fn, model_params = functorch.make_functional(model)
loss_function_fn, loss_function_fn_params = functorch.make_functional(loss_function)


def linearized_model(
    anchor: Tuple[torch.Tensor], params: Tuple[torch.Tensor], X: torch.Tensor
) -> torch.Tensor:
    """Evaluate the model at params, using its linearization around anchor."""

    def model_fn_params_only(params: Tuple[torch.Tensor]) -> torch.Tensor:
        return model_fn(params, X)

    diff = tuple(p - a for p, a in zip(params, anchor))
    model_at_anchor, jvp = functorch.jvp(model_fn_params_only, (anchor,), (diff,))

    return model_at_anchor + jvp


def linearized_loss(
    X: torch.Tensor,
    y: torch.Tensor,
    anchor: Tuple[torch.Tensor],
    params: Tuple[torch.Tensor],
) -> torch.Tensor:
    """Compute the loss given a mini-batch (X, y) under a linearized NN around anchor.

    Returns:
        f(X, θ₀) + (J_θ₀ f(X, θ₀)) @ (θ - θ₀) with f the neural network, θ₀ the anchor
        point of the linearization, and θ the evaluation point.
    """
    output = linearized_model(anchor, params, X)
    return loss_function_fn(loss_function_fn_params, output, y)


params_argnum = 3
GGN_functorch = functorch.hessian(linearized_loss, argnums=params_argnum)

# %%
#
# This yields a function that computes the GGN:

anchor = tuple(p.clone() for p in model_params)
GGN_mat = GGN_functorch(X, y, anchor, model_params)

# %%
#
# ``functorch``'s output is a nested tuple that contains the GGN blocks of
# :code:`(params[i], params[j])` at position :code:`[i, j]`. Let's convert the
# block representation into a matrix, and check that the multiplication onto
# ``v`` leads to the same result:

GGN_mat = blocks_to_matrix(GGN_mat).detach().cpu().numpy()

GGNv_functorch = GGN_mat @ v

if numpy.allclose(GGNv, GGNv_functorch):
    print("GGN-vector products match.")
else:
    raise ValueError("GGN-vector products don't match.")

# %%
# GGN-matrix products
# -------------------
#
# We can also compute the GGN matrix representation with the linear operator,
# simply by multiplying it onto the identity matrix. (Of course, this only
# works if the GGN is small enough.)
GGN_mat_from_linop = GGN @ numpy.eye(D)

# %%
#
# This should yield the same matrix as with :code:`functorch`.

if numpy.allclose(GGN_mat, GGN_mat_from_linop):
    print("GGNs match.")
else:
    raise ValueError("GGNs don't match.")

# %%
#
# Last, here's a visualization of the GGN.

plt.figure()
plt.title("GGN")
plt.imshow(GGN_mat)
plt.colorbar()

# %%
# Visual comparison: Hessian and GGN
# ----------------------------------
#
# To conclude, let's plot both the Hessian and GGN using the same limits

min_value = min(GGN_mat.min(), H_mat.min())
max_value = max(GGN_mat.max(), H_mat.max())

fig, ax = plt.subplots(ncols=2)
ax[0].set_title("Hessian")
ax[0].imshow(H_mat, vmin=min_value, vmax=max_value)
ax[1].set_title("GGN")
ax[1].imshow(GGN_mat, vmin=min_value, vmax=max_value)
