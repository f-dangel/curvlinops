r"""Inverse linear operators (natural gradient)
===============================================

This example demonstrates how to work with inverses of linear operators.

Concretely, we will compute the natural gradient :math:`\mathbf{\tilde{g}} =
\mathbf{F}^{-1} \mathbf{g}` that is defined by the inverse Fisher information
matrix :math:`\mathbf{F}^{-1}` and the gradient :math:`\mathbf{g}`. We can do
that with the GGN inverse, as it corresponds to the Fisher for common loss
functions like square and cross-entropy.

(The GGN is positive semi-definite, i.e. not full-rank. but we need a full-rank
matrix for the inverse. This is why we will add a damping term :math:`\delta
\mathvf{I}` to it before inverting.)

As always, let's first import the required functionality.
"""

import math
from typing import Tuple

import functorch
import matplotlib.pyplot as plt
import numpy
import torch
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator
from torch import nn

from curvlinops import CGInverseLinearOperator, GGNLinearOperator

# make deterministic
torch.manual_seed(0)
numpy.random.seed(0)


# %%
#
# Setup
# -----
#
# Let's create some toy data, consisting of two mini-batches, a small MLP, and
# use mean-squared error as loss function.

N = 8
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
# Inverse GGN/Fisher
# ------------------
#
# Next, we set up the linear operator for the damped GGN/Fisher, and the linear
# operator of its inverse.

data = [(X1, y1), (X2, y2)]
GGN = GGNLinearOperator(model, loss_function, params, data)

delta = 1e-5
damping = aslinearoperator(delta * sparse.eye(GGN.shape[0]))

damped_GGN = GGN + damping

inverse_damped_GGN = CGInverseLinearOperator(damped_GGN)

# %%
#
# Gradient
# --------
#
# To compute the gradient, we can use a convenience function from the
# :code:`GGNLinearOperator`:

gradient, _ = GGN.gradient_and_loss()
# flatten, concatenate into a vector, and convert to scipy
gradient = numpy.concatenate([g.flatten().cpu().numpy() for g in gradient])

# %%
#
# Natural gradient
# ----------------
#
# Now we have all components together to compute the natural gradient with a
# simple matrix-vector product:

natural_gradient = inverse_damped_GGN @ gradient

# %%
#
# As a first sanity check, let's compare if the natural gradient satisfies
# :math:`\mathbf{F} \mathbf{\tilde{g}} = \mathbf{g}`

approx_gradient = damped_GGN @ natural_gradient

rtol, atol = 1e-4, 1e-5
if numpy.allclose(approx_gradient, gradient, rtol=rtol, atol=atol):
    print("Fisher applied onto the natural gradient matches gradient.")
else:
    for approx_g, g in zip(approx_gradient, gradient):
        if not numpy.isclose(approx_g, g, atol=atol, rtol=rtol):
            print(f"{approx_g} ≠ {g}")
    raise ValueError(
        "Fisher applied onto the natural gradient does not match the gradient."
    )


# %%
#
# Double-check
# ------------
#
# Finally, let's compute the GGN with :code:`functorch`, damp it, invert it,
# and multiply it onto the gradient to check if the code works.

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


# TODO Stack X1, X2 into X
# TODO Stack y1, y2 into y
# TODO Refactor: Remove code duplication in other example.


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
#  Let's convert the block representation into a matrix,

GGN_mat = blocks_to_matrix(GGN_mat).detach().cpu().numpy()

# %%
#
#  then damp it and invert it.

damping_mat = delta * numpy.eye(GGN_mat.shape[0])
inv_damped_GGN_mat = numpy.linalg.inv(GGN_mat + damping_mat)

# TODO Compute the gradient
# TODO Compute the natural gradient
# TODO Compare the results
