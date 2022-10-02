r"""Inverses (natural gradient)
===============================

This example demonstrates how to work with inverses of linear operators.

Concretely, we will compute the natural gradient :math:`\mathbf{\tilde{g}} =
\mathbf{F}^{-1} \mathbf{g}`,  defined by the inverse Fisher information
matrix :math:`\mathbf{F}^{-1}` and the gradient :math:`\mathbf{g}`. We can use
the GGN, as it corresponds to the Fisher for common loss functions like square
and cross-entropy loss.

.. note::
    The GGN is positive semi-definite, i.e. not full-rank. But we need a
    full-rank matrix to form the inverse. This is why we will add a damping term
    :math:`\delta \mathbf{I}` before inverting.

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


# %
#
# Next, let's compute the ingredients for the natural gradient.
#
# Inverse GGN/Fisher
# ------------------
#
# First,  we set up a linear operator for the damped GGN/Fisher

data = [(X1, y1), (X2, y2)]
GGN = GGNLinearOperator(model, loss_function, params, data)

delta = 1e-2
damping = aslinearoperator(delta * sparse.eye(GGN.shape[0]))

damped_GGN = GGN + damping

# %%
#
# and the linear operator of its inverse:

inverse_damped_GGN = CGInverseLinearOperator(damped_GGN)

# %%
#
# Gradient
# --------
#
# We can obtain the gradient via a convenience function of :code:`GGNLinearOperator`:

gradient, _ = GGN.gradient_and_loss()
# convert to numpy (vector) format
gradient = nn.utils.parameters_to_vector(gradient).cpu().detach()

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

print("Comparing gradient with Fisher @ natural gradient.")
report_nonclose(approx_gradient, gradient, rtol=1e-4, atol=1e-5)

# %%
#
# Verifying results
# -----------------
#
# To check if the code works, let's compute the GGN with :code:`functorch`,
# damp it, invert it, and multiply it onto the gradient.

# TODO Refactor: Remove code duplication in other example.

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


X = torch.cat([X for (X, _) in data])
y = torch.cat([y for (_, y) in data])


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
GGN_mat_functorch = GGN_functorch(X, y, anchor, model_params)

# %%
#
# ``functorch``'s output is a nested tuple that contains the GGN blocks of
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

GGN_mat_functorch = blocks_to_matrix(GGN_mat_functorch).detach().cpu().numpy()

# %%
#
#  then damp it and invert it.

damping_mat = delta * numpy.eye(GGN_mat_functorch.shape[0])
damped_GGN_mat = GGN_mat_functorch + damping_mat
inv_damped_GGN_mat = numpy.linalg.inv(damped_GGN_mat)


# %%
#
#  Next, let's compute the gradient with :code:`functorch`:


def loss(X: torch.Tensor, y: torch.Tensor, params: Tuple[torch.Tensor]) -> torch.Tensor:
    """Compute the loss given a mini-batch (X, y) and the neural network parameters."""
    output = model_fn(params, X)
    return loss_function_fn(loss_function_fn_params, output, y)


params_argnum = 2
H_functorch = functorch.hessian(loss, argnums=params_argnum)

gradient_functorch = functorch.grad(loss, argnums=params_argnum)(X, y, params)
# convert to numpy (vector) format
gradient_functorch = (
    nn.utils.parameters_to_vector(gradient_functorch).detach().cpu().numpy()
)

print("Comparing gradient with functorch's gradient.")
report_nonclose(gradient, gradient_functorch)

# %%
#
#  We can now compute the natural gradient from the :code:`functorch
#  `quantities. This should yield approximately the same result:

natural_gradient_functorch = inv_damped_GGN_mat @ gradient_functorch

print("Comparing natural gradient with functorch's natural gradient.")
report_nonclose(natural_gradient, natural_gradient_functorch, rtol=5e-3, atol=1e-5)

# %%
#
# Visual comparison
# -----------------
#
# Finally, let's visualize the damped Fisher/GGN and its inverse. For improved
# visibility, we take the logarithm of the absolute value of each element
# (blank pixels correspond to zeros).

fig, ax = plt.subplots(ncols=2)
plt.suptitle("Logarithm of absolute values")

ax[0].set_title("Damped GGN/Fisher")
image = ax[0].imshow(numpy.log10(numpy.abs(damped_GGN_mat)))
plt.colorbar(image, ax=ax[0], shrink=0.5)

ax[1].set_title("Inv. damped GGN/Fisher")
image = ax[1].imshow(numpy.log10(numpy.abs(inv_damped_GGN_mat)))
plt.colorbar(image, ax=ax[1], shrink=0.5)
