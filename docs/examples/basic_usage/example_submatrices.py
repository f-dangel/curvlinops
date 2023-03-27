r"""
Sub-matrices of linear operators
================================

This tutorial explains how to create linear operators that correspond to a sub-matrix
of another linear operator.

Specifically, given the linear operator :code:`A`, we are
interested in constructing the linear operator that corresponds to its sub-matrix
:code:`A[row_idxs, :][:, col_idxs]`, where :code:`row_idxs` contains the sub-matrix's
row indices, and :code:`col_idxs` contains the sub-matrix's column indices.

First, the imports.
"""

from time import time
from typing import List

import numpy
import torch
from torch import nn

from curvlinops import HessianLinearOperator
from curvlinops.examples.functorch import functorch_hessian
from curvlinops.examples.utils import report_nonclose
from curvlinops.submatrix import SubmatrixLinearOperator

# make deterministic
torch.manual_seed(0)
numpy.random.seed(0)

# %%
#
# Setup
# -----
#
# Let's create some toy data, a small MLP, and use mean-squared error as loss function.

N = 4
D_in = 7
D_hidden = 5
D_out = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X1, y1 = torch.rand(N, D_in).to(DEVICE), torch.rand(N, D_out).to(DEVICE)
X2, y2 = torch.rand(N, D_in).to(DEVICE), torch.rand(N, D_out).to(DEVICE)
data = [(X1, y1), (X2, y2)]

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
# We will investigate the Hessian. To make sure our results are correct, let's keep
# a Hessian matrix computed via :mod:`functorch` around.

H_functorch = (
    functorch_hessian(model, loss_function, params, data).detach().cpu().numpy()
)

# %%
#
# Here is the corresponding linear operator and a quick check that builds up
# its matrix representation through multiplication with the identity matrix,
# followed by comparison to the Hessian matrix computed via :mod:`functorch`.

H = HessianLinearOperator(model, loss_function, params, data)

report_nonclose(H_functorch, H @ numpy.eye(H.shape[1]))

# %%
#
# Diagonal blocks
# ---------------
#
# The Hessian consists of blocks :code:`(i, j)` that contain the second-order
# derivatives of the loss w.r.t. the parameters in :code:`(params[i], params[j])`.
#
# Let's define a function to extract these blocks from the Hessian:


def extract_block(
    mat: numpy.ndarray, params: List[torch.Tensor], i: int, j: int
) -> numpy.ndarray:
    """Extract the Hessian block from parameters ``i`` and ``j``.

    Args:
        mat: The matrix with block structure.
        params: The parameters defining the blocks.
        i: Row index of the block to be extracted.
        j: Column index of the block to be extracted.

    Returns:
        Block ``(i, j)``. Has shape ``[params[i].numel(), params[j].numel()]``.
    """
    param_dims = [p.numel() for p in params]
    row_start, row_end = sum(param_dims[:i]), sum(param_dims[: i + 1])
    col_start, col_end = sum(param_dims[:j]), sum(param_dims[: j + 1])

    return mat[row_start:row_end, :][:, col_start:col_end]


# %%
#
# As an example, let's extract the block that corresponds to the Hessian w.r.t.
# the first layer's weights in our model.

i, j = 0, 0
H_param0_functorch = extract_block(H_functorch, params, i, j)

# %%
#
# We can build a linear operator for this sub-Hessian by only providing the
# first layer's weight as parameter:

H_param0 = HessianLinearOperator(model, loss_function, [params[i]], data)

# %%
#
# Like this we can get blocks from the diagonal.
#
# Let's check that this linear operator works as expected by multiplying it
# onto the identity matrix and comparing the result to the block we extracted
# from our ground truth:

report_nonclose(H_param0_functorch, H_param0 @ numpy.eye(params[i].numel()))

# %%
#
# Now you might be wondering if we can also build up linear operators for
# off-diagonal blocks. These blocks contain mixed second-order derivatives and
# are not Hessians anymore. For instance, such a block is rectangular in
# general, and thus non-symmetric. Since we are not asking for a Hessian
# anymore, we cannot use the interface of :class:`HessianLinearOperator`.
#
# Luckily, there is a different way to achieve this.
#
# Off-diagonal blocks
# -------------------
#
# As an example,let's try to extract the Hessian block from the first and
# second parameters in our network (i.e. the weights and biases in the first
# layer). For that we need to slice the Hessian differently along its rows and
# columns. We can use the :class:`curvlinops.SubmatrixLinearOperator` class for
# that:

param_dims = [p.numel() for p in params]
i, j = 0, 1
row_start, row_end = sum(param_dims[:i]), sum(param_dims[: i + 1])
col_start, col_end = sum(param_dims[:j]), sum(param_dims[: j + 1])

row_idxs = list(range(row_start, row_end))  # keep the following row indices
col_idxs = list(range(col_start, col_end))  # keep the following column indices

H_param0_param1 = SubmatrixLinearOperator(H, row_idxs, col_idxs)

# %%
#
# As the following test shows, this linear operator indeed represents the
# desired rectangular Hessian block:

H_param0_param1_functorch = extract_block(H_functorch, params, i, j)

report_nonclose(
    H_param0_param1_functorch,
    H_param0_param1_functorch @ numpy.eye(param_dims[j]),
)

# %%
#
# Arbitrary sub-matrices
# ----------------------
#
# So far, we were constrained to blocks spanned by parameter tensors rather
# than arbitrary elements. As the name :class:`SubmatrixLinearOperator`
# suggests, we can use it to create arbitrary sub-matrices.
#
# As an example, let's say we want to keep rows :code:`[0, 13, 42]` of the
# Hessian, and columns :code:`[1, 2, 3]`. This works as follows:

row_idxs = [0, 13, 42]  # keep the following row indices
col_idxs = [1, 2, 3]  # keep the following column indices

H_sub = SubmatrixLinearOperator(H, row_idxs, col_idxs)
H_sub_functorch = H_functorch[row_idxs, :][:, col_idxs]

# %%
#
# Quick check to see if it worked:

report_nonclose(H_sub_functorch, H_sub @ numpy.eye(len(col_idxs)))

# %%
#
# Looks good.
#
# Performance remarks
# ----------------------
#
# By the way, using this interface, we could have also constructed the first
# parameter's Hessian as follows:

i, j = 0, 0
row_start, row_end = sum(param_dims[:i]), sum(param_dims[: i + 1])
col_start, col_end = sum(param_dims[:j]), sum(param_dims[: j + 1])

row_idxs = list(range(row_start, row_end))
col_idxs = list(range(col_start, col_end))

H_param0_alternative = SubmatrixLinearOperator(H, row_idxs, col_idxs)

report_nonclose(H_param0_functorch, H_param0_alternative @ numpy.eye(param_dims[0]))

# %%
#
# In general though, it is a good idea to first reduce the linear operator's
# size as much as possible (in our case, by restricting the parameters to the
# necessary ones using the :code:`params` argument in
# :class:`HessianLinearOperator`) and apply slicing afterwards to save
# computations.
#
# In our example, the matrix-vector product of :code:`H_param0` should
# therefore be faster than that of :code:`H_param0_alternative`:

x = numpy.random.rand(param_dims[0])

# less computations
start = time()
H_param0 @ x
end = time()
print(f"H_param0.matvec: {end - start:.2e} s")

# more computations
start = time()
H_param0_alternative @ x
end = time()
print(f"H_param0_alternative.matvec: {end - start:.2e} s")

# %%
#
# That's all for now.
