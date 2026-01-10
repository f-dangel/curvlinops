r"""
Matrix-vector products
======================

This tutorial contains a basic demonstration how to set up ``LinearOperators``
for the Hessian and the GGN and how to multiply them to a vector.

First, the imports.
"""

import matplotlib.pyplot as plt
from torch import cat, cuda, device, eye, manual_seed, nn, rand

from curvlinops import GGNLinearOperator, HessianLinearOperator
from curvlinops.examples.functorch import functorch_ggn, functorch_hessian
from curvlinops.utils import allclose_report

# make deterministic
manual_seed(0)

# %%
# Setup
# -----
# Let's create some toy data, a small MLP, and use mean-squared error as loss function.

N = 4
D_in = 7
D_hidden = 5
D_out = 3

DEVICE = device("cuda" if cuda.is_available() else "cpu")

X = rand(N, D_in, device=DEVICE)
y = rand(N, D_out, device=DEVICE)

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
# We can now multiply the Hessian onto a vector.

D = H.shape[0]
v = rand(D, device=DEVICE)

Hv = H @ v

# %%
#
# To verify the result, we compute the Hessian using ``functorch``, using a
# utility function from ``curvlinops.examples``:

H_mat = functorch_hessian(model, loss_function, params, data).detach()

# %%
#
# Let's check that the multiplication onto ``v`` leads to the same result:

Hv_functorch = H_mat @ v

print("Comparing Hessian-vector product with functorch's Hessian-vector product.")
assert allclose_report(Hv, Hv_functorch)


# %%
# Hessian-matrix products
# -----------------------
#
# We can also compute the Hessian's matrix representation with the linear
# operator, simply by multiplying it onto the identity matrix. (Of course, this
# only works if the Hessian is small enough.)
H_mat_from_linop = H @ eye(D, device=DEVICE)

# %%
#
# This should yield the same matrix as with :code:`functorch`.

print("Comparing Hessian with functorch's Hessian.")
assert allclose_report(H_mat, H_mat_from_linop)

# %%
#
# Last, here's a visualization of the Hessian.

plt.figure()
plt.title("Hessian")
plt.imshow(H_mat.cpu())
plt.colorbar()

# %%
# Accepted vector/matrix formats
# ------------------------------
#
# Curvature matrices are usually defined w.r.t. parameters of a neural net. In PyTorch,
# these parameters are split into multiple tensors (e.g. per layer). It is often more
# convenient to think and work with vectors/matrices defined in this list format, rather
# than in the flattened-and-concatenated parameter space.
#
# So far, we have only used vectors/matrices in the flattened-and-concatenated format.
# To account for the often more convenient list format, all linear operators in
# ``curvlinops`` can also handle vectors/matrices specified in tensor list format.
# In that format, a matrix is a list of tensors, each of which has the same shape as
# its corresponding parameter plus an additional trailing dimension for the matrix's
# column dimension.
#
# ``curvlinops`` preserves the format when performing matrix multiplies: If the input
# lived in the flattened-and-concatenated parameter space, the result will be as well.
# If the input lived in the tensor list parameter space, the result will be a tensor
# list as well.
#
# Let's make this concrete. First, set up the same matrix in flattened and list format:

num_columns = 3

print(f"Total network parameters: {D}")
print(f"Parameter shapes: {[p.shape for p in params]}")
print(f"Number of columns: {num_columns}")

# Matrix in tensor list format
M_list = [rand(*p.shape, num_columns, device=DEVICE) for p in params]
print(f"[Tensor list format] Matrix: {[m.shape for m in M_list]}")

# Matrix in flattened format (what we have been using before)
M_flat = cat([m.flatten(end_dim=-2) for m in M_list])
print(f"[Flat format] Matrix: {M_flat.shape}")

# %%
#
# Next, let's carry out the Hessian-matrix product and inspect the result's format:

HM_list = H @ M_list
print(f"[Tensor list format] Hessian-matrix product: {[hm.shape for hm in HM_list]}")

HM_flat = H @ M_flat
print(f"[Flat format] Hessian-matrix product: {HM_flat.shape}")

# %%
#
# As expected, this produces the same result:

HM_list_flattened = cat([hm.flatten(end_dim=-2) for hm in HM_list])

print("Comparing Hessian-matrix products across formats.")
assert allclose_report(HM_flat, HM_list_flattened)

# %%
#
# **Note:** Like in the early part of the tutorial, the column dimension is not
# necessary if we just want to multiply the Hessian onto a single vector.
#
# GGN-vector products
# -------------------
#
# Setting up a linear operator for the Fisher/GGN is identical to the Hessian.

GGN = GGNLinearOperator(model, loss_function, params, data)

# %%
#
# This is one of ``curvlinops``'s design features: All linear operators share the same
# interface, making it easy to switch between curvature matrices.
#
# Let's compute a GGN-vector product.

D = H.shape[0]
v = rand(D, device=DEVICE)

GGNv = GGN @ v

# %%
#
# To verify the result, we will use ``functorch`` to compute the GGN. For that,
# we use that the GGN corresponds to the Hessian if we replace the neural
# network by its linearization. This is implemented in a utility function of
# :code:`curvlinops.examples`:

GGN_mat = functorch_ggn(model, loss_function, params, data).detach()

GGNv_functorch = GGN_mat @ v

print("Comparing GGN-vector product with functorch's GGN-vector product.")
assert allclose_report(GGNv, GGNv_functorch)

# %%
# GGN-matrix products
# -------------------
#
# We can also compute the GGN matrix representation with the linear operator,
# simply by multiplying it onto the identity matrix. (Of course, this only
# works if the GGN is small enough.)
GGN_mat_from_linop = GGN @ eye(D, device=DEVICE)

# %%
#
# This should yield the same matrix as with :code:`functorch`.

print("Comparing GGN with functorch's GGN.")
assert allclose_report(GGN_mat, GGN_mat_from_linop)

# %%
#
# Last, here's a visualization of the GGN.

plt.figure()
plt.title("GGN")
plt.imshow(GGN_mat.cpu())
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
ax[0].imshow(H_mat.cpu(), vmin=min_value, vmax=max_value)
ax[1].set_title("GGN")
ax[1].imshow(GGN_mat.cpu(), vmin=min_value, vmax=max_value)
