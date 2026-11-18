r"""Influence functions (EKFAC)
==============================

This example demonstrates how to estimate `influence functions
<https://arxiv.org/abs/1703.04730>`_ with the eigenvalue-corrected
Kronecker-factored approximate curvature (EKFAC) linear operator.

Influence functions quantify how much an individual training point contributes to
a model's prediction on a test point. Removing a training point :math:`z_j` and
re-training is expensive, so we approximate the effect of an infinitesimal
up-weighting of :math:`z_j` on the loss at a test point :math:`z_{\text{test}}`
with

.. math::
    \mathcal{I}(z_{\text{test}}, z_j)
    = - \nabla_{\boldsymbol{\theta}}
    \ell(z_{\text{test}})^\top \mathbf{H}^{-1}
    \nabla_{\boldsymbol{\theta}} \ell(z_j)\,,

where :math:`\mathbf{H}` is the curvature (Hessian/GGN/Fisher) of the empirical
risk at the trained parameters :math:`\boldsymbol{\theta}`. A large positive
value means that up-weighting :math:`z_j` *increases* the test loss (the point is
harmful), while a large negative value means up-weighting :math:`z_j`
*decreases* the test loss (the point is helpful/influential).

The bottleneck is the inverse-curvature-vector product
:math:`\mathbf{H}^{-1} \nabla_{\boldsymbol{\theta}} \ell(z_j)`. Forming and
inverting the full curvature matrix is infeasible for realistic networks.
Instead, we use :code:`curvlinops`'
:py:class:`~curvlinops.EKFACLinearOperator`, whose
:py:meth:`~curvlinops.EKFACLinearOperator.inverse` method provides a cheap,
matrix-free inverse. This mirrors the influence-function recipe popularized by
`kronfluence <https://github.com/pomonam/kronfluence>`_.

As always, let's first import the required functionality.
"""

import matplotlib.pyplot as plt
from torch import (
    Tensor,
    device,
    float64,
    manual_seed,
    randn,
    zeros,
)
from torch.nn import Linear, MSELoss, Sequential, Tanh
from torch.nn.utils import parameters_to_vector
from torch.optim import Adam

from curvlinops import EKFACLinearOperator, FisherType
from curvlinops.examples import gradient_and_loss

# make deterministic
_ = manual_seed(0)

# %%
#
# Problem setup
# -------------
#
# We use a small synthetic regression problem: the inputs are drawn from a
# standard normal distribution and the targets follow a smooth non-linear
# function corrupted by a little noise. To make the influence analysis
# interesting, we deliberately inject a handful of *mislabelled* outliers whose
# targets are strongly corrupted. We expect these outliers to be flagged as
# harmful, i.e. to have large positive influence on the (clean) test loss.

DEVICE = device("cpu")
DTYPE = float64  # double precision improves the stability of the inverse

N_train = 32
N_test = 16
N_outliers = 3
D_in = 4
D_hidden = 8
D_out = 1


def target_function(inputs: Tensor) -> Tensor:
    """Smooth non-linear ground-truth mapping used to generate targets.

    Args:
        inputs: Input features of shape ``(N, D_in)``.

    Returns:
        Targets of shape ``(N, D_out)``.
    """
    return (inputs.sin().sum(dim=1, keepdim=True) + 0.5 * inputs[:, :1] ** 2).to(DTYPE)


X_train = randn(N_train, D_in, device=DEVICE, dtype=DTYPE)
y_train = target_function(X_train) + 0.05 * randn(
    N_train, D_out, device=DEVICE, dtype=DTYPE
)

# corrupt the last ``N_outliers`` targets to create harmful, mislabelled points
y_train[-N_outliers:] = -3.0 * y_train[-N_outliers:] - 3.0

# a clean held-out test set; influence is measured on the *mean* test loss
X_test = randn(N_test, D_in, device=DEVICE, dtype=DTYPE)
y_test = target_function(X_test)

model = Sequential(
    Linear(D_in, D_hidden),
    Tanh(),
    Linear(D_hidden, D_out),
).to(DEVICE, DTYPE)

loss_function = MSELoss(reduction="mean").to(DEVICE, DTYPE)

# %%
#
# Training the model
# ------------------
#
# We briefly train the model with Adam so that the parameters sit close to a
# local minimum, where the influence-function approximation is valid.

optimizer = Adam(model.parameters(), lr=1e-2)
for _ in range(300):
    optimizer.zero_grad()
    loss = loss_function(model(X_train), y_train)
    loss.backward()
    optimizer.step()

print(f"Final training loss: {loss.item():.5f}")

# collect the trained parameters (KFAC/EKFAC only supports Linear/Conv2d layers)
params = {n: p for n, p in model.named_parameters() if p.requires_grad}

# %%
#
# Curvature: EKFAC linear operator
# --------------------------------
#
# We approximate the curvature :math:`\mathbf{H}` of the empirical risk by the
# EKFAC approximation of the GGN/Fisher. We use the exact (type-2) Fisher so the
# operator is deterministic, and pass the full training set as a single-batch
# data loader.

data = [(X_train, y_train)]
ekfac = EKFACLinearOperator(
    model,
    loss_function,
    params,
    data,
    fisher_type=FisherType.TYPE2,
)

# %%
#
# The inverse curvature is obtained through
# :py:meth:`~curvlinops.EKFACLinearOperator.inverse`. Because the GGN/Fisher is
# only positive *semi*-definite, we add a small damping term to the eigenvalues
# before inverting, i.e. we invert :math:`\mathbf{H} + \delta \mathbf{I}`.

damping = 1e-3
ekfac_inv = ekfac.inverse(damping=damping)

# %%
#
# Per-example gradients
# ---------------------
#
# The influence formula requires the gradient of the mean loss over the clean
# test set and the loss gradient at every individual training point. We reuse
# :code:`curvlinops`' :code:`gradient_and_loss` convenience
# function to obtain these flat gradient vectors.


def flat_gradient(inputs: Tensor, targets: Tensor) -> Tensor:
    """Compute the flattened mean-loss gradient w.r.t. the parameters.

    Args:
        inputs: Input features of shape ``(N, D_in)``.
        targets: Targets of shape ``(N, D_out)``.

    Returns:
        Flat gradient vector matching the concatenated parameter shapes.
    """
    grad, _ = gradient_and_loss(model, loss_function, params, [(inputs, targets)])
    return parameters_to_vector(grad).detach()


grad_test = flat_gradient(X_test, y_test)

# %%
#
# Influence scores
# ----------------
#
# We first pre-compute the inverse-curvature-vector product with the test
# gradient, :math:`\mathbf{s} = \mathbf{H}^{-1} \nabla \ell(z_{\text{test}})`
# (the "influence embedding" of the test point). The influence of each training
# point is then a simple dot product
# :math:`\mathcal{I}(z_{\text{test}}, z_j) = - \mathbf{s}^\top \nabla \ell(z_j)`.

s_test = ekfac_inv @ grad_test

influences = zeros(N_train, device=DEVICE, dtype=DTYPE)
for j in range(N_train):
    grad_train_j = flat_gradient(X_train[j : j + 1], y_train[j : j + 1])
    influences[j] = -(s_test @ grad_train_j)

# %%
#
# Let's inspect the most and least influential training points. Positive scores
# flag harmful points (up-weighting them increases the test loss); negative
# scores flag helpful points. The injected outliers are the last
# ``N_outliers`` indices.

order = influences.argsort(descending=True)
outlier_indices = set(range(N_train - N_outliers, N_train))

print("\nMost harmful training points (largest positive influence):")
for rank in range(3):
    idx = order[rank].item()
    tag = " <- injected outlier" if idx in outlier_indices else ""
    print(f"  train point {idx:2d}: influence = {influences[idx]:+.4e}{tag}")

print("\nMost helpful training points (largest negative influence):")
for rank in range(3):
    idx = order[-(rank + 1)].item()
    tag = " <- injected outlier" if idx in outlier_indices else ""
    print(f"  train point {idx:2d}: influence = {influences[idx]:+.4e}{tag}")

# %%
#
# Visualizing the influences
# --------------------------
#
# Finally, we plot the influence score of every training point. The mislabelled
# outliers (highlighted) surface among the points with the largest positive
# influence: up-weighting them would most increase the clean test loss. This
# shows how EKFAC-based influence functions help flag harmful training data.
# Influence is only a first-order approximation, so the correspondence is not
# perfect: an occasional corrupted point may still align helpfully with a
# given test set.

fig, ax = plt.subplots()
ax.set_title("EKFAC influence of each training point on the test loss")
ax.set_xlabel("training point index")
ax.set_ylabel(r"influence $\mathcal{I}(z_\mathrm{test}, z_j)$")

colors = ["tab:red" if j in outlier_indices else "tab:blue" for j in range(N_train)]
ax.bar(range(N_train), influences.cpu(), color=colors)
ax.axhline(0.0, color="black", linewidth=0.8)

# legend proxies
ax.bar(0, 0, color="tab:blue", label="clean")
ax.bar(0, 0, color="tab:red", label="injected outlier")
_ = ax.legend()

# %%
#
# Verifying the inverse
# ---------------------
#
# As a sanity check, we confirm that the EKFAC inverse indeed acts as the
# inverse of the EKFAC operator on the test gradient, i.e.
# :math:`(\mathbf{H} + \delta \mathbf{I}) \mathbf{s} \approx
# \nabla \ell(z_{\text{test}})`.

reconstructed = ekfac @ s_test + damping * s_test
abs_err = (reconstructed - grad_test).abs().max().item()
print(f"\nMax abs. reconstruction error of the damped inverse: {abs_err:.2e}")
assert reconstructed.allclose(grad_test, rtol=1e-3, atol=1e-5)
