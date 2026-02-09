r"""Fisher-weighted Model Averaging
===================================

In this example we implement Fisher-weighted model averaging, a technique
described in this `NeurIPS 2022 paper <https://openreview.net/pdf?id=LSKlp_aceO>`_.
It requires Fisher-vector products, and multiplication with the inverse of a
sum of Fisher matrices. The paper uses a diagonal approximation of the Fisher
matrices. In addition, we will also use the exact Fisher matrices and rely on
matrix-free methods for applying the inverse.

.. note::
   In our setup, the Fisher equals the generalized Gauss-Newton matrix.
   Hence, we work with :py:class:`curvlinops.GGNLinearOperator`.

**Description:** We are given a set of :math:`T` tasks (represented by data
sets :math:`\mathcal{D}_t`), and train a model :math:`f_\mathbf{\theta}` on
each task independently using the same criterion function. This yields
:math:`T` parameters :math:`\mathbf{\theta}_1^\star, \dots,
\mathbf{\theta}_T^\star`, and we would like to combine them into a single model
:math:`f_\mathbf{\theta^\star}`. To do that, we use the Fisher information
matrices :math:`\mathbf{F}_t` of each task (given by the data set
:math:`\mathcal{D}_t` and the trained model parameters
:math:`\mathbf{\theta}_t^\star`). The merged parameters are given by

.. math::
   \mathbf{\theta}^\star = \left(\lambda \mathbf{I}
   + \sum_{t=1}^T \mathbf{F}_t \right)^{-1}
   \left( \sum_{t=1}^T \mathbf{F}_t \mathbf{\theta}_t^\star\right)\,.

This requires multiplying with the inverse of the sum of Fisher matrices
(extended with a damping term). If we approximate each Fisher with its diagonal,
this is easy, without this approximation, we will use
:py:class:`curvlinops.CGInverseLinearOperator` for inversion.

Let's start with the imports.
"""

from backpack.utils.convert_parameters import vector_to_parameter_list
from torch import cuda, device, manual_seed, rand
from torch.nn import Linear, MSELoss, ReLU, Sequential, Sigmoid
from torch.nn.utils import parameters_to_vector
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from curvlinops import (
    CGInverseLinearOperator,
    GGNDiagonalLinearOperator,
    GGNLinearOperator,
)
from curvlinops.examples import IdentityLinearOperator

# make deterministic
manual_seed(0)

DEVICE = device("cuda" if cuda.is_available() else "cpu")

# %%
#
# Setup
# -----
#
# First, we will create a bunch of synthetic regression tasks (i.e. data sets)
# and an untrained model for each of them.

T = 3  # number of tasks
D_in = 7  # input dimension of each task
D_hidden = 5  # hidden dimension of the architecture we will use
D_out = 3  # output dimension of each task
N = 20  # number of data per task
batch_size = 7


def make_architecture() -> Sequential:
    """Create a neural network.

    Returns:
        A neural network.
    """
    return Sequential(
        Linear(D_in, D_hidden),
        ReLU(),
        Linear(D_hidden, D_hidden),
        Sigmoid(),
        Linear(D_hidden, D_out),
    )


def make_dataset() -> TensorDataset:
    """Create a synthetic regression data set.

    Returns:
        A synthetic regression data set.
    """
    X, y = rand(N, D_in), rand(N, D_out)
    return TensorDataset(X, y)


models = [make_architecture().to(DEVICE) for _ in range(T)]
data_loaders = [DataLoader(make_dataset(), batch_size=batch_size) for _ in range(T)]
loss_functions = [MSELoss(reduction="mean").to(DEVICE) for _ in range(T)]

# %%
#
# Training
# --------
#
# Here, we train each model for a small number of epochs.

num_epochs = 10
log_epochs = [0, num_epochs - 1]

for task_idx in range(T):
    model = models[task_idx]
    data_loader = data_loaders[task_idx]
    loss_function = loss_functions[task_idx]
    optimizer = SGD(model.parameters(), lr=1e-2)

    for epoch in range(num_epochs):
        for batch_idx, (X, y) in enumerate(data_loader):
            optimizer.zero_grad()
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = loss_function(model(X), y)
            loss.backward()
            optimizer.step()

            if epoch in log_epochs and batch_idx == 0:
                print(f"Task {task_idx} batch loss at epoch {epoch}: {loss.item():.3f}")

# %%
#
# Linear operators
# ----------------
#
# We are now ready to set up the linear operators for the per-task Fishers:

# full Fisher matrices
fishers = [
    GGNLinearOperator(
        model,
        loss_function,
        [p for p in model.parameters() if p.requires_grad],
        data_loader,
    )
    for model, loss_function, data_loader in zip(models, loss_functions, data_loaders)
]

# Diagonal approximation as used in the seminal paper
# (Precisely speaking, the seminal paper uses a randomized approximation of the Fisher
# based on sampling that can be achieved with `fisher_type='mc'` and `mc_samples=1`.
# For simplicity we compute the exact GGN/Fisher diagonal here.)
diagonal_fishers = [
    GGNDiagonalLinearOperator(
        model,
        loss_function,
        [p for p in model.parameters() if p.requires_grad],
        data_loader,
    )
    for model, loss_function, data_loader in zip(models, loss_functions, data_loaders)
]


# %%
#
# Fisher-weighted Averaging
# -------------------------
#
# Next, we also need the trained parameters as vectors:

# flatten and concatenate
thetas = [
    parameters_to_vector((p for p in model.parameters() if p.requires_grad)).detach()
    for model in models
]

# %%
#
# We are ready to compute the sum of Fisher-weighted parameters (the right-hand
# side in the above equation):

rhs = sum(fisher @ theta for fisher, theta in zip(fishers, thetas))
diagonal_rhs = sum(fisher @ theta for fisher, theta in zip(diagonal_fishers, thetas))

# %%
#
# In the last step we need to normalize by multiplying with the inverse of the
# summed Fishers. Let's first create the linear operator and add a damping
# term:

dim = fishers[0].shape[0]
param_shapes = [p.shape for p in models[0].parameters() if p.requires_grad]
identity = IdentityLinearOperator(param_shapes, fishers[0].device, rhs.dtype)
damping = 1e-3

fisher_sum = damping * identity
for fisher in fishers:
    fisher_sum += fisher

# %%
#
# Finally, we define a linear operator for the inverse of the damped Fisher sum:

fisher_sum_inv = CGInverseLinearOperator(fisher_sum)

# %%
#
# .. note::
#    You may want to tweak the convergence criterion of CG using
#    :py:func:`curvlinops.CGInverseLinearOperator.set_cg_hyperparameters`. before
#    applying the matrix-vector product.

fisher_weighted_params = fisher_sum_inv @ rhs


# %%
#
# Summing the diagonal Fishers then inverting their damped version is even simpler:

diagonal_fisher_sum = diagonal_fishers[0]
for diagonal_fisher in diagonal_fishers[1:]:
    diagonal_fisher_sum = diagonal_fisher + diagonal_fisher_sum

diagonal_fisher_sum_inv = diagonal_fisher_sum.inverse(damping)

diagonal_fisher_weighted_params = diagonal_fisher_sum_inv @ diagonal_rhs

# %%
#
# Comparison
# ----------
#
# Let's compare the performance of the Fisher-averaged parameters with a naive
# average.

average_params = sum(thetas) / len(thetas)

# %%
#
# We initialize three neural networks with those parameters

# Using the full Fisher
fisher_model = make_architecture()

params = [p for p in fisher_model.parameters() if p.requires_grad]
theta_fisher = vector_to_parameter_list(fisher_weighted_params, params)
for theta, param in zip(theta_fisher, params):
    param.data = theta.to(param.device, param.dtype).data

# Using the diagonal Fisher
diagonal_fisher_model = make_architecture()

params = [p for p in diagonal_fisher_model.parameters() if p.requires_grad]
theta_diagonal_fisher = vector_to_parameter_list(
    diagonal_fisher_weighted_params, params
)
for theta, param in zip(theta_diagonal_fisher, params):
    param.data = theta.to(param.device, param.dtype).data


# Using averages (setting the Fisher matrix to be identity)
average_model = make_architecture()

params = [p for p in average_model.parameters() if p.requires_grad]
theta_average = vector_to_parameter_list(average_params, params)
for theta, param in zip(theta_average, params):
    param.data = theta.to(param.device, param.dtype).data

# %%
#
# and probe them on one batch of each task:


losses = {"Naive": [], "diag(F)": [], "F": []}
header = "\t" + "\t".join(losses.keys())
print(header)

for task_idx in range(T):
    data_loader = data_loaders[task_idx]
    loss_function = loss_functions[task_idx]
    X, y = next(iter(data_loader))

    losses["F"].append(loss_function(fisher_model(X), y).item())
    losses["diag(F)"].append(loss_function(diagonal_fisher_model(X), y).item())
    losses["Naive"].append(loss_function(average_model(X), y).item())
    assert losses["F"][-1] < losses["Naive"][-1]

    print(
        f"Task {task_idx}\t{losses['Naive'][-1]:.3f}\t{losses['diag(F)'][-1]:.3f}\t{losses['F'][-1]:.3f}"
    )

mean_losses = {key: sum(loss) / len(loss) for key, loss in losses.items()}
print(
    f"Avg\t{mean_losses['Naive']:.3f}\t{mean_losses['diag(F)']:.3f}\t{mean_losses['F']:.3f}"
)

# %%
#
# The Fisher-averaged parameters perform better than the naively averaged
# parameters; at least on the training data.
#
# That's all for now.
