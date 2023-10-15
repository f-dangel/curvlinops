r"""Fisher-weighted Model Averaging
===================================

In this example we implement Fisher-weighted model averaging, a technique
described in this `NeurIPS 2022 paper <https://openreview.net/pdf?id=LSKlp_aceO>`_.
It requires Fisher-vector products, and multiplication with the inverse of a
sum of Fisher matrices. The paper uses a diagonal approximation of the Fisher
matrices. Instead, we will use the exact Fisher matrices and rely on
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
(extended with a damping term). We will use
:py:class:`curvlinops.CGInverseLinearOperator` for that.

Let's start with the imports.

"""

import numpy
import torch
from backpack.utils.convert_parameters import vector_to_parameter_list
from scipy import sparse
from scipy.sparse.linalg import aslinearoperator
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from curvlinops import CGInverseLinearOperator, GGNLinearOperator

# make deterministic
torch.manual_seed(0)
numpy.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def make_architecture() -> nn.Sequential:
    """Create a neural network.

    Returns:
        A neural network.
    """
    return nn.Sequential(
        nn.Linear(D_in, D_hidden),
        nn.ReLU(),
        nn.Linear(D_hidden, D_hidden),
        nn.Sigmoid(),
        nn.Linear(D_hidden, D_out),
    )


def make_dataset() -> TensorDataset:
    """Create a synthetic regression data set.

    Returns:
        A synthetic regression data set.
    """
    X, y = torch.rand(N, D_in), torch.rand(N, D_out)
    return TensorDataset(X, y)


models = [make_architecture().to(DEVICE) for _ in range(T)]
data_loaders = [DataLoader(make_dataset(), batch_size=batch_size) for _ in range(T)]
loss_functions = [nn.MSELoss(reduction="mean").to(DEVICE) for _ in range(T)]

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
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

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

fishers = [
    GGNLinearOperator(
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
# Next, we also need the trained parameters as :py:mod:`scipy` vectors:

# flatten and convert to numpy
thetas = [
    nn.utils.parameters_to_vector((p for p in model.parameters() if p.requires_grad))
    for model in models
]
thetas = [theta.cpu().detach().numpy() for theta in thetas]

# %%
#
# We are ready to compute the sum of Fisher-weighted parameters (the right-hand
# side in the above equation):

rhs = sum(fisher @ theta for fisher, theta in zip(fishers, thetas))

# %%
#
# In the last step we need to normalize by multiplying with the inverse of the
# summed Fishers. Let's first create the linear operator and add a damping
# term:

dim = fishers[0].shape[0]
identity = sparse.eye(dim)
damping = 1e-3

fisher_sum = aslinearoperator(damping * identity)

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
# Comparison
# ----------
#
# Let's compare the performance of the Fisher-averaged parameters with a naive
# average.

average_params = sum(thetas) / len(thetas)

# %%
#
# We initialize two neural networks with those parameters

fisher_model = make_architecture()

params = [p for p in fisher_model.parameters() if p.requires_grad]
theta_fisher = vector_to_parameter_list(
    torch.from_numpy(fisher_weighted_params), params
)
for theta, param in zip(theta_fisher, params):
    param.data = theta.to(param.device).to(param.dtype).data

# same for the average-weighted parameters
average_model = make_architecture()

params = [p for p in average_model.parameters() if p.requires_grad]
theta_average = vector_to_parameter_list(torch.from_numpy(average_params), params)
for theta, param in zip(theta_average, params):
    param.data = theta.to(param.device).to(param.dtype).data

# %%
#
# and probe them on one batch of each task:

for task_idx in range(T):
    data_loader = data_loaders[task_idx]
    loss_function = loss_functions[task_idx]

    X, y = next(iter(data_loader))
    X, y = X.to(DEVICE), y.to(DEVICE)

    fisher_loss = loss_function(fisher_model(X), y)
    average_loss = loss_function(average_model(X), y)
    assert fisher_loss < average_loss

    print(f"Task {task_idx} batch loss with Fisher averaging: {fisher_loss.item():.3f}")
    print(f"Task {task_idx} batch loss with naive averaging: {average_loss.item():.3f}")

# %%
#
# The Fisher-averaged parameters perform better than the naively averaged
# parameters; at least on the training data.
#
# That's all for now.
