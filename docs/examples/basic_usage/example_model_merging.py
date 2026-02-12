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
:py:class:`curvlinops.CGInverseLinearOperator` for inversion. Naive averaging
corresponds to the special case where the Fisher is the identity.

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
# We are now ready to set up the linear operators for the per-task Fishers.
# We collect the per-task Fisher operators for all three strategies in a
# dictionary. Naive averaging corresponds to using the identity as Fisher:

per_task_fishers = {
    # Diagonal approximation as used in the seminal paper
    # (Precisely speaking, the seminal paper uses a randomized approximation of the
    # Fisher based on sampling that can be achieved with `fisher_type='mc'` and
    # `mc_samples=1`. For simplicity we compute the exact GGN/Fisher diagonal here.)
    "diag(F)": [
        GGNDiagonalLinearOperator(
            model,
            loss_function,
            [p for p in model.parameters() if p.requires_grad],
            data_loader,
        )
        for model, loss_function, data_loader in zip(
            models, loss_functions, data_loaders
        )
    ],
    "F": [
        GGNLinearOperator(
            model,
            loss_function,
            [p for p in model.parameters() if p.requires_grad],
            data_loader,
        )
        for model, loss_function, data_loader in zip(
            models, loss_functions, data_loaders
        )
    ],
}

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
# side in the above equation) for each strategy:

rhs = {
    key: sum(F @ theta for F, theta in zip(Fs, thetas))
    for key, Fs in per_task_fishers.items()
}

# %%
#
# In the last step we need to normalize by multiplying with the inverse of the
# summed Fishers. Let's first sum the per-task Fishers for each strategy:

fisher_sums = {}
for key, Fs in per_task_fishers.items():
    fisher_sums[key] = Fs[0]
    for F in Fs[1:]:
        fisher_sums[key] = F + fisher_sums[key]

# %%
#
# Finally, we compute the merged parameters by applying the inverse of the
# damped Fisher sum. For diagonal operators (naive and ``diag(F)``), the inverse
# is analytical. For the full Fisher, we use
# :py:class:`curvlinops.CGInverseLinearOperator`:
#
# .. note::
#    For the full Fisher, you may want to tweak the convergence criterion of CG
#    using :py:func:`curvlinops.CGInverseLinearOperator.set_cg_hyperparameters`
#    before applying the matrix-vector product.

damping = 1e-3

merged_params = {"Naive": sum(thetas) / len(thetas)}
for key in per_task_fishers:
    F_sum = fisher_sums[key]
    if hasattr(F_sum, "inverse"):
        fisher_sum_inv = F_sum.inverse(damping)
    else:
        identity = IdentityLinearOperator(
            [tuple(p.shape) for p in models[0].parameters() if p.requires_grad],
            DEVICE,
            next(models[0].parameters()).dtype,
        )
        fisher_sum_inv = CGInverseLinearOperator(F_sum + damping * identity)
    merged_params[key] = fisher_sum_inv @ rhs[key]

# %%
#
# Comparison
# ----------
#
# Let's compare the performance of the different strategies. We initialize a
# neural network for each:

merged_models = {}
for key, params_vec in merged_params.items():
    model = make_architecture()
    params = [p for p in model.parameters() if p.requires_grad]
    for theta, param in zip(vector_to_parameter_list(params_vec, params), params):
        param.data = theta.to(param.device, param.dtype).data
    merged_models[key] = model

# %%
#
# and probe them on one batch of each task:

losses = {key: [] for key in merged_params}
header = "\t" + "\t".join(losses.keys())
print(header)

for task_idx in range(T):
    data_loader = data_loaders[task_idx]
    loss_function = loss_functions[task_idx]
    X, y = next(iter(data_loader))

    for key, model in merged_models.items():
        losses[key].append(loss_function(model(X), y).item())
    assert losses["F"][-1] < losses["Naive"][-1]

    print(
        "\t".join(
            [f"Task {task_idx}"] + [f"{losses[key][-1]:.3f}" for key in merged_params]
        )
    )

mean_losses = {key: sum(loss) / len(loss) for key, loss in losses.items()}
print("\t".join(["Avg"] + [f"{mean_losses[key]:.3f}" for key in merged_params]))

# %%
#
# The Fisher-averaged parameters perform better than the naively averaged
# parameters; at least on the training data.
#
# That's all for now.
