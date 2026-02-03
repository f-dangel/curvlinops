r"""
Visual tour of curvature matrices
=================================

This tutorial visualizes different curvature matrices for a model with
sufficiently small parameter space.

First, the imports.
"""

from typing import Callable, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import cumsum
from torch import Tensor, cuda, device, eye, manual_seed, rand, randint
from torch.nn import (
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Linear,
    ReLU,
    Sequential,
    Sigmoid,
)
from torch.utils.data import DataLoader, TensorDataset
from tueplots import bundles

from curvlinops import (
    EFLinearOperator,
    FisherMCLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
)

# make deterministic
manual_seed(0)

DEVICE = device("cuda" if cuda.is_available() else "cpu")

# %%
# Setup
# -----
#
# We will create a synthetic classification task, a small CNN, and use
# cross-entropy error as loss function.

num_data = 50
batch_size = 20
in_channels = 3
in_features_shape = (in_channels, 10, 10)
num_classes = 5

# dataset
dataset = TensorDataset(
    rand(num_data, *in_features_shape),  # X
    randint(size=(num_data,), low=0, high=num_classes),  # y
)
dataloader = DataLoader(dataset, batch_size=batch_size)

# model
model = Sequential(
    Conv2d(in_channels, 4, 3, padding=1),
    ReLU(),
    Conv2d(4, 4, 5, padding=2, stride=2),
    Sigmoid(),
    Conv2d(4, 1, 3, padding=1),
    Flatten(),
    Linear(25, num_classes),
).to(DEVICE)

params = [p for p in model.parameters() if p.requires_grad]
num_params = sum(p.numel() for p in params)
num_params_layer = [
    sum(p.numel() for p in child.parameters()) for child in model.children()
]
num_tensors_layer = [len(list(child.parameters())) for child in model.children()]

loss_function = CrossEntropyLoss(reduction="mean").to(DEVICE)

print(f"Total parameters: {num_params}")
print(f"Layer parameters: {num_params_layer}")

# %%
# Computation
# -----------
#
# We can now set up linear operators for the curvature matrices we want to
# visualize, and compute them by multiplying the linear operator onto the
# identity matrix.
#
# First, create the linear operators:

Hessian_linop = HessianLinearOperator(model, loss_function, params, dataloader)
GGN_linop = GGNLinearOperator(model, loss_function, params, dataloader)
EF_linop = EFLinearOperator(model, loss_function, params, dataloader)
Hessian_blocked_linop = HessianLinearOperator(
    model,
    loss_function,
    params,
    dataloader,
    block_sizes=[s for s in num_tensors_layer if s != 0],
)
F_linop = FisherMCLinearOperator(model, loss_function, params, dataloader)
KFAC_linop = KFACLinearOperator(
    model, loss_function, params, dataloader, separate_weight_and_bias=False
)

# %%
#
# Then, compute the matrices

identity = eye(num_params, device=DEVICE)

Hessian_mat = Hessian_linop @ identity
GGN_mat = GGN_linop @ identity
EF_mat = EF_linop @ identity
Hessian_blocked_mat = Hessian_blocked_linop @ identity
F_mat = F_linop @ identity
KFAC_mat = KFAC_linop @ identity

# %%
# Visualization
# -------------
#
# We will show the matrix entries on a shared domain for better comparability.

matrices = [
    m.cpu()
    for m in (Hessian_mat, GGN_mat, EF_mat, Hessian_blocked_mat, F_mat, KFAC_mat)
]
titles = [
    "Hessian",
    "Generalized Gauss-Newton",
    "Empirical Fisher",
    "Block-diagonal Hessian",
    "Monte-Carlo Fisher",
    "KFAC",
]

rows, columns = 2, 3


def plot(
    transform: Callable[[Tensor], Tensor], transform_title: str = None
) -> Tuple[Figure, Axes]:
    """Visualize transformed curvature matrices using a shared domain.

    Args:
        transform: A transformation that will be applied to the matrices. Must
            accept a matrix and return a matrix of the same shape.
        transform_title: An optional string describing the transformation.
            Default: `None` (empty).

    Returns:
        Figure and axes of the created subplot.
    """
    min_value = min(transform(mat).min() for mat in matrices)
    max_value = max(transform(mat).max() for mat in matrices)

    fig, axes = plt.subplots(nrows=rows, ncols=columns, sharex=True, sharey=True)
    fig.supxlabel("Layer")
    fig.supylabel("Layer")

    for idx, (ax, mat, title) in enumerate(zip(axes.flat, matrices, titles)):
        ax.set_title(title)
        img = ax.imshow(transform(mat), vmin=min_value, vmax=max_value)

        # layer blocks
        boundaries = [0] + cumsum(num_params_layer).tolist()
        for pos in boundaries:
            if pos not in [0, num_params]:
                style = {"color": "w", "lw": 0.5, "ls": "-"}
                ax.axhline(y=pos - 1, xmin=0, xmax=num_params - 1, **style)
                ax.axvline(x=pos - 1, ymin=0, ymax=num_params - 1, **style)

        # label positions
        label_positions = [
            (boundaries[layer_idx] + boundaries[layer_idx + 1]) / 2
            for layer_idx in range(len(boundaries) - 1)
            if boundaries[layer_idx] != boundaries[layer_idx + 1]
        ]
        labels = [str(i + 1) for i in range(len(label_positions))]
        ax.set_xticks(label_positions)
        ax.set_xticklabels(labels)
        ax.set_yticks(label_positions)
        ax.set_yticklabels(labels)

        # colorbar
        last = idx == len(matrices) - 1
        if last:
            fig.colorbar(
                img, ax=axes.ravel().tolist(), label=transform_title, shrink=0.8
            )

    return fig, axes


# use `tueplots` to make the plot look pretty
plot_config = bundles.icml2024(column="full", nrows=1.5 * rows, ncols=columns)

# %%
#
# We will show their logarithmic absolute value:


def logabs(mat: Tensor, epsilon: float = 1e-6) -> Tensor:
    return mat.abs().clamp(min=epsilon).log10()


with plt.rc_context(plot_config):
    plot(logabs, transform_title="Logarithmic absolute entries")
    plt.savefig("curvature_matrices_log_abs.pdf", bbox_inches="tight")

# %%
#
# That's because it is hard to recognize structure in the unaltered entries:


def unchanged(mat):
    return mat


with plt.rc_context(plot_config):
    plot(unchanged, transform_title="Unaltered matrix entries")

# %%
#
# That's all for now.

plt.close("all")
