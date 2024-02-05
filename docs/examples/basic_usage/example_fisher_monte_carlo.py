r"""
Monte-Carlo approximation of the Fisher
=======================================

In this tutorial, we will compare two approaches to compute the Fisher
information matrix:

1. The Fisher as expected Hessian under the model's likelihood coincides with
   the generalized Gauss-Newton (GGN) matrix for common loss functions, like
   :class:`torch.nn.MSELoss` and :class:`torch.nn.CrossEntropyLoss`. For these settings,
   Fisher = GGN.

2. The Fisher can also be seen as expectation of the gradient outer product w.r.t. the
   model's likelihood. This expectation can be approximated by computing
   the outer product of 'would-be' gradients where the loss is evaluated on a
   label sampled from the model's likelihood, rather than the true label.

The first approach is implemented by :class:`curvlinops.GGNLinearOperator`, the
second by :class:`curvlinops.FisherMCLinearOperator`. We will see that both approaches
coincide as the Monte-Carlo approximation converges.

Let's get the imports out of our way.
"""

import matplotlib.pyplot as plt
import numpy
import torch
from matplotlib import animation
from torch import nn

from curvlinops import FisherMCLinearOperator, GGNLinearOperator

# make deterministic
torch.manual_seed(0)
numpy.random.seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Setup
# -----
#
# We will create a synthetic classification task, a small CNN, and use
# cross-entropy error as loss function.

Ns = [4, 6]
D_in = 7
D_hidden = 5
C = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = [
    (
        torch.rand(N, D_in).to(DEVICE),  # X
        torch.randint(low=0, high=C, size=(N,)).to(DEVICE),  # y
    )
    for N in Ns
]

model = nn.Sequential(
    nn.Linear(D_in, D_hidden),
    nn.ReLU(),
    nn.Linear(D_hidden, D_hidden),
    nn.Sigmoid(),
    nn.Linear(D_hidden, C),
).to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]

loss_function = nn.CrossEntropyLoss(reduction="mean").to(DEVICE)


# %%
# A first comparison
# ------------------
#
# Let's create linear operators for the GGN and the Monte-Carlo approximated
# Fisher and compute their matrix representations by multiplying them onto the
# identity matrix:

GGN = GGNLinearOperator(model, loss_function, params, data)
F = FisherMCLinearOperator(model, loss_function, params, data)

D = GGN.shape[0]
identity = numpy.eye(D)

GGN_mat = GGN @ identity
F_mat = F @ identity

# %%
#
# We can use the residual's Frobenius norm to quantify the approximation error
# of the Monte-Carlo estimator:

residual_norm = numpy.linalg.norm(GGN_mat - F_mat)
print(f"Residual (Frobenius) norm: {residual_norm:.5f}")

# %%
# Setting the number of MC samples
# --------------------------------
#
# To get more accurate estimates, we can use more samples in the MC approximation.
# This is achieved by specifying the optional :code:`mc_samples` argument to
# :class:`curvlinops.FisherMCLinearOperator`. The default value is :code:`1`.
#
# Here are the residual Frobenius norms when using more samples:

mc_samples = [1, 2, 4, 8]
residual_norms = []

for mc in mc_samples:
    F = FisherMCLinearOperator(model, loss_function, params, data, mc_samples=mc)
    F_mat = F @ identity
    residual_norms.append(numpy.linalg.norm(GGN_mat - F_mat))

for mc, norm in zip(mc_samples, residual_norms):
    print(f"mc_samples = {mc},\tresidual (Frobenius) norm = {norm:.5f}")

# %%
# Setting the random seed
# -----------------------
#
# You may have noticed above that the two linear operators created with
# :code:`mc_samples=1` yield identical residual Frobenius norms. This is
# because the two linear operators realize the same matrix, i.e. the same
# sample from the Monte-Carlo estimator.
#
# To see that :class:`curvlinops.FisherMCLinearOperator` indeed represents a
# deterministic matrix, let's create two linear operators with identical
# hyperparameters and compare their matrix representations. After creating the
# first linear operator, we generate some random numbers to show that the
# global random number generator does not influence the Monte-Carlo estimator:

F1_mat = FisherMCLinearOperator(model, loss_function, params, data) @ identity

# draw some random numbers to modify the global random number generator's state
torch.rand(123)

F2_mat = FisherMCLinearOperator(model, loss_function, params, data) @ identity

# still, we get the same deterministic approximation
residual_norm = numpy.linalg.norm(F1_mat - F2_mat)
if numpy.isclose(residual_norm, 0.0):
    print(residual_norm)
else:
    raise RuntimeError(f"Residual Frobenius norm should be 0. Got {residual_norm}.")

# %%
#
# This is because the class uses an internal random number generator to draw
# samples. Therefore, it will not be affected by changes to the global random
# number generator's state.
#
# You can get different realizations of the Monte-Carlo estimator by specifying
# the optional :code:`seed` argument. The above comparison with differently
# seeded linear operators leads to different matrices:

seed1 = 123456
F1_mat = (
    FisherMCLinearOperator(model, loss_function, params, data, seed=seed1) @ identity
)

seed2 = 654321
F2_mat = (
    FisherMCLinearOperator(model, loss_function, params, data, seed=seed2) @ identity
)

# now, we get two different deterministic approximations
residual_norm = numpy.linalg.norm(F1_mat - F2_mat)
if not numpy.isclose(residual_norm, 0.0):
    print(residual_norm)
else:
    raise RuntimeError(f"Residual Frobenius norm should be ≠0. Got {residual_norm}.")

# %%
#
# Approximation quality
# ---------------------
#
# Finally, let's combine what we have seen so far to visualize how well the
# Monte-Carlo approximated Fisher approximates the GGN.
#
# To do that, we will repeatedly draw samples for the Fisher information matrix
# and combine them to yield an estimate that incorporates all previous
# iterations. This approach allows to record snapshots of the estimator at a
# different number of total incorporated MC samples.
#
# We will use a :code:`logspace` for taking snapshots.

num_steps = 25
mc_samples = numpy.unique(
    numpy.logspace(0, 2, num_steps, endpoint=True, dtype=numpy.int32)
)
F_snapshots = []
F_accumulated = numpy.zeros((D, D))
start_seed = 123456789

for seed, mc in enumerate(range(mc_samples.max()), start=start_seed):
    # NOTE Only use `check_deterministic=False` if you know what you are doing
    # We do this here because we have previously convinced ourselves that the created
    # linear operators indeed realize deterministic matrices.
    F = FisherMCLinearOperator(
        model, loss_function, params, data, seed=seed, check_deterministic=False
    )
    F_accumulated += F @ identity
    if mc + 1 in mc_samples:
        F_snapshots.append(F_accumulated / (mc + 1))


# %%
#
# Let's visualize both the residual matrices and their Frobenius norms. To
# visualize the matrix, we will use the element-wise logarithm of its absolute
# value (shifted by a small constant to avoid taking the logarithm of 0):

residual_snapshots = [mat - GGN_mat for mat in F_snapshots]
residual_norms = [numpy.linalg.norm(res) for res in residual_snapshots]


def transform(mat: numpy.ndarray, epsilon: float = 1e-5) -> numpy.ndarray:
    """Transformation applied to the matrix before plotting.

    Applies element-wise absolute value, shifts by epsilon, then takes the
    element-wise logarithm.

    Args:
        mat: Matrix.
        epsilon: Small shift to avoid taking the log of 0.

    Returns:
        Transformed matrix
    """
    return numpy.log10(numpy.abs(mat) + epsilon)


# %%
#
# Here's the plotting code (feel free to skip to the visualization).

img_width = 4
rows, columns = 1, 2
fig, axes = plt.subplots(
    nrows=rows, ncols=columns, figsize=(columns * img_width, rows * img_width)
)
ax_img, ax_fro = axes[0], axes[1]

min_img = min(transform(res).min() for res in residual_snapshots)
max_img = max(transform(res).max() for res in residual_snapshots)

min_fro = 0.0
max_fro = max(residual_norms)

ax_fro.set_xlim(mc_samples.min(), mc_samples.max())
ax_fro.semilogx()
ax_fro.set_xlabel("MC samples")
ax_fro.set_ylabel("residual Frobenius norm")
ax_fro.set_ylim(min_fro, 1.05 * max_fro)
ln_style = "bo"

# collects artists to draw in each frame of the animation
artists = []

for frame_idx in range(len(mc_samples)):
    snapshot = residual_snapshots[frame_idx]
    img = ax_img.imshow(transform(snapshot), vmin=min_img, vmax=max_img, animated=True)

    # workaround for animated title: https://stackoverflow.com/a/47421938
    ax_img_title = plt.text(
        0.5,
        1.01,
        f"Residual ({mc_samples[frame_idx]} samples)",
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=ax_img.transAxes,
    )

    if frame_idx == 0:
        ax_img.imshow(transform(snapshot), vmin=min_img, vmax=max_img)
        plt.colorbar(img, ax=ax_img, label="logarithmic absolute entries", shrink=0.8)
        ax_fro.plot(
            mc_samples[: frame_idx + 1], residual_norms[: frame_idx + 1], ln_style
        )
        plt.subplots_adjust(wspace=0.5, bottom=0.2)

    ln_fro = ax_fro.plot(
        mc_samples[: frame_idx + 1], residual_norms[: frame_idx + 1], ln_style
    )

    artists.append([ax_img_title, img] + ln_fro)


ani = animation.ArtistAnimation(
    fig, artists, interval=1000, blit=False, repeat_delay=1000
)

# %%
#
# Here's a more qualitative comparison that contrasts the GGN matrix and the MC
# approximated Fisher (both transformed to logspace as described above):

img_width = 4
rows, columns = 1, 2
fig, axes = plt.subplots(
    nrows=rows, ncols=columns, figsize=(columns * img_width, rows * img_width)
)

min_value = min(transform(mat).min() for mat in F_snapshots + [GGN_mat])
max_value = max(transform(mat).max() for mat in F_snapshots + [GGN_mat])

# collects artists to draw in each frame of the animation
artists = []

ax_GGN, ax_F = axes[0], axes[1]
ax_GGN.set_title("GGN")


for frame_idx in range(len(mc_samples)):
    im_GGN = ax_GGN.imshow(
        transform(GGN_mat), vmin=min_value, vmax=max_value, animated=True
    )
    im_F = ax_F.imshow(
        transform(F_snapshots[frame_idx]), vmin=min_value, vmax=max_value, animated=True
    )

    # workaround for animated title: https://stackoverflow.com/a/47421938
    ax_F_title = plt.text(
        0.5,
        1.01,
        f"Fisher-MC ({mc_samples[frame_idx]} samples)",
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=ax_F.transAxes,
    )

    if frame_idx == 0:
        img = ax_GGN.imshow(transform(GGN_mat), vmin=min_value, vmax=max_value)
        ax_F.imshow(transform(F_snapshots[frame_idx]), vmin=min_value, vmax=max_value)
        fig.colorbar(
            img,
            ax=axes.ravel().tolist(),
            label="logarithmic absolute entries",
            shrink=0.8,
        )

    artists.append([ax_F_title, im_GGN, im_F])


ani = animation.ArtistAnimation(
    fig, artists, interval=1000, blit=False, repeat_delay=1000
)

# %%
#
# That's all for now.
