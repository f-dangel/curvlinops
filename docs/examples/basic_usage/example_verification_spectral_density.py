"""Spectral density estimation (verification)
=============================================

In this example we will use the spectral density estimation techniques of
`Papyan, 2020 <https://jmlr.org/papers/v21/20-933.html>`_ to reproduce the
plots for synthetic spectra shown in Figure 15 of the paper.

Here are the imports:
"""

from math import e
from os import getenv

import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from torch import (
    Tensor,
    as_tensor,
    linspace,
    logspace,
    manual_seed,
    randn,
    zeros,
)
from torch.distributions import Pareto
from torch.linalg import eigh
from tueplots import bundles

from curvlinops.examples import OuterProductLinearOperator, TensorLinearOperator
from curvlinops.papyan2020traces.spectrum import (
    LanczosApproximateLogSpectrumCached,
    LanczosApproximateSpectrumCached,
    lanczos_approximate_log_spectrum,
    lanczos_approximate_spectrum,
)

# LaTeX is not available in Github actions.
# Therefore, we are turning it off if the script executes on GHA.
USETEX = not getenv("CI")

manual_seed(0)

# %%
#
# Approximating a spectrum
# ------------------------
#
# The first subplot (Figure 15a) uses a synthetic matrix :math:`\mathbf{Y} =
# \mathbf{X} + \mathbf{Z} \mathbf{Z}^\top \in \mathbb{R}^{2000 \times 2000}`
# where :math:`\mathbf{X}_{1,1} = 5`, :math:`\mathbf{X}_{2,2} = 4`,
# :math:`\mathbf{X}_{3,3} = 3` and zero elsewhere, and the elements of
# :math:`\mathbf{Z} \in \mathbb{R}^{2000 \times 2000}` are standard normally
# distributed.
#
# Here is the function to draw a sample for :math:`\mathbf{Y}`:


def create_matrix(dim: int = 2000) -> Tensor:
    """Draw a matrix from the matrix distribution used in papyan2020traces, Figure 15a.

    Args:
        dim: Matrix dimension.

    Returns:
        A sample from the matrix distribution.k
    """
    X = zeros((dim, dim))
    X[0, 0] = 5
    X[1, 1] = 4
    X[2, 2] = 3

    Z = randn(dim, dim)

    return X + Z @ Z.T / dim


Y = create_matrix()

# %%
#
# This matrix is still reasonably small to compute its eigen-decomposition

print("Computing the full spectrum")
Y_evals, _ = eigh(Y)

# %%
#
# For an approximation of the eigenvalue spectrum we just need a
# :class:`PyTorchLinearOperator <curvlinops._torch_base.PyTorchLinearOperator>` of :code:`Y`:

Y_linop = TensorLinearOperator(Y)

# %%
#
# Without rank deflation
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We can now approximate the approximate spectrum using
# :func:`lanczos_approximate_spectrum <curvlinops.lanczos_approximate_spectrum>`
# and using the same hyperparameters as specified by the paper:

# spectral density hyperparameters
num_points = 200
ncv = 128
num_repeats = 10
kappa = 3
margin = 0.05

# %%
#
# For convenience, we feed the eigenvalues at the spectrum's edges as
# boundaries, so they don't get recomputed:

boundaries = (Y_evals.min().item(), Y_evals.max().item())

# %%
#
# Let's compute the approximate spectrum

print("Approximating density")
grid, density = lanczos_approximate_spectrum(
    Y_linop,
    ncv,
    num_points=num_points,
    num_repeats=num_repeats,
    kappa=kappa,
    boundaries=boundaries,
    margin=margin,
)

# %%
#
# and plot it with a histogram (same number of bins as in the paper) of the
# exact density:

# use `tueplots` to make the plot look pretty
plot_config = bundles.icml2024(column="half", usetex=USETEX)

with plt.rc_context(plot_config):
    plt.figure()
    plt.xlabel(r"Eigenvalue $\lambda$")
    plt.ylabel(r"Spectral density $\rho(\lambda)$")

    left, right = grid[0].item(), grid[-1].item()
    num_bins = 40
    bins = linspace(left, right, num_bins)
    plt.hist(
        Y_evals,
        bins=bins,
        log=True,
        density=True,
        label="Exact",
        edgecolor="white",
        lw=0.5,
    )

    plt.plot(grid, density, label="Approximate")
    plt.legend()

    # same ylimits as in the paper
    plt.ylim(bottom=1e-5, top=1e1)
    plt.savefig("toy_spectrum.pdf", bbox_inches="tight")

# %%
#
# For multiple hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# You may have noticed that there are multiple hyperparameters in the spectra
# estimation method. For instance, the :code:`kappa` parameter, which
# determines the width of the superimposed Gaussian bumps. In practice, this
# parameter requires tuning. But trying out another value with the above
# approach needs to re-evaluate the Lanczos iterations. This quickly becomes
# expensive, especially for larger matrices.
#
# As a solution, there exists a class :class:`LanczosApproximateSpectrumCached
# <curvlinops.LanczosApproximateSpectrumCached>` that computes and caches
# Lanczos iterations as we need them. This allows to quickly try out multiple
# hyperparameters.
#
# Let's try out different values for :code:`kappa`:

kappas = [1.1, 3, 10.0]
cache = LanczosApproximateSpectrumCached(Y_linop, ncv, boundaries)

# use `tueplots` to make the plot look pretty
plot_config = bundles.icml2024(column="full", ncols=len(kappas), usetex=USETEX)

with plt.rc_context(plot_config):
    fig, ax = plt.subplots(ncols=len(kappas), sharex=True, sharey=True)
    for idx, kappa in enumerate(kappas):
        grid, density = cache.approximate_spectrum(
            num_repeats=num_repeats, num_points=num_points, kappa=kappa, margin=margin
        )

        ax[idx].hist(
            Y_evals,
            bins=bins,
            log=True,
            density=True,
            label="Exact",
            edgecolor="white",
            lw=0.5,
        )
        ax[idx].plot(grid, density, label=rf"$\kappa = {kappa}$")
        ax[idx].legend()

        ax[idx].set_xlabel(r"Eigenvalue $\lambda$")
        if idx == 0:
            ax[idx].set_ylabel(r"Spectral density $\rho(\lambda)$")
        ax[idx].set_ylim(bottom=1e-5, top=1e1)

# %%
#
# With rank deflation
# ^^^^^^^^^^^^^^^^^^^
#
# As you can see in the above plot, the spectrum consists of a bulk and three
# outliers. We can project out the three (or in general :code:`k`) outliers to
# increase the approximation of the bulk. This technique is called rank
# deflation.

k = 3
print(f"Computing top-{k} eigenvalues of normalized operator")
Y_top_evals, Y_top_evecs = eigsh(Y_linop.to_scipy(), k=k, which="LA")

# Convert NumPy arrays to PyTorch tensors
Y_top_evals_tensor = as_tensor(Y_top_evals, dtype=Y.dtype, device=Y.device)
Y_top_evecs_tensor = as_tensor(Y_top_evecs, dtype=Y.dtype, device=Y.device)

Y_top_linop = OuterProductLinearOperator(Y_top_evals_tensor, Y_top_evecs_tensor)

Y_deflated_linop = Y_linop - Y_top_linop

# %%
#
# The procedure to estimate the deflated operator's spectral density is analogous:

print(f"Approximating density with eliminated top-{k} eigenspace")
grid_no_top, density_no_top = lanczos_approximate_spectrum(
    Y_deflated_linop,
    ncv,
    num_points=num_points,
    num_repeats=num_repeats,
    kappa=kappa,
    boundaries=boundaries,
    margin=0.05,
)

# %%
#
# Here is the visualization, with outliers marked separately:

# use `tueplots` to make the plot look pretty
plot_config = bundles.icml2024(column="half", usetex=USETEX)

with plt.rc_context(plot_config):
    plt.figure()
    plt.title(f"With rank deflation (top {k})")
    plt.xlabel(r"Eigenvalue $\lambda$")
    plt.ylabel(r"Spectral density $\rho(\lambda)$")

    plt.hist(
        Y_evals,
        bins=bins,
        log=True,
        density=True,
        label="Exact",
        edgecolor="white",
        lw=0.5,
    )
    plt.plot(grid_no_top, density_no_top, label="Approximate (deflated)")

    plt.plot(
        Y_top_evals,
        len(Y_top_evals) * [1 / Y_linop.shape[0]],
        linestyle="",
        marker="o",
        label=f"Top {k}",
    )

    # same ylimits as in the paper
    plt.ylim(bottom=1e-5, top=1e1)
    plt.legend()

# %%
#
# Approximating a log-spectrum
# ----------------------------
#
# The second subplot (Figure 15b) uses a synthetic matrix :math:`\mathbf{Y} =
# \frac{1}{1000} \mathbf{Z} \mathbf{Z}^\top \in \mathbb{R}^{500 \times 500}` where the
# elements of :math:`\mathbf{Z} \in \mathbb{R}^{500 \times 1000}` are following
# an i.i.d. Pareto distribution with parameter :math:`\alpha = 1`.
#
# Here is the function to draw a sample for :math:`\mathbf{Y}`:

manual_seed(0)


def create_matrix_log_spectrum(dim: int = 500) -> Tensor:
    """Draw a matrix from the matrix distribution used in papyan2020traces, Figure 15b.

    Args:
        dim: Matrix dimension.

    Returns:
        A sample from the matrix distribution.
    """
    pareto_dist = Pareto(1.0, 1.0)
    Z = pareto_dist.sample((dim, 2 * dim))

    return Z @ Z.T / (2 * dim)


Y = create_matrix_log_spectrum()

# %%
#
# As we will see below, the spectrum of such a matrix spans a large range. It
# is therefore interesting to estimate the spectral density not on a linear
# (:math:`p(\lambda)`), but on a logarithmic scale (:math:`p(\log|\lambda|)`).
#
# We will therefore consider estimating the spectral density of
# :math:`\log(|\mathbf{A}| + \epsilon \mathbf{I})` where the absolute value
# refers to replacing an eigenvalue of :math:`\mathbf{Y}` by its magnitude
# (likewise for the :math:`\log` operation). :math:`\epsilon` is a small
# shift to guarantee the logarithm exists.

epsilon = 1e-5

# %%
#
# Let's start by computing the exact spectrum and generating a linear operator:

print("Computing the full log-spectrum")
Y_evals, _ = eigh(Y)

Y_linop = TensorLinearOperator(Y)

# %%
#
# We can now approximate the approximate log-spectrum using
# :func:`lanczos_approximate_log_spectrum <curvlinops.lanczos_approximate_log_spectrum>`
# and using the same hyperparameters as specified by the paper:

# spectral density hyperparameters
num_points = 200
margin = 0.05
ncv = 256
num_repeats = 10
kappa = 1.04  # not specified in the paper → hand-tuned

# %%
#
# For convenience, we feed the eigenvalue magnitudes at the spectrum's edges as
# boundaries, so they don't get recomputed:

Y_abs_evals = Y_evals.abs()
boundaries = (Y_abs_evals.min().item(), Y_abs_evals.max().item())

print("Approximating log-spectrum")
grid, density = lanczos_approximate_log_spectrum(
    Y_linop,
    ncv,
    num_points=num_points,
    num_repeats=num_repeats,
    kappa=kappa,
    boundaries=boundaries,
    margin=margin,
    epsilon=epsilon,
)


# %%
#
# Now we can visualize the results:

# use `tueplots` to make the plot look pretty
plot_config = bundles.icml2024(column="half", usetex=USETEX)

with plt.rc_context(plot_config):
    plt.figure()
    plt.xlabel(r"Absolute eigenvalue $\nu = |\lambda| + \epsilon$")
    plt.ylabel(r"Spectral density $\rho(\log\nu)$")

    Y_log_abs_evals = (Y_evals.abs() + epsilon).log()

    xlimits_no_margin = (Y_log_abs_evals.min().item(), Y_log_abs_evals.max().item())
    width_no_margins = xlimits_no_margin[1] - xlimits_no_margin[0]
    xlimits = [
        xlimits_no_margin[0] - margin * width_no_margins,
        xlimits_no_margin[1] + margin * width_no_margins,
    ]

    plt.semilogx()
    num_bins = 40
    bins = logspace(xlimits[0], xlimits[1], num_bins, base=e)
    plt.hist(
        Y_log_abs_evals.exp(),
        bins=bins,
        log=True,
        density=True,
        label="Exact",
        edgecolor="white",
        lw=0.5,
    )

    plt.plot(grid, density, label="Approximate")

    # use same ylimits as in the paper
    plt.ylim(bottom=1e-14, top=1e-2)
    plt.legend()

    plt.savefig("toy_log_spectrum.pdf", bbox_inches="tight")

# %%
#
# For multiple hyperparameters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To efficiently produce such plots for multiple hyperparameters, there exists
# a class :class:`LanczosApproximateLogSpectrumCached
# <curvlinops.LanczosApproximateLogSpectrumCached>` that computes and caches
# Lanczos iterations as we need them.
#
# Let's try out different values for :code:`kappa`:

kappas = [1.01, 1.1, 3]
cache = LanczosApproximateLogSpectrumCached(Y_linop, ncv, boundaries)

# use `tueplots` to make the plot look pretty
plot_config = bundles.icml2024(column="full", ncols=len(kappas), usetex=USETEX)

with plt.rc_context(plot_config):
    fig, ax = plt.subplots(ncols=len(kappas), sharex=True, sharey=True)
    for idx, kappa in enumerate(kappas):
        grid, density = cache.approximate_log_spectrum(
            num_repeats=num_repeats,
            num_points=num_points,
            kappa=kappa,
            margin=margin,
            epsilon=epsilon,
        )

        ax[idx].hist(
            Y_log_abs_evals.exp(),
            bins=bins,
            log=True,
            density=True,
            label="Exact",
            edgecolor="white",
            lw=0.5,
        )
        ax[idx].loglog(grid, density, label=rf"$\kappa = {kappa}$")
        ax[idx].legend()

        ax[idx].set_xlabel(r"Absolute eigenvalue $\nu = |\lambda| + \epsilon$")
        if idx == 0:
            ax[idx].set_ylabel(r"Spectral density $\rho(\log \nu)$")
        ax[idx].set_ylim(bottom=1e-14, top=1e-2)
