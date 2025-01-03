"""
Trace and diagonal estimation
=============================

In this example we will use Hutchinson's method to estimate the trace of a matrix.
We will also compare Hutchinson's method with Hutch++ from
`Meyer et al., 2020 <https://arxiv.org/pdf/2010.09649>`_.
The tutorial mostly follows their experimental setup.

TODO Add diagonal estimation

Here are the imports:
"""

from os import getenv
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from numpy import (
    arange,
    array,
    diag,
    float64,
    logspace,
    mean,
    median,
    ndarray,
    percentile,
    trace,
    unique,
)
from numpy.linalg import qr
from numpy.random import randn, seed
from scipy.sparse.linalg import aslinearoperator
from tueplots import bundles

from curvlinops.trace.hutchinson import HutchinsonTraceEstimator
from curvlinops.trace.meyer2020hutch import HutchPPTraceEstimator

# LaTeX is not available on RTD and we also want to analyze smaller matrices
# to reduce build time
RTD = getenv("READTHEDOCS")

PLOT_CONFIG = bundles.icml2024(column="full" if RTD else "half", usetex=not RTD)

# Dimension of the matrices whose traces we will estimate
DIM = 1000 if RTD else 2000
# Number of repeats for the Hutchinson estimator to compute error bars
NUM_REPEATS = 20 if RTD else 100

seed(0)  # make deterministic

# %%
#
# Setup
# -----
#
# We will use power law matrices whose eigenvalues are given by :math:`\lambda_i = i^{-c}`,
# where :math:`i` is the index of the eigenvalue and :math:`c` is a constant that determines
# the rate of decay of the eigenvalues. A higher value of :math:`c` results in a faster decay
# of the eigenvalues.
#
# Here is a function that creates such a matrix:


def create_power_law_matrix(dim: int = DIM, c: float = 1.0) -> ndarray:
    """Draw a matrix with a power law spectrum.

    Eigenvalues λ_i are given by λ_i = i^(-c), where i is the index of the eigenvalue
    and c is a constant that determines the rate of decay of the eigenvalues.
    A higher value of c results in a faster decay of the eigenvalues.

    Args:
        dim: Matrix dimension.
        c: Power law constant. Default is ``1.0``.

    Returns:
        A sample matrix with a power law spectrum.
    """
    # Create the diagonal matrix Λ with Λii = i^(-c)
    L = diag(arange(1, dim + 1, dtype=float64) ** (-c))

    # Generate a random Gaussian matrix and orthogonalize it to get Q
    Q, _ = qr(randn(dim, dim))

    # Construct the matrix A = Q^T Λ Q
    return Q.T @ L @ Q


# %%
#
# Trace estimation
# ----------------
#
# Basics
# ^^^^^^
#
# To get started, let's create a power law matrix and turn it into a linear operator:

Y_mat = create_power_law_matrix()
Y = aslinearoperator(Y_mat)

# %%
#
# For reference, let's compute the exact trace:

exact_trace = trace(Y_mat)
print(f"Exact trace: {exact_trace:.3f}")

# %%
#
# Now, let's apply Hutchinson's method to estimate the trace.
#
# This method generates estimates from matrix-vector products with random vectors.
# To obtain better estimates we can draw multiple samples and average them.
# We usually want to repeat this process multiple times to get error estimates.
#
# Let's estimate the trace and see if the estimate is decent:

samples_per_estimate = 5

# Initialize the Hutchinson trace estimator
estimator = HutchinsonTraceEstimator(Y)

# Generate estimates by averaging samples, repeat this process multiple times
# so we have error bars.
estimates = []
for _ in range(NUM_REPEATS):
    samples = [estimator.sample() for _ in range(samples_per_estimate)]
    estimates.append(mean(samples))

# Calculate the median and quartiles (error bars) of the estimates
med = median(estimates)
quartile1 = percentile(estimates, 25)
quartile3 = percentile(estimates, 75)

# Print the exact trace and the statistical measures of the estimates
print(f"Exact trace: {exact_trace:.3f}")
print("Estimate:")
print(f"\t- Median: {med:.3f}")
print(f"\t- First quartile (25%): {quartile1:.3f}")
print(f"\t- Third quartile (75%): {quartile3:.3f}")

# Also print whether the true value lies between the quartiles
is_within_quartiles = quartile1 <= exact_trace <= quartile3
print(f"True value within interquartile range? {is_within_quartiles}")
assert is_within_quartiles

# %%
#
# Good! The estimate lies within the error bars.
#
# In the following, we will look at Hutchinson's method and Hutch++, a variance-reduced
# version of Hutchinson's method. This method deterministically computes the trace in a
# sub-space whose dimension we can specify, and uses Hutchinson's method to estimate the trace
# in the remaining space, which reduces the variance at the cost of storing the basis.
#
# We will analyze the quality of both methods on power law matrices of different decay rates.
#
# Hutchinson versus Hutch++
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generally speaking, Hutch++ yields better results when a matrix's spectrum decays rapidly,
# i.e. the trace is dominated by a small number of large eigenvalues. To reproduce this behavior,
# we will first consider a power law matrix with high decay rate :math:`c=2.0`:

Y_mat = create_power_law_matrix(c=2.0)

# %%
#
# We will compare three methods:
#
# - Vanilla Hutchinson
# - Hutch++ with basis dimension 10
# - Hutch++ with basis dimension 100
#
# Our goal is to look at the trace estimator's quality as a function of matrix-vector products.
# We will use the relative trace error, i.e. the absolute difference between the estimate and
# the true trace, normalized by the true trace.
# Here is a function that computes these relative trace errors for a given matrix:


def compute_relative_trace_errors(
    Y: ndarray,
    basis_dims: Tuple[int] = (10, 100),
    num_matvecs: ndarray = unique(logspace(0, 3, 40, dtype=int)),
) -> Dict[str, Dict[str, ndarray]]:
    """Compute the relative trace errors for Hutchinson's method and Hutch++.

    Args:
        Y: Matrix to estimate the trace of.
        basis_dims: Dimensions of the basis for Hutch++.
        num_matvecs: Number of matrix-vector products to use.

    Returns:
        Dictionary with the relative trace errors for Hutchinson's method and Hutch++.
    """
    Y = aslinearoperator(Y_mat)
    exact_trace = trace(Y_mat)

    # Initialize results dictionary
    results = {}

    # compute median and quartiles for Hutchinson's method
    num_samples = [n for n in num_matvecs]
    errors = []

    for _ in range(NUM_REPEATS):
        estimator = HutchinsonTraceEstimator(Y)
        samples = [estimator.sample() for _ in range(max(num_samples))]
        estimates = [mean(samples[:n]) for n in num_samples]
        errors.append([abs(est - exact_trace) / abs(exact_trace) for est in estimates])

    errors = array(errors)
    med = median(errors, axis=0)
    quartile1 = percentile(errors, 25, axis=0)
    quartile3 = percentile(errors, 75, axis=0)

    results["hutch"] = {
        "med": med,
        "quartile1": quartile1,
        "quartile3": quartile3,
        "num_matvecs": num_matvecs,
    }

    # compute median and quartiles for Hutch++ with different basis dimensions
    for basis_dim in basis_dims:
        # Hutch++ spends basis_dim matvecs to build the basis
        n_matvecs = [n for n in num_matvecs if n > basis_dim]
        num_samples = [n - basis_dim for n in n_matvecs]
        errors = []

        for _ in range(NUM_REPEATS):
            estimator = HutchPPTraceEstimator(Y, basis_dim=basis_dim)
            samples = [estimator.sample() for _ in range(max(num_samples))]
            estimates = [mean(samples[:n]) for n in num_samples]
            errors.append(
                [abs(est - exact_trace) / abs(exact_trace) for est in estimates]
            )

        errors = array(errors)
        med = median(errors, axis=0)
        quartile1 = percentile(errors, 25, axis=0)
        quartile3 = percentile(errors, 75, axis=0)
        results[f"hutch++_{basis_dim}"] = {
            "med": med,
            "quartile1": quartile1,
            "quartile3": quartile3,
            "num_matvecs": n_matvecs,
        }

    return results


# %%
#
# Let's compute the relative trace errors and look at them:

results = compute_relative_trace_errors(Y_mat)

print("Relative errors:")

for method, data in results.items():
    print(f"-\t{method}:")

    num_matvecs = data["num_matvecs"]
    med = data["med"]
    quartile1 = data["quartile1"]
    quartile3 = data["quartile3"]

    # print the first 3 values
    for i in range(3):
        print(
            f"\t\t- {num_matvecs[i]} matvecs: {med[i]:.5f} ({quartile1[i]:.3f} - {quartile3[i]:.3f})"
        )

# %%
#
# We should roughly see that the relative error decreases with more matrix-vector products.
#
# Let's visualize the convergence with the following function:


def plot_trace_estimation_results(
    results: Dict[str, Dict[str, ndarray]]
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the trace estimation results.

    Args:
        results: Dictionary with the relative trace errors for Hutchinson's method and Hutch++.

    Returns:
        Figure and axes.
    """
    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")

    for method, data in results.items():
        num_matvecs = data["num_matvecs"]
        med = data["med"]
        quartile1 = data["quartile1"]
        quartile3 = data["quartile3"]

        if method == "hutch":
            label = "Hutchinson"
        else:
            label = f"Hutch++ ({method.split('_')[1]}d basis)"

        ax.plot(num_matvecs, med, label=label)
        ax.fill_between(num_matvecs, quartile1, quartile3, alpha=0.3)

    ax.set_xlabel("Matrix-vector products")
    ax.set_ylabel("Relative trace error")
    ax.legend()

    return fig, ax


with plt.rc_context(PLOT_CONFIG):
    fig, ax = plot_trace_estimation_results(results)
    plt.savefig("trace_estimation_rapid_decay.pdf", bbox_inches="tight")


# %%
#
# As expected Hutch++ yields more accurate trace estimates compared to vanilla Hutchinson
# on this matrix with rapidly decaying eigenvalues. Using a larger basis further improves
# the results, but at the cost of storing a larger basis.
#
# Let's repeat the same for a matrix with a slower decay rate :math:`c=0.5`:

Y_mat = create_power_law_matrix(c=0.5)
results = compute_relative_trace_errors(Y_mat)

with plt.rc_context(PLOT_CONFIG):
    fig, ax = plot_trace_estimation_results(results)
    plt.savefig("trace_estimation_slow_decay.pdf", bbox_inches="tight")

# %%
#
# On this matrix, the benefits of Hutch++ are much less pronounced; indeed they 
# will completely disappear if the matrix's spectrum is completely flat, i.e. :math:`c=0`.
# Thankfully, many curvature matrices in deep learning have a decaying spectrum, which
# may allow Hutch++ to improve over Hutchinson.
#
# That's all for now.