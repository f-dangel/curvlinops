"""
Trace and diagonal estimation
=============================

In this example we will explore randomized estimators for the trace and diagonal
of a matrix using functionality from the `skerch <https://github.com/andres-fr/skerch>`_
library. This will also showcase the interoperability between Curvlinops and other
other linop-based libraries.

We will investigate three different methods to compute the trace and diagonal:
plain Hutchinson, Hutch++, and XDiag/XTrace.
We will compare their performances on synthetic matrices with different spectral
decays, observing that rank-deflation is most effective for rapidly decaying spectra,
and that exchangeability allows us to save 1 out of 3 measurements.

We begin with the imports, globals and defining the estimators by wrapping the
corresponding ``skerch`` functionality:
"""

from os import getenv
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from skerch.algorithms import hutch as _hutch
from skerch.algorithms import xhutchpp as _xhutchpp
from torch import (
    Tensor,
    arange,
    as_tensor,
    float64,
    int32,
    linspace,
    median,
    quantile,
    randn,
    stack,
)
from torch.linalg import qr
from tueplots import bundles

from curvlinops.examples import TensorLinearOperator

# LaTeX is not available on RTD and we also want to analyze smaller matrices
# to reduce build time
RTD = getenv("READTHEDOCS")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

PLOT_CONFIG = bundles.icml2024(
    column="full" if RTD else "half", usetex=not RTD, nrows=2
)

# Dimension of the matrices whose traces we will estimate
DIM = 200 if RTD else 1000
NUM_PLOT_POINTS = 30
# Number of repeats for the Hutchinson estimator to compute error bars
NUM_REPEATS = 50 if RTD else 200
SEEDS = [12345 + i for i in range(NUM_REPEATS)]
# x-axis for our line plots. Hutch++ requires matrix-vector products divisible by 3,
# and XTrace/XDiag require matrix-vector products divisible by 2
NUM_MATVECS_HUTCH = linspace(1, 100, NUM_PLOT_POINTS, dtype=int32).unique()
NUM_MATVECS_HUTCHPP = (NUM_MATVECS_HUTCH + (3 - NUM_MATVECS_HUTCH % 3)).unique()
NUM_MATVECS_X = (NUM_MATVECS_HUTCH + (2 - NUM_MATVECS_HUTCH % 2)).unique()


def hutch(lop, lop_device, lop_dtype, num_meas, seed):
    """Girard-Hutchinson estimator."""
    return _hutch(lop, lop_device, lop_dtype, num_meas, seed)


def hutchpp(lop, lop_device, lop_dtype, num_meas, seed):
    """Girard-Hutchinson with rank-deflation."""
    assert num_meas % 3 == 0, "num_meas must be a multiple of 3"
    # from xhutchpp we just use the deflation matrix Q here (num_meas // 3)
    Q = _xhutchpp(lop, lop_device, lop_dtype, num_meas // 3, 0, seed - num_meas)["Q"]
    result = _hutch(lop, lop_device, lop_dtype, num_meas // 3, seed, defl_Q=Q)
    result["diag"] += (Q.T * (Q.H @ lop)).sum(0)  # here another num_meas // 3
    result["tr"] = result["diag"].sum()
    return result


def xdiagtrace(lop, lop_device, lop_dtype, num_meas, seed):
    """XDiag/XTrace estimator."""
    assert num_meas % 2 == 0, "num_meas must be even"
    return _xhutchpp(lop, lop_device, lop_dtype, num_meas // 2, 0, seed)


# %%
#
# Synthetic matrices
# ------------------
#
# To conduct our experiments, we will sample matrices whose eigenvalues are given by
# :math:`\lambda_i = i^{-c}`, where :math:`i` is the index of the eigenvalue and
# :math:`c` is a constant that determines the rate of decay of the eigenvalues. A
# higher value of :math:`c` means that eigenvalues are decaying faster:


def create_power_law_matrix(
    dim: int = DIM, c: float = 1.0, dtype: torch.dtype = float64
) -> Tensor:
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
    L = arange(1, dim + 1, dtype=dtype) ** (-c)

    # Generate a random Gaussian matrix and orthogonalize it to get Q
    Q, _ = qr(randn(dim, dim, dtype=dtype))

    # Construct the matrix A = Q^H Λ Q
    return (Q.H * L) @ Q


# %%
#
# Dipping our toes
# ----------------
#
# To get started, let's create a power law matrix and turn it into a linear operator.
# For reference, we will also get its trace and diagonal:

Y_mat = create_power_law_matrix(dtype=DTYPE).to(DEVICE)
Y = TensorLinearOperator(Y_mat)
exact_diag = Y_mat.diag()
exact_trace = Y_mat.trace()

# %%
#
# The simplest method for stochastic trace and diagonal estimation is Hutchinson's.
# The idea is to estimate the trace from matrix-vector products with random vectors.
# To obtain better estimates, we can use more products.
# We can also repeat this process multiple times to get error estimates.
# Let's estimate the trace with just a few products:

# matrix-vector queries for one trace estimate
num_matvecs = 5

# Generate estimates, repeat process multiple times so we have error bars.
estimates = stack(
    [hutch(Y, Y.device, Y.dtype, num_matvecs, seed)["tr"] for seed in SEEDS]
)

# Calculate the median and quartiles (error bars) of the estimates
med = median(estimates)
quartile1 = quantile(estimates, 0.25)
quartile3 = quantile(estimates, 0.75)

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
# Good! As we can see, the estimate lies within the error bars.
#
# We can estimate the diagonal in an analogous fashion. But unlike the trace,
# the diagonal is a vector, which makes comparing the estimates by printing their
# entries tedious. Therefore, we report the relative error between exact and estimated
# diagonals:


def relative_diag_error(est: Tensor, exact: Tensor) -> Tensor:
    """Vectorized version of relative error.

    Args:
        est: Estimated vector.
        exact: Exact vector.

    Returns:
        Relative error: norm of the difference, divided by norm of ``exact``.
    """
    return (est - exact).norm() / exact.norm()


# Generate estimates, repeat process multiple times so we have error bars.
estimates = [hutch(Y, Y.device, Y.dtype, num_matvecs, seed)["diag"] for seed in SEEDS]
errors = stack([relative_diag_error(e, exact_diag) for e in estimates])

# Calculate the median and quartiles (error bars) of the estimates
med = median(errors)
quartile1 = quantile(errors, 0.25)
quartile3 = quantile(errors, 0.75)

# Print error stats
print("Relative diagonal errors:")
print(f"\t- Median: {med:.3f}")
print(f"\t- First quartile (25%): {quartile1:.3f}")
print(f"\t- Third quartile (75%): {quartile3:.3f}")

# %%
#
# As we can see here, a few Hutchinson products won't generally be very accurate.
# In the next section we will see what we can do about it.
#


# %%
#
# Comparison between methods
# --------------------------
#
# We will now do a more exhaustive comparison between Hutchinson's method and two other
# algorithms: Hutch++ and XTrace. Hutch++ combines vanilla Hutchinson with variance
# reduction, by deterministically computing the trace in a relevant sub-space, and
# using Hutchinson's method in the remaining "deflated" space. XTrace uses variance
# reduction from Hutch++ combined with the exchangeability principle (i.e. the
# estimate is identical when permuting the random test vectors), which reduces the
# number of measurements needed. All methods are unbiased, but Hutch++ and XTrace
# require additional memory to store the deflation basis.
#
# To compare their performance, we will create a matrix with a high
# spectral decay rate of :math:`c=2.0`, and a matrix with a slow spectral decay
# rate of :math:`c=0.5`, and run the different estimators with an increasing number
# of measurements.
#
# And since these are randomized estimators, we will repeat the estimation with
# different seeds to obtain error bars for each method, and investigate how their
# accuracy evolves as we increase the number of matrix-vector products, reporting
# relative error distributions.


def compute_relative_errors(
    Y_mat: Tensor,
) -> Tuple[Dict[str, Dict[str, Tensor]], Dict[str, Dict[str, Tensor]]]:
    """Compute the relative errors for Hutchinson, Hutch++, and XTrace.

    Args:
        Y_mat: Matrix to estimate the trace of.

    Returns:
        Dictionaries with the relative trace and diagonal errors.
    """
    Y = TensorLinearOperator(Y_mat)
    exact_diag = Y_mat.diag()
    exact_trace = exact_diag.sum()
    tr_results, diag_results = {}, {}
    for name, method, num_matvecs_method in zip(
        ("Hutchinson", "Hutch++", "Exchanged"),
        (hutch, hutchpp, xdiagtrace),
        (NUM_MATVECS_HUTCH, NUM_MATVECS_HUTCHPP, NUM_MATVECS_X),
    ):
        tr_results[name] = {
            "med": [],
            "quartile1": [],
            "quartile3": [],
            "num_matvecs": [],
        }
        diag_results[name] = {
            "med": [],
            "quartile1": [],
            "quartile3": [],
            "num_matvecs": [],
        }
        for n in num_matvecs_method:
            tr_errors, diag_errors = [], []
            for seed in SEEDS:
                est = method(Y, Y.device, Y.dtype, int(n), seed)
                tr_errors.append((est["tr"] - exact_trace).abs() / abs(exact_trace))
                diag_errors.append(relative_diag_error(est["diag"], exact_diag))
            tr_errors, diag_errors = as_tensor(tr_errors), as_tensor(diag_errors)
            tr_results[name]["med"].append(tr_errors.median())
            tr_results[name]["quartile1"].append(tr_errors.quantile(0.25))
            tr_results[name]["quartile3"].append(tr_errors.quantile(0.75))
            tr_results[name]["num_matvecs"].append(n)
            diag_results[name]["med"].append(diag_errors.median())
            diag_results[name]["quartile1"].append(diag_errors.quantile(0.25))
            diag_results[name]["quartile3"].append(diag_errors.quantile(0.75))
            diag_results[name]["num_matvecs"].append(n)
    return tr_results, diag_results


Y_mat_fast = create_power_law_matrix(c=2, dtype=DTYPE).to(DEVICE)  # Fast spectral decay
tr_results_fast, diag_results_fast = compute_relative_errors(Y_mat_fast)

Y_mat_slow = create_power_law_matrix(c=0.5, dtype=DTYPE).to(DEVICE)  # Slow decay
tr_results_slow, diag_results_slow = compute_relative_errors(Y_mat_slow)


# %%
#
# Great! now we are ready to plot and analyze the obtained errors:


def plot_estimation_results(
    results: Dict[str, Dict[str, Tensor]], ax: plt.Axes, target: str = "trace"
) -> None:
    """Plot the trace estimation results on the given Axes.

    Args:
        results: Dictionary with the relative trace errors.
        ax: The matplotlib Axes to plot on.
        target: The property that is approximated (used in ylabel).
            Default is ``'trace'``.
    """
    ax.set_yscale("log")

    for name, data in results.items():
        num_matvecs = data["num_matvecs"]
        med = data["med"]
        quartile1 = data["quartile1"]
        quartile3 = data["quartile3"]

        ax.plot(num_matvecs, med, label=name)
        ax.fill_between(num_matvecs, quartile1, quartile3, alpha=0.3)

    ax.set_xlabel("Matrix-vector products")
    ax.set_ylabel(f"Relative {target} error")
    ax.legend()


# Plot the trace errors for both fast and slow spectral decay
with plt.rc_context(PLOT_CONFIG):
    fig, axes = plt.subplots(nrows=2, sharex=True)
    plot_estimation_results(tr_results_fast, axes[0])
    plot_estimation_results(tr_results_slow, axes[1])
    axes[0].set_title("Fast spectral decay ($c=2$)")
    axes[1].set_title("Slow spectral decay ($c=0.5$)")

    # Remove xlabel from the first, and legend from the second, plot
    axes[0].set_xlabel(None)
    axes[1].legend().remove()

    plt.savefig("./trace_estimation.pdf", bbox_inches="tight")

# %%
#
# We observe that the relative error decreases with more matrix-vector
# products. For fast spectral decay, we also observe that Hutch++ and XTrace yield
# more accurate trace estimates than vanilla Hutchinson, while XTrace takes less
# measurements than Hutch++ thanks to the exchangeability principle. For slow
# spectral decay, the benefits of Hutch++ and XTrace
# disappear. Thankfully, many curvature matrices in deep learning exhibit a decaying
# spectrum, which may allow Hutch++ and XTrace to improve over Hutchinson (XTrace
# being generally superior due to having same memory but less measurement
# requirements).
#
# These observations also apply for the diagonal estimators, with less variance
# since the trace is itself a sum of all diagonal estimators:

with plt.rc_context(PLOT_CONFIG):
    fig, axes = plt.subplots(nrows=2, sharex=True)
    plot_estimation_results(diag_results_fast, axes[0], target="diagonal")
    plot_estimation_results(diag_results_slow, axes[1], target="diagonal")
    axes[0].set_title("Fast spectral decay ($c=2$)")
    axes[1].set_title("Slow spectral decay ($c=0.5$)")
    axes[0].set_xlabel(None)
    axes[1].legend().remove()

    plt.savefig("./diagonal_estimation.pdf", bbox_inches="tight")


# %%
#
# That's all for now.
