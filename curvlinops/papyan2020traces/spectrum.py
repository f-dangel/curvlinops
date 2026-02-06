"""Spectral analysis methods for PyTorch linear operators.

From Papyan, 2020:

- Traces of class/cross-class structure pervade deep learning spectra. Journal
  of Machine Learning Research (JMLR), https://jmlr.org/papers/v21/20-933.html
"""

from math import log, sqrt
from typing import List, Tuple, Union

from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh
from torch import (
    Tensor,
    as_tensor,
    diag_embed,
    linspace,
    randn,
    zeros,
    zeros_like,
)
from torch.distributions import Normal
from torch.linalg import eigh, vector_norm

from curvlinops._torch_base import PyTorchLinearOperator


def lanczos_approximate_spectrum(
    A: PyTorchLinearOperator,
    ncv: int,
    num_points: int = 1024,
    num_repeats: int = 1,
    kappa: float = 3.0,
    boundaries: Union[
        Tuple[float, float], Tuple[float, None], Tuple[None, float], None
    ] = None,
    margin: float = 0.05,
    boundaries_tol: float = 1e-2,
) -> Tuple[Tensor, Tensor]:
    """Approximate the spectral density p(λ) = 1/d ∑ᵢ δ(λ - λᵢ) of A ∈ Rᵈˣᵈ.

    Implements algorithm 2 (:code:`LanczosApproxSpec`) of Papyan, 2020
    (https://jmlr.org/papers/v21/20-933.html).

    Internally rescales the operator spectrum to the interval [-1; 1] such that
    the width ``kappa`` of the Gaussian bumps used to approximate the delta peaks
    need not be tweaked.

    Args:
        A: Symmetric linear operator.
        ncv: Number of Lanczos vectors (number of nodes/weights for the quadrature).
        num_points: Resolution.
        num_repeats: Number of Lanczos quadratures to average the density over.
            Default: ``1``. Taken from papyan2020traces, Section D.2.
        kappa: Width of the Gaussian used to approximate delta peaks in [-1; 1]. Must
            be greater than 1. Default: ``3``. Taken from papyan2020traces, Section D.2.
        boundaries: Estimates of the minimum and maximum eigenvalues of ``A``. If left
            unspecified, they will be estimated internally.
        margin: Relative margin added around the spectral boundary. Default: ``0.05``.
            Taken from papyan2020traces, Section D.2.
        boundaries_tol: (Only relevant if ``boundaries`` are not specified). Relative
            accuracy used to estimate the spectral boundary. ``0`` implies machine
            precision. Default: ``1e-2``, from
            https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.

    Returns:
        Grid points λ and approximated spectral density p(λ) of A.
    """
    boundaries = approximate_boundaries(A, tol=boundaries_tol, boundaries=boundaries)

    average_density = zeros(num_points, device=A.device, dtype=A.dtype)

    for n in range(num_repeats):
        lanczos_iter = fast_lanczos(A, ncv)
        grid, density = lanczos_approximate_spectrum_from_iter(
            lanczos_iter, boundaries, num_points, kappa, margin
        )

        average_density = (1 - 1 / (n + 1)) * average_density + density / (n + 1)

    return grid, average_density


def lanczos_approximate_spectrum_from_iter(
    lanczos_iter: Tuple[Tensor, Tensor],
    boundaries: Tuple[float, float],
    num_points: int,
    kappa: float,
    margin: float,
) -> Tuple[Tensor, Tensor]:
    """Compute a spectrum approximation from a Lanczos iteration.

    Args:
        lanczos_iter: Pair ``(evals, evecs)`` from a Lanczos run.
        boundaries: Approximate minimum and maximum eigenvalues of the operator.
        num_points: Number of grid points.
        kappa: Width parameter for the Gaussian bumps.
        margin: Relative margin added around the spectral boundary.

    Returns:
        Grid points and estimated spectral density.
    """
    eval_min, eval_max = boundaries
    _width = eval_max - eval_min
    _padding = margin * _width
    eval_min, eval_max = eval_min - _padding, eval_max + _padding

    # use normalized operator ``(A - c I) / d`` whose spectrum lies in [-1; 1]
    c = (eval_max + eval_min) / 2
    d = (eval_max - eval_min) / 2

    evals, evecs = lanczos_iter
    device, dtype = evals.device, evals.dtype

    # estimate on grid [-1; 1]
    grid_norm = linspace(-1, 1, num_points, device=device, dtype=dtype)
    density = zeros_like(grid_norm)

    ncv = evals.shape[0]
    nodes = (evals - c) / d
    # Repeat as ``(ncv, num_points)`` arrays to avoid broadcasting
    grid = grid_norm.reshape((1, num_points)).repeat(ncv, 1)
    nodes = nodes.reshape((ncv, 1)).repeat(1, num_points)
    weights = (evecs[0, :] ** 2 / d).reshape((ncv, 1)).repeat(1, num_points)

    # width of Gaussian bump in [-1; 1]
    sigma = 2 / (ncv - 1) / sqrt(8 * log(kappa))
    normal_dist = Normal(nodes, sigma)
    density = (weights * normal_dist.log_prob(grid).exp()).sum(0)

    return linspace(eval_min, eval_max, num_points, device=device, dtype=dtype), density


class _LanczosSpectrumCached:
    """Base class for approximating spectra with Lanczos iterations.

    Caches the Lanczos iterations to efficiently produce approximations with different
    hyperparameters.
    """

    def __init__(self, A: PyTorchLinearOperator, ncv: int):
        """Initialize.

        Args:
            A: Symmetric linear operator.
            ncv: Number of Lanczos vectors (number of nodes/weights for the quadrature).
        """
        self._A = A
        self._ncv = ncv
        self._lanczos_iters: List[Tuple[Tensor, Tensor]] = []

    def _get_lanczos_iters(self, num_iters: int) -> List[Tuple[Tensor, Tensor]]:
        while len(self._lanczos_iters) < num_iters:
            self._lanczos_iters.append(fast_lanczos(self._A, self._ncv))

        return self._lanczos_iters[:num_iters]


class LanczosApproximateSpectrumCached(_LanczosSpectrumCached):
    """Class to approximate the spectral density of p(λ) = 1/d ∑ᵢ δ(λ - λᵢ) of A ∈ Rᵈˣᵈ.

    Caches Lanczos iterations to efficiently produce spectral density approximations with
    different hyperparameters.
    """

    def __init__(
        self,
        A: PyTorchLinearOperator,
        ncv: int,
        boundaries: Union[
            Tuple[float, float], Tuple[float, None], Tuple[None, float], None
        ] = None,
        boundaries_tol: float = 1e-2,
    ):
        """Initialize.

        Args:
            A: Symmetric linear operator.
            ncv: Number of Lanczos vectors (number of nodes/weights for the quadrature).
            boundaries: Estimates of the minimum and maximum eigenvalues of ``A``. If
                left unspecified, they will be estimated internally.
            boundaries_tol: (Only relevant if ``boundaries`` are not specified).
                Relative accuracy used to estimate the spectral boundary. ``0`` implies
                machine precision. Default: ``1e-2``, from
                https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.
        """
        super().__init__(A, ncv)
        self._boundaries = approximate_boundaries(
            A, tol=boundaries_tol, boundaries=boundaries
        )

    def approximate_spectrum(
        self,
        num_repeats: int = 1,
        num_points: int = 1024,
        kappa: float = 3.0,
        margin: float = 0.05,
    ) -> Tuple[Tensor, Tensor]:
        """Approximate the spectal density of A.

        Args:
            num_repeats: Number of Lanczos quadratures to average the density over.
                Default: ``1``. Taken from papyan2020traces, Section D.2.
            num_points: Resolution. Default: ``1024``.
            kappa: Width of the Gaussian used to approximate delta peaks in [-1; 1]. Must
                be greater than 1. Default: ``3``. From papyan2020traces, Section D.2.
            margin: Relative margin added around the spectral boundary.
                Default: ``0.05``. Taken from papyan2020traces, Section D.2.

        Returns:
            Grid points λ and approximated spectral density p(λ) of A.
        """
        spectra = [
            lanczos_approximate_spectrum_from_iter(
                lanczos_iter, self._boundaries, num_points, kappa, margin
            )
            for lanczos_iter in self._get_lanczos_iters(num_repeats)
        ]
        grid = spectra[0][0]
        spectrum = sum(spectrum[1] for spectrum in spectra) / num_repeats

        return grid, spectrum


def lanczos_approximate_log_spectrum(
    A: PyTorchLinearOperator,
    ncv: int,
    num_points: int = 1024,
    num_repeats: int = 1,
    kappa: float = 1.04,
    boundaries: Union[
        Tuple[float, float], Tuple[float, None], Tuple[None, float], None
    ] = None,
    margin: float = 0.05,
    boundaries_tol: float = 1e-2,
    epsilon: float = 1e-5,
) -> Tuple[Tensor, Tensor]:
    """Approximate the spectral density ``p(λ) = 1/d ∑ᵢ δ(λ - λᵢ)`` of ``log(|A| + εI) ∈ Rᵈˣᵈ``.

    Follows the idea of Section C.7 in Papyan, 2020
    (https://jmlr.org/papers/v21/20-933.html).

    Here, log denotes the natural logarithm (i.e. base e).

    Internally rescales the operator spectrum to the interval [-1; 1] such that
    the width ``kappa`` of the Gaussian bumps used to approximate the delta peaks
    need not be tweaked.

    Args:
        A: Symmetric linear operator.
        ncv: Number of Lanczos vectors (number of nodes/weights for the quadrature).
        num_points: Resolution.
        num_repeats: Number of Lanczos quadratures to average the density over.
            Default: ``1``. Taken from papyan2020traces, Section D.2.
        kappa: Width of the Gaussian used to approximate delta peaks in [-1; 1]. Must
            be greater than 1. Default: ``1.04``. Obtained by tweaking while reproducing
            Fig. 15b from papyan2020traces (not specified by the paper).
        boundaries: Estimates of the minimum and maximum eigenvalues of :math:`|A|`. If left
            unspecified, they will be estimated internally.
        margin: Relative margin added around the spectral boundary. Default: ``0.05``.
            Taken from papyan2020traces, Section D.2.
        boundaries_tol: (Only relevant if ``boundaries`` are not specified). Relative
            accuracy used to estimate the spectral boundary. ``0`` implies machine
            precision. Default: ``1e-2``, from
            https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.
        epsilon: Shift to increase numerical stability. Default: ``1e-5``. Taken from
            papyan2020traces, Section D.2.

    Returns:
        Grid points λ and approximated spectral density p(λ) of ``log(|A| + εI)``.
    """
    boundaries = approximate_boundaries_abs(
        A, tol=boundaries_tol, boundaries=boundaries
    )

    average_density = zeros(num_points, device=A.device, dtype=A.dtype)

    for n in range(num_repeats):
        lanczos_iter = fast_lanczos(A, ncv)
        grid, density = lanczos_approximate_log_spectrum_from_iter(
            lanczos_iter, boundaries, num_points, kappa, margin, epsilon
        )

        average_density = (1 - 1 / (n + 1)) * average_density + density / (n + 1)

    return grid, average_density


def lanczos_approximate_log_spectrum_from_iter(
    lanczos_iter: Tuple[Tensor, Tensor],
    boundaries: Tuple[float, float],
    num_points: int,
    kappa: float,
    margin: float,
    epsilon: float,
) -> Tuple[Tensor, Tensor]:
    """Compute a log-spectrum approximation from a Lanczos iteration.

    Args:
        lanczos_iter: Pair ``(evals, evecs)`` from a Lanczos run.
        boundaries: Approximate spectral boundary of ``|A|``.
        num_points: Number of grid points.
        kappa: Width parameter for the Gaussian bumps.
        margin: Relative margin added around the boundary.
        epsilon: Positive shift for numerical stability.

    Returns:
        Grid points and estimated spectral density of ``log(|A| + εI)``.
    """
    log_eval_min, log_eval_max = (log(boundary + epsilon) for boundary in boundaries)
    _width = log_eval_max - log_eval_min
    _padding = margin * _width
    log_eval_min, log_eval_max = log_eval_min - _padding, log_eval_max + _padding

    # use normalized operator ``(log(|A| + εI) - c I) / d`` with spectrum in [-1; 1]
    c = (log_eval_max + log_eval_min) / 2
    d = (log_eval_max - log_eval_min) / 2

    evals, evecs = lanczos_iter
    device, dtype = evals.device, evals.dtype

    # estimate on grid [-1; 1]
    grid_norm = linspace(-1, 1, num_points, device=device, dtype=dtype)
    grid_out = (grid_norm * d + c).exp()

    abs_evals = evals.abs() + epsilon
    log_evals = abs_evals.log()
    nodes = (log_evals - c) / d

    # Repeat as ``(ncv, num_points)`` arrays to avoid broadcasting
    ncv = evals.shape[0]
    grid = grid_norm.reshape((1, num_points)).repeat(ncv, 1)
    nodes = nodes.reshape((ncv, 1)).repeat(1, num_points)
    weights = (evecs[0, :] ** 2).reshape((ncv, 1)).repeat(1, num_points)

    # width of Gaussian bump in [-1; 1]
    sigma = 2 / (ncv - 1) / sqrt(8 * log(kappa))
    normal_dist = Normal(nodes, sigma)
    density = (weights * normal_dist.log_prob(grid).exp()).sum(0) / (d * grid_out)

    return grid_out, density


class LanczosApproximateLogSpectrumCached(_LanczosSpectrumCached):
    """Class to approximate p(λ) = 1/d ∑ᵢ δ(λ - λᵢ) of log(|A| + εI) ∈ Rᵈˣᵈ.

    Caches Lanczos iterations to efficiently produce spectral density approximations with
    different hyperparameters.
    """

    def __init__(
        self,
        A: PyTorchLinearOperator,
        ncv: int,
        boundaries: Union[
            Tuple[float, float], Tuple[float, None], Tuple[None, float], None
        ] = None,
        boundaries_tol: float = 1e-2,
    ):
        """Initialize.

        Args:
            A: Symmetric linear operator.
            ncv: Number of Lanczos vectors (number of nodes/weights for the quadrature).
            boundaries: Estimates of the minimum and maximum eigenvalues of ``|A|``. If
                left unspecified, they will be estimated internally.
            boundaries_tol: (Only relevant if ``boundaries`` are not specified).
                Relative accuracy used to estimate the spectral boundary. ``0`` implies
                machine precision. Default: ``1e-2``, from
                https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.
        """
        super().__init__(A, ncv)
        self._boundaries = approximate_boundaries_abs(
            A, tol=boundaries_tol, boundaries=boundaries
        )

    def approximate_log_spectrum(
        self,
        num_repeats: int = 1,
        num_points: int = 1024,
        kappa: float = 3.0,
        margin: float = 0.05,
        epsilon: float = 1e-5,
    ) -> Tuple[Tensor, Tensor]:
        """Approximate the spectal density of A.

        Args:
            num_repeats: Number of Lanczos quadratures to average the density over.
                Default: ``1``. Taken from papyan2020traces, Section D.2.
            num_points: Resolution. Default: ``1024``.
            kappa: Width of the Gaussian used to approximate delta peaks in [-1; 1]. Must
                be greater than 1. Default: ``3``. From papyan2020traces, Section D.2.
            margin: Relative margin added around the spectral boundary.
                Default: ``0.05``. Taken from papyan2020traces, Section D.2.
            epsilon: Shift to increase numerical stability. Default: ``1e-5``. Taken from
                papyan2020traces, Section D.2.

        Returns:
            Grid points λ and approximated spectral density p(λ) of log(|A| + εI).
        """
        spectra = [
            lanczos_approximate_log_spectrum_from_iter(
                lanczos_iter, self._boundaries, num_points, kappa, margin, epsilon
            )
            for lanczos_iter in self._get_lanczos_iters(num_repeats)
        ]
        grid = spectra[0][0]
        spectrum = sum(spectrum[1] for spectrum in spectra) / num_repeats

        return grid, spectrum


def fast_lanczos(
    A: PyTorchLinearOperator, ncv: int, use_eigh_tridiagonal: bool = False
) -> Tuple[Tensor, Tensor]:
    """Lanczos iterations for large-scale problems (no reorthogonalization step).

    Implements algorithm 2 of Papyan, 2020 (https://jmlr.org/papers/v21/20-933.html).

    Args:
        A: Symmetric linear operator.
        ncv: Number of Lanczos vectors.
        use_eigh_tridiagonal: Whether to use eigh_tridiagonal to eigen-decompose the
            tri-diagonal matrix. Default: ``False``. Setting this value to ``True``
            results in faster eigen-decomposition, but is less stable.

    Returns:
        Eigenvalues and eigenvectors of the tri-diagonal matrix built up during
        Lanczos iterations. ``evecs[:, i]`` is normalized eigenvector of ``evals[i]``.
    """
    device, dtype = A.device, A.dtype
    alphas = zeros(ncv, device=device, dtype=dtype)
    betas = zeros(ncv - 1, device=device, dtype=dtype)

    dim = A.shape[1]
    v, v_prev = None, None

    for m in range(ncv):
        if m == 0:
            v = randn(dim, device=device, dtype=dtype)
            v /= vector_norm(v)
            v_next = A @ v

        else:
            v_next = A @ v - betas[m - 1] * v_prev

        alphas[m] = (v_next * v).sum()
        v_next -= alphas[m] * v

        last = m == ncv - 1
        if not last:
            betas[m] = vector_norm(v_next)
            v_next /= betas[m]
            v_prev = v
            v = v_next

    if use_eigh_tridiagonal:
        # Convert to NumPy for SciPy operations
        evals_np, evecs_np = eigh_tridiagonal(
            alphas.detach().cpu().numpy(), betas.detach().cpu().numpy()
        )
        # Convert back to PyTorch tensors
        evals = as_tensor(evals_np, device=device, dtype=dtype)
        evecs = as_tensor(evecs_np, device=device, dtype=dtype)
    else:
        # Build tridiagonal matrix using PyTorch
        T = (
            diag_embed(alphas)
            + diag_embed(betas, offset=1)
            + diag_embed(betas, offset=-1)
        )
        evals, evecs = eigh(T)

    return evals, evecs


def approximate_boundaries(
    A: PyTorchLinearOperator,
    tol: float = 1e-2,
    boundaries: Union[
        Tuple[float, float], Tuple[float, None], Tuple[None, float], None
    ] = None,
) -> Tuple[float, float]:
    """Approximate λₘᵢₙ(A) and λₘₐₓ(A) using SciPy's ``eigsh``.

    Args:
        A: Symmetric linear operator.
        tol: Relative accuracy used by ``eigsh``. ``0`` implies machine precision.
            Default: ``1e-2``, from
            https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.
        boundaries: A tuple of floats that specifies known parts of the boundaries
            which consequently won't be recomputed. Default: ``None``.

    Returns:
        Estimates of λₘᵢₙ and λₘₐₓ.
    """
    eigsh_kwargs = {"tol": tol, "return_eigenvectors": False}
    A_scipy = A.to_scipy()

    if boundaries is None:
        eval_min, eval_max = eigsh(A_scipy, k=2, which="BE", **eigsh_kwargs)
    else:
        eval_min, eval_max = boundaries

        if eval_min is None:
            (eval_min,) = eigsh(A_scipy, k=1, which="SA", **eigsh_kwargs)
        if eval_max is None:
            (eval_max,) = eigsh(A_scipy, k=1, which="LA", **eigsh_kwargs)

    return eval_min, eval_max


def approximate_boundaries_abs(
    A: PyTorchLinearOperator,
    tol: float = 1e-2,
    boundaries: Union[
        Tuple[float, float], Tuple[float, None], Tuple[None, float], None
    ] = None,
) -> Tuple[float, float]:
    """Approximate λₘᵢₙ(|A|) and λₘₐₓ(|A|) using SciPy's ``eigsh``.

    Args:
        A: Symmetric linear operator.
        tol: Relative accuracy used by ``eigsh``. ``0`` implies machine precision.
            Default: ``1e-2``, from
            https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html#examples.
        boundaries: A tuple of floats that specifies known parts of the boundaries
            which consequently won't be recomputed. Default: ``None``.

    Returns:
        Estimates of λₘᵢₙ and λₘₐₓ of :math:`|A|`.
    """
    eval_min, eval_max = (None, None) if boundaries is None else boundaries

    eigsh_kwargs = {"tol": tol, "return_eigenvectors": False}
    A_scipy = A.to_scipy()

    if eval_max is None:
        (eval_max,) = eigsh(A_scipy, k=1, which="LM", **eigsh_kwargs)
    if eval_min is None:
        (eval_min,) = eigsh(A_scipy, k=1, which="SM", **eigsh_kwargs)

    return abs(eval_min), abs(eval_max)
