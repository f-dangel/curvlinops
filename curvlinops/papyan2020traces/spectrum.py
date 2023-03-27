"""Spectral analysis methods for SciPy linear operators.

From Papyan, 2020:

- Traces of class/cross-class structure pervade deep learning spectra. Journal
  of Machine Learning Research (JMLR), https://jmlr.org/papers/v21/20-933.html
"""

from typing import List, Tuple, Union

from numpy import exp, inner, linspace, log, ndarray, pi, sqrt, zeros, zeros_like
from numpy.linalg import norm
from numpy.random import randn
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, eigsh


def lanczos_approximate_spectrum(
    A: LinearOperator,
    ncv: int,
    num_points: int = 1024,
    num_repeats: int = 1,
    kappa: float = 3.0,
    boundaries: Union[
        Tuple[float, float], Tuple[float, None], Tuple[None, float], None
    ] = None,
    margin: float = 0.05,
    boundaries_tol: float = 1e-2,
) -> Tuple[ndarray, ndarray]:
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

    average_density = zeros(num_points)

    for n in range(num_repeats):
        lanczos_iter = fast_lanczos(A, ncv)
        grid, density = lanczos_approximate_spectrum_from_iter(
            lanczos_iter, boundaries, num_points, kappa, margin
        )

        average_density = (1 - 1 / (n + 1)) * average_density + density / (n + 1)

    return grid, average_density


def lanczos_approximate_spectrum_from_iter(
    lanczos_iter: Tuple[ndarray, ndarray],
    boundaries: Tuple[float, float],
    num_points: int,
    kappa: float,
    margin: float,
) -> Tuple[ndarray, ndarray]:
    eval_min, eval_max = boundaries
    _width = eval_max - eval_min
    _padding = margin * _width
    eval_min, eval_max = eval_min - _padding, eval_max + _padding

    # use normalized operator ``(A - c I) / d`` whose spectrum lies in [-1; 1]
    c = (eval_max + eval_min) / 2
    d = (eval_max - eval_min) / 2

    # estimate on grid [-1; 1]
    grid_norm = linspace(-1, 1, num_points, endpoint=True)
    density = zeros_like(grid_norm)

    evals, evecs = lanczos_iter
    ncv = evals.shape[0]
    nodes = (evals - c) / d
    # Repeat as ``(ncv, num_points)`` arrays to avoid broadcasting
    grid = grid_norm.reshape((1, num_points)).repeat(ncv, axis=0)
    nodes = nodes.reshape((ncv, 1)).repeat(num_points, axis=1)
    weights = (evecs[0, :] ** 2 / d).reshape((ncv, 1)).repeat(num_points, axis=1)

    # width of Gaussian bump in [-1; 1]
    sigma = 2 / (ncv - 1) / sqrt(8 * log(kappa))
    density = (weights * _gaussian(grid, nodes, sigma)).sum(0)

    return linspace(eval_min, eval_max, num_points, endpoint=True), density


class _LanczosSpectrumCached:
    """Base class for approximating spectra with Lanczos iterations.

    Caches the Lanczos iterations to efficiently produce approximations with different
    hyperparameters.
    """

    def __init__(self, A: LinearOperator, ncv: int):
        """Initialize.

        Args:
            A: Symmetric linear operator.
            ncv: Number of Lanczos vectors (number of nodes/weights for the quadrature).
        """
        self._A = A
        self._ncv = ncv
        self._lanczos_iters: List[Tuple[ndarray, ndarray]] = []

    def _get_lanczos_iters(self, num_iters: int) -> List[Tuple[ndarray, ndarray]]:
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
        A: LinearOperator,
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
    ) -> Tuple[ndarray, ndarray]:
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
    A: LinearOperator,
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
) -> Tuple[ndarray, ndarray]:
    """Approximate the spectral density p(λ) = 1/d ∑ᵢ δ(λ - λᵢ) of log(|A| + εI) ∈ Rᵈˣᵈ.

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
        boundaries: Estimates of the minimum and maximum eigenvalues of ``|A|``. If left
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
        Grid points λ and approximated spectral density p(λ) of log(|A| + εI).
    """
    boundaries = approximate_boundaries_abs(
        A, tol=boundaries_tol, boundaries=boundaries
    )

    average_density = zeros(num_points)

    for n in range(num_repeats):
        lanczos_iter = fast_lanczos(A, ncv)
        grid, density = lanczos_approximate_log_spectrum_from_iter(
            lanczos_iter, boundaries, num_points, kappa, margin, epsilon
        )

        average_density = (1 - 1 / (n + 1)) * average_density + density / (n + 1)

    return grid, average_density


def lanczos_approximate_log_spectrum_from_iter(
    lanczos_iter: Tuple[ndarray, ndarray],
    boundaries: Tuple[float, float],
    num_points: int,
    kappa: float,
    margin: float,
    epsilon: float,
) -> Tuple[ndarray, ndarray]:
    log_eval_min, log_eval_max = (log(boundary + epsilon) for boundary in boundaries)
    _width = log_eval_max - log_eval_min
    _padding = margin * _width
    log_eval_min, log_eval_max = log_eval_min - _padding, log_eval_max + _padding

    # use normalized operator ``(log(|A| + εI) - c I) / d`` with spectrum in [-1; 1]
    c = (log_eval_max + log_eval_min) / 2
    d = (log_eval_max - log_eval_min) / 2

    # estimate on grid [-1; 1]
    grid_norm = linspace(-1, 1, num_points, endpoint=True)
    grid_out = exp(grid_norm * d + c)

    evals, evecs = lanczos_iter

    abs_evals = abs(evals) + epsilon
    log_evals = log(abs_evals)
    nodes = (log_evals - c) / d

    # Repeat as ``(ncv, num_points)`` arrays to avoid broadcasting
    ncv = evals.shape[0]
    grid = grid_norm.reshape((1, num_points)).repeat(ncv, axis=0)
    nodes = nodes.reshape((ncv, 1)).repeat(num_points, axis=1)
    weights = (evecs[0, :] ** 2).reshape((ncv, 1)).repeat(num_points, axis=1)

    # width of Gaussian bump in [-1; 1]
    sigma = 2 / (ncv - 1) / sqrt(8 * log(kappa))
    density = (weights * _gaussian(grid, nodes, sigma)).sum(0) / (d * grid_out)

    return grid_out, density


class LanczosApproximateLogSpectrumCached(_LanczosSpectrumCached):
    """Class to approximate p(λ) = 1/d ∑ᵢ δ(λ - λᵢ) of log(|A| + εI) ∈ Rᵈˣᵈ.

    Caches Lanczos iterations to efficiently produce spectral density approximations with
    different hyperparameters.
    """

    def __init__(
        self,
        A: LinearOperator,
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
    ) -> Tuple[ndarray, ndarray]:
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
    A: LinearOperator, ncv: int, use_eigh_tridiagonal: bool = False
) -> Tuple[ndarray, ndarray]:
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
    alphas, betas = zeros(ncv), zeros(ncv - 1)

    dim = A.shape[1]
    v, v_prev = None, None

    for m in range(ncv):
        if m == 0:
            v = randn(dim)
            v /= norm(v)
            v_next = A @ v

        else:
            v_next = A @ v - betas[m - 1] * v_prev

        alphas[m] = inner(v_next, v)
        v_next -= alphas[m] * v

        last = m == ncv - 1
        if not last:
            betas[m] = norm(v_next)
            v_next /= betas[m]
            v_prev = v
            v = v_next

    if use_eigh_tridiagonal:
        evals, evecs = eigh_tridiagonal(alphas, betas)
    else:
        T = diags([betas, alphas, betas], offsets=[-1, 0, 1]).todense()
        evals, evecs = eigh(T)

    return evals, evecs


def approximate_boundaries(
    A: LinearOperator,
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

    if boundaries is None:
        eval_min, eval_max = eigsh(A, k=2, which="BE", **eigsh_kwargs)
    else:
        eval_min, eval_max = boundaries

        if eval_min is None:
            (eval_min,) = eigsh(A, k=1, which="SA", **eigsh_kwargs)
        if eval_max is None:
            (eval_max,) = eigsh(A, k=1, which="LA", **eigsh_kwargs)

    return eval_min, eval_max


def approximate_boundaries_abs(
    A: LinearOperator,
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
        Estimates of λₘᵢₙ and λₘₐₓ of |A|.
    """
    eval_min, eval_max = (None, None) if boundaries is None else boundaries

    eigsh_kwargs = {"tol": tol, "return_eigenvectors": False}
    if eval_max is None:
        (eval_max,) = eigsh(A, k=1, which="LM", **eigsh_kwargs)
    if eval_min is None:
        (eval_min,) = eigsh(A, k=1, which="SM", **eigsh_kwargs)

    return abs(eval_min), abs(eval_max)


def _gaussian(x: ndarray, mu: ndarray, sigma: float) -> ndarray:
    """Normal distribution pdf.

    Args:
        x: Position to evaluate.
        mu: Mean values.
        sigma: Standard deviation.

    Returns:
        Values of normal distribution.
    """
    return exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * sqrt(2 * pi))
