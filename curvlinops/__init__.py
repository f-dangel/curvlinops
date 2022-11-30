"""``curvlinops`` library API."""

from curvlinops.fisher import FisherMCLinearOperator
from curvlinops.ggn import GGNLinearOperator
from curvlinops.gradient_moments import EFLinearOperator
from curvlinops.hessian import HessianLinearOperator
from curvlinops.inverse import CGInverseLinearOperator
from curvlinops.papyan2020traces.spectrum import (
    LanczosApproximateLogSpectrumCached,
    LanczosApproximateSpectrumCached,
    lanczos_approximate_log_spectrum,
    lanczos_approximate_spectrum,
)

__all__ = [
    "HessianLinearOperator",
    "GGNLinearOperator",
    "EFLinearOperator",
    "FisherMCLinearOperator",
    "CGInverseLinearOperator",
    "lanczos_approximate_spectrum",
    "lanczos_approximate_log_spectrum",
    "LanczosApproximateSpectrumCached",
    "LanczosApproximateLogSpectrumCached",
]
