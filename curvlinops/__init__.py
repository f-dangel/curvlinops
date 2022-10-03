"""``curvlinops`` library API."""

from curvlinops.ggn import GGNLinearOperator
from curvlinops.hessian import HessianLinearOperator
from curvlinops.inverse import CGInverseLinearOperator
from curvlinops.papyan2020traces.spectrum import (
    lanczos_approximate_log_spectrum,
    lanczos_approximate_spectrum,
)

__all__ = [
    "HessianLinearOperator",
    "GGNLinearOperator",
    "CGInverseLinearOperator",
    "lanzcos_approximate_spectrum",
    "lanzcos_approximate_log_spectrum",
]
