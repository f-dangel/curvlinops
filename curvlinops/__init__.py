"""``curvlinops`` library API."""

from curvlinops.diagonal.hutchinson import HutchinsonDiagonalEstimator
from curvlinops.fisher import FisherMCLinearOperator
from curvlinops.ggn import GGNLinearOperator
from curvlinops.gradient_moments import EFLinearOperator
from curvlinops.hessian import HessianLinearOperator
from curvlinops.inverse import CGInverseLinearOperator, NeumannInverseLinearOperator
from curvlinops.jacobian import JacobianLinearOperator, TransposedJacobianLinearOperator
from curvlinops.kfac import KFACLinearOperator
from curvlinops.papyan2020traces.spectrum import (
    LanczosApproximateLogSpectrumCached,
    LanczosApproximateSpectrumCached,
    lanczos_approximate_log_spectrum,
    lanczos_approximate_spectrum,
)
from curvlinops.submatrix import SubmatrixLinearOperator
from curvlinops.trace.hutchinson import HutchinsonTraceEstimator
from curvlinops.trace.meyer2020hutch import HutchPPTraceEstimator

__all__ = [
    "HessianLinearOperator",
    "GGNLinearOperator",
    "EFLinearOperator",
    "FisherMCLinearOperator",
    "KFACLinearOperator",
    "JacobianLinearOperator",
    "TransposedJacobianLinearOperator",
    "CGInverseLinearOperator",
    "NeumannInverseLinearOperator",
    "SubmatrixLinearOperator",
    "lanczos_approximate_spectrum",
    "lanczos_approximate_log_spectrum",
    "LanczosApproximateSpectrumCached",
    "LanczosApproximateLogSpectrumCached",
    "HutchinsonTraceEstimator",
    "HutchPPTraceEstimator",
    "HutchinsonDiagonalEstimator",
]
