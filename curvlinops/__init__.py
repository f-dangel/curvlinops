"""``curvlinops`` library API."""

from curvlinops.diagonal.epperly2024xtrace import xdiag
from curvlinops.diagonal.hutchinson import hutchinson_diag
from curvlinops.ekfac import EKFACLinearOperator
from curvlinops.fisher import FisherMCLinearOperator
from curvlinops.ggn import GGNLinearOperator
from curvlinops.gradient_moments import EFLinearOperator
from curvlinops.hessian import HessianLinearOperator
from curvlinops.inverse import (
    CGInverseLinearOperator,
    KFACInverseLinearOperator,
    LSMRInverseLinearOperator,
    NeumannInverseLinearOperator,
)
from curvlinops.jacobian import JacobianLinearOperator, TransposedJacobianLinearOperator
from curvlinops.kfac import FisherType, KFACLinearOperator, KFACType
from curvlinops.norm.hutchinson import hutchinson_squared_fro
from curvlinops.papyan2020traces.spectrum import (
    LanczosApproximateLogSpectrumCached,
    LanczosApproximateSpectrumCached,
    lanczos_approximate_log_spectrum,
    lanczos_approximate_spectrum,
)
from curvlinops.submatrix import SubmatrixLinearOperator
from curvlinops.trace.epperly2024xtrace import xtrace
from curvlinops.trace.hutchinson import hutchinson_trace
from curvlinops.trace.meyer2020hutch import hutchpp_trace

__all__ = [
    # linear operators
    "HessianLinearOperator",
    "GGNLinearOperator",
    "EFLinearOperator",
    "FisherMCLinearOperator",
    "KFACLinearOperator",
    "EKFACLinearOperator",
    "JacobianLinearOperator",
    "TransposedJacobianLinearOperator",
    # Enums
    "FisherType",
    "KFACType",
    # inversion
    "CGInverseLinearOperator",
    "LSMRInverseLinearOperator",
    "NeumannInverseLinearOperator",
    "KFACInverseLinearOperator",
    # slicing
    "SubmatrixLinearOperator",
    # spectral properties
    "lanczos_approximate_spectrum",
    "lanczos_approximate_log_spectrum",
    "LanczosApproximateSpectrumCached",
    "LanczosApproximateLogSpectrumCached",
    # trace estimation
    "hutchinson_trace",
    "hutchpp_trace",
    "xtrace",
    # diagonal estimation
    "hutchinson_diag",
    "xdiag",
    # norm estimation
    "hutchinson_squared_fro",
]
