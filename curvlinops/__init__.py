"""``curvlinops`` library API."""

from curvlinops.ekfac import EKFACLinearOperator
from curvlinops.ggn import GGNLinearOperator
from curvlinops.ggn_diagonal import GGNDiagonalLinearOperator
from curvlinops.gradient_moments import EFLinearOperator
from curvlinops.hessian import HessianLinearOperator
from curvlinops.inverse import (
    CGInverseLinearOperator,
    LSMRInverseLinearOperator,
    NeumannInverseLinearOperator,
)
from curvlinops.jacobian import JacobianLinearOperator, TransposedJacobianLinearOperator
from curvlinops.kfac import KFACLinearOperator
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.kfoc import KFOCLinearOperator
from curvlinops.papyan2020traces.spectrum import (
    LanczosApproximateLogSpectrumCached,
    LanczosApproximateSpectrumCached,
    lanczos_approximate_log_spectrum,
    lanczos_approximate_spectrum,
)
from curvlinops.submatrix import SubmatrixLinearOperator

__all__ = [
    # linear operators
    "HessianLinearOperator",
    "GGNLinearOperator",
    "GGNDiagonalLinearOperator",
    "EFLinearOperator",
    "KFACLinearOperator",
    "EKFACLinearOperator",
    "KFOCLinearOperator",
    "JacobianLinearOperator",
    "TransposedJacobianLinearOperator",
    # Enums
    "FisherType",
    "KFACType",
    # inversion
    "CGInverseLinearOperator",
    "LSMRInverseLinearOperator",
    "NeumannInverseLinearOperator",
    # slicing
    "SubmatrixLinearOperator",
    # spectral properties
    "lanczos_approximate_spectrum",
    "lanczos_approximate_log_spectrum",
    "LanczosApproximateSpectrumCached",
    "LanczosApproximateLogSpectrumCached",
]
