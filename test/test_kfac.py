"""Contains tests for ``curvlinops.kfac``."""

from typing import Iterable, List, Tuple

from numpy import eye
from scipy.linalg import block_diag
from torch import Tensor
from torch.nn import Module, MSELoss, Parameter

from curvlinops.examples.utils import report_nonclose
from curvlinops.ggn import GGNLinearOperator
from curvlinops.kfac import KFACLinearOperator


def test_kfac(
    kfac_expand_exact_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ]
):
    model, loss_func, params, data = kfac_expand_exact_case

    ggn_blocks = []  # list of per-parameter GGNs
    for param in params:
        ggn = GGNLinearOperator(model, loss_func, [param], data)
        ggn_blocks.append(ggn @ eye(ggn.shape[1]))
    ggn = block_diag(*ggn_blocks)

    kfac = KFACLinearOperator(model, loss_func, params, data, mc_samples=2_000)
    kfac_mat = kfac @ eye(kfac.shape[1])

    atol = {"sum": 5e-1, "mean": 5e-3}[loss_func.reduction]
    rtol = {"sum": 2e-2, "mean": 2e-2}[loss_func.reduction]

    report_nonclose(ggn, kfac_mat, rtol=rtol, atol=atol)
