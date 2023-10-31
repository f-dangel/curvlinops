"""Contains tests for ``curvlinops.kfac``."""

from typing import Iterable, List, Tuple

from numpy import eye
from pytest import mark
from scipy.linalg import block_diag
from torch import Tensor, randperm
from torch.nn import Module, MSELoss, Parameter

from curvlinops.examples.utils import report_nonclose
from curvlinops.ggn import GGNLinearOperator
from curvlinops.gradient_moments import EFLinearOperator
from curvlinops.kfac import KFACLinearOperator


@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac(
    kfac_expand_exact_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ],
    shuffle: bool,
):
    """Test the KFAC implementation against the exact GGN.

    Args:
        kfac_expand_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
    """
    model, loss_func, params, data = kfac_expand_exact_case

    if shuffle:
        permutation = randperm(len(params))
        params = [params[i] for i in permutation]

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


def test_kfac_one_datum(
    kfac_expand_exact_one_datum_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ]
):
    model, loss_func, params, data = kfac_expand_exact_one_datum_case

    ggn_blocks = []  # list of per-parameter GGNs
    for param in params:
        ggn = GGNLinearOperator(model, loss_func, [param], data)
        ggn_blocks.append(ggn @ eye(ggn.shape[1]))
    ggn = block_diag(*ggn_blocks)

    kfac = KFACLinearOperator(model, loss_func, params, data, mc_samples=10_000)
    kfac_mat = kfac @ eye(kfac.shape[1])

    atol = {"sum": 1e-3, "mean": 1e-3}[loss_func.reduction]
    rtol = {"sum": 3e-2, "mean": 3e-2}[loss_func.reduction]

    report_nonclose(ggn, kfac_mat, rtol=rtol, atol=atol)


def test_kfac_ef_one_datum(
    kfac_ef_exact_one_datum_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ]
):
    model, loss_func, params, data = kfac_ef_exact_one_datum_case

    ef_blocks = []  # list of per-parameter EFs
    for param in params:
        ef = EFLinearOperator(model, loss_func, [param], data)
        ef_blocks.append(ef @ eye(ef.shape[1]))
    ef = block_diag(*ef_blocks)

    kfac = KFACLinearOperator(model, loss_func, params, data, fisher_type="empirical")
    kfac_mat = kfac @ eye(kfac.shape[1])

    report_nonclose(ef, kfac_mat)
