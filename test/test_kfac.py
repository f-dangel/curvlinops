"""Contains tests for ``curvlinops.kfac``."""

from typing import Iterable, List, Tuple

from numpy import eye
from pytest import mark
from scipy.linalg import block_diag
from torch import Tensor, randperm
from torch.nn import Module, MSELoss, Parameter

from curvlinops.examples.utils import report_nonclose
from curvlinops.ggn import GGNLinearOperator
from curvlinops.kfac import KFACLinearOperator


@mark.parametrize("include_weights", [True, False], ids=["weights", "no_weights"])
@mark.parametrize("shuffle", [False, True], ids=["", "shuffled"])
def test_kfac(
    kfac_expand_exact_case: Tuple[
        Module, MSELoss, List[Parameter], Iterable[Tuple[Tensor, Tensor]]
    ],
    shuffle: bool,
    include_weights: bool,
):
    """Test the KFAC implementation against the exact GGN.

    Args:
        kfac_expand_exact_case: A fixture that returns a model, loss function, list of
            parameters, and data.
        shuffle: Whether to shuffle the parameters before computing the KFAC matrix.
        include_weights: Whether to include weight parameters in the KFAC matrix.
    """
    model, loss_func, params, data = kfac_expand_exact_case

    if not include_weights:
        names = {p.data_ptr(): name for name, p in model.named_parameters()}
        params = [p for p in params if "weight" not in names[p.data_ptr()]]

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

    # Check that input covariances were not computed
    if not include_weights:
        assert len(kfac._input_covariances) == 0


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
