"""Tests data type issue from https://github.com/f-dangel/curvlinops/issues/132."""

from test.cases import DEVICES, DEVICES_IDS

from numpy import random
from pytest import mark, raises
from torch import device, float64, manual_seed, rand
from torch.nn import Linear, MSELoss

from curvlinops import KFACLinearOperator


@mark.parametrize("dev", DEVICES, ids=DEVICES_IDS)
def test_bug_132_dtype_deterministic_checks(dev: device):
    """Test whether the vectors used in the deterministic checks have correct data type.

    This bug was reported in https://github.com/f-dangel/curvlinops/issues/132.

    Args:
        dev: The device to run the test on.
    """
    # make deterministic
    manual_seed(0)
    random.seed(0)

    # create a toy problem, load everything to float64
    dt = float64
    N = 4
    D_in = 3
    D_out = 2

    X = rand(N, D_in, dtype=dt, device=dev)
    y = rand(N, D_out, dtype=dt, device=dev)
    data = [(X, y)]

    model = Linear(D_in, D_out).to(dev, dt)
    params = [p for p in model.parameters() if p.requires_grad]

    loss_func = MSELoss().to(dev, dt)

    with raises(RuntimeError):
        _ = KFACLinearOperator(model, loss_func, params, data, check_deterministic=True)
