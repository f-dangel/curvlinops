r"""Tests for :class:`LayerIO`'s unflattened IO collection.

Focus is on the ``intermediate_as_batch=False`` setting. The per-sample
rank-one decomposition the IO collector returns must reconstruct the exact
per-parameter GGN block. The check runs on two fixture pools: the
``GGNLinearOperator`` ``case`` fixture (broad loss and architecture coverage)
and the ``kfac_weight_sharing_exact_case`` fixture (closes the
convolutional-weight-sharing gap).
"""

from collections.abc import Iterable

from einops import einsum
from torch import Tensor, block_diag, float64
from torch.nn import Module, MSELoss

from curvlinops import GGNLinearOperator
from curvlinops.computers.io_collector import LayerIO, LayerIOSnapshot
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import allclose_report, make_functional_call
from test.utils import Conv2dModel, WeightShareModel, block_diagonal, change_dtype


def _reconstruct_ggn_blocks(
    snap: LayerIOSnapshot, mapping: list[dict[str, str]]
) -> dict[str, Tensor]:
    """Reconstruct per-parameter GGN blocks from the unflattened IO collection.

    For each parameter group ``{"W": wname}`` or ``{"b": bname}``, form the
    per-sample rank-one sum
    ``sum_{v, n, t} g_{v, n, t} g_{v, n, t}^T (otimes) a_{n, t} a_{n, t}^T``
    (for joint groups the returned ``a`` already includes a ``1`` in the bias
    padding slot, so the same formula applies).

    Args:
        snap: Snapshot produced by :meth:`LayerIO.snapshot` after
            :meth:`LayerIO.populate` ran with
            ``intermediate_as_batch=False`` and ``FisherType.TYPE2``.
        mapping: Parameter groups (use ``io.mapping``).

    Returns:
        A dict mapping each parameter name to the reconstructed
        ``(numel, numel)`` GGN block.
    """
    blocks: dict[str, Tensor] = {}
    for group in mapping:
        a, g = snap.standardized_io(group)
        if "W" in group:
            # Per-sample vec(W) gradient: sum_t g (otimes) a, reshaped to
            # a ``(vec*batch, d_out*d_in)`` matrix so its Gram is the GGN block.
            per_sample_grads = einsum(
                g, a, "vec batch shared out, batch shared inp -> vec batch out inp"
            ).reshape(-1, g.shape[-1] * a.shape[-1])
            blocks[group["W"]] = per_sample_grads.T @ per_sample_grads
        else:
            blocks[group["b"]] = einsum(
                g, g, "vec batch shared row, vec batch shared col -> row col"
            )
    return blocks


def _assert_io_unflattened_reconstructs_ggn(
    model: Module,
    loss_func: Module,
    params: dict[str, Tensor],
    X,
    y: Tensor,
    batch_size_fn,
) -> None:
    """Run the IO collector and verify exact per-parameter GGN reconstruction.

    With ``intermediate_as_batch=False`` + ``FisherType.TYPE2``, the
    per-sample rank-one decomposition must equal the exact per-parameter
    block of the GGN (for position-wise layers).

    Args:
        model: The neural network (or callable ``model_func``).
        loss_func: The loss function.
        params: Named model parameters.
        X: A single input batch (tensor or dict).
        y: The corresponding target batch.
        batch_size_fn: Function extracting the batch size from ``X``.
    """
    ggn = block_diagonal(
        GGNLinearOperator,
        model,
        loss_func,
        params,
        [(X, y)],
        batch_size_fn=batch_size_fn,
    )

    for p in params.values():
        p.requires_grad_(True)
    model_func = make_functional_call(model) if isinstance(model, Module) else model
    io = LayerIO(
        model_func,
        loss_func,
        params,
        X,
        fisher_type=FisherType.TYPE2,
        kfac_approx=KFACType.EXPAND,
        intermediate_as_batch=False,
        batch_size_fn=batch_size_fn,
    )
    snap = io.snapshot(*io.populate(params, X, y))

    blocks = _reconstruct_ggn_blocks(snap, io.mapping)
    reconstructed = block_diag(*(blocks[name] for name in params))
    assert allclose_report(reconstructed, ggn)


def test_kfac_io_unflattened_reconstructs_ggn(
    case: tuple[
        Module,
        Module,
        dict[str, Tensor],
        Iterable[tuple[Tensor, Tensor]],
        object,
    ],
):
    """IO collector with ``intermediate_as_batch=False`` reconstructs the GGN.

    Parametrized over the same cases as ``GGNLinearOperator``'s matvec test so
    CE/BCE/MSE, both reductions, 2D/3D inputs, and dict-style inputs are all
    exercised.

    Args:
        case: Fixture providing model, loss, parameters, data, and optional
            batch-size function.
    """
    model, loss_func, params, data, batch_size_fn = change_dtype(case, float64)
    X, y = next(iter(data))
    _assert_io_unflattened_reconstructs_ggn(
        model, loss_func, params, X, y, batch_size_fn
    )


def test_kfac_io_unflattened_reconstructs_ggn_weight_sharing(
    kfac_weight_sharing_exact_case: tuple[
        WeightShareModel | Conv2dModel,
        MSELoss,
        dict[str, Tensor],
        dict[str, Iterable[tuple[Tensor, Tensor]]],
        object,
    ],
):
    """Same reconstruction check, but on the weight-sharing fixture.

    Covers Conv2d (via unfolded per-position linear layers in the IO
    collector) and the expand-setting linear weight-sharing models. Restricted
    to the EXPAND setting, where each output row depends only on its matching
    input row.

    Args:
        kfac_weight_sharing_exact_case: Fixture providing a weight-sharing
            model + MSE, and data keyed by KFAC setting.
    """
    model, loss_func, params, data, batch_size_fn = kfac_weight_sharing_exact_case
    model.setting = KFACType.EXPAND
    if isinstance(model, Conv2dModel):
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    data = data[KFACType.EXPAND]
    model, loss_func, params, data, batch_size_fn = change_dtype(
        (model, loss_func, params, data, batch_size_fn), float64
    )
    X, y = next(iter(data))
    _assert_io_unflattened_reconstructs_ggn(
        model, loss_func, params, X, y, batch_size_fn
    )
