r"""Tests for :func:`make_compute_kfac_io_batch` IO collection.

Focus is on the ``intermediate_as_batch`` flag. When turned off, the
per-sample rank-one decomposition the IO collector returns should reconstruct
the exact per-parameter GGN block. The check runs on two fixture pools: the
``GGNLinearOperator`` ``case`` fixture (broad loss and architecture coverage)
and the ``kfac_weight_sharing_exact_case`` fixture (closes the
convolutional-weight-sharing gap).
"""

from collections.abc import Iterable

from einops import einsum
from pytest import raises
from torch import Tensor, block_diag, float64, manual_seed, rand
from torch.nn import Linear, Module, MSELoss, Sequential

from curvlinops import GGNLinearOperator
from curvlinops.computers.kfac_make_fx import (
    make_compute_kfac_io_batch,
    make_group_gatherers,
)
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import allclose_report, make_functional_call
from test.utils import Conv2dModel, WeightShareModel, block_diagonal, change_dtype


def _reconstruct_ggn_blocks(
    layer_inputs: dict[str, Tensor],
    layer_output_grads: dict[str, Tensor],
    mapping: list[dict[str, str]],
    io_groups: dict,
    io_param_names: dict,
    layer_hparams: dict,
    kfac_approx: str,
) -> dict[str, Tensor]:
    """Reconstruct per-parameter GGN blocks from the unflattened IO collection.

    For each parameter group ``{"W": wname}`` or ``{"b": bname}``, form the
    per-sample rank-one sum
    ``sum_{v, n, t} g_{v, n, t} g_{v, n, t}^T (otimes) a_{n, t} a_{n, t}^T``
    (for joint groups the returned ``a`` already includes a ``1`` in the bias
    padding slot, so the same formula applies).

    Args:
        layer_inputs: Per-IO-layer inputs from the collector.
        layer_output_grads: Per-IO-layer backpropped grad_outputs from the
            collector (already scaled by ``make_compute_kfac_io_batch`` for
            the given reduction).
        mapping: Parameter groups returned by the collector.
        io_groups: IO-layer grouping from the collector.
        io_param_names: Per-IO-layer parameter names from the collector.
        layer_hparams: Per-IO-layer hyperparameters from the collector.
        kfac_approx: KFAC approximation setting for the group gatherers.

    Returns:
        A dict mapping each parameter name to the reconstructed
        ``(numel, numel)`` GGN block.
    """
    group_inputs, group_grads = make_group_gatherers(
        io_groups, io_param_names, layer_hparams, kfac_approx
    )
    blocks: dict[str, Tensor] = {}
    for group in mapping:
        g = group_grads(group, layer_output_grads)  # (V, N, T, d_out)
        if "W" in group:
            a = group_inputs(group, layer_inputs)  # (N, T, d_in)
            d_out = g.shape[-1]
            d_in = a.shape[-1]
            block = einsum(
                g,
                g,
                a,
                a,
                "vec batch shared out_row, vec batch shared out_col, "
                "batch shared in_row, batch shared in_col "
                "-> out_row in_row out_col in_col",
            ).reshape(d_out * d_in, d_out * d_in)
            blocks[group["W"]] = block
        else:
            blocks[group["b"]] = einsum(
                g,
                g,
                "vec batch shared out_row, vec batch shared out_col -> out_row out_col",
            )
    return blocks


def _assert_io_unflattened_reconstructs_ggn(
    model: Module,
    loss_func: Module,
    params: dict[str, Tensor],
    X,
    y: Tensor,
    batch_size_fn,
    kfac_approx: str,
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
        kfac_approx: KFAC approximation setting for the group gatherers.
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
    (fn, mapping, io_groups, io_pnames, layer_hparams) = make_compute_kfac_io_batch(
        model_func,
        loss_func,
        params,
        X,
        FisherType.TYPE2,
        intermediate_as_batch=False,
    )
    layer_inputs, layer_output_grads = fn(params, X, y)

    blocks = _reconstruct_ggn_blocks(
        layer_inputs,
        layer_output_grads,
        mapping,
        io_groups,
        io_pnames,
        layer_hparams,
        kfac_approx,
    )

    reconstructed = block_diag(*(blocks[name] for name in params))
    assert allclose_report(reconstructed, ggn, atol=1e-10, rtol=1e-11)


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
        model, loss_func, params, X, y, batch_size_fn, KFACType.EXPAND
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
        model, loss_func, params, X, y, batch_size_fn, KFACType.EXPAND
    )


def test_empirical_rejects_unflattened():
    """``intermediate_as_batch=False`` raises with ``FisherType.EMPIRICAL``."""
    manual_seed(0)
    model = Sequential(Linear(4, 2))
    loss_func = MSELoss()
    params = dict(model.named_parameters())
    X = rand(2, 4)
    with raises(ValueError, match="EMPIRICAL"):
        make_compute_kfac_io_batch(
            make_functional_call(model),
            loss_func,
            params,
            X,
            fisher_type=FisherType.EMPIRICAL,
            intermediate_as_batch=False,
        )
