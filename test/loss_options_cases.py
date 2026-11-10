"""Cases and references for testing loss-function options (``weight``, ...).

Each case pairs a criterion that uses exactly one non-default option with a model
and data. See https://github.com/f-dangel/curvlinops/issues/311.
"""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from einops.layers.torch import Rearrange
from pytest import param
from torch import Tensor, cat, float64, manual_seed, rand, randint, randn, stack, tensor
from torch.func import functional_call, jacrev
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Linear,
    Module,
    ReLU,
    Sequential,
)

REDUCTIONS = ["mean", "sum"]

NUM_CLASSES = 3  # for cross-entropy cases
NUM_OUTPUTS = 2  # for BCE cases
IN_DIM = 5
SEQ_LEN = 3
BATCH_SIZES = (3, 4)

LossOptionCase = dict[str, Any]


def _model_func(
    out_dim: int, seed: int, deep_linear: bool = False, seq: bool = False
) -> Callable[[], Module]:
    """Create a function that builds a small double-precision model.

    Args:
        out_dim: Output dimension.
        seed: Seed for the parameter initialization.
        deep_linear: Whether to omit the nonlinearity (KFAC is exact for one datum).
        seq: Whether to make the model emit ``(batch, out_dim, seq_len)``.

    Returns:
        Function without arguments that returns the model.
    """

    def build() -> Module:
        manual_seed(seed)
        layers = [Linear(IN_DIM, 4)]
        if not deep_linear:
            layers.append(ReLU())
        layers.append(Linear(4, out_dim))
        if seq:
            layers.append(Rearrange("b l c -> b c l"))
        return Sequential(*layers).to(float64)

    return build


def _targets(kind: str, shape: tuple[int, ...], num_ignored: int) -> Tensor:
    """Create targets of a given kind.

    Args:
        kind: One of ``'class'``, ``'ignore_default'``, ``'ignore_custom'``,
            ``'soft'``, ``'binary'``.
        shape: Shape of the (class-index) targets, e.g. ``(batch,)``. For ``'soft'``
            and ``'binary'`` this is the full shape of the targets.
        num_ignored: Number of entries that are ignored (flattened order from the
            front). Only used for the ``'ignore_*'`` kinds.

    Returns:
        The targets.

    Raises:
        ValueError: If the kind is unknown.
    """
    if kind == "class":
        return randint(0, NUM_CLASSES, shape)
    if kind == "ignore_default":
        y = randint(0, NUM_CLASSES, shape).flatten()
        y[:num_ignored] = -100
        return y.reshape(shape)
    if kind == "ignore_custom":
        # non-ignored entries must differ from the ignore index 1
        y = (2 * randint(0, 2, shape)).flatten()
        y[:num_ignored] = 1
        return y.reshape(shape)
    if kind == "soft":
        return randn(shape, dtype=float64).softmax(dim=-1)
    if kind == "binary":
        return rand(shape, dtype=float64)
    raise ValueError(kind)


def _make_case(
    case_id: str,
    loss_func: Callable[[str], Module],
    target_kind: str,
    out_dim: int,
    seed: int,
    family: str,
    seq: bool = False,
    num_ignored: tuple[int, int] = (0, 0),
) -> LossOptionCase:
    """Assemble a case dictionary.

    Args:
        case_id: Identifier of the case.
        loss_func: Function mapping a reduction to the criterion.
        target_kind: Passed to ``_targets``.
        out_dim: Model output dimension.
        seed: Seed for the model and the data.
        family: ``'dense'``, ``'one_datum'`` or ``'metamorphic'``.
        seq: Whether outputs are sequences ``(batch, out_dim, seq_len)``.
        num_ignored: Number of ignored target entries in the two batches.

    Returns:
        The case dictionary.
    """
    deep_linear = family == "one_datum"

    def data() -> list[tuple[Tensor, Tensor]]:
        manual_seed(seed + 1)
        batches = []
        for batch_size, n_ign in zip(
            (1,) if family == "one_datum" else BATCH_SIZES, num_ignored
        ):
            X_shape = (batch_size, SEQ_LEN, IN_DIM) if seq else (batch_size, IN_DIM)
            X = rand(*X_shape, dtype=float64)
            if target_kind in {"soft", "binary"}:
                y = _targets(target_kind, (batch_size, out_dim), 0)
            else:
                y_shape = (batch_size, SEQ_LEN) if seq else (batch_size,)
                y = _targets(target_kind, y_shape, n_ign)
                # at least one observed target per batch (mean would be NaN)
                assert (y != _ignore_index(target_kind)).any()
            batches.append((X, y))
        return batches

    case = {
        "id": case_id,
        "loss_func": loss_func,
        "model_func": _model_func(out_dim, seed, deep_linear=deep_linear, seq=seq),
        "data": data,
        "seed": seed,
        "family": family,
    }
    return case


def _ignore_index(target_kind: str) -> int:
    """Return the ignored target value for a target kind.

    Args:
        target_kind: The kind of targets.

    Returns:
        The ignored value, or ``-1000`` if the kind has none.
    """
    return {"ignore_default": -100, "ignore_custom": 1}.get(target_kind, -1000)


def _drop_ignored(data_func: Callable, target_kind: str) -> Callable:
    """Return a data function that removes rows with ignored targets.

    Args:
        data_func: Function that returns the original mini-batches.
        target_kind: The kind of targets.

    Returns:
        Function that returns the mini-batches without ignored rows.
    """

    def data_reference() -> list[tuple[Tensor, Tensor]]:
        idx = _ignore_index(target_kind)
        out = []
        for X, y in data_func():
            keep = y != idx
            out.append((X[keep], y[keep]))
        return out

    return data_reference


def _ce(**kwargs) -> Callable[[str], Module]:
    return lambda reduction: CrossEntropyLoss(reduction=reduction, **kwargs)


def _bce(**kwargs) -> Callable[[str], Module]:
    return lambda reduction: BCEWithLogitsLoss(reduction=reduction, **kwargs)


# (id, criterion factory, target kind, out_dim, seq, ignored entries per batch)
_OPTION_ROWS = [
    ("ce_default", _ce(), "class", NUM_CLASSES, False, (0, 0)),
    ("ce_ignore_default", _ce(), "ignore_default", NUM_CLASSES, False, (1, 2)),
    (
        "ce_ignore_custom",
        _ce(ignore_index=1),
        "ignore_custom",
        NUM_CLASSES,
        False,
        (1, 2),
    ),
    (
        "ce_weight",
        _ce(weight=tensor([0.5, 2.0, 1.5], dtype=float64)),
        "class",
        NUM_CLASSES,
        False,
        (0, 0),
    ),
    (
        "ce_label_smoothing",
        _ce(label_smoothing=0.1),
        "class",
        NUM_CLASSES,
        False,
        (0, 0),
    ),
    ("ce_soft_targets", _ce(), "soft", NUM_CLASSES, False, (0, 0)),
    ("ce_seq_ignore_default", _ce(), "ignore_default", NUM_CLASSES, True, (2, 5)),
    ("bce_default", _bce(), "binary", NUM_OUTPUTS, False, (0, 0)),
    (
        "bce_weight",
        _bce(weight=tensor([0.5, 2.0], dtype=float64)),
        "binary",
        NUM_OUTPUTS,
        False,
        (0, 0),
    ),
    (
        "bce_pos_weight",
        _bce(pos_weight=tensor([3.0, 0.5], dtype=float64)),
        "binary",
        NUM_OUTPUTS,
        False,
        (0, 0),
    ),
]


def _build_cases() -> tuple[list, list, list]:
    """Build the dense, one-datum and metamorphic case lists.

    Returns:
        The dense, one-datum, and metamorphic cases.
    """
    dense, one_datum, metamorphic = [], [], []
    for i, (cid, lf, kind, out_dim, seq, n_ign) in enumerate(_OPTION_ROWS):
        seed = 100 + i
        dense.append(_make_case(cid, lf, kind, out_dim, seed, "dense", seq, n_ign))
        if "ignore" not in cid:
            one_datum.append(_make_case(cid, lf, kind, out_dim, seed, "one_datum"))
        if cid in {"ce_ignore_default", "ce_ignore_custom"}:
            case = _make_case(cid, lf, kind, out_dim, seed, "metamorphic", seq, n_ign)
            case["data_reference"] = _drop_ignored(case["data"], kind)
            case["loss_func_reference"] = _ce()
            metamorphic.append(case)
    return dense, one_datum, metamorphic


DENSE_CASES, ONE_DATUM_CASES, METAMORPHIC_CASES = _build_cases()


def case_params(cases: list[LossOptionCase]) -> list:
    """Wrap cases into ``pytest.param`` objects with readable ids.

    Args:
        cases: The cases.

    Returns:
        List of ``pytest.param``.
    """
    return [param(c, id=c["id"]) for c in cases]


def reference_ef(
    model: Module,
    loss_func: Module,
    params: dict[str, Tensor],
    data: list[tuple[Tensor, Tensor]],
) -> Tensor:
    """Compute the empirical Fisher of a criterion with its options.

    The EF is ``c * sum_k grad(l_k) grad(l_k)^T``. Loss terms ``l_k`` are one per
    sequence position for ``CrossEntropyLoss`` and one per output row for
    ``BCEWithLogitsLoss``; each is the ``reduction='sum'`` criterion (with the same
    options) on that term alone. For ``'sum'``, ``c = 1``. For ``'mean'``, ``c`` is
    the ratio of the mean- and sum-reduced losses on all data, which reproduces
    PyTorch's denominator (observed targets, class weights, ...).

    Args:
        model: The neural network.
        loss_func: The criterion, including its reduction.
        params: Parameters w.r.t. which the EF is computed.
        data: List of mini-batches.

    Returns:
        The EF as a dense matrix.
    """
    # NOTE(#311): with `weight`/`pos_weight` this uses PyTorch-literal semantics
    # (the per-term gradient carries w once, so the EF carries w^2). Whether the
    # EF should instead carry w (data duplication) is an open question (ask Felix).
    X = cat([x for x, _ in data])
    y = cat([t for _, t in data])
    params_dict = {n: p.detach() for n, p in params.items()}

    loss_sum = deepcopy(loss_func)
    loss_sum.reduction = "sum"

    def terms(p: dict[str, Tensor]) -> Tensor:
        out = functional_call(model, p, (X,))
        if isinstance(loss_func, CrossEntropyLoss):
            soft = y.ndim == out.ndim
            out_flat = out.movedim(1, -1).reshape(-1, out.shape[1])
            y_flat = (
                y.movedim(1, -1).reshape(-1, out.shape[1]) if soft else y.reshape(-1)
            )
        else:
            out_flat = out.reshape(-1, out.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])
        return stack([loss_sum(o, t) for o, t in zip(out_flat, y_flat)])

    jac = jacrev(terms)(params_dict)
    jac = cat([j.flatten(start_dim=1) for j in jac.values()], dim=1)

    scale = 1.0
    if loss_func.reduction == "mean":
        out = functional_call(model, params_dict, (X,))
        scale = (loss_func(out, y) / loss_sum(out, y)).item()
    return scale * jac.T @ jac
