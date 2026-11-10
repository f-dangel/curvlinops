"""Loss-function options (`ignore_index`, `weight`, etc.) and linear operators
combinations. This is to test that Curvlinops respects the semantics of
PyTorch's loss functions.

Each ``Option`` is a criterion with one non-default argument (or `"none"`, as a
control) plus the targets it is used with. Each ``Operator`` is a curvature
operator configuration. ``EXPECTED_FAILURES`` lists the cells where Curvlinops
does not yet match the ground truth.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, auto
from functools import partial

from pytest import FixtureRequest, mark
from torch import Tensor, float64, manual_seed, rand, randint, randn, tensor
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Identity,
    Linear,
    Module,
    ReLU,
    Sequential,
)

from curvlinops import (
    EFLinearOperator,
    EKFACLinearOperator,
    FisherType,
    GGNDiagonalLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
    KFOCLinearOperator,
)
from curvlinops._torch_base import PyTorchLinearOperator

REDUCTIONS = ("mean", "sum")
SEED = 0

IN_DIM, HIDDEN_DIM, SEQ_LEN = 5, 4, 3
NUM_CLASSES, NUM_OUTPUTS = 3, 2
BATCH_SIZES = (3, 4)

KFAC_BACKENDS = tuple(KFACLinearOperator._BACKENDS)
DENSE_MC_SAMPLES, KFAC_MC_SAMPLES = 100, 3_000


class Targets(Enum):
    """What the targets of a criterion look like."""

    CLASS = auto()
    SOFT = auto()
    BINARY = auto()

    @property
    def out_dim(self) -> int:
        """Number of model outputs (per position).

        Returns:
            The output dimension.
        """
        return NUM_OUTPUTS if self is Targets.BINARY else NUM_CLASSES


@dataclass(frozen=True, eq=False)
class _OptionSpec:
    """A criterion with its options and the kind of targets it is used with."""

    criterion: Callable[..., Module]
    targets: Targets = Targets.CLASS
    ignore_index: int | None = None
    seq: bool = False


class Option(Enum):
    """Loss-function option under test."""

    # Plain cross entropy
    CE_DEFAULT = _OptionSpec(CrossEntropyLoss)

    # Cross entropy with a default ignore index (-100)
    CE_IGNORE_DEFAULT = _OptionSpec(CrossEntropyLoss, ignore_index=-100)

    # Cross entropy with a custom ignore index (1)
    CE_IGNORE_CUSTOM = _OptionSpec(
        partial(CrossEntropyLoss, ignore_index=1), ignore_index=1
    )

    # Cross entropy with custom class weights
    CE_WEIGHT = _OptionSpec(
        partial(CrossEntropyLoss, weight=tensor([0.5, 2.0, 1.5], dtype=float64))
    )

    # Cross entropy with label smoothing
    CE_LABEL_SMOOTHING = _OptionSpec(partial(CrossEntropyLoss, label_smoothing=0.1))

    # Cross entropy with soft targets, i.e., probability vector instead of class index
    CE_SOFT_TARGETS = _OptionSpec(CrossEntropyLoss, Targets.SOFT)

    # Cross entropy with ignore index on sequence outputs, e.g., `(B, L, K)` output,
    # `(B, L)` target, some are `-100`.
    CE_SEQ_IGNORE_DEFAULT = _OptionSpec(CrossEntropyLoss, ignore_index=-100, seq=True)

    # Binary cross entropy, default options
    BCE_DEFAULT = _OptionSpec(BCEWithLogitsLoss, Targets.BINARY)

    # BCE loss with loss weights applied to `(B, 2)`
    BCE_WEIGHT = _OptionSpec(
        partial(BCEWithLogitsLoss, weight=tensor([0.5, 2.0], dtype=float64)),
        Targets.BINARY,
    )

    # BCE loss with weight added to the positive class in `(B, 2)`
    BCE_POS_WEIGHT = _OptionSpec(
        partial(BCEWithLogitsLoss, pos_weight=tensor([3.0, 0.5], dtype=float64)),
        Targets.BINARY,
    )

    def criterion(self, reduction: str) -> Module:
        """Create the criterion.

        Args:
            reduction: ``'mean'`` or ``'sum'``.

        Returns:
            The criterion with this option.
        """
        return self.value.criterion(reduction=reduction)

    def plain_criterion(self, reduction: str) -> Module:
        """Create the same kind of criterion without any non-default option.

        Args:
            reduction: ``'mean'`` or ``'sum'``.

        Returns:
            ``BCEWithLogitsLoss`` for binary targets, else ``CrossEntropyLoss``.
        """
        match self.value.targets:
            case Targets.BINARY:
                return BCEWithLogitsLoss(reduction=reduction)
            case _:
                return CrossEntropyLoss(reduction=reduction)

    def model(self, deep_linear: bool = False) -> Module:
        """Create a small double-precision model whose outputs fit the criterion.

        Args:
            deep_linear: Whether to drop the nonlinearity. KFAC is then exact on
                one datum.

        Returns:
            The model.
        """
        manual_seed(SEED)

        return Sequential(
            Linear(IN_DIM, HIDDEN_DIM),
            Identity() if deep_linear else ReLU(),
            Linear(HIDDEN_DIM, self.value.targets.out_dim),
            _ClassDimSecond() if self.value.seq else Identity(),
        ).to(float64)

    def data(self, one_datum: bool = False) -> list[tuple[Tensor, Tensor]]:
        """Create mini-batches.

        With an ignore index, every other target of the first batch is ignored.
        The batches then have different fractions of observed targets, which a
        per-batch normalization of ``reduction='mean'`` gets wrong.

        Args:
            one_datum: Whether to create a single batch with a single datum.

        Returns:
            List of ``(X, y)`` mini-batches.
        """
        manual_seed(SEED)

        spec = self.value
        seq = (SEQ_LEN,) if spec.seq else ()
        data = []

        for i, batch_size in enumerate((1,) if one_datum else BATCH_SIZES):
            X = rand(batch_size, *seq, IN_DIM, dtype=float64)
            y = self._targets(batch_size, seq)

            if i == 0 and spec.ignore_index is not None:
                y.view(-1)[::2] = spec.ignore_index

            data.append((X, y))

        return data

    def _targets(self, batch_size: int, seq: tuple[int, ...]) -> Tensor:
        """Draw random targets for one mini-batch.

        Args:
            batch_size: Number of data points.
            seq: Sequence dimensions of the outputs.

        Returns:
            The targets.
        """
        spec = self.value

        match spec.targets:
            case Targets.CLASS:
                # observed targets must differ from the ignore index
                classes = tensor([
                    c for c in range(NUM_CLASSES) if c != spec.ignore_index
                ])
                return classes[randint(len(classes), (batch_size, *seq))]
            case Targets.SOFT:
                return randn(batch_size, NUM_CLASSES, dtype=float64).softmax(-1)
            case Targets.BINARY:
                return rand(batch_size, NUM_OUTPUTS, dtype=float64)


class _ClassDimSecond(Module):
    """Move the class dimension from last to second, ``(B, L, C) -> (B, C, L)``."""

    def forward(self, x: Tensor) -> Tensor:
        """Swap the last two axes (works with and without batch axis).

        Args:
            x: Input of shape ``(..., L, C)``.

        Returns:
            Tensor of shape ``(..., C, L)``.
        """
        return x.movedim(-1, -2)


def drop_ignored(
    data: list[tuple[Tensor, Tensor]], ignore_index: int
) -> list[tuple[Tensor, Tensor]]:
    """Remove the data points whose target is ignored.

    Args:
        data: Mini-batches with targets of shape ``(B,)``.
        ignore_index: The ignored target value.

    Returns:
        The mini-batches without ignored data points.
    """
    return [(X[y != ignore_index], y[y != ignore_index]) for X, y in data]


@dataclass(frozen=True, eq=False)
class _OperatorSpec:
    """How to build a curvature operator."""

    linop: type[PyTorchLinearOperator]
    fisher_type: FisherType | None = None
    mc_samples: int | None = None
    backends: tuple[str | None, ...] = (None,)
    single_batch: bool = False

    def exact(self) -> "_OperatorSpec":
        """Swap MC sampling for the exact type-2 Fisher.

        Returns:
            The exact counterpart, or the operator itself if it does not sample.
        """
        if self.mc_samples is None:
            return self

        return replace(self, fisher_type=FisherType.TYPE2, mc_samples=None)

    def build(
        self,
        model: Module,
        loss_func: Module,
        params: dict[str, Tensor],
        data: list[tuple[Tensor, Tensor]],
        backend: str | None = None,
    ) -> PyTorchLinearOperator:
        """Build the operator.

        Args:
            model: The neural network.
            loss_func: The criterion.
            params: Parameters.
            data: Mini-batches.
            backend: Backend of KFAC-type operators. ``None`` uses the default.

        Returns:
            The linear operator.
        """
        kwargs = {
            "fisher_type": self.fisher_type,
            "mc_samples": self.mc_samples,
            "backend": backend,
        }

        return self.linop(
            model,
            loss_func,
            params,
            data,
            **{key: value for key, value in kwargs.items() if value is not None},
        )


class Operator(Enum):
    """Curvature operator under test."""

    HESSIAN = _OperatorSpec(HessianLinearOperator)
    GGN = _OperatorSpec(GGNLinearOperator)
    GGN_DIAG = _OperatorSpec(GGNDiagonalLinearOperator)
    EF = _OperatorSpec(EFLinearOperator)
    GGN_MC = _OperatorSpec(GGNLinearOperator, mc_samples=DENSE_MC_SAMPLES)
    # draws its samples once at construction, so it needs more of them
    GGN_DIAG_MC = _OperatorSpec(
        GGNDiagonalLinearOperator, mc_samples=40 * DENSE_MC_SAMPLES
    )
    KFAC_TYPE2 = _OperatorSpec(
        KFACLinearOperator, FisherType.TYPE2, backends=KFAC_BACKENDS
    )
    KFAC_MC = _OperatorSpec(
        KFACLinearOperator, FisherType.MC, KFAC_MC_SAMPLES, KFAC_BACKENDS
    )
    KFAC_EF = _OperatorSpec(
        KFACLinearOperator, FisherType.EMPIRICAL, backends=KFAC_BACKENDS
    )
    KFAC_FORWARD_ONLY = _OperatorSpec(
        KFACLinearOperator, FisherType.FORWARD_ONLY, backends=KFAC_BACKENDS
    )
    EKFAC_TYPE2 = _OperatorSpec(
        EKFACLinearOperator, FisherType.TYPE2, backends=KFAC_BACKENDS
    )
    EKFAC_EF = _OperatorSpec(
        EKFACLinearOperator, FisherType.EMPIRICAL, backends=KFAC_BACKENDS
    )
    KFOC_MC = _OperatorSpec(
        KFOCLinearOperator, FisherType.MC, KFAC_MC_SAMPLES, single_batch=True
    )

    @property
    def is_mc(self) -> bool:
        """Whether the operator is a Monte-Carlo estimate.

        Returns:
            ``True`` for MC operators.
        """
        return self.value.mc_samples is not None


class Reason(Enum):
    """Why a cell of the test matrix fails. ``test_xfail_reason`` checks each one."""

    OPTION_IGNORED = "the operator gives the same result without the option"
    MEAN_NORMALIZATION = "'sum' is right, 'mean' divides by the wrong number"
    IGNORED_INPUTS_COUNTED = "ignored data points still enter the input factor"
    CRASHES = "the operator raises"


_TBD_EF_WEIGHT = "should the EF carry w (data duplication) or w^2 (PyTorch)?"
_TBD_MC_WEIGHT = "what should MC sampling do with a target-dependent loss Hessian?"


@dataclass(frozen=True)
class XFail:
    """An expected failure. ``None`` for reduction or backend matches all.

    ``raises`` pins the exception of a crash. ``tbd`` notes an open question about
    what the correct result is.
    """

    operator: Operator
    option: Option
    reason: Reason
    reduction: str | None = None
    backend: str | None = None
    raises: type[Exception] | None = None
    tbd: str | None = None

    def __post_init__(self):
        """Check that exactly the crashes pin an exception.

        Raises:
            ValueError: If ``raises`` is set for a non-crash or missing for a crash.
        """
        if (self.reason is Reason.CRASHES) != (self.raises is not None):
            raise ValueError(f"Set `raises` exactly for crashes: {self}.")

    def cells(self) -> list[tuple[str, str | None]]:
        """List the ``(reduction, backend)`` cells this entry covers.

        Returns:
            The covered cells.
        """
        reductions = REDUCTIONS if self.reduction is None else (self.reduction,)
        backends = (
            self.operator.value.backends if self.backend is None else (self.backend,)
        )

        return [(r, b) for r in reductions for b in backends]

    def matches(
        self, operator: Operator, option: Option, reduction: str, backend: str | None
    ) -> bool:
        """Check whether this entry covers a test cell.

        Args:
            operator: The operator under test.
            option: The loss option under test.
            reduction: The reduction under test.
            backend: The backend under test, if any.

        Returns:
            Whether the cell is expected to fail.
        """
        return (
            (self.operator, self.option) == (operator, option)
            and self.reduction in (None, reduction)
            and self.backend in (None, backend)
        )


# Notes on the table:
# - With one datum, the class weight cancels for `CE_WEIGHT` + `'mean'`, so those
#   one-datum cells pass trivially.
# - Some MC cells with `'mean'` pass within MC tolerance although the option is
#   ignored (`CE_SEQ_IGNORE_DEFAULT`, `CE_WEIGHT`). Their `'sum'` cells catch it.
EXPECTED_FAILURES: list[XFail] = [
    XFail(
        Operator.HESSIAN,
        Option.CE_IGNORE_DEFAULT,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
    ),
    XFail(
        Operator.HESSIAN,
        Option.CE_IGNORE_CUSTOM,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
    ),
    XFail(
        Operator.HESSIAN, Option.CE_WEIGHT, Reason.MEAN_NORMALIZATION, reduction="mean"
    ),
    XFail(
        Operator.HESSIAN,
        Option.CE_SEQ_IGNORE_DEFAULT,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
    ),
    XFail(
        Operator.GGN,
        Option.CE_IGNORE_DEFAULT,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
    ),
    XFail(
        Operator.GGN,
        Option.CE_IGNORE_CUSTOM,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
    ),
    XFail(Operator.GGN, Option.CE_WEIGHT, Reason.MEAN_NORMALIZATION, reduction="mean"),
    XFail(
        Operator.GGN,
        Option.CE_SEQ_IGNORE_DEFAULT,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
    ),
    XFail(Operator.GGN_DIAG, Option.CE_IGNORE_DEFAULT, Reason.OPTION_IGNORED),
    XFail(Operator.GGN_DIAG, Option.CE_IGNORE_CUSTOM, Reason.OPTION_IGNORED),
    XFail(Operator.GGN_DIAG, Option.CE_WEIGHT, Reason.OPTION_IGNORED),
    XFail(Operator.GGN_DIAG, Option.CE_SEQ_IGNORE_DEFAULT, Reason.OPTION_IGNORED),
    XFail(Operator.GGN_DIAG, Option.BCE_WEIGHT, Reason.OPTION_IGNORED),
    XFail(Operator.GGN_DIAG, Option.BCE_POS_WEIGHT, Reason.OPTION_IGNORED),
    XFail(
        Operator.EF,
        Option.CE_IGNORE_DEFAULT,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
    ),
    XFail(
        Operator.EF,
        Option.CE_IGNORE_CUSTOM,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
    ),
    XFail(
        Operator.EF,
        Option.CE_WEIGHT,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
        tbd=_TBD_EF_WEIGHT,
    ),
    XFail(Operator.EF, Option.CE_SOFT_TARGETS, Reason.CRASHES, raises=ValueError),
    XFail(
        Operator.EF,
        Option.CE_SEQ_IGNORE_DEFAULT,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
    ),
    XFail(Operator.GGN_MC, Option.CE_IGNORE_DEFAULT, Reason.OPTION_IGNORED),
    XFail(Operator.GGN_MC, Option.CE_IGNORE_CUSTOM, Reason.OPTION_IGNORED),
    XFail(
        Operator.GGN_MC,
        Option.CE_WEIGHT,
        Reason.OPTION_IGNORED,
        reduction="sum",
        tbd=_TBD_MC_WEIGHT,
    ),
    XFail(
        Operator.GGN_MC,
        Option.CE_SEQ_IGNORE_DEFAULT,
        Reason.OPTION_IGNORED,
        reduction="sum",
    ),
    XFail(
        Operator.GGN_MC, Option.BCE_WEIGHT, Reason.OPTION_IGNORED, tbd=_TBD_MC_WEIGHT
    ),
    XFail(
        Operator.GGN_MC,
        Option.BCE_POS_WEIGHT,
        Reason.OPTION_IGNORED,
        tbd=_TBD_MC_WEIGHT,
    ),
    XFail(Operator.GGN_DIAG_MC, Option.CE_IGNORE_DEFAULT, Reason.OPTION_IGNORED),
    XFail(Operator.GGN_DIAG_MC, Option.CE_IGNORE_CUSTOM, Reason.OPTION_IGNORED),
    XFail(
        Operator.GGN_DIAG_MC,
        Option.CE_WEIGHT,
        Reason.OPTION_IGNORED,
        reduction="sum",
        tbd=_TBD_MC_WEIGHT,
    ),
    XFail(
        Operator.GGN_DIAG_MC,
        Option.CE_SEQ_IGNORE_DEFAULT,
        Reason.OPTION_IGNORED,
        reduction="sum",
    ),
    XFail(
        Operator.GGN_DIAG_MC,
        Option.BCE_WEIGHT,
        Reason.OPTION_IGNORED,
        tbd=_TBD_MC_WEIGHT,
    ),
    XFail(
        Operator.GGN_DIAG_MC,
        Option.BCE_POS_WEIGHT,
        Reason.OPTION_IGNORED,
        tbd=_TBD_MC_WEIGHT,
    ),
    XFail(Operator.KFAC_TYPE2, Option.CE_IGNORE_DEFAULT, Reason.OPTION_IGNORED),
    XFail(Operator.KFAC_TYPE2, Option.CE_IGNORE_CUSTOM, Reason.OPTION_IGNORED),
    XFail(
        Operator.KFAC_TYPE2, Option.CE_WEIGHT, Reason.OPTION_IGNORED, reduction="sum"
    ),
    XFail(
        Operator.KFAC_TYPE2, Option.CE_SOFT_TARGETS, Reason.CRASHES, raises=ValueError
    ),
    XFail(Operator.KFAC_TYPE2, Option.BCE_WEIGHT, Reason.OPTION_IGNORED),
    XFail(Operator.KFAC_TYPE2, Option.BCE_POS_WEIGHT, Reason.OPTION_IGNORED),
    XFail(Operator.KFAC_MC, Option.CE_IGNORE_DEFAULT, Reason.OPTION_IGNORED),
    XFail(Operator.KFAC_MC, Option.CE_IGNORE_CUSTOM, Reason.OPTION_IGNORED),
    XFail(
        Operator.KFAC_MC,
        Option.CE_WEIGHT,
        Reason.OPTION_IGNORED,
        reduction="sum",
        tbd=_TBD_MC_WEIGHT,
    ),
    XFail(Operator.KFAC_MC, Option.CE_SOFT_TARGETS, Reason.CRASHES, raises=ValueError),
    XFail(
        Operator.KFAC_MC, Option.BCE_WEIGHT, Reason.OPTION_IGNORED, tbd=_TBD_MC_WEIGHT
    ),
    XFail(
        Operator.KFAC_MC,
        Option.BCE_POS_WEIGHT,
        Reason.OPTION_IGNORED,
        tbd=_TBD_MC_WEIGHT,
    ),
    XFail(Operator.KFAC_EF, Option.CE_IGNORE_DEFAULT, Reason.IGNORED_INPUTS_COUNTED),
    XFail(Operator.KFAC_EF, Option.CE_IGNORE_CUSTOM, Reason.IGNORED_INPUTS_COUNTED),
    XFail(
        Operator.KFAC_EF,
        Option.CE_WEIGHT,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
        tbd=_TBD_EF_WEIGHT,
    ),
    XFail(Operator.KFAC_EF, Option.CE_SOFT_TARGETS, Reason.CRASHES, raises=ValueError),
    XFail(Operator.KFAC_FORWARD_ONLY, Option.CE_IGNORE_DEFAULT, Reason.OPTION_IGNORED),
    XFail(Operator.KFAC_FORWARD_ONLY, Option.CE_IGNORE_CUSTOM, Reason.OPTION_IGNORED),
    XFail(Operator.EKFAC_TYPE2, Option.CE_IGNORE_DEFAULT, Reason.OPTION_IGNORED),
    XFail(Operator.EKFAC_TYPE2, Option.CE_IGNORE_CUSTOM, Reason.OPTION_IGNORED),
    XFail(
        Operator.EKFAC_TYPE2, Option.CE_WEIGHT, Reason.OPTION_IGNORED, reduction="sum"
    ),
    XFail(
        Operator.EKFAC_TYPE2,
        Option.CE_SOFT_TARGETS,
        Reason.CRASHES,
        backend="make_fx",
        raises=ValueError,
    ),
    XFail(
        Operator.EKFAC_TYPE2,
        Option.CE_SOFT_TARGETS,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
        backend="hooks",
    ),
    XFail(Operator.EKFAC_TYPE2, Option.BCE_WEIGHT, Reason.OPTION_IGNORED),
    XFail(Operator.EKFAC_TYPE2, Option.BCE_POS_WEIGHT, Reason.OPTION_IGNORED),
    XFail(Operator.EKFAC_EF, Option.CE_IGNORE_DEFAULT, Reason.IGNORED_INPUTS_COUNTED),
    XFail(Operator.EKFAC_EF, Option.CE_IGNORE_CUSTOM, Reason.IGNORED_INPUTS_COUNTED),
    XFail(
        Operator.EKFAC_EF,
        Option.CE_WEIGHT,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
        tbd=_TBD_EF_WEIGHT,
    ),
    XFail(
        Operator.EKFAC_EF,
        Option.CE_SOFT_TARGETS,
        Reason.CRASHES,
        backend="make_fx",
        raises=ValueError,
    ),
    XFail(
        Operator.EKFAC_EF,
        Option.CE_SOFT_TARGETS,
        Reason.MEAN_NORMALIZATION,
        reduction="mean",
        backend="hooks",
    ),
    XFail(Operator.KFOC_MC, Option.CE_IGNORE_DEFAULT, Reason.OPTION_IGNORED),
    XFail(Operator.KFOC_MC, Option.CE_IGNORE_CUSTOM, Reason.OPTION_IGNORED),
]


def maybe_xfail(
    request: FixtureRequest,
    operator: Operator,
    option: Option,
    reduction: str,
    backend: str | None = None,
):
    """Mark the running test as a strict expected failure if the table lists it.

    Args:
        request: The pytest request of the running test.
        operator: The operator under test.
        option: The loss option under test.
        reduction: The reduction under test.
        backend: The backend under test, if any.
    """
    for xfail in EXPECTED_FAILURES:
        if xfail.matches(operator, option, reduction, backend):
            request.applymarker(
                mark.xfail(strict=True, reason=xfail.reason.value, raises=xfail.raises)
            )
            return
