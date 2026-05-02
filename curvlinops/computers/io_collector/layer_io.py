"""High-level orchestration over the IO collector for KFAC-style operators."""

from __future__ import annotations

from collections import UserDict
from collections.abc import Callable, MutableMapping
from contextlib import contextmanager
from math import sqrt

from einops import einsum
from torch import Tensor, autograd
from torch.nn import CrossEntropyLoss

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers._base import ParamGroup, _BaseKFACComputer
from curvlinops.computers.io_collector.collector import with_kfac_io
from curvlinops.computers.io_collector.groups import (
    _build_param_groups_from_io,
    make_group_gatherers,
)
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _enable_requires_grad


class LayerIO:
    r"""Setup-once orchestrator for per-layer IO collection.

    Owns shape-independent metadata (parameter groups, IO-layer mappings) and
    a per-batch-size cache of FX-traced ``io_fn``\ s built by
    :func:`with_kfac_io`; new batch sizes trigger a fresh trace via
    :meth:`ensure_io_fn`. Operators construct one :class:`LayerIO` per
    ``compute()`` call, then per batch call :meth:`populate` for raw layer IO
    and :meth:`snapshot` to wrap it as a :class:`LayerIOSnapshot` exposing
    per-group accessors at three granularities (raw, standardized weight-
    sharing format, expanded per-sample ``vec(W)`` gradients).

    Three integration patterns: (1) trace everything (KFAC) — wrap the
    per-batch reduction in :func:`_make_fx` under :meth:`enable_param_grads`;
    (2) trace IO only (KFOC) — wrap :meth:`populate` only, replay under
    :func:`torch.no_grad`; (3) fully eager — skip :func:`_make_fx`.

    Args:
        model_func: Functional model ``(params, X) -> prediction``.
        loss_func: Loss function (``MSELoss``, ``CrossEntropyLoss``, or
            ``BCEWithLogitsLoss``).
        params: Named parameter dict. Used at construction to bootstrap the
            first ``io_fn`` trace; param values are not retained.
        X_example: Example input tensor for the bootstrap trace. Determines
            the cache key for the initial ``io_fn``.
        fisher_type: Type of Fisher information. Defaults to
            :attr:`FisherType.MC`.
        mc_samples: Number of Monte-Carlo samples (only used with
            :attr:`FisherType.MC`). Defaults to ``1``.
        kfac_approx: KFAC approximation type for per-group standardization
            (:attr:`KFACType.EXPAND` or :attr:`KFACType.REDUCE`). Defaults to
            :attr:`KFACType.EXPAND`.
        separate_weight_and_bias: Whether to treat weights and biases as
            separate parameter groups. Defaults to ``True``.
        intermediate_as_batch: Whether to flatten the model output's
            intermediate (non-batch, non-class) axes into the batch axis when
            forming ``grad_outputs``. ``True`` (default) reproduces
            KFAC-expand. ``False`` keeps the intermediate axes separate
            (consumed by KFOC). Not supported with
            :attr:`FisherType.EMPIRICAL`.
        batch_size_fn: Maps an input batch to its size. Used as the
            ``io_fn`` cache key. Defaults to ``X.shape[0]`` (must be supplied
            for non-Tensor inputs like :class:`UserDict`).

    Raises:
        ValueError: If ``intermediate_as_batch=False`` is combined with
            :attr:`FisherType.EMPIRICAL`.
    """

    def __init__(
        self,
        model_func: Callable,
        loss_func: Callable,
        params: dict[str, Tensor],
        X_example: Tensor | MutableMapping,
        fisher_type: FisherType = FisherType.MC,
        mc_samples: int = 1,
        kfac_approx: str = KFACType.EXPAND,
        separate_weight_and_bias: bool = True,
        intermediate_as_batch: bool = True,
        batch_size_fn: Callable[[Tensor | MutableMapping], int] | None = None,
    ):
        """Bootstrap shape-independent metadata and trace the first ``io_fn``.

        Raises:
            ValueError: If ``intermediate_as_batch=False`` is combined with
                :attr:`FisherType.EMPIRICAL`.
        """
        if not intermediate_as_batch and fisher_type == FisherType.EMPIRICAL:
            raise ValueError(
                "intermediate_as_batch=False is not supported with "
                "FisherType.EMPIRICAL because the per-datum loss helper "
                "assumes a 1d prediction shape."
            )

        self._model_func = model_func
        self._loss_func = loss_func
        self.fisher_type = fisher_type
        self._mc_samples = mc_samples
        self.kfac_approx = kfac_approx
        self._separate_weight_and_bias = separate_weight_and_bias
        self._intermediate_as_batch = intermediate_as_batch
        self._batch_size_fn = batch_size_fn or (lambda X: X.shape[0])

        self._grad_outputs_computer = _BaseKFACComputer._set_up_grad_outputs_computer(
            loss_func, fisher_type, mc_samples
        )

        # Bootstrap: trace once to seed shape-independent metadata; subsequent
        # batch sizes go through ensure_io_fn which validates against the seed.
        if isinstance(X_example, UserDict):
            _register_userdict_as_pytree()
        io_fn, self.io_param_names, self.layer_hparams = with_kfac_io(
            model_func, X_example, params, fisher_type
        )
        self._io_fns: dict[int, Callable] = {self._batch_size_fn(X_example): io_fn}

        self.mapping, self.io_groups = _build_param_groups_from_io(
            self.io_param_names, separate_weight_and_bias
        )
        self.group_inputs, self.group_grads = make_group_gatherers(
            self.io_groups, self.io_param_names, self.layer_hparams, kfac_approx
        )

    def ensure_io_fn(
        self, X: Tensor | MutableMapping, params: dict[str, Tensor]
    ) -> Callable:
        """Return the FX-traced ``io_fn`` for ``X``'s batch size, building if needed.

        Validates that re-traced metadata matches the seed from construction
        to detect violations of the shape-independence assumption.

        Args:
            X: Input batch (only its batch size is consulted for the cache key).
            params: Named parameters, passed through to :func:`with_kfac_io`.

        Returns:
            A traced callable ``(params, X) -> (output, layer_inputs, layer_outputs)``.

        Raises:
            RuntimeError: If re-traced metadata for a new batch size disagrees
                with the bootstrap metadata.
        """
        key = self._batch_size_fn(X)
        if key in self._io_fns:
            return self._io_fns[key]

        if isinstance(X, UserDict):
            _register_userdict_as_pytree()

        io_fn, io_param_names, layer_hparams = with_kfac_io(
            self._model_func, X, params, self.fisher_type
        )
        if io_param_names != self.io_param_names:
            raise RuntimeError(
                "IO-collector parameter-name metadata changed across batch sizes. "
                f"Expected {self.io_param_names}, got {io_param_names}."
            )
        if layer_hparams != self.layer_hparams:
            raise RuntimeError(
                "IO-collector layer hyperparameters changed across batch sizes. "
                f"Expected {self.layer_hparams}, got {layer_hparams}."
            )
        self._io_fns[key] = io_fn
        return io_fn

    def populate(
        self,
        params: dict[str, Tensor],
        X: Tensor | MutableMapping,
        y: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Run forward + backward; return per-layer raw IO.

        Reuses the cached ``io_fn`` for ``X``'s batch size (or builds one).
        Backpropagates the Fisher-type-specific gradient outputs to produce
        per-layer inputs and batched output gradients. ``layer_output_grads``
        is scaled so that ``sum_v g g^T`` equals the batch loss Hessian
        (mean-reduction's ``1/N`` is folded in once as ``1/sqrt(N)`` per
        vector).

        Returns plain ``(dict, dict)`` rather than a :class:`LayerIOSnapshot`
        so the function is trivially trace-friendly with
        :func:`torch.fx.experimental.proxy_tensor.make_fx`. Build the snapshot
        separately via :meth:`snapshot`.

        Args:
            params: Named model parameters.
            X: Input batch.
            y: Target batch.

        Returns:
            ``(layer_inputs, layer_output_grads)`` dicts keyed by IO layer name.
            ``layer_output_grads`` is empty for :attr:`FisherType.FORWARD_ONLY`.
        """
        io_fn = self.ensure_io_fn(X, params)
        output, layer_inputs, layer_outputs = io_fn(params, X)

        if self.fisher_type == FisherType.FORWARD_ONLY:
            return layer_inputs, {}

        if self._intermediate_as_batch:
            # ``CrossEntropyLoss`` expects class dim second; other losses last.
            if isinstance(self._loss_func, CrossEntropyLoss):
                output_for_grad = output.movedim(1, -1).flatten(0, -2)
                y_for_grad = y.flatten()
            else:
                output_for_grad = output.flatten(0, -2)
                y_for_grad = y.flatten(0, -2)
        else:
            output_for_grad, y_for_grad = output, y

        grad_outputs = self._grad_outputs_computer(
            output_for_grad.detach(), y_for_grad, None
        )
        # Equivalent to the hooks backend's two-step scaling: combining both
        # into a single ``1/sqrt(N)`` per vector squares to the same ``1/N``
        # on ``ggT``.
        mean_scale = 1.0 / sqrt(output_for_grad.shape[0])
        grad_outputs.mul_({"sum": 1.0, "mean": mean_scale}[self._loss_func.reduction])

        io_layer_names, output_tensors = zip(*layer_outputs.items())
        layer_output_grads_list = autograd.grad(
            output_for_grad,
            output_tensors,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
        )
        layer_output_grads = dict(zip(io_layer_names, layer_output_grads_list))
        return layer_inputs, layer_output_grads

    def snapshot(
        self,
        layer_inputs: dict[str, Tensor],
        layer_output_grads: dict[str, Tensor],
    ) -> LayerIOSnapshot:
        """Wrap raw per-batch IO in a snapshot with per-group accessors.

        Args:
            layer_inputs: Per IO-layer input tensors (from :meth:`populate`).
            layer_output_grads: Per IO-layer batched output grads.

        Returns:
            A :class:`LayerIOSnapshot` bound to this :class:`LayerIO`.
        """
        return LayerIOSnapshot(self, layer_inputs, layer_output_grads)

    @contextmanager
    def enable_param_grads(self, params: dict[str, Tensor]):
        """Temporarily enable ``requires_grad`` on ``params`` for the trace boundary.

        Required around ``_make_fx`` calls that include :meth:`populate`, so
        the ``autograd.grad`` call inside has differentiable inputs at trace
        time. Prior ``requires_grad`` state is restored on exit.

        Args:
            params: Named parameters whose ``requires_grad`` should be
                temporarily enabled.

        Yields:
            None.
        """
        with _enable_requires_grad(list(params.values())):
            yield


class LayerIOSnapshot:
    """Per-batch raw IO with on-demand per-group accessors.

    Built via :meth:`LayerIO.snapshot`. Provides two granularities of
    per-group access at increasing computational cost (raw per IO-layer dicts
    are exposed as :attr:`layer_inputs` / :attr:`layer_output_grads`):

    1. :meth:`standardized_io` — per-group ``(a, g)`` in weight-sharing format.
    2. :meth:`per_sample_grads` — per-group per-sample ``vec(W)`` gradients.
    """

    def __init__(
        self,
        owner: LayerIO,
        layer_inputs: dict[str, Tensor],
        layer_output_grads: dict[str, Tensor],
    ):
        """Store the owner :class:`LayerIO` and the raw IO dicts.

        Args:
            owner: The :class:`LayerIO` whose metadata + gatherers this
                snapshot delegates to.
            layer_inputs: Per IO-layer input tensors.
            layer_output_grads: Per IO-layer batched output grads.
        """
        self._owner = owner
        self.layer_inputs = layer_inputs
        self.layer_output_grads = layer_output_grads

    def standardized_io(self, group: ParamGroup) -> tuple[Tensor | None, Tensor | None]:
        r"""Return per-group ``(a, g)`` in weight-sharing format.

        Concatenates contributions from all IO layers that share this group's
        weight, applies bias padding for joint W+b groups, and converts to
        the ``[batch, shared, *]`` layout consumed by KFAC-style operators.

        Args:
            group: Parameter group dict (``{"W": ..., "b": ...}`` or subset).

        Returns:
            Tuple ``(a, g)`` where:

            * ``a`` has shape ``[B, S, d_in]`` if ``"W" in group``, else ``None``
              (bias-only groups have no input factor).
            * ``g`` has shape ``[V, B, S, d_out]`` for stochastic Fisher types,
              or ``None`` for :attr:`FisherType.FORWARD_ONLY`.
        """
        a = self._owner.group_inputs(group, self.layer_inputs) if "W" in group else None
        g = (
            self._owner.group_grads(group, self.layer_output_grads)
            if self._owner.fisher_type != FisherType.FORWARD_ONLY
            else None
        )
        return a, g

    def per_sample_grads(self, group: ParamGroup) -> Tensor:
        r"""Return per-sample parameter gradients for a group.

        For W-containing groups, builds the explicit ``vec(W)`` per-sample
        gradient tensor :math:`P_{v,n} = \sum_t g_{v,n,t} a_{n,t}^\top` of
        shape ``[V, B, d_out, d_in]``. Memory: ``V * B * d_out * d_in *
        dtype.itemsize`` per group — caller should free between groups.

        For bias-only groups, returns per-sample bias gradients
        :math:`b_{v,n} = \sum_t g_{v,n,t}` of shape ``[V, B, d_out]`` (sum
        over the shared axis).

        Args:
            group: Parameter group dict.

        Returns:
            Per-sample parameter gradient tensor; shape varies with group
            structure (see above).

        Raises:
            RuntimeError: If the owning :class:`LayerIO` was constructed with
                :attr:`FisherType.FORWARD_ONLY` (no gradient outputs were
                collected, so per-sample param grads are undefined).
        """
        a, g = self.standardized_io(group)
        if g is None:
            raise RuntimeError(
                "per_sample_grads is undefined for FisherType.FORWARD_ONLY: "
                "the IO collector did not backpropagate gradient outputs."
            )
        if "W" in group:
            return einsum(
                g, a, "vec batch shared out, batch shared inp -> vec batch out inp"
            )
        # Bias-only: sum over the shared axis.
        return einsum(g, "vec batch shared row -> vec batch row")
