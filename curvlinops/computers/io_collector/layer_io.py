r"""High-level orchestration over the IO collector for KFAC-style operators.

Provides:

- :class:`LayerIO`: setup-once owner of shape-independent metadata
  (parameter groups, IO-layer mappings) plus a per-shape cache of FX-traced
  ``io_fn``\s. Operators consume :class:`LayerIO` to obtain per-batch raw IO
  without re-deriving the plumbing.
- :class:`LayerIOSnapshot`: per-batch wrapper over raw layer inputs and output
  gradients, exposing on-demand per-group accessors at three granularities
  (raw, standardized weight-sharing format, expanded per-sample ``vec(W)``
  gradients).
- :func:`LayerIO.trace_context`: context manager wrapping
  :func:`_enable_requires_grad` for the trace boundary, so callers don't have
  to import the autograd-ownership helper directly.

The two-tier state in :class:`LayerIO` separates concerns by shape dependence:

* **Shape-independent (tier 1)** — built once at construction:
  ``mapping``, ``io_groups``, ``io_param_names``, ``layer_hparams``, the
  ``grad_outputs`` computer, and the per-group ``(group_inputs, group_grads)``
  gatherers.
* **Shape-specialized (tier 2)** — cached per unique input shape:
  the ``io_fn`` returned by :func:`with_kfac_io`. New shapes trigger a fresh
  trace via :meth:`LayerIO.ensure_io_fn`.

Operators have three integration patterns:

1. **Trace everything** (KFAC-style): wrap the entire per-batch reduction
   (``populate`` + per-group einsums) in :func:`_make_fx`, inside
   :meth:`LayerIO.trace_context`.
2. **Trace IO only** (KFOC-style): wrap just :meth:`LayerIO.populate` in
   :func:`_make_fx`, replay under :func:`torch.no_grad`, then process per-group
   eagerly.
3. **Fully eager**: skip :func:`_make_fx` entirely.
"""

from __future__ import annotations

from collections import UserDict
from collections.abc import Callable, MutableMapping
from contextlib import contextmanager
from math import sqrt
from typing import Any

from einops import einsum
from torch import Tensor, autograd
from torch.nn import CrossEntropyLoss

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _BaseKFACComputer
from curvlinops.computers.io_collector.collector import with_kfac_io
from curvlinops.computers.io_collector.groups import (
    _build_param_groups_from_io,
    make_group_gatherers,
)
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _enable_requires_grad

ShapeKey = tuple


class LayerIO:
    r"""Setup-once orchestrator for per-layer IO collection.

    Owns shape-independent metadata and a per-shape cache of FX-traced
    ``io_fn``\ s. Operators construct one :class:`LayerIO` per ``compute()``
    call and consume :meth:`populate` / :meth:`snapshot` per batch.

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
        self._fisher_type = fisher_type
        self._mc_samples = mc_samples
        self._kfac_approx = kfac_approx
        self._separate_weight_and_bias = separate_weight_and_bias
        self._intermediate_as_batch = intermediate_as_batch

        self._grad_outputs_computer = _BaseKFACComputer._set_up_grad_outputs_computer(
            loss_func, fisher_type, mc_samples
        )

        # Bootstrap: trace one ``io_fn`` to extract shape-independent metadata.
        if isinstance(X_example, UserDict):
            _register_userdict_as_pytree()
        first_io_fn, io_param_names, layer_hparams = with_kfac_io(
            model_func, X_example, params, fisher_type
        )
        self._io_param_names = io_param_names
        self._layer_hparams = layer_hparams
        self._mapping, self._io_groups = _build_param_groups_from_io(
            io_param_names, separate_weight_and_bias
        )
        self._group_inputs, self._group_grads = make_group_gatherers(
            self._io_groups, self._io_param_names, self._layer_hparams, kfac_approx
        )
        self._io_fns: dict[ShapeKey, Callable] = {
            self._shape_key(X_example): first_io_fn
        }

    @property
    def mapping(self) -> list[ParamGroup]:
        """Parameter groups (list of dicts mapping role → param name)."""
        return self._mapping

    @property
    def io_groups(self) -> dict[ParamGroupKey, list[str]]:
        """Per-group IO-layer name lists (for weight-tied layers)."""
        return self._io_groups

    @property
    def io_param_names(self) -> dict[str, dict[str, str]]:
        """Per-IO-layer parameter name mappings."""
        return self._io_param_names

    @property
    def layer_hparams(self) -> dict[str, dict[str, Any]]:
        """Per-IO-layer hyperparameter dicts."""
        return self._layer_hparams

    @property
    def fisher_type(self) -> FisherType:
        """Fisher type passed at construction."""
        return self._fisher_type

    @property
    def kfac_approx(self) -> str:
        """KFAC approximation type used by the per-group gatherers."""
        return self._kfac_approx

    @staticmethod
    def _shape_key(X: Tensor | MutableMapping) -> ShapeKey:
        """Hashable shape signature for cache keying.

        Args:
            X: Input tensor or dict of input tensors.

        Returns:
            A hashable tuple representing ``X``'s shape structure.
        """
        if isinstance(X, Tensor):
            return tuple(X.shape)
        return tuple((k, tuple(v.shape)) for k, v in sorted(X.items()))

    def ensure_io_fn(
        self, X: Tensor | MutableMapping, params: dict[str, Tensor]
    ) -> Callable:
        """Return the FX-traced ``io_fn`` for ``X``'s shape, building if needed.

        Re-traces ``with_kfac_io`` for unseen shapes and caches the result.
        Asserts that re-traced metadata matches the bootstrap to detect
        violations of the shape-independence assumption.

        Args:
            X: Input batch (only its shape is consulted for the cache key).
            params: Named parameters, passed through to :func:`with_kfac_io`.

        Returns:
            A traced callable ``(params, X) -> (output, layer_inputs, layer_outputs)``.

        Raises:
            RuntimeError: If re-traced metadata for a new shape disagrees with
                the bootstrap metadata.
        """
        key = self._shape_key(X)
        if key in self._io_fns:
            return self._io_fns[key]

        if isinstance(X, UserDict):
            _register_userdict_as_pytree()

        io_fn, io_param_names, layer_hparams = with_kfac_io(
            self._model_func, X, params, self._fisher_type
        )
        if io_param_names != self._io_param_names:
            raise RuntimeError(
                "IO-collector parameter-name metadata changed across shapes. "
                f"Expected {self._io_param_names}, got {io_param_names}."
            )
        if layer_hparams != self._layer_hparams:
            raise RuntimeError(
                "IO-collector layer hyperparameters changed across shapes. "
                f"Expected {self._layer_hparams}, got {layer_hparams}."
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

        Reuses the cached ``io_fn`` for ``X``'s shape (or builds one).
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

        if self._fisher_type == FisherType.FORWARD_ONLY:
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

        io_layer_names = list(layer_outputs)
        output_tensors = list(layer_outputs.values())
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
    def trace_context(self, params: dict[str, Tensor]):
        """Context manager around ``_make_fx`` calls that include :meth:`populate`.

        Wraps :func:`_enable_requires_grad` so the ``autograd.grad`` call
        inside :meth:`populate` has differentiable inputs at trace time.
        Restores prior ``requires_grad`` state on exit, satisfying the
        "owner-of-autograd" contract: callers don't have to import the
        helper directly.

        Args:
            params: Named parameters whose ``requires_grad`` should be
                temporarily enabled (typically the same dict passed to
                :meth:`populate`).

        Yields:
            None.
        """
        with _enable_requires_grad(list(params.values())):
            yield


class LayerIOSnapshot:
    """Per-batch raw IO with on-demand per-group accessors.

    Built via :meth:`LayerIO.snapshot`. Provides three granularities of
    access at increasing computational cost:

    1. :meth:`raw_inputs` / :meth:`raw_output_grads` — per IO-layer view.
    2. :meth:`standardized_io` — per-group ``(a, g)`` in weight-sharing format.
    3. :meth:`per_sample_grads` — per-group per-sample ``vec(W)`` gradients.
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
        self._layer_inputs = layer_inputs
        self._layer_output_grads = layer_output_grads

    @property
    def layer_inputs(self) -> dict[str, Tensor]:
        """Raw per-IO-layer input tensors."""
        return self._layer_inputs

    @property
    def layer_output_grads(self) -> dict[str, Tensor]:
        """Raw per-IO-layer output gradient tensors."""
        return self._layer_output_grads

    def raw_inputs(self, io_layer_name: str) -> Tensor:
        """Return the raw input tensor for a single IO layer.

        Args:
            io_layer_name: IO-layer name (e.g. ``"Linear0"``).

        Returns:
            The raw input tensor stored under ``io_layer_name``.
        """
        return self._layer_inputs[io_layer_name]

    def raw_output_grads(self, io_layer_name: str) -> Tensor:
        """Return the raw batched output-grad tensor for a single IO layer.

        Args:
            io_layer_name: IO-layer name (e.g. ``"Linear0"``).

        Returns:
            The raw batched output-gradient tensor stored under ``io_layer_name``.
        """
        return self._layer_output_grads[io_layer_name]

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
        a = (
            self._owner._group_inputs(group, self._layer_inputs)
            if "W" in group
            else None
        )
        g = (
            self._owner._group_grads(group, self._layer_output_grads)
            if self._owner._fisher_type != FisherType.FORWARD_ONLY
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
        """
        a, g = self.standardized_io(group)
        if "W" in group:
            return einsum(
                g, a, "vec batch shared out, batch shared inp -> vec batch out inp"
            )
        # Bias-only: sum over the shared axis.
        return einsum(g, "vec batch shared row -> vec batch row")
