"""KFAC computer using FX graph tracing instead of hooks.

This module provides ``MakeFxKFACComputer``, which computes Kronecker factors
by tracing the model's forward+backward pass with ``torch.fx`` via the IO
collector, collecting layer inputs/outputs rather than using forward/backward
hooks. The entire per-batch computation (IO collection, backward pass, and
covariance einsums) is traced with ``make_fx``, allowing ``torch.compile``
to optimize the full per-batch kernel.
"""

from collections import UserDict, defaultdict
from collections.abc import Callable

from einops import einsum
from torch import Tensor, autograd, cat, manual_seed, no_grad
from torch.nn import CrossEntropyLoss, Module

from curvlinops._checks import _register_userdict_as_pytree
from curvlinops.computers._base import ParamGroup, ParamGroupKey, _BaseKFACComputer
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _make_fx, make_functional_call


def _build_param_groups_from_io(
    io_param_names: dict[str, dict[str, str]],
    separate_weight_and_bias: bool,
) -> tuple[list[ParamGroup], dict[ParamGroupKey, list[str]]]:
    """Build parameter groups and IO-layer mapping from IO collector detections.

    Groups IO layers by weight name to form virtual modules. Each virtual
    module produces one group (joint) or separate W/b groups (separate).
    Also maps each group key to its contributing IO layer names.

    Args:
        io_param_names: Per-IO-layer parameter names from the IO collector.
        separate_weight_and_bias: Whether to treat weight and bias separately.

    Returns:
        Tuple of (parameter groups, IO-layer mapping).

    Raises:
        ValueError: If joint treatment and a weight has conflicting biases.
    """
    # Build virtual modules: one per unique weight (or standalone bias).
    # Each has a ParamGroup and a list of contributing IO layer names.
    modules: dict[str, ParamGroup] = {}
    module_io: dict[str, list[str]] = defaultdict(list)

    for io_name, pnames in io_param_names.items():
        w, b = pnames.get("W"), pnames.get("b")
        if w is not None:
            modules.setdefault(w, {"W": w})
            module_io[w].append(io_name)
            if b is not None:
                existing = modules[w].get("b")
                if not separate_weight_and_bias and existing and existing != b:
                    raise ValueError(
                        f"Weight '{w}' is used with conflicting biases "
                        f"'{existing}' and '{b}' under joint treatment. "
                        f"Use separate_weight_and_bias=True."
                    )
                modules[w]["b"] = b
        elif b is not None:
            modules[b] = {"b": b}
            module_io[b].append(io_name)

    # Convert virtual modules to parameter groups and IO-layer mapping
    groups: list[ParamGroup] = []
    io_groups: dict[ParamGroupKey, list[str]] = {}

    for mod_key, mod in modules.items():
        io = module_io[mod_key]
        param_dicts = (
            [{r: n} for r, n in mod.items()] if separate_weight_and_bias else [mod]
        )
        for pd in param_dicts:
            groups.append(pd)
            key = tuple(pd.values())
            # Bias-only groups: only IO layers that actually use the bias
            io_groups[key] = (
                [n for n in io if "b" in io_param_names[n]] if "W" not in pd else io
            )

    return groups, io_groups


def _bias_pad(has_joint_wb: bool, io_layer_params: dict[str, str]) -> int | None:
    """Determine bias padding for a specific IO layer usage.

    Args:
        has_joint_wb: Whether the parameter group has joint weight+bias.
        io_layer_params: Parameter roles for this IO layer.

    Returns:
        ``1`` if bias active, ``0`` if joint but bias inactive, ``None`` otherwise.
    """
    if not has_joint_wb:
        return None
    return 1 if "b" in io_layer_params else 0


def make_compute_kfac_io_batch(
    model_func: Callable,
    loss_func: Callable,
    params: dict[str, Tensor],
    X: Tensor,
    fisher_type: FisherType = FisherType.MC,
    mc_samples: int = 1,
    separate_weight_and_bias: bool = True,
    output_check_fn: Callable[[Tensor], None] | None = None,
) -> tuple[
    Callable[
        [dict[str, Tensor], Tensor, Tensor],
        tuple[dict[str, Tensor], dict[str, Tensor]],
    ],
    list[ParamGroup],
    dict[ParamGroupKey, list[str]],
    dict[str, dict[str, str]],
    dict[str, dict],
]:
    """Set up per-batch IO collection and backward pass for KFAC.

    Returns an **untraced** closure that, given ``(params, X, y)``, runs the
    forward pass with IO collection and backpropagates the Fisher-type-specific
    gradient outputs to produce per-layer inputs and batched output gradients.

    The closure can be:

    * composed with covariance computation and traced for KFAC
      (see :func:`make_compute_kfac_batch`),
    * called eagerly for EKFAC eigenvalue correction,
    * traced directly for compile tests.

    Args:
        model_func: Functional model ``(params, X) -> prediction``, or an
            ``nn.Module`` (converted internally via ``make_functional_call``).
        loss_func: Loss function (``MSELoss``, ``CrossEntropyLoss``, or
            ``BCEWithLogitsLoss``).
        params: Named parameter dict (example for IO tracing).
        X: Example input tensor (shapes determine the traced graph).
        fisher_type: Type of Fisher information. Defaults to
            ``FisherType.MC``.
        mc_samples: Number of Monte-Carlo samples (only used when
            ``fisher_type=FisherType.MC``). Defaults to ``1``.
        separate_weight_and_bias: Whether to treat weights and biases
            separately. Defaults to ``True``.
        output_check_fn: Optional callback ``(output) -> None`` called with
            the model output during setup. Raise inside this callback to
            reject unsupported output shapes (e.g., EKFAC's 2d restriction).

    Returns:
        Tuple of ``(io_batch_fn, mapping, io_groups, io_param_names,
        layer_hparams)`` where ``io_batch_fn(params, X, y)`` returns
        ``(layer_inputs, layer_output_grads)`` dicts keyed by IO layer name.
    """
    if isinstance(model_func, Module):
        model_func = make_functional_call(model_func)

    grad_outputs_computer = _BaseKFACComputer._set_up_grad_outputs_computer(
        loss_func, fisher_type, mc_samples
    )

    if isinstance(loss_func, CrossEntropyLoss):

        def rearrange_fn(output: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
            return output.movedim(1, -1).flatten(0, -2), y.flatten()

    else:

        def rearrange_fn(output: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
            return output.flatten(0, -2), y.flatten(0, -2)

    loss_reduction = loss_func.reduction
    is_forward_only = fisher_type == FisherType.FORWARD_ONLY

    io_fn, io_param_names, layer_hparams = with_kfac_io(
        model_func, X, params, fisher_type
    )

    if output_check_fn is not None:
        output_check_fn(model_func(params, X))

    mapping, io_groups = _build_param_groups_from_io(
        io_param_names, separate_weight_and_bias
    )

    def io_batch(
        params: dict[str, Tensor], X: Tensor, y: Tensor
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        output, layer_inputs, layer_outputs = io_fn(params, X)

        if is_forward_only:
            return layer_inputs, {}

        output_local, y_local = rearrange_fn(output, y)
        grad_outputs = grad_outputs_computer(output_local.detach(), y_local, None)
        num_loss_terms = output_local.shape[0]
        scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[loss_reduction]
        grad_outputs.mul_(scale)

        io_layer_names = list(layer_outputs)
        output_tensors = list(layer_outputs.values())
        layer_output_grads_list = autograd.grad(
            output_local,
            output_tensors,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
        )
        layer_output_grads = dict(zip(io_layer_names, layer_output_grads_list))
        return layer_inputs, layer_output_grads

    return io_batch, mapping, io_groups, io_param_names, layer_hparams


def make_compute_kfac_batch(
    model_func: Callable,
    loss_func: Callable,
    params: dict[str, Tensor],
    X: Tensor,
    y: Tensor,
    fisher_type: FisherType = FisherType.MC,
    mc_samples: int = 1,
    kfac_approx: str = KFACType.EXPAND,
    separate_weight_and_bias: bool = True,
) -> tuple[
    Callable[[dict[str, Tensor], Tensor, Tensor], tuple[list[Tensor], list[Tensor]]],
    list[ParamGroup],
    list[ParamGroupKey],
    list[ParamGroupKey],
]:
    """Set up and trace the per-batch KFAC Kronecker factor computation.

    Builds on :func:`make_compute_kfac_io_batch` by adding the covariance
    computation (weight-sharing format conversion + einsum) and tracing the
    entire pipeline with ``make_fx`` into a single FX graph with zero graph
    breaks.

    Args:
        model_func: Functional model ``(params, X) -> prediction``, or an
            ``nn.Module`` (converted internally via ``make_functional_call``).
        loss_func: Loss function (``MSELoss``, ``CrossEntropyLoss``, or
            ``BCEWithLogitsLoss``).
        params: Named parameter dict.
        X: Example input tensor.
        y: Example target tensor.
        fisher_type: Type of Fisher information. Defaults to
            ``FisherType.MC``.
        mc_samples: Number of Monte-Carlo samples (only used when
            ``fisher_type=FisherType.MC``). Defaults to ``1``.
        kfac_approx: KFAC approximation type (``KFACType.EXPAND`` or
            ``KFACType.REDUCE``). Defaults to ``KFACType.EXPAND``.
        separate_weight_and_bias: Whether to treat weights and biases
            separately. Defaults to ``True``.

    Returns:
        Tuple of ``(traced_fn, mapping, weight_group_keys, all_group_keys)``
        where ``traced_fn`` is a compiled function
        ``(params, X, y) -> (input_covs, gradient_covs)`` (each a list of
        tensors), ``mapping`` is the list of parameter groups, and the keys
        index the covariance lists.
    """
    io_batch, mapping, io_groups, io_param_names, layer_hparams = (
        make_compute_kfac_io_batch(
            model_func,
            loss_func,
            params,
            X,
            fisher_type,
            mc_samples,
            separate_weight_and_bias,
        )
    )
    weight_group_keys = [tuple(g.values()) for g in mapping if "W" in g]
    all_group_keys = [tuple(g.values()) for g in mapping]

    # Infer num_per_example_loss_terms from (y, loss_func)
    batch_size = y.shape[0]
    if isinstance(loss_func, CrossEntropyLoss):
        num_per_example_loss_terms = y.numel() // batch_size
    else:
        num_per_example_loss_terms = y.shape[:-1].numel() // batch_size

    loss_reduction = loss_func.reduction

    # Pre-compute group structure for stable ordering in the trace
    weight_group_info = []
    for group in mapping:
        if "W" not in group:
            continue
        group_key = tuple(group.values())
        io_names = io_groups.get(group_key, [])
        has_joint_wb = "b" in group
        bias_pads = [_bias_pad(has_joint_wb, io_param_names[n]) for n in io_names]
        hparams = [layer_hparams[n] for n in io_names]
        weight_group_info.append((io_names, bias_pads, hparams))

    grad_group_info = []
    for group in mapping:
        group_key = tuple(group.values())
        io_names = io_groups.get(group_key, [])
        hparams = [layer_hparams[n] for n in io_names]
        grad_group_info.append((io_names, hparams))

    def compute_batch(
        params: dict[str, Tensor], X: Tensor, y: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        layer_inputs, layer_output_grads = io_batch(params, X, y)

        input_covs = []
        for io_names, bias_pads, hparams in weight_group_info:
            xs = [
                input_to_weight_sharing_format(
                    layer_inputs[n].data.detach(),
                    kfac_approx,
                    layer_hyperparams=hp,
                    bias_pad=bp,
                )
                for n, bp, hp in zip(io_names, bias_pads, hparams)
            ]
            x = cat(xs, dim=1)
            scale = x.shape[1]
            xxT = einsum(x, x, "batch shared i, batch shared j -> i j")
            input_covs.append(xxT.div_(scale))

        if not layer_output_grads:
            return input_covs, []

        gradient_covs = []
        for io_names, hparams in grad_group_info:
            gs = [
                grad_to_weight_sharing_format(
                    layer_output_grads[n].data.detach(),
                    kfac_approx,
                    layer_hyperparams=hp,
                    num_leading_dims=2,
                )
                for n, hp in zip(io_names, hparams)
            ]
            g = cat(gs, dim=2)
            correction = compute_loss_correction(
                g.shape[1],
                num_per_example_loss_terms,
                loss_reduction,
            )
            ggT = einsum(g, g, "v batch shared i, v batch shared j -> i j").mul_(
                correction
            )
            gradient_covs.append(ggT)

        return input_covs, gradient_covs

    traced_fn = _make_fx(compute_batch)(params, X, y)

    return traced_fn, mapping, weight_group_keys, all_group_keys


class MakeFxKFACComputer(_BaseKFACComputer):
    """KFAC computer that uses FX graph tracing instead of hooks.

    Uses the IO collector (``with_kfac_io``) to detect affine layers and collect
    their inputs/outputs via ``torch.fx``, then computes Kronecker factors from
    these collected values. The entire per-batch computation (forward pass with
    IO collection, backward pass, and covariance einsums) is traced with
    ``make_fx``, allowing ``torch.compile`` to optimize the full per-batch
    kernel.

    Supports both ``nn.Module`` and plain callable ``model_func``.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and enable gradients on params for autograd.grad."""
        super().__init__(*args, **kwargs)
        for p in self._params.values():
            p.requires_grad_(True)

    def _trace_batch_functions(
        self,
    ) -> tuple[
        dict[int, Callable],
        list[ParamGroup],
        list[ParamGroupKey],
        list[ParamGroupKey],
    ]:
        """Trace per-batch KFAC computation for all batch sizes in the data.

        Iterates over the data once, tracing one FX graph per unique batch
        size.

        Returns:
            Tuple of ``(traced_fns, mapping, weight_group_keys, all_group_keys)``.
        """
        traced_fns: dict[int, Callable] = {}
        mapping: list[ParamGroup] | None = None
        weight_group_keys: list[ParamGroupKey] | None = None
        all_group_keys: list[ParamGroupKey] | None = None

        for X, y in self._loop_over_data(desc="FX tracing"):
            batch_size = self._batch_size_fn(X)
            if batch_size not in traced_fns:
                if isinstance(X, UserDict):
                    _register_userdict_as_pytree()
                io_batch, mapping, io_groups, io_param_names, layer_hparams = (
                    make_compute_kfac_io_batch(
                        self._model_func,
                        self._loss_func,
                        self._params,
                        X,
                        self._fisher_type,
                        self._mc_samples,
                        self._separate_weight_and_bias,
                    )
                )
                weight_group_keys = [tuple(g.values()) for g in mapping if "W" in g]
                all_group_keys = [tuple(g.values()) for g in mapping]

                # Pre-compute group structure for stable ordering in the trace
                weight_group_info = []
                for group in mapping:
                    if "W" not in group:
                        continue
                    group_key = tuple(group.values())
                    io_names = io_groups.get(group_key, [])
                    has_joint_wb = "b" in group
                    bias_pads = [
                        _bias_pad(has_joint_wb, io_param_names[n]) for n in io_names
                    ]
                    hparams = [layer_hparams[n] for n in io_names]
                    weight_group_info.append((io_names, bias_pads, hparams))

                grad_group_info = []
                for group in mapping:
                    group_key = tuple(group.values())
                    io_names = io_groups.get(group_key, [])
                    hparams = [layer_hparams[n] for n in io_names]
                    grad_group_info.append((io_names, hparams))

                kfac_approx = self._kfac_approx
                loss_reduction = self._loss_func.reduction
                num_per_example_loss_terms = self._num_per_example_loss_terms

                def compute_batch(
                    params: dict[str, Tensor], X: Tensor, y: Tensor
                ) -> tuple[list[Tensor], list[Tensor]]:
                    layer_inputs, layer_output_grads = io_batch(params, X, y)

                    input_covs = []
                    for io_names, bias_pads, hparams in weight_group_info:
                        xs = [
                            input_to_weight_sharing_format(
                                layer_inputs[n].data.detach(),
                                kfac_approx,
                                layer_hyperparams=hp,
                                bias_pad=bp,
                            )
                            for n, bp, hp in zip(io_names, bias_pads, hparams)
                        ]
                        x = cat(xs, dim=1)
                        scale = x.shape[1]
                        xxT = einsum(x, x, "batch shared i, batch shared j -> i j")
                        input_covs.append(xxT.div_(scale))

                    if not layer_output_grads:
                        return input_covs, []

                    gradient_covs = []
                    for io_names, hparams in grad_group_info:
                        gs = [
                            grad_to_weight_sharing_format(
                                layer_output_grads[n].data.detach(),
                                kfac_approx,
                                layer_hyperparams=hp,
                                num_leading_dims=2,
                            )
                            for n, hp in zip(io_names, hparams)
                        ]
                        g = cat(gs, dim=2)
                        correction = compute_loss_correction(
                            g.shape[1],
                            num_per_example_loss_terms,
                            loss_reduction,
                        )
                        ggT = einsum(
                            g,
                            g,
                            "v batch shared i, v batch shared j -> i j",
                        ).mul_(correction)
                        gradient_covs.append(ggT)

                    return input_covs, gradient_covs

                traced_fns[batch_size] = _make_fx(compute_batch)(self._params, X, y)

        return traced_fns, mapping, weight_group_keys, all_group_keys

    def compute(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Compute KFAC's Kronecker factors.

        Traces IO collection and batch computation in a single data pass,
        then accumulates factors in a second pass.

        Returns:
            Tuple of ``(input_covariances, gradient_covariances, mapping)``.
        """
        return self._compute_kronecker_factors(self._trace_batch_functions())

    def _compute_kronecker_factors(
        self,
        traced_batch: tuple[
            dict[int, Callable],
            list[ParamGroup],
            list[ParamGroupKey],
            list[ParamGroupKey],
        ],
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Accumulate KFAC's Kronecker factors using pre-traced batch functions.

        Runs the pre-traced per-batch functions and accumulates input and
        gradient covariances across all batches.

        Args:
            traced_batch: Pre-traced batch functions from
                :func:`trace_kfac_batch`.

        Returns:
            Tuple of (input_covariances, gradient_covariances, mapping).
        """
        traced_fns, mapping, weight_group_keys, all_group_keys = traced_batch

        grad_normalization = {
            "sum": 1.0,
            "mean": 1.0 / self._N_data,
        }[self._loss_func.reduction]

        input_covariances: dict[ParamGroupKey, Tensor] = {}
        gradient_covariances: dict[ParamGroupKey, Tensor] = {}

        manual_seed(self._seed)

        # no_grad: the traced graph already contains explicit backward ops from
        # make_fx; disabling the outer autograd avoids retaining intermediates.
        with no_grad():
            for X, y in self._loop_over_data(desc="KFAC matrices"):
                batch_size = self._batch_size_fn(X)
                input_covs, gradient_covs = traced_fns[batch_size](self._params, X, y)

                for key, xxT in zip(weight_group_keys, input_covs):
                    self._set_or_add_(input_covariances, key, xxT.div_(self._N_data))

                for key, ggT in zip(all_group_keys, gradient_covs):
                    self._set_or_add_(
                        gradient_covariances, key, ggT.mul_(grad_normalization)
                    )

        if self._fisher_type == FisherType.FORWARD_ONLY:
            self._set_gradient_covariances_to_identity(gradient_covariances, mapping)

        return input_covariances, gradient_covariances, mapping
