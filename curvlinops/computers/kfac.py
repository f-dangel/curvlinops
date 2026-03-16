"""Computer for the Fisher/GGN's Kronecker-factored approximation (KFAC).

Kronecker-Factored Approximate Curvature (KFAC) was originally introduced for MLPs in

- Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored
  approximate curvature. International Conference on Machine Learning (ICML),

extended to CNNs in

- Grosse, R., & Martens, J. (2016). A kronecker-factored approximate Fisher matrix for
  convolution layers. International Conference on Machine Learning (ICML),

and generalized to all linear layers with weight sharing in

- Eschenhagen, R., Immer, A., Turner, R. E., Schneider, F., Hennig, P. (2023).
  Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures (NeurIPS).
"""

from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any

from einops import einsum, rearrange
from torch import Generator, Tensor, autograd, eye
from torch.func import vmap
from torch.nn import (
    BCEWithLogitsLoss,
    Conv2d,
    CrossEntropyLoss,
    Linear,
    Module,
    MSELoss,
    Parameter,
)
from torch.utils.hooks import RemovableHandle

from curvlinops._empirical_risk import _EmpiricalRiskMixin
from curvlinops.computers.io_collector.conv import CONV_STR
from curvlinops.computers.io_collector.linear import LINEAR_STR
from curvlinops.computers.kfac_math import (
    compute_loss_correction,
    grad_to_weight_sharing_format,
    input_to_weight_sharing_format,
)
from curvlinops.ggn_utils import make_grad_output_fn
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _seed_generator


@dataclass(frozen=True)
class ParameterUsage:
    """Describes how a group of parameters is used in an affine layer.

    Bundles the operation type, parameter name mapping, and layer hyperparameters
    into a single object. Both the hook-based and make_fx-based KFAC backends
    produce ``list[ParameterUsage]``.

    Attributes:
        op: The operation string, matching ``AffineLayerInfo.operation``.
            Either ``"Linear(y=W@x+b)"`` or ``"Conv(y=W*x+b)"``.
        params: Maps local parameter roles to full qualified parameter names.
            Keys are ``"W"`` (weight) and optionally ``"b"`` (bias).
            Values are full parameter names, e.g. ``{"W": "0.weight", "b": "0.bias"}``.
        hyperparams: Layer hyperparameters. Empty dict for linear layers.
            For convolutions, contains ``kernel_size``, ``stride``, ``padding``,
            ``dilation``, ``groups``.
    """

    op: str
    params: dict[str, str]
    hyperparams: dict[str, Any]


def _module_name_from_param(param_name: str) -> str:
    """Derive the module name from a fully qualified parameter name.

    Args:
        param_name: Full parameter name, e.g. ``"0.weight"`` or ``"weight"``.

    Returns:
        Module name, e.g. ``"0"`` or ``""`` (root module).
    """
    return param_name.rsplit(".", 1)[0] if "." in param_name else ""


class KFACComputer(_EmpiricalRiskMixin):
    r"""Computes KFAC's Kronecker factors for the Fisher/GGN.

    This class handles the data iteration and computation logic for KFAC. It computes
    the input and gradient covariances (Kronecker factors) and the parameter mapping.

    KFAC approximates the per-layer Fisher/GGN with a Kronecker product:
    Consider a weight matrix :math:`\\mathbf{W}` and a bias vector :math:`\\mathbf{b}`
    in a single layer. The layer's Fisher :math:`\\mathbf{F}(\\mathbf{\\theta})` for

    .. math::
        \\mathbf{\\theta}
        =
        \\begin{pmatrix}
        \\mathrm{vec}(\\mathbf{W}) \\\\ \\mathbf{b}
        \\end{pmatrix}

    where :math:`\\mathrm{vec}` denotes column-stacking is approximated as

    .. math::
        \\mathbf{F}(\\mathbf{\\theta})
        \\approx
        \\mathbf{A}_{(\\text{KFAC})} \\otimes \\mathbf{B}_{(\\text{KFAC})}

    (see :class:`curvlinops.GGNLinearOperator` with ``mc_samples > 0``).
    Loosely speaking, the first Kronecker factor is the un-centered covariance of the
    inputs to a layer. The second Kronecker factor is the un-centered covariance of
    'would-be' gradients w.r.t. the layer's output. Those 'would-be' gradients result
    from sampling labels from the model's distribution and computing their gradients.

    Attributes:
        _SUPPORTED_LOSSES: Tuple of supported loss functions.
        _SUPPORTED_MODULES: Tuple of supported layers.
        _SUPPORTED_FISHER_TYPE: Enum class of supported Fisher types.
        _SUPPORTED_KFAC_APPROX: Enum class of supported KFAC approximation types.
    """

    _SUPPORTED_LOSSES = (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)
    _SUPPORTED_MODULES = (Linear, Conv2d)
    _SUPPORTED_FISHER_TYPE: FisherType = FisherType
    _SUPPORTED_KFAC_APPROX: KFACType = KFACType
    NEEDS_NUM_PER_EXAMPLE_LOSS_TERMS: bool = True

    def __init__(
        self,
        model_func: Module,
        loss_func: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
        params: list[Parameter],
        data: Iterable[tuple[Tensor | MutableMapping, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        seed: int = 2_147_483_647,
        fisher_type: str = FisherType.MC,
        mc_samples: int = 1,
        kfac_approx: str = KFACType.EXPAND,
        num_per_example_loss_terms: int | None = None,
        separate_weight_and_bias: bool = True,
        num_data: int | None = None,
        batch_size_fn: Callable[[MutableMapping | Tensor], int] | None = None,
    ):
        """Set up the KFAC computer.

        Warning:
            This is an early proto-type with limitations:
                - Only Linear and Conv2d modules are supported.

        Args:
            model_func: The neural network. Must consist of modules.
            loss_func: The loss function.
            params: The parameters defining the Fisher/GGN that will be approximated
                through KFAC.
            data: A data loader containing the data of the Fisher/GGN.
            progressbar: Whether to show a progress bar when computing the Kronecker
                factors. Defaults to ``False``.
            check_deterministic: Whether to check that the data and model are
                deterministic. Defaults to ``True``.
            seed: The seed for the random number generator used to draw labels
                from the model's predictive distribution. Defaults to ``2147483647``.
            fisher_type: The type of Fisher/GGN to approximate.
                If ``FisherType.TYPE2``, the exact Hessian of the loss w.r.t. the model
                outputs is used. This requires as many backward passes as the output
                dimension, i.e. the number of classes for classification. This is
                sometimes also called type-2 Fisher. If ``FisherType.MC``, the
                expectation is approximated by sampling ``mc_samples`` labels from the
                model's predictive distribution. If ``FisherType.EMPIRICAL``, the
                empirical gradients are used which corresponds to the uncentered
                gradient covariance, or the empirical Fisher.
                If ``FisherType.FORWARD_ONLY``, the gradient covariances will be
                identity matrices, see the FOOF method in
                `Benzing, 2022 <https://arxiv.org/abs/2201.12250>`_ or ISAAC in
                `Petersen et al., 2023 <https://arxiv.org/abs/2305.00604>`_.
                Defaults to ``FisherType.MC``.
            mc_samples: The number of Monte-Carlo samples to use per data point.
                Has to be set to ``1`` when ``fisher_type != FisherType.MC``.
                Defaults to ``1``.
            kfac_approx: A string specifying the KFAC approximation that should
                be used for linear weight-sharing layers, e.g. ``Conv2d`` modules
                or ``Linear`` modules that process matrix- or higher-dimensional
                features.
                Possible values are ``KFACType.EXPAND`` and ``KFACType.REDUCE``.
                See `Eschenhagen et al., 2023 <https://arxiv.org/abs/2311.00636>`_
                for an explanation of the two approximations.
                Defaults to ``KFACType.EXPAND``.
            num_per_example_loss_terms: Number of per-example loss terms, e.g., the
                number of tokens in a sequence. The model outputs will have
                ``num_data * num_per_example_loss_terms * C`` entries, where ``C`` is
                the dimension of the random variable we define the likelihood over --
                for the ``CrossEntropyLoss`` it will be the number of classes, for the
                ``MSELoss`` and ``BCEWithLogitsLoss`` it will be the size of the last
                dimension of the the model outputs/targets (our convention here).
                If ``None``, ``num_per_example_loss_terms`` is inferred from the data at
                the cost of one traversal through the data loader. It is expected to be
                the same for all examples. Defaults to ``None``.
            separate_weight_and_bias: Whether to treat weights and biases separately.
                Defaults to ``True``.
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified. The intended behavior is to consume the first
                entry of the iterates from ``data`` and return their batch size.

        Raises:
            ValueError: If the loss function is not supported.
            ValueError: If ``fisher_type != FisherType.MC`` and ``mc_samples != 1``.
            ValueError: If ``kfac_approx`` is not supported.
        """
        if not isinstance(loss_func, self._SUPPORTED_LOSSES):
            raise ValueError(
                f"Invalid loss: {loss_func}. Supported: {self._SUPPORTED_LOSSES}."
            )
        if fisher_type not in self._SUPPORTED_FISHER_TYPE:
            raise ValueError(
                f"Invalid fisher_type: {fisher_type}. "
                f"Supported: {self._SUPPORTED_FISHER_TYPE}."
            )
        if fisher_type != FisherType.MC and mc_samples != 1:
            raise ValueError(
                f"Invalid mc_samples: {mc_samples}. "
                "Only mc_samples=1 is supported for `fisher_type != FisherType.MC`."
            )
        if kfac_approx not in self._SUPPORTED_KFAC_APPROX:
            raise ValueError(
                f"Invalid kfac_approx: {kfac_approx}. "
                f"Supported: {self._SUPPORTED_KFAC_APPROX}."
            )

        self._seed = seed
        self._generator: None | Generator = None
        self._separate_weight_and_bias = separate_weight_and_bias
        self._fisher_type = fisher_type
        self._mc_samples = mc_samples
        self._kfac_approx = kfac_approx
        self._mapping = self.compute_parameter_groups(
            params, model_func, separate_weight_and_bias
        )

        # Function (prediction_batch, label_batch) -> grad_outputs for backpropagation
        self._grad_outputs_computer = self._set_up_grad_outputs_computer(
            loss_func, fisher_type, mc_samples
        )

        super().__init__(
            model_func,
            loss_func,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            num_data=num_data,
            num_per_example_loss_terms=num_per_example_loss_terms,
            batch_size_fn=batch_size_fn,
        )

    def compute(
        self,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], list[ParameterUsage]]:
        """Compute the Kronecker factors.

        Returns:
            Tuple of ``(input_covariances, gradient_covariances, mapping)`` where the
            first two are dictionaries mapping layer names to covariance matrices and
            ``mapping`` is a list of ``ParameterUsage`` objects.
        """
        return (*self._compute_kronecker_factors(), self._mapping)

    def _get_module(self, usage: ParameterUsage) -> Module:
        """Get the module corresponding to a parameter usage.

        Derives the module name from the first parameter's full qualified name.

        Args:
            usage: Parameter usage info for the layer.

        Returns:
            The module object.
        """
        mod_name = _module_name_from_param(next(iter(usage.params.values())))
        return self._model_func.get_submodule(mod_name)

    @cached_property
    def _usage_by_module(self) -> dict[str, ParameterUsage]:
        """Lookup from module name to ``ParameterUsage``.

        Derives the module name from the first parameter's full qualified name.

        Returns:
            Dictionary mapping module names to ``ParameterUsage`` objects.
        """
        return {
            _module_name_from_param(next(iter(u.params.values()))): u
            for u in self._mapping
        }

    @cached_property
    def _mapping_by_key(self) -> dict[tuple[str, ...], ParameterUsage]:
        """Lookup from parameter group key to ``ParameterUsage``.

        Returns:
            Dictionary mapping group keys to ``ParameterUsage`` objects.
        """
        return {tuple(u.params.values()): u for u in self._mapping}

    @staticmethod
    def _set_up_grad_outputs_computer(
        loss_func: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
        fisher_type: FisherType,
        mc_samples: int,
    ) -> Callable[[Tensor, Tensor, Generator | None], Tensor]:
        """Set up the function that computes network output gradients for KFAC.

        Args:
            loss_func: The loss function.
            fisher_type: The Fisher type.
            mc_samples: Number of MC samples (used when ``fisher_type`` is ``MC``).

        Returns:
            A function ``(output_batch, y_batch, generator) -> grad_outputs``
            that computes the gradients to be backpropagated from the network's
            output, with shape ``[num_vectors, batch, *output_shape]``.
        """
        mode = {
            FisherType.MC: "mc",
            FisherType.TYPE2: "exact",
            FisherType.EMPIRICAL: "empirical",
            FisherType.FORWARD_ONLY: "forward-only",
        }[fisher_type]

        grad_output_fn = make_grad_output_fn(loss_func, mode, mc_samples)
        randomness = {
            "mc": "different",
            "exact": "same",
            "empirical": "same",
            "forward-only": "same",
        }[mode]
        return vmap(
            grad_output_fn, in_dims=(0, 0, None), out_dims=1, randomness=randomness
        )

    def _compute_kronecker_factors(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Compute KFAC's Kronecker factors.

        Warning:
            This hooks-based implementation assumes each module is called exactly
            once per forward pass. Weight tying (same module called multiple
            times) will silently produce incorrect results because the hook fires
            multiple times with wrong normalization. The backend cannot detect
            this. Use the ``make_fx`` backend for weight-tied architectures.

        Returns:
            Tuple containing (input_covariances, gradient_covariances) dictionaries.
        """
        # Create empty dictionaries to be populated by hooks
        input_covariances: dict[str, Tensor] = {}
        gradient_covariances: dict[str, Tensor] = {}

        # install forward and backward hooks
        hook_handles: list[RemovableHandle] = []

        for usage in self._mapping:
            module = self._get_module(usage)

            # input covariance only required for weights
            if "W" in usage.params:
                hook_handles.append(
                    module.register_forward_pre_hook(
                        partial(
                            self._hook_accumulate_input_covariance,
                            usage=usage,
                            input_covariances=input_covariances,
                        )
                    )
                )

            # gradient covariance required for weights and biases
            hook_handles.append(
                module.register_forward_hook(
                    partial(
                        self._register_tensor_hook_on_output_to_accumulate_gradient_covariance,
                        usage=usage,
                        gradient_covariances=gradient_covariances,
                    )
                )
            )

        # loop over data set, computing the Kronecker factors
        self._generator = _seed_generator(self._generator, self.device, self._seed)

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            output = self._model_func(X)
            output, y = self._rearrange_for_larger_than_2d_output(output, y)
            self._compute_loss_and_backward(output, y)

        # clean up
        for handle in hook_handles:
            handle.remove()

        # Handle FORWARD_ONLY case
        if self._fisher_type == FisherType.FORWARD_ONLY:
            self._set_gradient_covariances_to_identity(gradient_covariances)

        return input_covariances, gradient_covariances

    def _set_gradient_covariances_to_identity(
        self, gradient_covariances: dict[str, Tensor]
    ) -> None:
        """Set gradient covariances to identity for forward-only KFAC.

        For the FOOF/ISAAC method, the gradient covariance is the identity.
        We set it explicitly for simplicity, though this could be more efficient.

        Args:
            gradient_covariances: Dictionary to populate with identity matrices.
        """
        for usage in self._mapping:
            param = self._params[next(iter(usage.params.values()))]
            gradient_covariances[tuple(usage.params.values())] = eye(
                param.shape[0], dtype=param.dtype, device=self.device
            )

    def _rearrange_for_larger_than_2d_output(
        self, output: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Rearrange the output and target if output is >2d.

        This will determine what kind of Fisher/GGN is approximated.

        Args:
            output: The model's prediction
                :math:`\{f_\mathbf{\theta}(\mathbf{x}_n)\}_{n=1}^N`.
            y: The labels :math:`\{\mathbf{y}_n\}_{n=1}^N`.

        Returns:
            The rearranged output and target.
        """
        if isinstance(self._loss_func, CrossEntropyLoss):
            output = rearrange(output, "batch c ... -> (batch ...) c")
            y = rearrange(y, "batch ... -> (batch ...)")
        else:
            output = rearrange(output, "batch ... c -> (batch ...) c")
            y = rearrange(y, "batch ... c -> (batch ...) c")
        return output, y

    def _compute_loss_and_backward(self, output: Tensor, y: Tensor):
        r"""Compute the loss and the backward pass(es) required for KFAC.

        Args:
            output: The model's prediction
                :math:`\{f_\mathbf{\theta}(\mathbf{x}_n)\}_{n=1}^N`.
            y: The labels :math:`\{\mathbf{y}_n\}_{n=1}^N`.

        Raises:
            ValueError: If the output is not 2d and y is not 1d/2d.
        """
        if output.ndim != 2 or y.ndim not in {1, 2}:
            raise ValueError(
                "Only 2d output and 1d/2d target are supported. "
                f"Got {output.ndim=} and {y.ndim=}."
            )

        # Compute the gradients w.r.t. the network's output that will be
        # backpropagated to compute the KFAC approximation.
        # Detach output: we only need values for the backpropagated vectors.
        grad_outputs = self._grad_outputs_computer(output.detach(), y, self._generator)

        # Fix scaling caused by the batch dimension
        num_loss_terms = output.shape[0]
        reduction = self._loss_func.reduction
        scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[reduction]
        grad_outputs.mul_(scale)

        # Backpropagate all vectors (0 for forward-only, 1 for empirical,
        # mc_samples for MC, and C (number of output features per datum) for TYPE2).
        num_vectors = grad_outputs.shape[0]
        for v in range(num_vectors):
            autograd.grad(
                output,
                list(self._params.values()),
                grad_outputs=grad_outputs[v],
                retain_graph=v < num_vectors - 1,
            )

    def _register_tensor_hook_on_output_to_accumulate_gradient_covariance(
        self,
        module: Module,
        inputs: tuple[Tensor],
        output: Tensor,
        usage: ParameterUsage,
        gradient_covariances: dict[str, Tensor],
    ):
        """Register tensor hook on layer's output to accumulate the grad. covariance.

        Note:
            The easier way to compute the gradient covariance would be via a full
            backward hook on the module itself which performs the computation.
            However, this approach breaks down if the output of a layer feeds into an
            activation with `inplace=True` (see
            https://github.com/pytorch/pytorch/issues/61519). Hence we use the
            workaround
            https://github.com/pytorch/pytorch/issues/61519#issuecomment-883524237, and
            install a module hook which installs a tensor hook on the module's output
            tensor, which performs the accumulation of the gradient covariance.

        Args:
            module: Layer onto whose output a tensor hook to accumulate the gradient
                covariance will be installed.
            inputs: The layer's input tensors.
            output: The layer's output tensor.
            usage: Parameter usage info for this layer.
            gradient_covariances: Dictionary to store gradient covariances.
        """
        tensor_hook = partial(
            self._accumulate_gradient_covariance,
            usage=usage,
            gradient_covariances=gradient_covariances,
        )
        output.register_hook(tensor_hook)

    def _accumulate_gradient_covariance(
        self,
        grad_output: Tensor,
        usage: ParameterUsage,
        gradient_covariances: dict[str, Tensor],
    ):
        """Accumulate the gradient covariance for a layer's output.

        Updates the provided ``gradient_covariances`` dictionary.

        Args:
            grad_output: The gradient w.r.t. the output.
            usage: Parameter usage info for this layer.
            gradient_covariances: Dictionary to store gradient covariances.
        """
        g = grad_output.data.detach()
        batch_size = g.shape[0]

        g = grad_to_weight_sharing_format(
            g, self._kfac_approx, layer_hyperparams=usage.hyperparams
        )

        # Note: mc_samples scaling is already handled inside make_grad_output_fn.
        correction = compute_loss_correction(
            batch_size,
            self._num_per_example_loss_terms,
            self._loss_func.reduction,
            self._N_data,
        )

        covariance = einsum(g, g, "batch shared i, batch shared j -> i j").mul_(
            correction
        )
        self._set_or_add_(
            gradient_covariances, tuple(usage.params.values()), covariance
        )

    def _hook_accumulate_input_covariance(
        self,
        module: Module,
        inputs: tuple[Tensor, ...],
        usage: ParameterUsage,
        input_covariances: dict[tuple[str, ...], Tensor],
    ):
        """Pre-forward hook that accumulates the input covariance of a layer.

        Updates the provided ``input_covariances`` dictionary.

        Args:
            module: Module on which the hook is called.
            inputs: Inputs to the module.
            usage: Parameter usage info for this layer.
            input_covariances: Dictionary to store input covariances.

        Raises:
            ValueError: If the module has multiple inputs.
        """
        if len(inputs) != 1:
            raise ValueError("Modules with multiple inputs are not supported.")
        x = inputs[0].data.detach()

        has_joint_wb = "b" in usage.params and "W" in usage.params

        x = input_to_weight_sharing_format(
            x,
            self._kfac_approx,
            layer_hyperparams=usage.hyperparams,
            bias_pad=1 if has_joint_wb else None,
        )
        scale = x.shape[1]
        covariance = einsum(x, x, "batch shared i, batch shared j -> i j").div_(
            self._N_data * scale
        )
        self._set_or_add_(input_covariances, tuple(usage.params.values()), covariance)

    @staticmethod
    def _set_or_add_(
        dictionary: dict[Any, Tensor], key: Any, value: Tensor
    ) -> dict[str, Tensor]:
        """Set or add a value to a dictionary entry.

        Args:
            dictionary: The dictionary to update.
            key: The key to update.
            value: The value to add.

        Returns:
            The updated dictionary.

        Raises:
            ValueError: If the types of the value and the dictionary entry are
                incompatible.
        """
        if key not in dictionary:
            dictionary[key] = value
        elif isinstance(dictionary[key], Tensor) and isinstance(value, Tensor):
            dictionary[key].add_(value)
        else:
            raise ValueError(
                "Incompatible types for addition: dictionary value of type "
                f"{type(dictionary[key])} and value to be added of type {type(value)}."
            )
        return dictionary

    @classmethod
    def compute_parameter_groups(
        cls,
        params: list[Tensor | Parameter],
        model_func: Module,
        separate_weight_and_bias: bool = True,
    ) -> list[ParameterUsage]:
        """Construct parameter groups for the model's layers.

        Each supported module produces one group (joint treatment) or two
        groups (separate treatment). Joint treatment (``separate_weight_and_bias
        =False``) stores fewer Kronecker factors and is recommended for
        performance.

        Args:
            params: List of parameters.
            model_func: The model function.
            separate_weight_and_bias: Whether to treat weight and bias as
                separate parameter groups.

        Returns:
            List of ``ParameterUsage`` objects, one per parameter group.

        Raises:
            NotImplementedError: If parameters are found outside supported layers.
        """
        # Map PyTorch's parameter names to short role keys
        _role = {"weight": "W", "bias": "b"}
        # Map module types to operation strings and hparam extractors
        _module_info: dict[type, tuple[str, Callable[[Module], dict[str, Any]]]] = {
            Linear: (LINEAR_STR, lambda _: {}),
            Conv2d: (
                CONV_STR,
                lambda m: dict(
                    kernel_size=m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                ),
            ),
        }

        param_ids = {p.data_ptr() for p in params}
        ptr_to_name = {
            p.data_ptr(): name
            for name, p in model_func.named_parameters()
            if p.data_ptr() in param_ids
        }
        groups: list[ParameterUsage] = []
        processed = set()

        for _, mod in model_func.named_modules():
            if isinstance(mod, cls._SUPPORTED_MODULES) and any(
                p.data_ptr() in param_ids for p in mod.parameters()
            ):
                param_roles = {}
                for p_name, p in mod.named_parameters(recurse=False):
                    p_id = p.data_ptr()
                    if p_id in param_ids:
                        param_roles[_role[p_name]] = ptr_to_name[p_id]
                        processed.add(p_id)
                op, hparam_fn = next(
                    v for t, v in _module_info.items() if isinstance(mod, t)
                )
                param_dicts = (
                    [{r: n} for r, n in param_roles.items()]
                    if separate_weight_and_bias
                    else [param_roles]
                )
                for params_dict in param_dicts:
                    groups.append(
                        ParameterUsage(
                            op=op, params=params_dict, hyperparams=hparam_fn(mod)
                        )
                    )

        if len(processed) != len(param_ids):
            raise NotImplementedError("Found parameters in un-supported layers.")

        return groups
