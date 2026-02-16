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
from enum import Enum, EnumMeta
from functools import partial
from typing import Any

from einops import einsum, rearrange, reduce
from torch import Generator, Tensor, autograd, cat, eye
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
from curvlinops.kfac_utils import (
    _check_binary_if_BCEWithLogitsLoss,
    extract_averaged_patches,
    extract_patches,
    make_grad_output_fn,
)
from curvlinops.utils import _seed_generator


class MetaEnum(EnumMeta):
    """Metaclass for the Enum class for desired behavior of the `in` operator."""

    def __contains__(cls, item):
        """Return whether ``item`` is a valid Enum value.

        Args:
            item: Candidate value.

        Returns:
            ``True`` if ``item`` is a valid Enum value.
        """
        try:
            cls(item)
        except ValueError:
            return False
        return True


class FisherType(str, Enum, metaclass=MetaEnum):
    """Enum for the Fisher type.

    Attributes:
        TYPE2 (str): ``'type-2'`` - Type-2 Fisher, i.e. the exact Hessian of the
            loss w.r.t. the model outputs is used. This requires as many backward
            passes as the output dimension, i.e. the number of classes for
            classification.
        MC (str): ``'mc'`` - Monte-Carlo approximation of the expectation by sampling
            ``mc_samples`` labels from the model's predictive distribution.
        EMPIRICAL (str): ``'empirical'`` - Empirical gradients are used which
            corresponds to the uncentered gradient covariance, or the empirical Fisher.
        FORWARD_ONLY (str): ``'forward-only'`` - The gradient covariances will be
            identity matrices, see the FOOF method in
            `Benzing, 2022 <https://arxiv.org/abs/2201.12250>`_ or ISAAC in
            `Petersen et al., 2023 <https://arxiv.org/abs/2305.00604>`_.
    """

    TYPE2 = "type-2"
    MC = "mc"
    EMPIRICAL = "empirical"
    FORWARD_ONLY = "forward-only"


class KFACType(str, Enum, metaclass=MetaEnum):
    """Enum for the KFAC approximation type.

    KFAC-expand and KFAC-reduce are defined in
    `Eschenhagen et al., 2023 <https://arxiv.org/abs/2311.00636>`_.

    Attributes:
        EXPAND (str): ``'expand'`` - KFAC-expand approximation.
        REDUCE (str): ``'reduce'`` - KFAC-reduce approximation.
    """

    EXPAND = "expand"
    REDUCE = "reduce"


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

    (see :class:`curvlinops.FisherMCLinearOperator` for the Fisher's definition).
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
        self._mapping = self.compute_parameter_mapping(params, model_func)

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
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, dict[str, int]]]:
        """Compute the Kronecker factors.

        Returns:
            Tuple of ``(input_covariances, gradient_covariances, mapping)`` where the
            first two are dictionaries mapping module names to covariance matrices and
            ``mapping`` maps module names to dictionaries of parameter names and their
            positions.
        """
        return (*self._compute_kronecker_factors(), self._mapping)

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
        batched_grad_output_fn = vmap(
            grad_output_fn, in_dims=(0, 0, None), out_dims=1, randomness=randomness
        )

        def compute_grad_outputs(
            output: Tensor, y: Tensor, generator: Generator | None = None
        ) -> Tensor:
            """Compute the gradients that are backpropagated from the network's output.

            Args:
                output: Neural network prediction with batch axis.
                y: Target labels with batch axis.
                generator: Random generator (used for MC mode).

            Returns:
                Gradients to be backpropagated from the network's output as a tensor of
                shape ``[num_vectors, *output.shape]`` where ``num_vectors`` depends on
                the Fisher type.
            """
            # Binary label check is data-dependent and not supported in vmap,
            # so we check outside and disable it inside (via make_grad_output_fn).
            _check_binary_if_BCEWithLogitsLoss(y, loss_func)
            return batched_grad_output_fn(output, y, generator)

        return compute_grad_outputs

    def _compute_kronecker_factors(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Compute KFAC's Kronecker factors.

        Returns:
            Tuple containing (input_covariances, gradient_covariances) dictionaries.
        """
        # Create empty dictionaries to be populated by hooks
        input_covariances: dict[str, Tensor] = {}
        gradient_covariances: dict[str, Tensor] = {}

        # install forward and backward hooks
        hook_handles: list[RemovableHandle] = []

        for mod_name, param_pos in self._mapping.items():
            module = self._model_func.get_submodule(mod_name)

            # input covariance only required for weights
            if "weight" in param_pos:
                hook_handles.append(
                    module.register_forward_pre_hook(
                        partial(
                            self._hook_accumulate_input_covariance,
                            module_name=mod_name,
                            input_covariances=input_covariances,
                        )
                    )
                )

            # gradient covariance required for weights and biases
            hook_handles.append(
                module.register_forward_hook(
                    partial(
                        self._register_tensor_hook_on_output_to_accumulate_gradient_covariance,
                        module_name=mod_name,
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

        # Handle FORWARD_ONLY case by setting gradient covariances to identity
        if self._fisher_type == FisherType.FORWARD_ONLY:
            # We choose to set the gradient covariance to the identity explicitly
            # for the sake of simplicity, but this could be done more efficiently.
            for mod_name, param_pos in self._mapping.items():
                # We iterate over _mapping to get the module names corresponding
                # to the parameters. We only need the output dimension of the
                # module, but don't know whether the parameter is a weight or
                # bias; therefore, we just call `next(iter(param_pos.values()))`
                # to get the first parameter.
                param = self._params[next(iter(param_pos.values()))]
                gradient_covariances[mod_name] = eye(
                    param.shape[0], dtype=param.dtype, device=self.device
                )

        return input_covariances, gradient_covariances

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
                self._params,
                grad_outputs=grad_outputs[v],
                retain_graph=v < num_vectors - 1,
            )

    def _register_tensor_hook_on_output_to_accumulate_gradient_covariance(
        self,
        module: Module,
        inputs: tuple[Tensor],
        output: Tensor,
        module_name: str,
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
            module_name: The name of the layer in the neural network.
            gradient_covariances: Dictionary to store gradient covariances.
        """
        tensor_hook = partial(
            self._accumulate_gradient_covariance,
            module=module,
            module_name=module_name,
            gradient_covariances=gradient_covariances,
        )
        output.register_hook(tensor_hook)

    def _accumulate_gradient_covariance(
        self,
        grad_output: Tensor,
        module: Module,
        module_name: str,
        gradient_covariances: dict[str, Tensor],
    ):
        """Accumulate the gradient covariance for a layer's output.

        Updates the provided ``gradient_covariances`` dictionary.

        Args:
            grad_output: The gradient w.r.t. the output.
            module: The layer whose output's gradient covariance will be accumulated.
            module_name: The name of the layer in the neural network.
            gradient_covariances: Dictionary to store gradient covariances.
        """
        g = grad_output.data.detach()
        batch_size = g.shape[0]
        if isinstance(module, Conv2d):
            g = rearrange(g, "batch c o1 o2 -> batch o1 o2 c")

        if self._kfac_approx == KFACType.EXPAND:
            # KFAC-expand approximation
            g = rearrange(g, "batch ... d_out -> (batch ...) d_out")
        else:
            # KFAC-reduce approximation
            g = reduce(g, "batch ... d_out -> batch d_out", "sum")

        # Compute correction for the loss scaling depending on the loss reduction used.
        # Note: mc_samples scaling is already handled inside make_grad_output_fn.
        num_loss_terms = batch_size * self._num_per_example_loss_terms
        correction = {
            "sum": 1.0,
            "mean": num_loss_terms**2
            / (self._N_data * self._num_per_example_loss_terms),
        }[self._loss_func.reduction]

        covariance = einsum(g, g, "b i,b j->i j").mul_(correction)
        self._set_or_add_(gradient_covariances, module_name, covariance)

    def _hook_accumulate_input_covariance(
        self,
        module: Module,
        inputs: tuple[Tensor],
        module_name: str,
        input_covariances: dict[str, Tensor],
    ):
        """Pre-forward hook that accumulates the input covariance of a layer.

        Updates the provided ``input_covariances`` dictionary.

        Args:
            module: Module on which the hook is called.
            inputs: Inputs to the module.
            module_name: Name of the module in the neural network.
            input_covariances: Dictionary to store input covariances.

        Raises:
            ValueError: If the module has multiple inputs.
        """
        if len(inputs) != 1:
            raise ValueError("Modules with multiple inputs are not supported.")
        x = inputs[0].data.detach()

        if isinstance(module, Conv2d):
            patch_extractor_fn = {
                KFACType.EXPAND: extract_patches,
                KFACType.REDUCE: extract_averaged_patches,
            }[self._kfac_approx]
            x = patch_extractor_fn(
                x,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )

        if self._kfac_approx == KFACType.EXPAND:
            # KFAC-expand approximation
            scale = x.shape[1:-1].numel()  # weight-sharing dimensions size
            x = rearrange(x, "batch ... d_in -> (batch ...) d_in")
        else:
            # KFAC-reduce approximation
            scale = 1.0  # since we use a mean reduction
            x = reduce(x, "batch ... d_in -> batch d_in", "mean")

        params = self._mapping[module_name]
        if not self._separate_weight_and_bias and {"weight", "bias"} == set(
            params.keys()
        ):
            x = cat([x, x.new_ones(x.shape[0], 1)], dim=1)

        covariance = einsum(x, x, "b i,b j -> i j").div_(self._N_data * scale)
        self._set_or_add_(input_covariances, module_name, covariance)

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
    def compute_parameter_mapping(
        cls, params: list[Tensor | Parameter], model_func: Module
    ) -> dict[str, dict[str, int]]:
        """Construct the mapping between layers, their parameters, and positions.

        Args:
            params: List of parameters.
            model_func: The model function.

        Returns:
            A dictionary of dictionaries. The outer dictionary's keys are the names of
            the layers that contain parameters. The interior dictionary's keys are the
            parameter names, and the values their respective positions.

        Raises:
            NotImplementedError: If parameters are found outside supported layers.
        """
        param_ids = [p.data_ptr() for p in params]
        positions = {}
        processed = set()

        for mod_name, mod in model_func.named_modules():
            if isinstance(mod, cls._SUPPORTED_MODULES) and any(
                p.data_ptr() in param_ids for p in mod.parameters()
            ):
                param_positions = {}
                for p_name, p in mod.named_parameters():
                    p_id = p.data_ptr()
                    if p_id in param_ids:
                        pos = param_ids.index(p_id)
                        param_positions[p_name] = pos
                        processed.add(p_id)
                positions[mod_name] = param_positions

        # check that all parameters are in known modules
        if len(processed) != len(param_ids):
            raise NotImplementedError("Found parameters in un-supported layers.")

        return positions
