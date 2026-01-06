"""Linear operator for the Fisher/GGN's Kronecker-factored approximation.

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

from __future__ import annotations

from collections.abc import MutableMapping
from enum import Enum, EnumMeta
from functools import partial
from math import sqrt
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

from einops import einsum, rearrange, reduce
from torch import Generator, Tensor, cat, eye, randn, stack
from torch.autograd import grad
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

from curvlinops._torch_base import (
    CurvatureLinearOperator,
    PyTorchLinearOperator,
    _ChainPyTorchLinearOperator,
)
from curvlinops.blockdiagonal import BlockDiagonalLinearOperator
from curvlinops.kfac_utils import (
    extract_averaged_patches,
    extract_patches,
    loss_hessian_matrix_sqrt,
)
from curvlinops.kronecker import KroneckerProductLinearOperator

FactorType = TypeVar(
    "FactorType", Optional[Tensor], Tuple[Optional[Tensor], Optional[Tensor]]
)


class MetaEnum(EnumMeta):
    """Metaclass for the Enum class for desired behavior of the `in` operator."""

    def __contains__(cls, item):
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


class KFACLinearOperator(CurvatureLinearOperator):
    r"""Linear operator to multiply with the Fisher/GGN's KFAC approximation.

    KFAC approximates the per-layer Fisher/GGN with a Kronecker product:
    Consider a weight matrix :math:`\mathbf{W}` and a bias vector :math:`\mathbf{b}`
    in a single layer. The layer's Fisher :math:`\mathbf{F}(\mathbf{\theta})` for

    .. math::
        \mathbf{\theta}
        =
        \begin{pmatrix}
        \mathrm{vec}(\mathbf{W}) \\ \mathbf{b}
        \end{pmatrix}

    where :math:`\mathrm{vec}` denotes column-stacking is approximated as

    .. math::
        \mathbf{F}(\mathbf{\theta})
        \approx
        \mathbf{A}_{(\text{KFAC})} \otimes \mathbf{B}_{(\text{KFAC})}

    (see :class:`curvlinops.FisherMCLinearOperator` for the Fisher's definition).
    Loosely speaking, the first Kronecker factor is the un-centered covariance of the
    inputs to a layer. The second Kronecker factor is the un-centered covariance of
    'would-be' gradients w.r.t. the layer's output. Those 'would-be' gradients result
    from sampling labels from the model's distribution and computing their gradients.

    Kronecker-Factored Approximate Curvature (KFAC) was originally introduced for MLPs in

    - Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored
    approximate curvature. International Conference on Machine Learning (ICML),

    extended to CNNs in

    - Grosse, R., & Martens, J. (2016). A kronecker-factored approximate Fisher matrix for
    convolution layers. International Conference on Machine Learning (ICML),

    and generalized to all linear layers with weight sharing in

    - Eschenhagen, R., Immer, A., Turner, R. E., Schneider, F., Hennig, P. (2023).
    Kronecker-Factored Approximate Curvature for Modern Neural Network Architectures (NeurIPS).

    Attributes:
        _SUPPORTED_LOSSES: Tuple of supported loss functions.
        _SUPPORTED_MODULES: Tuple of supported layers.
        _SUPPORTED_FISHER_TYPE: Enum of supported Fisher types.
        _SUPPORTED_KFAC_APPROX: Enum of supported KFAC approximation types.
        SELF_ADJOINT: Whether the operator is self-adjoint. ``True`` for KFAC.
    """

    _SUPPORTED_LOSSES = (MSELoss, CrossEntropyLoss, BCEWithLogitsLoss)
    _SUPPORTED_MODULES = (Linear, Conv2d)
    _SUPPORTED_FISHER_TYPE: FisherType = FisherType
    _SUPPORTED_KFAC_APPROX: KFACType = KFACType
    SELF_ADJOINT: bool = True

    def __init__(
        self,
        model_func: Module,
        loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        seed: int = 2147483647,
        fisher_type: str = FisherType.MC,
        mc_samples: int = 1,
        kfac_approx: str = KFACType.EXPAND,
        num_per_example_loss_terms: Optional[int] = None,
        separate_weight_and_bias: bool = True,
        num_data: Optional[int] = None,
        batch_size_fn: Optional[Callable[[Union[MutableMapping, Tensor]], int]] = None,
    ):
        """Kronecker-factored approximate curvature (KFAC) proxy of the Fisher/GGN.

        Warning:
            If the model's parameters change, e.g. during training, you need to
            create a fresh instance of this object. This is because, for performance
            reasons, the Kronecker factors are computed once and cached during the
            first matrix-vector product. They will thus become outdated if the model
            changes.

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
            check_deterministic: Whether to check that the linear operator is
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
            ValueError: If ``X`` is not a tensor and ``batch_size_fn`` is not specified.
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
        self._generator: Union[None, Generator] = None
        self._separate_weight_and_bias = separate_weight_and_bias
        self._fisher_type = fisher_type
        self._mc_samples = mc_samples
        self._kfac_approx = kfac_approx
        self._mapping = self.compute_parameter_mapping(params, model_func)

        super().__init__(
            model_func,
            loss_func,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=False,
            num_data=num_data,
            batch_size_fn=batch_size_fn,
        )

        self._set_num_per_example_loss_terms(num_per_example_loss_terms)

        # Create the block diagonal operator
        self._block_diagonal_operator = self._create_block_diagonal_operator()

        # Create canonical form transformation operators
        self._to_canonical = _ToCanonicalLinearOperator(
            self._params, self._mapping, self._separate_weight_and_bias
        )
        self._from_canonical = self._to_canonical.adjoint()

        # Build the operator that represents KFAC
        self._operator: _ChainPyTorchLinearOperator = (
            self._from_canonical @ self._block_diagonal_operator @ self._to_canonical
        )

        if check_deterministic:
            self._check_deterministic()

    def _set_num_per_example_loss_terms(
        self, num_per_example_loss_terms: Optional[int]
    ):
        """Set the number of per-example loss terms.

        Args:
            num_per_example_loss_terms: Number of per-example loss terms. If ``None``,
                it is inferred from the data at the cost of one traversal through the
                data loader.

        Raises:
            ValueError: If the number of loss terms is not divisible by the number of
                data points.
        """
        if num_per_example_loss_terms is None:
            # Determine the number of per-example loss terms
            num_loss_terms = sum(
                (
                    y.numel()
                    if isinstance(self._loss_func, CrossEntropyLoss)
                    else y.shape[:-1].numel()
                )
                for (_, y) in self._loop_over_data(desc="_num_per_example_loss_terms")
            )
            if num_loss_terms % self._N_data != 0:
                raise ValueError(
                    "The number of loss terms must be divisible by the number of data "
                    f"points; num_loss_terms={num_loss_terms}, N_data={self._N_data}."
                )
            self._num_per_example_loss_terms = num_loss_terms // self._N_data
        else:
            self._num_per_example_loss_terms = num_per_example_loss_terms

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Apply KFAC to a matrix (multiple vectors) in tensor list format.

        This method now uses the block-diagonal structure internally.

        Args:
            M: Matrix for multiplication in tensor list format. Each entry has the
                same shape as a parameter with an additional trailing dimension of size
                ``K`` for the columns, i.e. ``[(*p1.shape, K), (*p2.shape, K), ...]``.

        Returns:
            Matrix-multiplication result ``KFAC @ M`` in tensor list format. Has the
            same shapes as the input.
        """
        return self._operator._matmat(M)

    def _create_block_diagonal_operator(self) -> BlockDiagonalLinearOperator:
        """Create block-diagonal linear operator from Kronecker factors.

        Each block corresponds to a layer and is a KroneckerProductLinearOperator.

        Returns:
            Block-diagonal linear operator with Kronecker product blocks.
        """
        input_covariances, gradient_covariances = self._compute_kronecker_factors()

        factors = []

        for mod_name, param_pos in self._mapping.items():
            aaT = input_covariances.pop(mod_name, None)
            ggT = gradient_covariances.pop(mod_name)

            # Handle joint weight+bias case
            if not self._separate_weight_and_bias and {"weight", "bias"} == set(
                param_pos.keys()
            ):
                # Single Kronecker product block for weight+bias
                factors.append([ggT, aaT])
            else:
                # Separate blocks for weight and bias
                for p_name in param_pos:
                    factors.append([ggT, aaT] if p_name == "weight" else [ggT])

        # Create Kronecker product linear operators for each block
        blocks = [KroneckerProductLinearOperator(*fs) for fs in factors]

        return BlockDiagonalLinearOperator(blocks)

    def _setup_generator(self):
        """Initialize and seed the random number generator if needed.

        Creates a new generator on the correct device if one doesn't exist or if
        the existing generator is on the wrong device. Always seeds the generator
        with the stored seed value.
        """
        if self._generator is None or self._generator.device != self.device:
            self._generator = Generator(device=self.device)
        self._generator.manual_seed(self._seed)

    def _compute_kronecker_factors(
        self,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Compute and return KFAC's Kronecker factors.

        Returns:
            Tuple of (input_covariances, gradient_covariances) dictionaries.
        """
        input_covariances: Dict[str, Tensor] = {}
        gradient_covariances: Dict[str, Tensor] = {}

        # install forward and backward hooks
        hook_handles: List[RemovableHandle] = []

        for mod_name, param_pos in self._mapping.items():
            module = self._model_func.get_submodule(mod_name)

            # input covariance only required for weights
            if "weight" in param_pos:
                hook_handles.append(
                    module.register_forward_pre_hook(
                        partial(
                            self._hook_accumulate_input_covariance,
                            module_name=mod_name,
                            covariances_dict=input_covariances,
                        )
                    )
                )

            # gradient covariance required for weights and biases
            hook_handles.append(
                module.register_forward_hook(
                    partial(
                        self._register_tensor_hook_on_output_to_accumulate_gradient_covariance,
                        module_name=mod_name,
                        covariances_dict=gradient_covariances,
                    )
                )
            )

        # loop over data set, computing the Kronecker factors
        self._setup_generator()

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            output = self._model_func(X)
            output, y = self._rearrange_for_larger_than_2d_output(output, y)
            self._compute_loss_and_backward(output, y, gradient_covariances)

        # clean up
        for handle in hook_handles:
            handle.remove()

        return input_covariances, gradient_covariances

    def _rearrange_for_larger_than_2d_output(
        self, output: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor]:
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

    def _maybe_adjust_loss_scale(self, loss: Tensor, output: Tensor) -> Tensor:
        """Adjust the scale of the loss tensor if necessary.

        The ``BCEWithLogitsLoss`` and ``MSELoss`` also average over the output dimension
        in addition to the batch dimension. We adjust the scale of the loss to correct
        for this.

        Args:
            loss: The loss tensor to adjust.
            output: The model's output.

        Returns:
            The scaled loss tensor.
        """
        if (
            isinstance(self._loss_func, (BCEWithLogitsLoss, MSELoss))
            and self._loss_func.reduction == "mean"
        ):
            # ``BCEWithLogitsLoss`` and ``MSELoss`` also average over non-batch
            # dimensions. We have to scale the loss to incorporate this scaling.
            _, C = output.shape
            loss *= sqrt(C)
        return loss

    def _compute_loss_and_backward(
        self, output: Tensor, y: Tensor, gradient_covariances: Dict[str, Tensor]
    ):
        r"""Compute the loss and the backward pass(es) required for KFAC.

        Args:
            output: The model's prediction
                :math:`\{f_\mathbf{\theta}(\mathbf{x}_n)\}_{n=1}^N`.
            y: The labels :math:`\{\mathbf{y}_n\}_{n=1}^N`.
            gradient_covariances: Dictionary to store computed gradient covariances.

        Raises:
            ValueError: If the output is not 2d and y is not 1d/2d.
            ValueError: If ``fisher_type`` is not ``FisherType.TYPE2``,
                ``FisherType.MC``, ``FisherType.EMPIRICAL``, or
                ``FisherType.FORWARD_ONLY``.
        """
        if output.ndim != 2 or y.ndim not in {1, 2}:
            raise ValueError(
                "Only 2d output and 1d/2d target are supported. "
                f"Got {output.ndim=} and {y.ndim=}."
            )

        if self._fisher_type == FisherType.TYPE2:
            # Compute per-sample Hessian square root, then concatenate over samples.
            # Result has shape `(batch_size, num_classes, num_classes)`
            hessian_sqrts = stack(
                [
                    loss_hessian_matrix_sqrt(out.detach(), target, self._loss_func)
                    for out, target in zip(output.split(1), y.split(1))
                ]
            )

            # Fix scaling caused by the batch dimension
            num_loss_terms = output.shape[0]
            reduction = self._loss_func.reduction
            scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[reduction]
            hessian_sqrts.mul_(scale)

            # For each column `c` of the matrix square root we need to backpropagate,
            # but we can do this for all samples in parallel
            num_cols = hessian_sqrts.shape[-1]
            for c in range(num_cols):
                batched_column = hessian_sqrts[:, :, c]
                grad(
                    (output * batched_column).sum(),
                    self._params,
                    retain_graph=c < num_cols - 1,
                )

        elif self._fisher_type == FisherType.MC:
            for mc in range(self._mc_samples):
                y_sampled = self.draw_label(output)
                loss = self._loss_func(output, y_sampled)
                loss = self._maybe_adjust_loss_scale(loss, output)
                grad(loss, self._params, retain_graph=mc != self._mc_samples - 1)

        elif self._fisher_type == FisherType.EMPIRICAL:
            loss = self._loss_func(output, y)
            loss = self._maybe_adjust_loss_scale(loss, output)
            grad(loss, self._params)

        elif self._fisher_type == FisherType.FORWARD_ONLY:
            # Since FOOF sets the gradient covariance Kronecker factors to the identity,
            # we don't need to do a backward pass. See https://arxiv.org/abs/2201.12250.
            # We choose to set the gradient covariance to the identity explicitly for
            # the sake of simplicity, such that the rest of the code here and for
            # `KFACInverseLinearOperator` does not have to be adapted. This could be
            # changed to decrease the memory costs.
            for mod_name, param_pos in self._mapping.items():
                # We iterate over _mapping to get the module names corresponding to the
                # parameters. We only need the output dimension of the module, but
                # don't know whether the parameter is a weight or bias; therefore, we
                # just call `next(iter(param_pos.values()))` to get the first parameter.
                param = self._params[next(iter(param_pos.values()))]
                gradient_covariances[mod_name] = eye(
                    param.shape[0], dtype=param.dtype, device=self.device
                )

        else:
            raise ValueError(
                f"Invalid fisher_type: {self._fisher_type}. "
                + f"Supported: {self._SUPPORTED_FISHER_TYPE}."
            )

    def draw_label(self, output: Tensor) -> Tensor:
        r"""Draw a sample from the model's predictive distribution.

        The model's distribution is implied by the (negative log likelihood) loss
        function. For instance, ``MSELoss`` implies a Gaussian distribution with
        constant variance, and ``CrossEntropyLoss`` implies a categorical distribution.

        Args:
            output: The model's prediction
                :math:`\{f_\mathbf{\theta}(\mathbf{x}_n)\}_{n=1}^N`.

        Returns:
            A sample
            :math:`\{\mathbf{y}_n\}_{n=1}^N` drawn from the model's predictive
            distribution :math:`p(\mathbf{y} \mid \mathbf{x}, \mathbf{\theta})`. Has
            the same shape as the labels that would be fed into the loss function
            together with ``output``.

        Raises:
            ValueError: If the output is not 2d.
            NotImplementedError: If the loss function is not supported.
        """
        if output.ndim != 2:
            raise ValueError("Only a 2d output is supported.")

        if isinstance(self._loss_func, MSELoss):
            std = sqrt(0.5)
            perturbation = std * randn(
                output.shape,
                device=output.device,
                dtype=output.dtype,
                generator=self._generator,
            )
            return output.clone().detach() + perturbation

        elif isinstance(self._loss_func, CrossEntropyLoss):
            probs = output.softmax(dim=1)
            labels = probs.multinomial(
                num_samples=1, generator=self._generator
            ).squeeze(-1)
            return labels

        elif isinstance(self._loss_func, BCEWithLogitsLoss):
            probs = output.sigmoid()
            labels = probs.bernoulli(generator=self._generator)
            return labels

        else:
            raise NotImplementedError

    def _register_tensor_hook_on_output_to_accumulate_gradient_covariance(
        self,
        module: Module,
        inputs: Tuple[Tensor],
        output: Tensor,
        module_name: str,
        covariances_dict: Dict[str, Tensor],
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
            covariances_dict: Dictionary to store computed covariances.
        """
        tensor_hook = partial(
            self._accumulate_gradient_covariance,
            module=module,
            module_name=module_name,
            covariances_dict=covariances_dict,
        )
        output.register_hook(tensor_hook)

    def _accumulate_gradient_covariance(
        self,
        grad_output: Tensor,
        module: Module,
        module_name: str,
        covariances_dict: Dict[str, Tensor],
    ):
        """Accumulate the gradient covariance for a layer's output.

        Args:
            grad_output: The gradient w.r.t. the output.
            module: The layer whose output's gradient covariance will be accumulated.
            module_name: The name of the layer in the neural network.
            covariances_dict: Dictionary to store computed covariances.
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

        # Compute correction for the loss scaling depending on the loss reduction used
        num_loss_terms = batch_size * self._num_per_example_loss_terms
        # self._mc_samples will be 1 if fisher_type != FisherType.MC
        correction = {
            "sum": 1.0 / self._mc_samples,
            "mean": num_loss_terms**2
            / (self._N_data * self._mc_samples * self._num_per_example_loss_terms),
        }[self._loss_func.reduction]

        covariance = einsum(g, g, "b i,b j->i j").mul_(correction)
        self._set_or_add_(covariances_dict, module_name, covariance)

    def _hook_accumulate_input_covariance(
        self,
        module: Module,
        inputs: Tuple[Tensor],
        module_name: str,
        covariances_dict: Dict[str, Tensor],
    ):
        """Pre-forward hook that accumulates the input covariance of a layer.

        Args:
            module: Module on which the hook is called.
            inputs: Inputs to the module.
            module_name: Name of the module in the neural network.
            covariances_dict: Dictionary to store computed covariances.

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
        self._set_or_add_(covariances_dict, module_name, covariance)

    @staticmethod
    def _set_or_add_(
        dictionary: Dict[str, Tensor], key: str, value: Tensor
    ) -> Dict[str, Tensor]:
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
        cls, params: List[Union[Tensor, Parameter]], model_func: Module
    ) -> Dict[str, Dict[str, int]]:
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

    def trace(self) -> Tensor:
        r"""Trace of the KFAC approximation.

        Returns:
            Trace of the KFAC approximation.
        """
        return self._block_diagonal_operator.trace

    def det(self) -> Tensor:
        r"""Determinant of the KFAC approximation.

        Returns:
            Determinant of the KFAC approximation.
        """
        return self._block_diagonal_operator.det

    def logdet(self) -> Tensor:
        r"""Log determinant of the KFAC approximation.

        More numerically stable than the ``det`` property.

        Returns:
            Log determinant of the KFAC approximation.
        """
        return self._block_diagonal_operator.logdet

    def frobenius_norm(self) -> Tensor:
        r"""Frobenius norm of the KFAC approximation.

        Returns:
            Frobenius norm of the KFAC approximation.
        """
        return self._block_diagonal_operator.frobenius_norm

    def inverse(
        self,
        damping: float = 0.0,
        use_heuristic_damping: bool = False,
        min_damping: float = 1e-8,
        use_exact_damping: bool = False,
        retry_double_precision: bool = True,
    ) -> _ChainPyTorchLinearOperator:
        """Return the inverse of the KFAC linear operator.

        Args:
            damping: Damping value for all input and gradient covariances.
                Default: ``0.0``.
            use_heuristic_damping: Whether to use a heuristic damping strategy by
                `Martens and Grosse, 2015 <https://arxiv.org/abs/1503.05671>`_
                (Section 6.3). Default: ``False``.
            min_damping: Minimum damping value. Only used if
                ``use_heuristic_damping`` is ``True``. Default: ``1e-8``.
            use_exact_damping: Whether to use exact damping, i.e. to invert
                :math:`(A \\otimes B) + \\text{damping}\\; \\mathbf{I}`.
                Default: ``False``.
            retry_double_precision: Whether to retry Cholesky decomposition used for
                inversion in double precision. Default: ``True``.

        Returns:
            Linear operator representing the inverse of KFAC.

        Raises:
            ValueError: If both heuristic and exact damping are selected.
            ValueError: If heuristic or exact damping is used and the damping value
                is a tuple.
        """
        # Invert the blocks of self._block_diagonal_operator
        inverse_blocks = [
            block.inverse(
                damping=damping,
                use_heuristic_damping=use_heuristic_damping,
                min_damping=min_damping,
                use_exact_damping=use_exact_damping,
                retry_double_precision=retry_double_precision,
            )
            for block in self._block_diagonal_operator._blocks
        ]

        # Create the inverse block diagonal operator
        inverse_block_diagonal = BlockDiagonalLinearOperator(inverse_blocks)

        return self._from_canonical @ inverse_block_diagonal @ self._to_canonical


class _ToCanonicalLinearOperator(PyTorchLinearOperator):
    """Linear operator that transforms parameters from original to canonical form.

    Canonical form orders parameters by layer, with proper grouping and flattening.
    This is the adjoint of _FromCanonicalLinearOperator.
    """

    def __init__(
        self,
        params: List[Parameter],
        mapping: Dict[str, Dict[str, int]],
        separate_weight_and_bias: bool,
    ):
        """Initialize the canonical form transformation operator.

        Args:
            params: List of model parameters in original order.
            mapping: Parameter mapping from layer names to parameter positions.
            separate_weight_and_bias: Whether to treat weights and biases separately.
        """
        self._params = params
        self._mapping = mapping
        self._separate_weight_and_bias = separate_weight_and_bias

        in_shape = [tuple(p.shape) for p in params]
        out_shape = self._compute_canonical_shapes()

        super().__init__(in_shape, out_shape)

    def _compute_canonical_shapes(self) -> List[Tuple[int, ...]]:
        """Compute the shapes in canonical form.

        Returns:
            List of shapes after canonical transformation.
        """
        canonical_shapes = []

        for param_pos in self._mapping.values():
            # Handle joint weight+bias case
            if not self._separate_weight_and_bias and {"weight", "bias"} == set(
                param_pos.keys()
            ):
                w_pos = param_pos["weight"]
                w = self._params[w_pos]
                # Combined weight+bias gets flattened to 1D
                total_params = w.numel() + w.shape[0]  # weight + bias
                canonical_shapes.append((total_params,))
            else:
                # Handle separate weight and bias
                for p_name in param_pos:
                    pos = param_pos[p_name]
                    # Each parameter gets flattened to 1D
                    canonical_shapes.append((self._params[pos].numel(),))

        return canonical_shapes

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Transform parameter tensors to canonical form.

        Args:
            M: Parameter tensors in original order.

        Returns:
            Parameter tensors in canonical form (flattened and reordered).
        """
        canonical_M = []

        for param_pos in self._mapping.values():
            # Handle joint weight+bias case
            if not self._separate_weight_and_bias and {"weight", "bias"} == set(
                param_pos.keys()
            ):
                w_pos, b_pos = param_pos["weight"], param_pos["bias"]
                # Flatten weight tensor into matrix and concatenate bias
                w_flat = M[w_pos].flatten(start_dim=1, end_dim=-2)
                # Add bias as additional row
                combined = cat([w_flat, M[b_pos].unsqueeze(1)], dim=1)
                # Flatten parameter space dimension
                canonical_M.append(combined.flatten(end_dim=-2))
            else:
                # Handle separate weight and bias
                for p_name in param_pos:
                    pos = param_pos[p_name]
                    canonical_M.append(M[pos].flatten(end_dim=-2))

        return canonical_M

    def _adjoint(self) -> _FromCanonicalLinearOperator:
        """Return the adjoint transformation operator.

        Returns:
            Linear operator that transforms from canonical to parameter form.
        """
        return _FromCanonicalLinearOperator(
            self._params, self._mapping, self._separate_weight_and_bias
        )

    @property
    def device(self):
        """Infer device from parameters.

        Returns:
            The device of the parameters.

        Raises:
            RuntimeError: If parameters are on different devices.
        """
        devices = {p.device for p in self._params}
        if len(devices) != 1:
            raise RuntimeError(f"Could not infer device. Parameters live on {devices}.")
        return devices.pop()

    @property
    def dtype(self):
        """Infer dtype from parameters.

        Returns:
            The dtype of the parameters.

        Raises:
            RuntimeError: If parameters have different dtypes.
        """
        dtypes = {p.dtype for p in self._params}
        if len(dtypes) != 1:
            raise RuntimeError(f"Could not infer dtype. Parameters have {dtypes}.")
        return dtypes.pop()


class _FromCanonicalLinearOperator(PyTorchLinearOperator):
    """Linear operator that transforms parameters from canonical to original form.

    This is the adjoint of _ToCanonicalLinearOperator.
    """

    def __init__(
        self,
        params: List[Parameter],
        mapping: Dict[str, Dict[str, int]],
        separate_weight_and_bias: bool,
    ):
        """Initialize the canonical form reverse transformation operator.

        Args:
            params: List of model parameters in original order.
            mapping: Parameter mapping from layer names to parameter positions.
            separate_weight_and_bias: Whether to treat weights and biases separately.
        """
        self._params = params
        self._mapping = mapping
        self._separate_weight_and_bias = separate_weight_and_bias

        # Input and output shapes are swapped compared to ToCanonical
        out_shape = [tuple(p.shape) for p in params]
        in_shape = self._compute_canonical_shapes()

        super().__init__(in_shape, out_shape)

    def _compute_canonical_shapes(self) -> List[Tuple[int, ...]]:
        """Compute the shapes in canonical form.

        Returns:
            List of shapes after canonical transformation.
        """
        canonical_shapes = []

        for param_pos in self._mapping.values():
            # Handle joint weight+bias case
            if not self._separate_weight_and_bias and {"weight", "bias"} == set(
                param_pos.keys()
            ):
                w_pos = param_pos["weight"]
                w = self._params[w_pos]
                # Combined weight+bias gets flattened to 1D
                total_params = w.numel() + w.shape[0]  # weight + bias
                canonical_shapes.append((total_params,))
            else:
                # Handle separate weight and bias
                for p_name in param_pos:
                    pos = param_pos[p_name]
                    # Each parameter gets flattened to 1D
                    canonical_shapes.append((self._params[pos].numel(),))

        return canonical_shapes

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Transform parameter tensors from canonical form back to original order.

        Args:
            M: Parameter tensors in canonical form.

        Returns:
            Parameter tensors in original order with proper shapes.
        """
        original_M = [None] * len(self._params)
        (num_columns,) = {m.shape[-1] for m in M}
        processed = 0

        for param_pos in self._mapping.values():
            # Handle joint weight+bias case
            if not self._separate_weight_and_bias and {"weight", "bias"} == set(
                param_pos.keys()
            ):
                w_pos, b_pos = param_pos["weight"], param_pos["bias"]
                combined = M[processed]

                # Get original weight shape
                w = self._params[w_pos]
                w_rows, w_cols = w.shape[0], w.shape[1:].numel()

                # Reshape combined tensor back to (weight + bias) matrix
                combined = combined.reshape(w_rows, w_cols + 1, num_columns)
                w_part, b_part = combined.split([w_cols, 1], dim=1)

                # Reshape into parameter shape
                original_M[w_pos] = w_part.reshape(*w.shape, num_columns)
                original_M[b_pos] = b_part.reshape(w_rows, num_columns)
                processed += 1
            else:
                # Handle separate weight and bias
                for p_name in param_pos:
                    pos = param_pos[p_name]
                    original_M[pos] = M[processed].reshape(
                        *self._params[pos].shape, num_columns
                    )
                    processed += 1

        return original_M

    def _adjoint(self) -> _ToCanonicalLinearOperator:
        """Return the adjoint transformation operator.

        Returns:
            Linear operator that transforms from parameter to canonical form.
        """
        return _ToCanonicalLinearOperator(
            self._params, self._mapping, self._separate_weight_and_bias
        )

    @property
    def device(self):
        """Infer device from parameters.

        Returns:
            The device of the parameters.

        Raises:
            RuntimeError: If parameters are on different devices.
        """
        devices = {p.device for p in self._params}
        if len(devices) != 1:
            raise RuntimeError(f"Could not infer device. Parameters live on {devices}.")
        return devices.pop()

    @property
    def dtype(self):
        """Infer dtype from parameters.

        Returns:
            The dtype of the parameters.

        Raises:
            RuntimeError: If parameters have different dtypes.
        """
        dtypes = {p.dtype for p in self._params}
        if len(dtypes) != 1:
            raise RuntimeError(f"Could not infer dtype. Parameters have {dtypes}.")
        return dtypes.pop()
