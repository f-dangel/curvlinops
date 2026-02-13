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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from warnings import warn

from einops import einsum, rearrange, reduce
from torch import Generator, Tensor, cat, eye
from torch.autograd import grad
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

from curvlinops._torch_base import (
    CurvatureLinearOperator,
    PyTorchLinearOperator,
    _ChainPyTorchLinearOperator,
)
from curvlinops.blockdiagonal import BlockDiagonalLinearOperator
from curvlinops.kfac_utils import (
    ToCanonicalLinearOperator,
    _check_binary_if_BCEWithLogitsLoss,
    extract_averaged_patches,
    extract_patches,
    make_grad_output_fn,
)
from curvlinops.kronecker import KroneckerProductLinearOperator
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
    NEEDS_NUM_PER_EXAMPLE_LOSS_TERMS: bool = True

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
        self._representation = None

        # Function (prediction_batch, label_batch) -> grad_outputs for backpropagation
        self._grad_outputs_computer = (
            self._set_up_grad_outputs_computer(loss_func, fisher_type, mc_samples)
            # TODO Implement grad_output sampler for empirical case and remove
            # _maybe_adjust_loss_scale
            if fisher_type in {FisherType.MC, FisherType.TYPE2}
            else None
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

    @property
    def representation(self) -> Dict[str, PyTorchLinearOperator]:
        """Return the internal representation of the linear operator.

        This attribute is lazily evaluated and cached after the first access.

        Returns:
            A dictionary containing the linear operators converting from parameter to
            canonical space and back, as well as the canonical KFAC operator (block-
            diagonal Kronecker-factored).
        """
        if self._representation is None:
            input_covariances, gradient_covariances = self.compute_kronecker_factors()
            # KFAC in the canonical basis
            canonical_op = self._setup_canonical_operator(
                input_covariances, gradient_covariances
            )

            # Set up converters from parameter to canonical space and back
            to_canonical_op = ToCanonicalLinearOperator(
                [p.shape for p in self._params],
                list(self._mapping.values()),
                self._separate_weight_and_bias,
                self.device,
                self.dtype,
            )
            from_canonical_op = to_canonical_op.adjoint()

            self._representation = {
                "canonical_op": canonical_op,
                "to_canonical_op": to_canonical_op,
                "from_canonical_op": from_canonical_op,
            }
        return self._representation

    def refresh_representation(self):
        """Refresh the internal representation of the linear operator.

        Re-computes the Kronecker factors.
        """
        self._representation = None
        # Accessing the property triggers the re-computation
        _ = self.representation

    @staticmethod
    def _set_up_grad_outputs_computer(
        loss_func: Union[MSELoss, CrossEntropyLoss, BCEWithLogitsLoss],
        fisher_type: FisherType,
        mc_samples: int,
    ) -> Callable[[Tensor, Tensor, Optional[Generator]], Tensor]:
        """Set up the function that computes network output gradients for KFAC.

        Args:
            loss_func: The loss function.
            fisher_type: The Fisher type (``TYPE2`` or ``MC``).
            mc_samples: Number of MC samples (used when ``fisher_type`` is ``MC``).

        Returns:
            A function ``(output_batch, y_batch, generator) -> grad_outputs``
            that computes the gradients to be backpropagated from the network's
            output, with shape ``[num_vectors, batch, *output_shape]``.
        """
        mode = {FisherType.MC: "mc", FisherType.TYPE2: "exact"}[fisher_type]

        grad_output_fn = make_grad_output_fn(loss_func, mode, mc_samples)
        randomness = {"mc": "different", "exact": "same"}[mode]
        batched_grad_output_fn = vmap(
            grad_output_fn, in_dims=(0, 0, None), out_dims=1, randomness=randomness
        )

        def compute_grad_outputs(
            output: Tensor, y: Tensor, generator: Optional[Generator] = None
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

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Apply KFAC to a matrix (multiple vectors) in tensor list format.

        This allows for matrix-matrix products with the KFAC approximation in PyTorch
        without converting tensors to numpy arrays, which avoids unnecessary
        device transfers when working with GPUs and flattening/concatenating.

        Args:
            M: Matrix for multiplication in tensor list format. Each entry has the
                same shape as a parameter with an additional trailing dimension of size
                ``K`` for the columns, i.e. ``[(*p1.shape, K), (*p2.shape, K), ...]``.

        Returns:
            Matrix-multiplication result ``KFAC @ M`` in tensor list format. Has the same
            shapes as the input.
        """
        P = self.representation["from_canonical_op"]
        PT = self.representation["to_canonical_op"]
        K = self.representation["canonical_op"]
        kfac = P @ K @ PT
        return kfac @ M

    def compute_kronecker_factors(self) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Compute KFAC's Kronecker factors.

        Returns:
            Tuple containing (input_covariances, gradient_covariances) dictionaries.
        """
        # Create empty dictionaries to be populated by hooks
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

    def _compute_loss_and_backward(self, output: Tensor, y: Tensor):
        r"""Compute the loss and the backward pass(es) required for KFAC.

        Args:
            output: The model's prediction
                :math:`\{f_\mathbf{\theta}(\mathbf{x}_n)\}_{n=1}^N`.
            y: The labels :math:`\{\mathbf{y}_n\}_{n=1}^N`.

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

        if self._fisher_type in {FisherType.TYPE2, FisherType.MC}:
            # Compute the gradients w.r.t. the network's output that will be
            # backpropagated to compute the KFAC approximation
            # - TYPE2: Hessian square root columns
            # - MC: Monte-Carlo approximation of the the Hessian square root
            # Detach output: we only need values for the backpropagated vectors.
            grad_outputs = self._grad_outputs_computer(
                output.detach(), y, self._generator
            )

            # Fix scaling caused by the batch dimension
            num_loss_terms = output.shape[0]
            reduction = self._loss_func.reduction
            scale = {"sum": 1.0, "mean": 1.0 / num_loss_terms}[reduction]
            grad_outputs.mul_(scale)

            # Backpropagate all vectors
            num_vectors = grad_outputs.shape[0]
            for v in range(num_vectors):
                grad(
                    output,
                    self._params,
                    grad_outputs=grad_outputs[v],
                    retain_graph=v < num_vectors - 1,
                )

        elif self._fisher_type == FisherType.EMPIRICAL:
            loss = self._loss_func(output, y)
            loss = self._maybe_adjust_loss_scale(loss, output)
            grad(loss, self._params)

        elif self._fisher_type == FisherType.FORWARD_ONLY:
            # No backward passes required for forward-only KFAC
            pass

        else:
            raise ValueError(
                f"Invalid fisher_type: {self._fisher_type}. "
                + f"Supported: {self._SUPPORTED_FISHER_TYPE}."
            )

    def _register_tensor_hook_on_output_to_accumulate_gradient_covariance(
        self,
        module: Module,
        inputs: Tuple[Tensor],
        output: Tensor,
        module_name: str,
        gradient_covariances: Dict[str, Tensor],
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
        gradient_covariances: Dict[str, Tensor],
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
        inputs: Tuple[Tensor],
        module_name: str,
        input_covariances: Dict[str, Tensor],
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
        dictionary: Dict[Any, Tensor], key: Any, value: Tensor
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

    def _setup_canonical_operator(
        self,
        input_covariances: Dict[str, Tensor],
        gradient_covariances: Dict[str, Tensor],
    ) -> BlockDiagonalLinearOperator:
        """Set up the canonical KFAC operator from Kronecker factors.

        Args:
            input_covariances: Dictionary mapping module names to input covariances.
            gradient_covariances: Dictionary mapping module names to gradient
                covariances.

        Returns:
            Block diagonal linear operator representing KFAC in canonical basis.
        """
        # Set up Kronecker operators for each block
        factors = []

        for mod_name, param_pos in self._mapping.items():
            aaT = input_covariances.get(mod_name, None)
            ggT = gradient_covariances[mod_name]

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

        # KFAC in the canonical basis
        return BlockDiagonalLinearOperator(blocks)

    def trace(self) -> Tensor:
        """Trace of the KFAC approximation.

        Returns:
            Trace of the KFAC approximation.
        """
        return self.representation["canonical_op"].trace()

    def det(self) -> Tensor:
        """Compute the determinant of the KFAC approximation.

        Returns:
            Determinant of the KFAC approximation.
        """
        return self.representation["canonical_op"].det()

    def logdet(self) -> Tensor:
        """Log determinant of the KFAC approximation.

        More numerically stable than the ``det`` method.

        Returns:
            Log determinant of the KFAC approximation.
        """
        return self.representation["canonical_op"].logdet()

    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the KFAC approximation.

        Returns:
            Frobenius norm of the KFAC approximation.
        """
        return self.representation["canonical_op"].frobenius_norm()

    def inverse(
        self,
        damping: float = 0.0,
        use_heuristic_damping: bool = False,
        min_damping: float = 1e-8,
        use_exact_damping: bool = False,
        retry_double_precision: bool = True,
    ) -> _ChainPyTorchLinearOperator:
        r"""Return the inverse of the KFAC approximation.

        Inverts each Kronecker-factored block of the canonical operator
        and returns the result in parameter space.

        Args:
            damping: Damping value applied to all Kronecker factors. Default: ``0.0``.
            use_heuristic_damping: Whether to use a heuristic damping strategy by
                `Martens and Grosse, 2015 <https://arxiv.org/abs/1503.05671>`_
                (Section 6.3). Only supported for exactly two factors.
            min_damping: Minimum damping value. Only used if
                ``use_heuristic_damping`` is ``True``.
            use_exact_damping: Whether to use exact damping, i.e. to invert
                :math:`(A \\otimes B) + \\text{damping}\\; \\mathbf{I}`.
            retry_double_precision: Whether to retry Cholesky decomposition used for
                inversion in double precision.

        Returns:
            Inverse of the KFAC approximation as a linear operator ``P @ K^-1 @ PT``.
        """
        P = self.representation["from_canonical_op"]
        PT = self.representation["to_canonical_op"]
        K = self.representation["canonical_op"]
        K_inv = BlockDiagonalLinearOperator([
            block.inverse(
                damping=damping,
                use_heuristic_damping=use_heuristic_damping,
                min_damping=min_damping,
                use_exact_damping=use_exact_damping,
                retry_double_precision=retry_double_precision,
            )
            for block in K._blocks
        ])
        return P @ K_inv @ PT

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the KFAC linear operator.

        Returns:
            State dictionary.
        """
        loss_type = {
            MSELoss: "MSELoss",
            CrossEntropyLoss: "CrossEntropyLoss",
            BCEWithLogitsLoss: "BCEWithLogitsLoss",
        }[type(self._loss_func)]
        return {
            # Model and loss function
            "model_func_state_dict": self._model_func.state_dict(),
            "loss_type": loss_type,
            "loss_reduction": self._loss_func.reduction,
            # Attributes
            "progressbar": self._progressbar,
            "seed": self._seed,
            "fisher_type": self._fisher_type,
            "mc_samples": self._mc_samples,
            "kfac_approx": self._kfac_approx,
            "num_per_example_loss_terms": self._num_per_example_loss_terms,
            "separate_weight_and_bias": self._separate_weight_and_bias,
            "num_data": self._N_data,
            # Representation (Kronecker factors), if computed
            "_representation": self._representation,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the state of the KFAC linear operator.

        Warning:
            Loading a state dict will overwrite the parameters of the model underlying
            the linear operator!

        Args:
            state_dict: State dictionary.

        Raises:
            ValueError: If the loss function does not match the state dict.
            ValueError: If the loss function reduction does not match the state dict.
        """
        warn(
            "Loading a state dict will overwrite the parameters of the model underlying the linear operator!",
            stacklevel=2,
        )
        self._model_func.load_state_dict(state_dict["model_func_state_dict"])
        # Verify that the loss function and its reduction match the state dict
        loss_func_type = {
            "MSELoss": MSELoss,
            "CrossEntropyLoss": CrossEntropyLoss,
            "BCEWithLogitsLoss": BCEWithLogitsLoss,
        }[state_dict["loss_type"]]
        if not isinstance(self._loss_func, loss_func_type):
            raise ValueError(
                f"Loss function mismatch: {loss_func_type} != {type(self._loss_func)}."
            )
        if state_dict["loss_reduction"] != self._loss_func.reduction:
            raise ValueError(
                "Loss function reduction mismatch: "
                f"{state_dict['loss_reduction']} != {self._loss_func.reduction}."
            )

        # Set attributes
        self._progressbar = state_dict["progressbar"]
        self._seed = state_dict["seed"]
        self._fisher_type = state_dict["fisher_type"]
        self._mc_samples = state_dict["mc_samples"]
        self._kfac_approx = state_dict["kfac_approx"]
        self._num_per_example_loss_terms = state_dict["num_per_example_loss_terms"]
        self._separate_weight_and_bias = state_dict["separate_weight_and_bias"]
        self._N_data = state_dict["num_data"]
        self._representation = state_dict["_representation"]

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, Any],
        model_func: Module,
        params: List[Parameter],
        data: Iterable[Tuple[Union[Tensor, MutableMapping], Tensor]],
        check_deterministic: bool = True,
        batch_size_fn: Optional[Callable[[Union[MutableMapping, Tensor]], int]] = None,
    ) -> KFACLinearOperator:
        """Load a KFAC linear operator from a state dictionary.

        Args:
            state_dict: State dictionary.
            model_func: The model function.
            params: The model's parameters that KFAC is computed for.
            data: A data loader containing the data of the Fisher/GGN.
            check_deterministic: Whether to check that the linear operator is
                deterministic. Defaults to ``True``.
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified. The intended behavior is to consume the first
                entry of the iterates from ``data`` and return their batch size.

        Returns:
            Linear operator of KFAC approximation.
        """
        loss_func = {
            "MSELoss": MSELoss,
            "CrossEntropyLoss": CrossEntropyLoss,
            "BCEWithLogitsLoss": BCEWithLogitsLoss,
        }[state_dict["loss_type"]](reduction=state_dict["loss_reduction"])
        kfac = cls(
            model_func,
            loss_func,
            params,
            data,
            batch_size_fn=batch_size_fn,
            check_deterministic=False,
            progressbar=state_dict["progressbar"],
            seed=state_dict["seed"],
            fisher_type=state_dict["fisher_type"],
            mc_samples=state_dict["mc_samples"],
            kfac_approx=state_dict["kfac_approx"],
            num_per_example_loss_terms=state_dict["num_per_example_loss_terms"],
            separate_weight_and_bias=state_dict["separate_weight_and_bias"],
            num_data=state_dict["num_data"],
        )
        kfac.load_state_dict(state_dict)

        # Potentially call `check_deterministic` after the state dict is loaded
        if check_deterministic:
            kfac._check_deterministic()

        return kfac
