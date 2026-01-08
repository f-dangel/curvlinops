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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
from warnings import warn

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

from curvlinops._torch_base import CurvatureLinearOperator
from curvlinops.kfac_utils import (
    extract_averaged_patches,
    extract_patches,
    loss_hessian_matrix_sqrt,
)

FactorType = TypeVar(
    "FactorType", Optional[Tensor], Tuple[Optional[Tensor], Optional[Tensor]]
)


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
        self._input_covariances: Dict[str, Tensor] = {}
        self._gradient_covariances: Dict[str, Tensor] = {}
        self._mapping = self.compute_parameter_mapping(params, model_func)

        # Properties of the full matrix KFAC approximation are initialized to `None`
        self._reset_matrix_properties()

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

    def _reset_matrix_properties(self):
        """Reset matrix properties."""
        self._trace = None
        self._det = None
        self._logdet = None
        self._frobenius_norm = None

    @staticmethod
    def _left_and_right_multiply(
        M: Tensor,
        aaT: FactorType,
        ggT: FactorType,
        eigenvalues: Optional[Tensor] = None,
    ) -> Tensor:
        """Left and right multiply matrix with Kronecker factors.

        Args:
            M: (Batched) Matrix for multiplication. Shape will be
                (ggT.shape[0], aaT.shape[0], K), where K is the number of vectors/the
                batch dimension of the batched matrix product.
            aaT: Input covariance Kronecker factor or its eigenvectors. ``None`` for
                biases.
            ggT: Gradient covariance Kronecker factor or its eigenvectors.
            eigenvalues: Eigenvalues of the (E)KFAC approximation when multiplying with
                the eigendecomposition of the KFAC approximation. ``None`` for the
                non-decomposed KFAC approximation. Defaults to ``None``.

        Returns:
            Matrix-multiplication result.
        """
        if eigenvalues is None:
            M = einsum(ggT, M, aaT, "i j, j k v, k l -> i l v")
        else:
            # Perform preconditioning in KFE, e.g. see equation (21) in
            # https://arxiv.org/abs/2308.03296.
            aaT_eigvecs = aaT
            ggT_eigvecs = ggT
            # Transform in eigenbasis.
            M = einsum(ggT_eigvecs, M, aaT_eigvecs, "i j, i k v, k l -> j l v")
            # Multiply (broadcasted) by eigenvalues.
            M.mul_(eigenvalues.unsqueeze(-1))
            # Transform back to standard basis.
            M = einsum(ggT_eigvecs, M, aaT_eigvecs, "i j, j k v, l k -> i l v")
        return M

    @staticmethod
    def _separate_left_and_right_multiply(
        KM: List[Tensor],
        M: List[Tensor],
        param_pos: Dict[str, int],
        aaT: FactorType,
        ggT: FactorType,
        eigenvalues: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Multiply matrix with Kronecker factors for separated weight and bias.

        Args:
            KM: List to write the matrix-multiplication result to.
            M: List of matrices for multiplication.
            param_pos: Dictionary with positions of the weight and bias parameters.
            aaT: Input covariance Kronecker factor or its eigenvectors. ``None`` for
                biases.
            ggT: Gradient covariance Kronecker factor or its eigenvectors.
            eigenvalues: Eigenvalues of the (E)KFAC approximation when multiplying with
                the eigendecomposition of the KFAC approximation. ``None`` for the
                non-decomposed KFAC approximation. Defaults to ``None``.
        """
        for p_name, pos in param_pos.items():
            # for weights we need to multiply from the right with aaT
            # for weights and biases we need to multiply from the left with ggT
            if p_name == "weight":
                M_w = rearrange(M[pos], "c_out ... v -> c_out (...) v")
                # If `eigenvalues` is not `None`, we transform to eigenbasis here
                KM[pos] = einsum(M_w, aaT, "c_out j v, j k -> c_out k v")
            else:
                KM[pos] = M[pos]

            # If `eigenvalues` is not `None`, we convert to eigenbasis here
            KM[pos] = einsum(
                ggT.T if eigenvalues else ggT, KM[pos], "j k, k ... v -> j ... v"
            )

            if eigenvalues is not None:
                # Multiply (broadcasted) by eigenvalues, convert back to original basis
                KM[pos].mul_(eigenvalues[pos].unsqueeze(-1))
                if p_name == "weight":
                    KM[pos] = einsum(KM[pos], aaT, "c_out j v, k j -> c_out k v")
                KM[pos] = einsum(ggT, KM[pos], "j k, k ... v -> j ... v")

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
        if not (self._input_covariances or self._gradient_covariances):
            self.compute_kronecker_factors()

        KM: List[Tensor | None] = [None] * len(M)

        for mod_name, param_pos in self._mapping.items():
            # cache the weight shape to ensure correct shapes are returned
            if "weight" in param_pos:
                weight_shape = M[param_pos["weight"]].shape

            # get the Kronecker factors for the current module
            # aaT does not exist when weight matrix is excluded
            aaT = self._input_covariances.get(mod_name)
            # ggT always exists
            ggT = self._gradient_covariances[mod_name]

            # bias and weights are treated jointly
            if (
                not self._separate_weight_and_bias
                and "weight" in param_pos.keys()
                and "bias" in param_pos.keys()
            ):
                w_pos, b_pos = param_pos["weight"], param_pos["bias"]
                # v denotes the free dimension for treating multiple vectors in parallel
                M_w = rearrange(M[w_pos], "c_out ... v -> c_out (...) v")
                M_joint = cat([M_w, M[b_pos].unsqueeze(-2)], dim=-2)
                M_joint = self._left_and_right_multiply(M_joint, aaT, ggT)
                w_cols = M_w.shape[1]
                KM[w_pos], KM[b_pos] = M_joint.split([w_cols, 1], dim=-2)
                KM[b_pos].squeeze_(1)
            else:
                self._separate_left_and_right_multiply(KM, M, param_pos, aaT, ggT)

            # restore original shapes
            if "weight" in param_pos:
                KM[param_pos["weight"]] = KM[param_pos["weight"]].view(weight_shape)

        return KM

    def compute_kronecker_factors(self):
        """Compute and cache KFAC's Kronecker factors for future ``matmat``s."""
        self._reset_matrix_properties()

        # install forward and backward hooks
        hook_handles: List[RemovableHandle] = []

        for mod_name, param_pos in self._mapping.items():
            module = self._model_func.get_submodule(mod_name)

            # input covariance only required for weights
            if "weight" in param_pos.keys():
                hook_handles.append(
                    module.register_forward_pre_hook(
                        partial(
                            self._hook_accumulate_input_covariance, module_name=mod_name
                        )
                    )
                )

            # gradient covariance required for weights and biases
            hook_handles.append(
                module.register_forward_hook(
                    partial(
                        self._register_tensor_hook_on_output_to_accumulate_gradient_covariance,
                        module_name=mod_name,
                    )
                )
            )

        # loop over data set, computing the Kronecker factors
        if self._generator is None or self._generator.device != self.device:
            self._generator = Generator(device=self.device)
        self._generator.manual_seed(self._seed)

        for X, y in self._loop_over_data(desc="KFAC matrices"):
            output = self._model_func(X)
            output, y = self._rearrange_for_larger_than_2d_output(output, y)
            self._compute_loss_and_backward(output, y)

        # clean up
        for handle in hook_handles:
            handle.remove()

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
                self._gradient_covariances[mod_name] = eye(
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
        self, module: Module, inputs: Tuple[Tensor], output: Tensor, module_name: str
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
        """
        tensor_hook = partial(
            self._accumulate_gradient_covariance, module=module, module_name=module_name
        )
        output.register_hook(tensor_hook)

    def _accumulate_gradient_covariance(
        self, grad_output: Tensor, module: Module, module_name: str
    ):
        """Accumulate the gradient covariance for a layer's output.

        Updates ``self._gradient_covariances``.

        Args:
            grad_output: The gradient w.r.t. the output.
            module: The layer whose output's gradient covariance will be accumulated.
            module_name: The name of the layer in the neural network.
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
        self._gradient_covariances = self._set_or_add_(
            self._gradient_covariances, module_name, covariance
        )

    def _hook_accumulate_input_covariance(
        self, module: Module, inputs: Tuple[Tensor], module_name: str
    ):
        """Pre-forward hook that accumulates the input covariance of a layer.

        Updates ``self._input_covariances``.

        Args:
            module: Module on which the hook is called.
            inputs: Inputs to the module.
            module_name: Name of the module in the neural network.

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
        if (
            "weight" in params.keys()
            and "bias" in params.keys()
            and not self._separate_weight_and_bias
        ):
            x = cat([x, x.new_ones(x.shape[0], 1)], dim=1)

        covariance = einsum(x, x, "b i,b j -> i j").div_(self._N_data * scale)
        self._input_covariances = self._set_or_add_(
            self._input_covariances, module_name, covariance
        )

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

    @property
    def trace(self) -> Tensor:
        r"""Trace of the KFAC approximation.

        Will call ``compute_kronecker_factors`` if it has not been called before and
        will cache the trace until ``compute_kronecker_factors`` is called again.
        Uses the property of the Kronecker product that
        :math:`\text{tr}(A \otimes B) = \text{tr}(A) \text{tr}(B)`.

        Returns:
            Trace of the KFAC approximation.
        """
        if self._trace is not None:
            return self._trace

        if not (self._input_covariances or self._gradient_covariances):
            self.compute_kronecker_factors()

        self._trace = 0.0
        for mod_name, param_pos in self._mapping.items():
            tr_ggT = self._gradient_covariances[mod_name].trace()
            if (
                not self._separate_weight_and_bias
                and "weight" in param_pos.keys()
                and "bias" in param_pos.keys()
            ):
                self._trace += self._input_covariances[mod_name].trace() * tr_ggT
            else:
                for p_name in param_pos.keys():
                    self._trace += tr_ggT * (
                        self._input_covariances[mod_name].trace()
                        if p_name == "weight"
                        else 1
                    )
        return self._trace

    @property
    def det(self) -> Tensor:
        r"""Determinant of the KFAC approximation.

        Will call ``compute_kronecker_factors`` if it has not been called before and
        will cache the determinant until ``compute_kronecker_factors`` is called again.
        Uses the property of the Kronecker product that
        :math:`\det(A \otimes B) = \det(A)^{m} \det(B)^{n}`,
        where
        :math:`A \in \mathbb{R}^{n \times n}` and :math:`B \in \mathbb{R}^{m \times m}`.

        Returns:
            Determinant of the KFAC approximation.
        """
        if self._det is not None:
            return self._det

        if not (self._input_covariances or self._gradient_covariances):
            self.compute_kronecker_factors()

        self._det = 1.0
        for mod_name, param_pos in self._mapping.items():
            m = self._gradient_covariances[mod_name].shape[0]
            det_ggT = self._gradient_covariances[mod_name].det()
            if (
                not self._separate_weight_and_bias
                and "weight" in param_pos.keys()
                and "bias" in param_pos.keys()
            ):
                n = self._input_covariances[mod_name].shape[0]
                det_aaT = self._input_covariances[mod_name].det()
                self._det *= det_aaT.pow(m) * det_ggT.pow(n)
            else:
                for p_name in param_pos.keys():
                    n = (
                        self._input_covariances[mod_name].shape[0]
                        if p_name == "weight"
                        else 1
                    )
                    self._det *= det_ggT.pow(n) * (
                        self._input_covariances[mod_name].det().pow(m)
                        if p_name == "weight"
                        else 1
                    )
        return self._det

    @property
    def logdet(self) -> Tensor:
        r"""Log determinant of the KFAC approximation.

        More numerically stable than the ``det`` property.
        Will call ``compute_kronecker_factors`` if it has not been called before and
        will cache the log determinant until ``compute_kronecker_factors`` is called
        again. Uses the property of the Kronecker product that
        :math:`\log \det(A \otimes B) = m \log \det(A) + n \log \det(B)`, where
        :math:`A \in \mathbb{R}^{n \times n}` and :math:`B \in \mathbb{R}^{m \times m}`.

        Returns:
            Log determinant of the KFAC approximation.
        """
        if self._logdet is not None:
            return self._logdet

        if not (self._input_covariances or self._gradient_covariances):
            self.compute_kronecker_factors()

        self._logdet = 0.0
        for mod_name, param_pos in self._mapping.items():
            m = self._gradient_covariances[mod_name].shape[0]
            logdet_ggT = self._gradient_covariances[mod_name].logdet()
            if (
                not self._separate_weight_and_bias
                and "weight" in param_pos.keys()
                and "bias" in param_pos.keys()
            ):
                n = self._input_covariances[mod_name].shape[0]
                logdet_aaT = self._input_covariances[mod_name].logdet()
                self._logdet += m * logdet_aaT + n * logdet_ggT
            else:
                for p_name in param_pos.keys():
                    n = (
                        self._input_covariances[mod_name].shape[0]
                        if p_name == "weight"
                        else 1
                    )
                    self._logdet += n * logdet_ggT + (
                        m * self._input_covariances[mod_name].logdet()
                        if p_name == "weight"
                        else 0
                    )
        return self._logdet

    @property
    def frobenius_norm(self) -> Tensor:
        r"""Frobenius norm of the KFAC approximation.

        Will call ``compute_kronecker_factors`` if it has not been called before and
        will cache the Frobenius norm until ``compute_kronecker_factors`` is called again.
        Uses the property of the Kronecker product that
        :math:`\|A \otimes B\|_F = \|A\|_F \|B\|_F`.

        Returns:
            Frobenius norm of the KFAC approximation.
        """
        if self._frobenius_norm is not None:
            return self._frobenius_norm

        if not (self._input_covariances or self._gradient_covariances):
            self.compute_kronecker_factors()

        self._frobenius_norm = 0.0
        for mod_name, param_pos in self._mapping.items():
            squared_frob_ggT = self._gradient_covariances[mod_name].square().sum()
            if (
                not self._separate_weight_and_bias
                and "weight" in param_pos.keys()
                and "bias" in param_pos.keys()
            ):
                squared_frob_aaT = self._input_covariances[mod_name].square().sum()
                self._frobenius_norm += squared_frob_aaT * squared_frob_ggT
            else:
                for p_name in param_pos.keys():
                    self._frobenius_norm += squared_frob_ggT * (
                        self._input_covariances[mod_name].square().sum()
                        if p_name == "weight"
                        else 1
                    )
        self._frobenius_norm.sqrt_()
        return self._frobenius_norm

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
            # Kronecker factors (if computed)
            "input_covariances": self._input_covariances,
            "gradient_covariances": self._gradient_covariances,
            # Properties (not necessarily computed)
            "trace": self._trace,
            "det": self._det,
            "logdet": self._logdet,
            "frobenius_norm": self._frobenius_norm,
        }

    def _check_if_keys_match_mapping_keys(self, dictionary: dict):
        """Check if the keys of a dictionary match the mapping keys of the linear operator.

        Args:
            dictionary: Dictionary to check.

        Raises:
            ValueError: If the keys do not match the mapping keys.
        """
        dictionary_keys = set(dictionary.keys())
        mapping_keys = set(self._mapping.keys())
        if dictionary_keys and dictionary_keys != mapping_keys:
            raise ValueError(
                "Keys in dictionary do not match mapping keys of linear operator. "
                f"Difference: {dictionary_keys - mapping_keys}."
            )

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

        # Set Kronecker factors (if computed)
        self._check_if_keys_match_mapping_keys(state_dict["input_covariances"])
        self._check_if_keys_match_mapping_keys(state_dict["gradient_covariances"])
        self._input_covariances = state_dict["input_covariances"]
        self._gradient_covariances = state_dict["gradient_covariances"]

        # Set properties (not necessarily computed)
        self._trace = state_dict["trace"]
        self._det = state_dict["det"]
        self._logdet = state_dict["logdet"]
        self._frobenius_norm = state_dict["frobenius_norm"]

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
