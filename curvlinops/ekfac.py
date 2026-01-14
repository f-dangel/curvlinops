"""Contains LinearOperator implementation of EKFAC approximation of the Fisher/GGN."""

from __future__ import annotations

from collections.abc import MutableMapping
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from einops import einsum, rearrange
from torch import Generator, Tensor, cat
from torch.linalg import eigh
from torch.nn import (
    BCEWithLogitsLoss,
    Conv2d,
    CrossEntropyLoss,
    Module,
    MSELoss,
    Parameter,
)
from torch.utils.hooks import RemovableHandle

from curvlinops.kfac import (
    FisherType,
    KFACLinearOperator,
    KFACType,
)
from curvlinops.kfac_utils import extract_patches


def compute_eigenvalue_correction_linear_weight_sharing(
    g: Tensor,
    ggT_eigvecs: Tensor,
    a: Tensor,
    aaT_eigvecs: Union[Tensor, None],
    _force_strategy: Optional[str] = None,
) -> Tensor:
    r"""Computes eigenvalue corrections for a linear layer with weight sharing.

    Chooses between two computational strategies depending on memory requirements.

    Args:
        g: Output gradients of the layer with shape
            ``[N, S, D1]``, where ``N`` is the batch size, ``S`` the weight sharing
            dimension, and ``D1`` the output dimension.
        ggT_eigvecs: Eigenvectors of the gradient covariance with shape
            ``[D1, D1]``.
        a: Layer inputs with shape ``[N, S, D2]``, where ``D2`` is the input dimension.
        aaT_eigvecs: Eigenvectors of the input covariance with shape
            ``[D2, D2]`` or ``None`` if the layer has no weights (bias only).
        _force_strategy: If specified, forces the use of either ``'gramian'`` or
            ``'per_example_gradients'`` strategy. If ``None``, the strategy is chosen
            based on memory requirements. Defaults to ``None``. This flag serves mainly
            for testing purposes.

    Returns:
        The eigencorrection with shape ``[D1, D2]`` (or ``[D1]`` for the bias case).

    Raises:
        ValueError: If an invalid ``_force_strategy`` is provided.

    Below we explain the mathematical details of what this function does. The mapping is
    is as follows: (``g``, :math:`\mathbf{Y}`), (``ggT_eigvecs``, :math:`\mathbf{Q}_1`),
    (``a``, :math:`\mathbf{X}`), (``aaT_eigvecs``, :math:`\mathbf{Q}_2`).

    Note:
        **Introduction:** In the following, let :math:`D_1` be the output dimension
        of the layer, :math:`D_2` the input dimension, :math:`S` the weight sharing
        dimension, and :math:`N` the batch size.

        Given the layer inputs :math:`\mathbf{X} \in \mathbb{R}^{N \times S \times D_2}`,
        output gradients :math:`\mathbf{Y} \in \mathbb{R}^{N \times S \times D_1}`, and a
        Kronecker-factored basis :math:`\mathbf{Q}_1 \otimes \mathbf{Q}_2` with factors
        :math:`\mathbf{Q}_i \in \mathbb{R}^{D_i \times D_i}`, our goal is to compute the
        eigencorrection :math:`\mathbf{E} \in \mathbb{R}^{D_1 \times D_2}` which has the
        same shape as the layer's weights.

        The common way to do that is to compute the per-example gradients
        :math:`\mathbf{G} \in \mathbb{R}^{N \times D_1 \times D_2}` with

        .. math::
            \mathbf{G}_{n,d_1,d_2}
            =
            \sum_s \mathbf{Y}_{n,s,d_1} \mathbf{X}_{n,s,d_2},

        rotate them into the Kronecker-factored basis,

        .. math::
            \mathbf{\tilde{G}}_{n,d_1,d_2}
            =
            \sum_{i,j} \mathbf{G}_{n,i,j} \mathbf{Q}_{1,i,d_1} \mathbf{Q}_{2,j,d_2},

        and compute the correction by squaring and summing out the batch dimension,

        .. math::
            \mathbf{E}_{d_1,d_2}
            =
            \sum_{n} \mathbf{\tilde{G}}_{n,d_1,d_2}^2.

        Building up the per-example gradients can be extremely memory-costly.
        Therefore, we also consider an alternative approach which can have smaller
        memory footprint if the weight sharing is mild.

    Note:
        **(1) Cost analysis of per-example gradient approach:** The peak memory of
        building up per-example gradients is dominated by :math:`N D_1 D_2`.

        We have two options to compute the rotated per-example gradient.

        1. First compute :math:`\mathbf{G}` and then rotate it. The first step costs
           :math:`N S D_1 D_2` time and the rotation costs
           :math:`N D_1 D_2 (D_1 + D_2)` time.
        2. Rotate the activations and output gradients, then compute the rotated
           per-example gradient. The rotations cost :math:`N S (D_1^2 + D_2^2)` time
           The last step is :math:`N S D_1 D_2` time.

        So in practise, we should prefer the first approach over the second if

        .. math::
            D_1 D_2 (D_1 + D_2) < S (D_1^2 + D_2^2).

        In the implementation, ``opt-einsum`` will automatically do that for us.

        Adding the cost for squaring and contracting, the overall cost is
        :math:`N S D_1 D_2 + N \min(S (D_1^2 + D_2^2), D_1 D_2 (D_1 + D_2)) + 2 N D_1 D_2`.

    Note:
        **(2) Cost analysis of Gramian contraction approach:** A way to avoid building
        up per-example gradients is to write the eigencorrection as big contraction of
        the rotated activations :math:`\mathbf{\tilde{X}}, \mathbf{\tilde{Y}}` and then
        rearrange the contractions such that the batch dimension can be directly summed:

        .. math::
            \mathbf{E}_{d_1,d_2}
            =
            \sum_{n}
            \left(
            \sum_{s} \mathbf{\tilde{Y}}_{n,s,d_1} \mathbf{\tilde{X}}_{n,s,d_2}
            \right)
            \left(
            \sum_{t} \mathbf{\tilde{Y}}_{n,t,d_1} \mathbf{\tilde{X}}_{n,t,d_2}
            \right)
            \\
            =
            \sum_{n} \sum_{s} \sum_{t}
            \left(
            \mathbf{\tilde{Y}}_{n,s,d_1} \mathbf{\tilde{Y}}_{n,t,d_1}
            \right)
            \left(
            \mathbf{\tilde{X}}_{n,s,d_2} \mathbf{\tilde{X}}_{n,t,d_2}
            \right)

        This requires building up the Gramians

        .. math::
            \mathbf{G^Y}_{n,s,t,d_1}
            =
            \mathbf{\tilde{Y}}_{n,s,d_1} \mathbf{\tilde{Y}}_{n,t,d_1},
            \\
            \mathbf{G^X}_{n,s,t,d_2}
            =
            \mathbf{\tilde{X}}_{n,s,d_2} \mathbf{\tilde{X}}_{n,t,d_2}.

        Peak memory is dominated by :math:`N S^2 (D_1 + D_2)`.
        The time is :math:`N S (D_1^2 + D_2^2)` for the rotations,
        :math:`N S^2 (D_1 + D_2)` for building up the Gramians, and
        :math:`N S^2 D_1 D_2` for the final contraction.
        In total, this is :math:`N S (D_1^2 + D_2^2) + N S^2 (D_1 + D_2 + D_1 D_2)`.

    **We select the approach with the smaller memory footprint**, i.e. Gramian
    contraction if :math:`S^2 (D_1 + D_2) < D_1 D_2`, and squaring per-example
    gradients otherwise. So generally speaking, the more weight sharing, the
    better building up per-example gradients will be.

    In the extreme case :math:`S=1` (no weight sharing), the Gramian
    contraction approach uses only :math:`N (D_1 + D_2) < N D_1 D_2` memory
    compared to the per-example gradient approach. In terms of time, the
    Gramian contraction uses :math:`N (D_1^2 + D_2^2 + D_1 + D_2 + D_1 D_2) < N
    (3 D_1 D_2 + D_1^2 + D_2^2)` compared to the per-example gradient.
    """
    strategies = {"gramian", "per_example_gradients", None}
    if _force_strategy not in strategies:
        raise ValueError(
            f"Invalid _force_strategy: {_force_strategy}. Supported: {strategies}."
        )

    Q1, Q2 = ggT_eigvecs, aaT_eigvecs
    Y, X = g, a

    if Q2 is None:  # -> 1d (bias) case
        eigencorrection = (
            einsum(Q1, Y, "j d1, batch shared j -> batch d1").square_().sum(0)
        )

    else:  # -> 2d (weight or weight+bias) case
        # Determine approach: Gramian contraction or per-example gradients
        (_, S, D1), (_, _, D2) = g.shape, a.shape

        # Determine approach based on _force_strategy or memory requirements
        use_gramian = (
            _force_strategy == "gramian"
            if _force_strategy is not None
            # Memory of per-example gradients is dominated by N * D1 * D2
            # Memory of Gramian contraction is dominated by N * S^2 * (D1 + D2)
            # We choose the approach that requires less memory.
            else S**2 * (D1 + D2) < D1 * D2
        )

        if use_gramian:  # -> Gramian approach
            X_rot = einsum(X, Q2, "batch shared j, j d2 -> batch shared d2")
            Y_rot = einsum(Y, Q1, "batch shared i, i d1 -> batch shared d1")
            # In the absence of weight sharing (S=1), this simply computes
            # (Q^T X_rot)^2 and (Q^T Y_rot)^2, then computes the correction
            X_gram = einsum(X_rot, X_rot, "batch s d2, batch t d2 -> batch s t d2")
            Y_gram = einsum(Y_rot, Y_rot, "batch s d1, batch t d1 -> batch s t d1")
            eigencorrection = einsum(
                Y_gram, X_gram, "batch s t d1, batch s t d2 -> d1 d2"
            )

        else:  # -> per-example gradient approach
            rotated_per_example_gradient = einsum(
                Q1,
                Y,
                X,
                Q2,
                "i d1, batch shared i, batch shared j, j d2 -> batch d1 d2",
            )
            eigencorrection = rotated_per_example_gradient.square_().sum(dim=0)

    return eigencorrection


class EKFACLinearOperator(KFACLinearOperator):
    """Linear operator to multiply with the Fisher/GGN's EKFAC approximation.

    Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EKFAC) was originally
    introduced in

    - George, T., Laurent, C., Bouthillier, X., Ballas, N., Vincent, P. (2018).
      Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis (NeurIPS)

    and concurrently in the context of continual learning in

    - Liu, X., Masana, M., Herranz, L., Van de Weijer, J., Lopez, A., Bagdanov, A. (2018).
      Rotate your networks: Better weight consolidation and less catastrophic forgetting
      (ICPR).

    Attributes:
        _SUPPORTED_FISHER_TYPE: Tuple with supported Fisher types.
    """

    _SUPPORTED_FISHER_TYPE: Tuple[FisherType] = (
        FisherType.TYPE2,
        FisherType.MC,
        FisherType.EMPIRICAL,
    )

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
        batch_size_fn: Optional[Callable[[MutableMapping], int]] = None,
    ):
        """Eigenvalue-corrected KFAC (EKFAC) proxy of the Fisher/GGN.

        Warning:
            If the model's parameters change, e.g. during training, you need to
            create a fresh instance of this object. This is because, for performance
            reasons, the Kronecker factors are computed once and cached during the
            first matrix-vector product. They will thus become outdated if the model
            changes.

        Warning:
            This is an early proto-type with limitations:
                - Only Linear and Conv2d modules are supported.
                - Only models with 2d output are supported.

        Args:
            model_func: The neural network. Must consist of modules.
            loss_func: The loss function.
            params: The parameters defining the Fisher/GGN that will be approximated
                through EKFAC.
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
                sometimes also called type-2 Fisher.
                If ``FisherType.MC``, the expectation is approximated by sampling
                ``mc_samples`` labels from the model's predictive distribution.
                If ``FisherType.EMPIRICAL``, the empirical gradients are used which
                corresponds to the uncentered gradient covariance/empirical Fisher.
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
        """
        super().__init__(
            model_func=model_func,
            loss_func=loss_func,
            params=params,
            data=data,
            progressbar=progressbar,
            check_deterministic=False,
            seed=seed,
            fisher_type=fisher_type,
            mc_samples=mc_samples,
            kfac_approx=kfac_approx,
            num_per_example_loss_terms=num_per_example_loss_terms,
            separate_weight_and_bias=separate_weight_and_bias,
            num_data=num_data,
            batch_size_fn=batch_size_fn,
        )

        # Initialize the eigenvectors of the Kronecker factors
        self._input_covariances_eigenvectors: Dict[str, Tensor] = {}
        self._gradient_covariances_eigenvectors: Dict[str, Tensor] = {}
        # Initialize the cache for activations
        self._cached_activations: Dict[str, Tensor] = {}
        # Initialize the corrected eigenvalues for EKFAC
        self._corrected_eigenvalues: Dict[str, Union[Tensor, Dict[str, Tensor]]] = {}

        if check_deterministic:
            self._check_deterministic()

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
            The rearranged outputs and targets.

        Raises:
            ValueError: If the output is not 2d and y is not 1d/2d.
        """
        # Our individual gradient implementation for EKFAC does not support computing
        # the individual gradients for any loss terms that might dependent on each other,
        # i.e., loss terms other than the per-data point loss terms.
        if output.ndim != 2 or y.ndim not in {1, 2}:
            raise ValueError(
                "Only 2d output and 1d/2d target are supported for EKFAC. "
                f"Got {output.ndim=} and {y.ndim=}."
            )
        return output, y

    def _maybe_compute_ekfac(self):
        """Compute the EKFAC approximation when necessary."""
        if not self._corrected_eigenvalues:
            if not (self._input_covariances or self._gradient_covariances):
                self.compute_kronecker_factors()
            self.compute_eigenvalue_correction()

    def _matmat(self, M: List[Tensor]) -> List[Tensor]:
        """Apply EKFAC to a matrix (multiple vectors) in tensor list format.

        This allows for matrix-matrix products with the EKFAC approximation in PyTorch
        without converting tensors to numpy arrays, which avoids unnecessary
        device transfers when working with GPUs and flattening/concatenating.

        Args:
            M: Matrix for multiplication in tensor list format. Each entry has the
                same shape as a parameter with an additional trailing dimension of size
                ``K`` for the columns, i.e. ``[(*p1.shape, K), (*p2.shape, K), ...]``.

        Returns:
            Matrix-multiplication result ``EKFAC @ M`` in tensor list format. Has the same
            shapes as the input.
        """
        self._maybe_compute_ekfac()

        KM: List[Tensor | None] = [None] * len(M)

        for mod_name, param_pos in self._mapping.items():
            # cache the weight shape to ensure correct shapes are returned
            if "weight" in param_pos:
                weight_shape = M[param_pos["weight"]].shape

            # Get the EKFAC approximation components for the current module
            # aaT_eigenvectors does not exist if the weight matrix is excluded
            aaT_eigenvectors = self._input_covariances_eigenvectors.get(mod_name)
            # ggT_eigenvectors and corrected_eigenvalues always exists
            ggT_eigenvectors = self._gradient_covariances_eigenvectors[mod_name]
            corrected_eigenvalues = self._corrected_eigenvalues[mod_name]

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
                M_joint = self._left_and_right_multiply(
                    M_joint, aaT_eigenvectors, ggT_eigenvectors, corrected_eigenvalues
                )
                w_cols = M_w.shape[1]
                KM[w_pos], KM[b_pos] = M_joint.split([w_cols, 1], dim=-2)
                KM[b_pos].squeeze_(1)
            else:
                self._separate_left_and_right_multiply(
                    KM,
                    M,
                    param_pos,
                    aaT_eigenvectors,
                    ggT_eigenvectors,
                    corrected_eigenvalues,
                )

            # restore original shapes
            if "weight" in param_pos:
                KM[param_pos["weight"]] = KM[param_pos["weight"]].view(weight_shape)

        return KM

    def _compute_eigenvectors(self):
        """Compute the eigenvectors of the KFAC approximation."""
        if not (self._input_covariances or self._gradient_covariances):
            self.compute_kronecker_factors()

        for mod_name in self._mapping.keys():
            for source, destination in zip(
                (self._input_covariances, self._gradient_covariances),
                (
                    self._input_covariances_eigenvectors,
                    self._gradient_covariances_eigenvectors,
                ),
            ):
                factor = source.pop(mod_name, None)
                if factor is not None:
                    destination[mod_name] = eigh(factor).eigenvectors

    def compute_eigenvalue_correction(self):
        """Compute and cache the corrected eigenvalues for EKFAC."""
        self._reset_matrix_properties()

        # Compute the eigenvectors of the KFAC approximation
        if not (
            self._input_covariances_eigenvectors
            or self._gradient_covariances_eigenvectors
        ):
            self._compute_eigenvectors()

        # install forward and backward hooks
        hook_handles: List[RemovableHandle] = []

        for mod_name, param_pos in self._mapping.items():
            module = self._model_func.get_submodule(mod_name)

            # cache activations for computing per-example gradients
            if "weight" in param_pos.keys():
                hook_handles.append(
                    module.register_forward_pre_hook(
                        partial(self._hook_cache_inputs, module_name=mod_name)
                    )
                )

            # compute the corrected eigenvalues using the per-example gradients
            hook_handles.append(
                module.register_forward_hook(
                    partial(
                        self._register_tensor_hook_on_output_to_accumulate_corrected_eigenvalues,
                        module_name=mod_name,
                    )
                )
            )

        if self._generator is None or self._generator.device != self.device:
            self._generator = Generator(device=self.device)
        self._generator.manual_seed(self._seed)

        # loop over data set, computing the corrected eigenvalues
        for X, y in self._loop_over_data(desc="Eigenvalue correction"):
            output = self._model_func(X)
            output, y = self._rearrange_for_larger_than_2d_output(output, y)
            self._compute_loss_and_backward(output, y)

        # Clear the cached activations
        self._cached_activations.clear()

        # clean up
        for handle in hook_handles:
            handle.remove()

    def _hook_cache_inputs(
        self, module: Module, inputs: Tuple[Tensor], module_name: str
    ):
        """Pre-forward hook that caches the inputs of a layer.

        Updates ``self._cached_activations``.

        Args:
            module: Module on which the hook is called.
            inputs: Inputs to the module.
            module_name: Name of the module in the neural network.

        Raises:
            ValueError: If the module has multiple inputs.
        """
        if len(inputs) != 1:
            raise ValueError("Modules with multiple inputs are not supported.")
        self._cached_activations[module_name] = inputs[0].data.detach()

    def _register_tensor_hook_on_output_to_accumulate_corrected_eigenvalues(
        self, module: Module, inputs: Tuple[Tensor], output: Tensor, module_name: str
    ):
        """Register tensor hook on layer's output to accumulate the corrected eigenvalues.

        Note:
            The easier way to compute the corrected eigenvalues would be via a full
            backward hook on the module itself which performs the computation.
            However, this approach breaks down if the output of a layer feeds into an
            activation with `inplace=True` (see
            https://github.com/pytorch/pytorch/issues/61519). Hence we use the
            workaround
            https://github.com/pytorch/pytorch/issues/61519#issuecomment-883524237, and
            install a module hook which installs a tensor hook on the module's output
            tensor, which performs the accumulation of the gradient covariance.

        Args:
            module: Layer onto whose output a tensor hook to accumulate the corrected
                eigenvalues will be installed.
            inputs: The layer's input tensors.
            output: The layer's output tensor.
            module_name: The name of the layer in the neural network.
        """
        tensor_hook = partial(
            self._accumulate_corrected_eigenvalues,
            module=module,
            module_name=module_name,
        )
        output.register_hook(tensor_hook)

    def _accumulate_corrected_eigenvalues(
        self, grad_output: Tensor, module: Module, module_name: str
    ):
        r"""Accumulate the corrected eigenvalues.

        The corrected eigenvalues are computed as
        :math:`\lambda_{\text{corrected}} = (Q_g^T G Q_a)^2`, where
        :math:`Q_a` and :math:`Q_g` are the eigenvectors of the input and gradient
        covariances, respectively, and ``G`` is the gradient matrix. The corrected
        eigenvalues are used to correct the eigenvalues of the KFAC approximation
        (EKFAC).

        Updates ``self._corrected_eigenvalues``.

        Args:
            grad_output: The gradient w.r.t. the output.
            module: The layer for which corrected eigenvalues will be accumulated.
            module_name: The name of the layer in the neural network.
        """
        g = grad_output.data.detach()
        batch_size = g.shape[0]
        if isinstance(module, Conv2d):
            g = rearrange(g, "batch c o1 o2 -> batch o1 o2 c")
        g = rearrange(g, "batch ... d_out -> batch (...) d_out")

        # Compute correction for the loss scaling depending on the loss reduction used
        num_loss_terms = batch_size * self._num_per_example_loss_terms
        # self._mc_samples will be 1 if fisher_type != FisherType.MC
        correction = {
            "sum": 1.0 / self._mc_samples,
            "mean": num_loss_terms**2
            / (self._N_data * self._mc_samples * self._num_per_example_loss_terms),
        }[self._loss_func.reduction]

        # Compute the corrected eigenvalues for the EKFAC approximation
        param_pos = self._mapping[module_name]
        # aaT_eigenvectors does not exist if the weight matrix of the module is excluded
        aaT_eigenvectors = self._input_covariances_eigenvectors.get(module_name)
        # ggT_eigenvectors always exists
        ggT_eigenvectors = self._gradient_covariances_eigenvectors[module_name]

        # Rearrange the activations for computing per-example gradients
        activations = self._cached_activations.get(module_name)
        if activations is not None:
            if isinstance(module, Conv2d):
                activations = extract_patches(
                    activations,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                )
            activations = rearrange(activations, "batch ... d_in -> batch (...) d_in")

        if (
            not self._separate_weight_and_bias
            and "weight" in param_pos.keys()
            and "bias" in param_pos.keys()
        ):
            a_augmented = cat(
                [activations, activations.new_ones(*activations.shape[:-1], 1)], dim=-1
            )
            eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                g, ggT_eigenvectors, a_augmented, aaT_eigenvectors
            )
            self._corrected_eigenvalues = self._set_or_add_(
                self._corrected_eigenvalues,
                module_name,
                eigencorrection.mul_(correction),
            )

        else:
            if module_name not in self._corrected_eigenvalues:
                self._corrected_eigenvalues[module_name] = {}
            for p_name, pos in param_pos.items():
                eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                    g,
                    ggT_eigenvectors,
                    activations,
                    aaT_eigvecs=None if p_name == "bias" else aaT_eigenvectors,
                )
                self._corrected_eigenvalues[module_name] = self._set_or_add_(
                    self._corrected_eigenvalues[module_name],
                    pos,
                    eigencorrection.mul_(correction),
                )

    @property
    def trace(self) -> Tensor:
        r"""Trace of the EKFAC approximation.

        Will call ``compute_kronecker_factors`` and ``compute_eigenvalue_correction`` if
        either of them has not been called before and will cache the trace until one of
        them is called again.

        Returns:
            Trace of the EKFAC approximation.
        """
        if self._trace is not None:
            return self._trace

        self._maybe_compute_ekfac()

        # Compute the trace using the corrected eigenvalues
        self._trace = 0.0
        for corrected_eigenvalues in self._corrected_eigenvalues.values():
            if isinstance(corrected_eigenvalues, dict):
                for val in corrected_eigenvalues.values():
                    self._trace += val.sum()
            else:
                self._trace += corrected_eigenvalues.sum()

        return self._trace

    @property
    def det(self) -> Tensor:
        r"""Determinant of the EKFAC approximation.

        Will call ``compute_kronecker_factors`` and ``compute_eigenvalue_correction`` if
        either of them has not been called before and will cache the determinant until
        one of them is called again.

        Returns:
            Determinant of the EKFAC approximation.
        """
        if self._det is not None:
            return self._det

        self._maybe_compute_ekfac()

        # Compute the determinant using the corrected eigenvalues
        self._det = 1.0
        for corrected_eigenvalues in self._corrected_eigenvalues.values():
            if isinstance(corrected_eigenvalues, dict):
                for val in corrected_eigenvalues.values():
                    self._det *= val.prod()
            else:
                self._det *= corrected_eigenvalues.prod()

        return self._det

    @property
    def logdet(self) -> Tensor:
        r"""Log determinant of the EKFAC approximation.

        More numerically stable than the ``det`` property.
        Will call ``compute_kronecker_factors`` and ``compute_eigenvalue_correction`` if
        either of them has not been called before and will cache the logdet until one of
        them is called again.

        Returns:
            Log determinant of the EKFAC approximation.
        """
        if self._logdet is not None:
            return self._logdet

        self._maybe_compute_ekfac()

        # Compute the log determinant using the corrected eigenvalues
        self._logdet = 0.0
        for corrected_eigenvalues in self._corrected_eigenvalues.values():
            if isinstance(corrected_eigenvalues, dict):
                for val in corrected_eigenvalues.values():
                    self._logdet += val.log().sum()
            else:
                self._logdet += corrected_eigenvalues.log().sum()

        return self._logdet

    @property
    def frobenius_norm(self) -> Tensor:
        r"""Frobenius norm of the EKFAC approximation.

        Will call ``compute_kronecker_factors`` and ``compute_eigenvalue_correction`` if
        either of them has not been called before and will cache the Frobenius norm
        until one of them is called again.

        Returns:
            Frobenius norm of the EKFAC approximation.
        """
        if self._frobenius_norm is not None:
            return self._frobenius_norm

        self._maybe_compute_ekfac()

        # Compute the Frobenius norm using the corrected eigenvalues
        self._frobenius_norm = 0.0
        for corrected_eigenvalues in self._corrected_eigenvalues.values():
            if isinstance(corrected_eigenvalues, dict):
                for val in corrected_eigenvalues.values():
                    self._frobenius_norm += val.square().sum()
            else:
                self._frobenius_norm += corrected_eigenvalues.square().sum()

        return self._frobenius_norm.sqrt_()

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the EKFAC linear operator.

        Returns:
            State dictionary.
        """
        state_dict = super().state_dict()
        # Add quantities specifically for EKFAC (if computed)
        state_dict.update(
            {
                "input_covariances_eigenvectors": self._input_covariances_eigenvectors,
                "gradient_covariances_eigenvectors": self._gradient_covariances_eigenvectors,
                "cached_activations": self._cached_activations,
                "corrected_eigenvalues": self._corrected_eigenvalues,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the state of the EKFAC linear operator.

        Args:
            state_dict: State dictionary.
        """
        super().load_state_dict(state_dict)

        # Set EKFAC-specific quantities
        self._check_if_keys_match_mapping_keys(
            state_dict["input_covariances_eigenvectors"]
        )
        self._check_if_keys_match_mapping_keys(
            state_dict["gradient_covariances_eigenvectors"]
        )
        self._input_covariances_eigenvectors = state_dict[
            "input_covariances_eigenvectors"
        ]
        self._gradient_covariances_eigenvectors = state_dict[
            "gradient_covariances_eigenvectors"
        ]
        self._cached_activations = state_dict["cached_activations"]
        self._corrected_eigenvalues = state_dict["corrected_eigenvalues"]
