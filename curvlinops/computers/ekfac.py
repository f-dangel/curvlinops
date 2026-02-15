"""Computer for the Fisher/GGN's eigenvalue-corrected KFAC (EKFAC) approximation."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

from einops import einsum, rearrange
from torch import Tensor, cat
from torch.linalg import eigh
from torch.nn import Conv2d, Module
from torch.utils.hooks import RemovableHandle

from curvlinops.computers.kfac import FisherType, KFACComputer
from curvlinops.kfac_utils import extract_patches
from curvlinops.utils import _seed_generator


def compute_eigenvalue_correction_linear_weight_sharing(
    g: Tensor,
    ggT_eigvecs: Tensor,
    a: Union[Tensor, None],
    aaT_eigvecs: Union[Tensor, None],
    _force_strategy: Optional[str] = None,
) -> Tensor:
    r"""Compute eigenvalue corrections for a linear layer with weight sharing.

    Chooses between two computational strategies depending on memory requirements.

    Args:
        g: Output gradients of the layer with shape
            ``[N, S, D1]``, where ``N`` is the batch size, ``S`` the weight sharing
            dimension, and ``D1`` the output dimension.
        ggT_eigvecs: Eigenvectors of the gradient covariance with shape
            ``[D1, D1]``.
        a: Layer inputs with shape ``[N, S, D2]``, where ``D2`` is the input dimension
            or ``None`` if the layer has no weights (bias only). In that case,
            `aaT_eigvecs` has to be `None`, too.
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
        ValueError: If only one of ``a`` and ``aaT_eigvecs`` is ``None``.

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
    if (a is None and aaT_eigvecs is not None) or (
        a is not None and aaT_eigvecs is None
    ):
        raise ValueError(
            "Both (a, aaT_eigvecs) must be None or Tensor. "
            + f"Got {(type(a), type(aaT_eigvecs))}."
        )

    Q1, Q2 = ggT_eigvecs, aaT_eigvecs
    Y, X = g, a

    if Q2 is None and X is None:  # -> 1d (bias) case
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


class EKFACComputer(KFACComputer):
    """Computes EKFAC's eigenvalue-corrected Kronecker factors for the Fisher/GGN.

    Extends :class:`KFACComputer` with eigenvalue decomposition of the Kronecker
    factors and eigenvalue correction computation.

    Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EKFAC) was originally
    introduced in

    - George, T., Laurent, C., Bouthillier, X., Ballas, N., Vincent, P. (2018).
      Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis
      (NeurIPS)

    and concurrently in the context of continual learning in

    - Liu, X., Masana, M., Herranz, L., Van de Weijer, J., Lopez, A., Bagdanov, A.
      (2018). Rotate your networks: Better weight consolidation and less catastrophic
      forgetting (ICPR).

    Attributes:
        _SUPPORTED_FISHER_TYPE: Tuple of supported Fisher types.
    """

    _SUPPORTED_FISHER_TYPE: Tuple[FisherType, ...] = (
        FisherType.TYPE2,
        FisherType.MC,
        FisherType.EMPIRICAL,
    )

    def compute(
        self,
    ) -> Tuple[
        Dict[str, Tensor],
        Dict[str, Tensor],
        Dict[str, Union[Tensor, Dict[int, Tensor]]],
        Dict[str, Dict[str, int]],
    ]:
        """Compute eigenvalue-corrected Kronecker factors.

        Returns:
            Tuple of ``(input_covariance_eigenvectors, gradient_covariance_eigenvectors,
            corrected_eigenvalues, mapping)`` where the first two are dictionaries
            mapping module names to eigenvector matrices, the third maps module names to
            eigenvalue corrections, and ``mapping`` maps module names to dictionaries of
            parameter names and their positions.
        """
        input_covariances, gradient_covariances, mapping = super().compute()
        input_covariances = self._eigenvectors_(input_covariances)
        gradient_covariances = self._eigenvectors_(gradient_covariances)
        corrected_eigenvalues = self.compute_eigenvalue_correction(
            input_covariances, gradient_covariances
        )
        return input_covariances, gradient_covariances, corrected_eigenvalues, mapping

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

    @staticmethod
    def _eigenvectors_(dictionary: Dict[Any, Tensor]) -> Dict[Any, Tensor]:
        """Replace all matrix values with their eigenvectors (inplace).

        Args:
            dictionary: A dictionary mapping module names to square matrices.

        Returns:
            The modified dictionary mapping module names to the eigenvectors of the
            input matrices.
        """
        for key, value in dictionary.items():
            dictionary[key] = eigh(value).eigenvectors

        return dictionary

    def compute_eigenvalue_correction(
        self,
        input_covariances_eigenvectors: Dict[str, Tensor],
        gradient_covariances_eigenvectors: Dict[str, Tensor],
    ) -> Dict[str, Union[Tensor, Dict[int, Tensor]]]:
        """Compute the corrected eigenvalues for EKFAC.

        Args:
            input_covariances_eigenvectors: Dictionary mapping module names to input
                covariance eigenvectors.
            gradient_covariances_eigenvectors: Dictionary mapping module names to
                gradient covariance eigenvectors.

        Returns:
            Dictionary containing corrected eigenvalues for each module.
        """
        # Create empty dictionary to be populated by hooks
        corrected_eigenvalues: Dict[str, Union[Tensor, Dict[int, Tensor]]] = {}

        # install forward hooks
        hook_handles: List[RemovableHandle] = []

        for mod_name in self._mapping:
            module = self._model_func.get_submodule(mod_name)

            # compute the corrected eigenvalues using the per-example gradients
            hook_handles.append(
                module.register_forward_hook(
                    partial(
                        self._register_tensor_hook_on_output_to_accumulate_corrected_eigenvalues,
                        module_name=mod_name,
                        input_covariances_eigenvectors=input_covariances_eigenvectors,
                        gradient_covariances_eigenvectors=gradient_covariances_eigenvectors,
                        corrected_eigenvalues=corrected_eigenvalues,
                    )
                )
            )

        self._generator = _seed_generator(self._generator, self.device, self._seed)

        # loop over data set, computing the corrected eigenvalues
        for X, y in self._loop_over_data(desc="Eigenvalue correction"):
            output = self._model_func(X)
            output, y = self._rearrange_for_larger_than_2d_output(output, y)
            self._compute_loss_and_backward(output, y)

        # clean up
        for handle in hook_handles:
            handle.remove()

        return corrected_eigenvalues

    def _register_tensor_hook_on_output_to_accumulate_corrected_eigenvalues(
        self,
        module: Module,
        inputs: Tuple[Tensor],
        output: Tensor,
        module_name: str,
        input_covariances_eigenvectors: Dict[str, Tensor],
        gradient_covariances_eigenvectors: Dict[str, Tensor],
        corrected_eigenvalues: Dict[str, Union[Tensor, Dict[int, Tensor]]],
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
            input_covariances_eigenvectors: Dictionary containing input covariance
                eigenvectors.
            gradient_covariances_eigenvectors: Dictionary containing gradient
                covariance eigenvectors.
            corrected_eigenvalues: Dictionary to store corrected eigenvalues.
        """
        tensor_hook = partial(
            self._accumulate_corrected_eigenvalues,
            module=module,
            module_name=module_name,
            input_covariances_eigenvectors=input_covariances_eigenvectors,
            gradient_covariances_eigenvectors=gradient_covariances_eigenvectors,
            corrected_eigenvalues=corrected_eigenvalues,
            inputs=inputs,
        )
        output.register_hook(tensor_hook)

    def _accumulate_corrected_eigenvalues(
        self,
        grad_output: Tensor,
        module: Module,
        module_name: str,
        input_covariances_eigenvectors: Dict[str, Tensor],
        gradient_covariances_eigenvectors: Dict[str, Tensor],
        corrected_eigenvalues: Dict[str, Union[Tensor, Dict[int, Tensor]]],
        inputs: Tuple[Tensor],
    ):
        r"""Accumulate the corrected eigenvalues.

        The corrected eigenvalues are computed as
        :math:`\lambda_{\text{corrected}} = (Q_g^T G Q_a)^2`, where
        :math:`Q_a` and :math:`Q_g` are the eigenvectors of the input and gradient
        covariances, respectively, and ``G`` is the gradient matrix. The corrected
        eigenvalues are used to correct the eigenvalues of the KFAC approximation
        (EKFAC).

        Updates the provided ``corrected_eigenvalues`` dictionary.

        Args:
            grad_output: The gradient w.r.t. the output.
            module: The layer for which corrected eigenvalues will be accumulated.
            module_name: The name of the layer in the neural network.
            input_covariances_eigenvectors: Dictionary containing input covariance
                eigenvectors.
            gradient_covariances_eigenvectors: Dictionary containing gradient
                covariance eigenvectors.
            corrected_eigenvalues: Dictionary to store corrected eigenvalues.
            inputs: A tuple containing the layer's inputs.

        Raises:
            ValueError: If the module has multiple inputs.
        """
        g = grad_output.data.detach()
        batch_size = g.shape[0]
        if isinstance(module, Conv2d):
            g = rearrange(g, "batch c o1 o2 -> batch o1 o2 c")
        g = rearrange(g, "batch ... d_out -> batch (...) d_out")

        # We only need layer inputs to extract information w.r.t. the weights
        param_pos = self._mapping[module_name]
        a_required = "weight" in param_pos

        if len(inputs) != 1:
            raise ValueError("Modules with multiple inputs are not supported.")
        a = inputs[0].data.detach() if a_required else None

        if a_required:
            # Perform patch extraction for convolution
            if isinstance(module, Conv2d):
                a = extract_patches(
                    a,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                )
            # Rearrange the activations for computing per-example gradients
            a = rearrange(a, "batch ... d_in -> batch (...) d_in")

        # Compute correction for the loss scaling depending on the loss reduction used
        num_loss_terms = batch_size * self._num_per_example_loss_terms
        correction = {
            "sum": 1.0,
            "mean": num_loss_terms**2
            / (self._N_data * self._num_per_example_loss_terms),
        }[self._loss_func.reduction]

        # Compute the corrected eigenvalues for the EKFAC approximation
        # aaT_eigenvectors does not exist if the weight matrix of the module is excluded
        aaT_eigenvectors = input_covariances_eigenvectors.get(module_name)
        # ggT_eigenvectors always exists
        ggT_eigenvectors = gradient_covariances_eigenvectors[module_name]

        if not self._separate_weight_and_bias and {"weight", "bias"} == set(
            param_pos.keys()
        ):
            a_augmented = cat([a, a.new_ones(*a.shape[:-1], 1)], dim=-1)
            eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                g, ggT_eigenvectors, a_augmented, aaT_eigenvectors
            )
            self._set_or_add_(
                corrected_eigenvalues,
                module_name,
                eigencorrection.mul_(correction),
            )

        else:
            if module_name not in corrected_eigenvalues:
                corrected_eigenvalues[module_name] = {}
            for p_name, pos in param_pos.items():
                eigencorrection = compute_eigenvalue_correction_linear_weight_sharing(
                    g,
                    ggT_eigenvectors,
                    None if p_name == "bias" else a,
                    aaT_eigvecs=None if p_name == "bias" else aaT_eigenvectors,
                )
                self._set_or_add_(
                    corrected_eigenvalues[module_name],
                    pos,
                    eigencorrection.mul_(correction),
                )
