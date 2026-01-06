"""Contains LinearOperator implementation of EKFAC approximation of the Fisher/GGN."""

from __future__ import annotations

from functools import partial
from typing import Dict, List, Tuple, Union

from einops import einsum, rearrange
from torch import Tensor, cat
from torch.linalg import eigh
from torch.nn import (
    Conv2d,
    Module,
)
from torch.utils.hooks import RemovableHandle

from curvlinops._torch_base import _ChainPyTorchLinearOperator
from curvlinops.blockdiagonal import BlockDiagonalLinearOperator
from curvlinops.eigh import EighDecomposedLinearOperator
from curvlinops.kfac import (
    FisherType,
    KFACLinearOperator,
)
from curvlinops.kfac_utils import extract_patches
from curvlinops.kronecker import KroneckerProductLinearOperator


class EKFACLinearOperator(KFACLinearOperator):
    """Linear operator to multiply with the Fisher/GGN's EKFAC approximation.

    Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EKFAC) was originally
    introduced in

    - George, T., Laurent, C., Bouthillier, X., Ballas, N., Vincent, P. (2018).
    Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis (NeurIPS)

    and concurrently in the context of continual learning in

    Liu, X., Masana, M., Herranz, L., Van de Weijer, J., Lopez, A., Bagdanov, A. (2018).
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

    def _create_block_diagonal_operator(self) -> BlockDiagonalLinearOperator:
        """Create block-diagonal linear operator from EKFAC factors.

        Each block corresponds to a layer and is an EighDecomposedLinearOperator with
        Kronecker-factored eigenvectors.

        Returns:
            Block-diagonal linear operator with eigendecomposition blocks.
        """
        # Compute the Kronecker-factored bases
        Qs = self._compute_bases()

        # Compute the correction terms in the above bases
        lambdas = self._compute_basis_correction(Qs)

        # Build up block-diagonal operator with eigen-decomposed Kronecker products
        blocks = []

        for mod_name, param_pos in self._mapping.items():
            lam = lambdas.pop(mod_name)

            # Handle joint weight+bias case
            if not self._separate_weight_and_bias and {"weight", "bias"} == set(
                param_pos.keys()
            ):
                Q = Qs[mod_name]
                blocks.append([lam, Q])
            else:
                # Separate blocks for weight and bias
                for p_name in param_pos:
                    Q, correction = Qs[mod_name][p_name], lam[p_name]
                    blocks.append([correction, Q])

        # Convert them into EighDecomposedLinearOperator's
        blocks = [EighDecomposedLinearOperator(*block) for block in blocks]

        return BlockDiagonalLinearOperator(blocks)

    @staticmethod
    def _eigenbasis(covariances: Dict[str, Tensor]) -> None:
        """Helper method to compute eigenvectors from a single covariance dictionary.

        Modifies the input dictionary in-place, replacing covariance matrices with
        their corresponding eigenvectors.

        Args:
            covariances: Dictionary of covariance matrices. Will be modified in-place
                to contain eigenvectors instead of covariance matrices.
        """
        for name in list(covariances.keys()):
            _, eigenvectors = eigh(covariances[name])
            covariances[name] = eigenvectors

    def _compute_bases(
        self,
    ) -> Dict[
        str,
        Union[
            KroneckerProductLinearOperator,
            Dict[str, KroneckerProductLinearOperator],
        ],
    ]:
        """Compute the rotation operators (bases) for EKFAC eigenvalue correction.

        Returns:
            Dictionary of KroneckerProductLinearOperator rotation operators.
        """
        # Always compute Kronecker factors
        input_covariances, gradient_covariances = self._compute_kronecker_factors()

        # Compute eigenvectors from the factors (modifies dictionaries in-place)
        self._eigenbasis(input_covariances)
        self._eigenbasis(gradient_covariances)

        # Build up KroneckerProductLinearOperators for the bases
        Qs: Dict[
            str,
            Union[
                KroneckerProductLinearOperator,
                Dict[str, KroneckerProductLinearOperator],
            ],
        ] = {}

        for mod_name, param_pos in self._mapping.items():
            Q_a = input_covariances.pop(mod_name, None)
            Q_g = gradient_covariances.pop(mod_name)

            if not self._separate_weight_and_bias and {"weight", "bias"} == set(
                param_pos.keys()
            ):
                # Single operator for joint weight+bias
                Qs[mod_name] = KroneckerProductLinearOperator(Q_g, Q_a)
            else:
                # Separate operators for weight and bias
                Qs[mod_name] = {}
                for p_name in param_pos:
                    factors = [Q_g, Q_a] if p_name == "weight" else [Q_g]
                    Qs[mod_name][p_name] = KroneckerProductLinearOperator(*factors)

        return Qs

    def _compute_basis_correction(
        self,
        bases: Dict[
            str,
            Union[
                KroneckerProductLinearOperator,
                Dict[str, KroneckerProductLinearOperator],
            ],
        ],
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        """Compute and return the corrections for EKFAC.

        Args:
            bases: Dictionary of KroneckerProductLinearOperator's for
                basis transformation, pre-built from eigenvectors.

        Returns:
            Dictionary of corrections.
        """
        corrections: Dict[str, Union[Tensor, Dict[str, Tensor]]] = {}

        # install forward hooks
        hook_handles: List[RemovableHandle] = []

        for mod_name in self._mapping:
            module = self._model_func.get_submodule(mod_name)

            # compute the corrections using the per-example gradients
            hook_handles.append(
                module.register_forward_hook(
                    partial(
                        self._register_tensor_hook_on_output_to_accumulate_corrections,
                        module_name=mod_name,
                        bases=bases,
                        corrections=corrections,
                    )
                )
            )

        self._setup_generator()

        # loop over data set, computing the corrected basis values
        for X, y in self._loop_over_data(desc="Basis correction"):
            output = self._model_func(X)
            output, y = self._rearrange_for_larger_than_2d_output(output, y)
            self._compute_loss_and_backward(output, y, {})

        # clean up
        for handle in hook_handles:
            handle.remove()

        return corrections

    def _register_tensor_hook_on_output_to_accumulate_corrections(
        self,
        module: Module,
        inputs: Tuple[Tensor],
        output: Tensor,
        module_name: str,
        bases: Dict[
            str,
            Union[
                KroneckerProductLinearOperator,
                Dict[str, KroneckerProductLinearOperator],
            ],
        ],
        corrections: Dict[str, Union[Tensor, Dict[str, Tensor]]],
    ):
        """Register tensor hook on layer's output to accumulate the corrections.

        Note:
            The easier way to compute the corrections would be via a full
            backward hook on the module itself which performs the computation.
            However, this approach breaks down if the output of a layer feeds into an
            activation with `inplace=True` (see
            https://github.com/pytorch/pytorch/issues/61519). Hence we use the
            workaround
            https://github.com/pytorch/pytorch/issues/61519#issuecomment-883524237, and
            install a module hook which installs a tensor hook on the module's output
            tensor, which performs the accumulation of the gradient covariance.

        Args:
            module: Layer onto whose output a tensor hook to accumulate the
                corrections will be installed.
            inputs: The layer's input tensors.
            output: The layer's output tensor.
            module_name: The name of the layer in the neural network.
            rotation_operators: Dictionary of KroneckerProductLinearOperator's for
                gradient rotation.
            corrections: Dictionary to store corrections.

        Raises:
            ValueError: If the module has multiple inputs.
        """
        if len(inputs) != 1:
            raise ValueError("Modules with multiple inputs are not supported.")

        # Capture the input tensor for the backward hook
        layer_input = inputs[0].data.detach()

        tensor_hook = partial(
            self._accumulate_corrected_basis_values,
            module=module,
            module_name=module_name,
            layer_input=layer_input,
            bases=bases,
            corrections=corrections,
        )
        output.register_hook(tensor_hook)

    def _accumulate_corrected_basis_values(
        self,
        grad_output: Tensor,
        module: Module,
        module_name: str,
        layer_input: Tensor,
        bases: Dict[
            str,
            Union[
                KroneckerProductLinearOperator,
                Dict[str, KroneckerProductLinearOperator],
            ],
        ],
        corrections: Dict[str, Union[Tensor, Dict[str, Tensor]]],
    ):
        r"""Accumulate the corrections.

        The corrections are computed by transforming per-example gradients to
        the basis using KroneckerProductLinearOperator's, then squaring and
        summing.

        Args:
            grad_output: The gradient w.r.t. the output.
            module: The layer for which corrections will be accumulated.
            module_name: The name of the layer in the neural network.
            layer_input: The input tensor to the layer.
            bases: Dictionary of KroneckerProductLinearOperator's for
                basis transformation.
            corrections: Dictionary to store corrections.
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

        # Compute the corrections for the EKFAC approximation
        param_pos = self._mapping[module_name]

        # Process the layer input for computing per-example gradients
        activations = layer_input
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

        if not self._separate_weight_and_bias and {"weight", "bias"} == set(
            param_pos.keys()
        ):
            activations = cat(
                [activations, activations.new_ones(*activations.shape[:-1], 1)], dim=-1
            )
            # Compute per-example gradient using the layer input
            per_example_gradient = einsum(
                g,
                activations,
                "batch shared d_out, batch shared d_in -> d_out d_in batch",
            ).flatten(end_dim=-2)

            # Apply basis transformation
            Q = bases[module_name]
            per_example_gradient = Q.adjoint() @ per_example_gradient

            # Compute corrections
            self._set_or_add_(
                corrections,
                module_name,
                per_example_gradient.square_().sum(dim=1).mul_(correction),
            )
        else:
            if module_name not in corrections:
                corrections[module_name] = {}
            for p_name in param_pos:
                # Compute per-example gradient using the layer input
                per_example_gradient = (
                    einsum(
                        g,
                        activations,
                        "batch shared d_out, batch shared d_in -> d_out d_in batch",
                    ).flatten(end_dim=-2)
                    if p_name == "weight"
                    else einsum(g, "batch shared d_out -> d_out batch")
                )

                # Apply basis transformation
                Q = bases[module_name][p_name]
                per_example_gradient = Q.adjoint() @ per_example_gradient

                # Compute corrections
                self._set_or_add_(
                    corrections[module_name],
                    p_name,
                    per_example_gradient.square_().sum(dim=1).mul_(correction),
                )

    def inverse(
        self,
        damping: float = 0.0,
    ) -> _ChainPyTorchLinearOperator:
        """Return the inverse of the EKFAC linear operator.

        Args:
            damping: Damping value added to eigenvalues before inversion.
                Default: ``0.0``.

        Returns:
            Linear operator representing the inverse of EKFAC.
        """
        # Invert the blocks of self._block_diagonal_operator
        inverse_blocks = [
            block.inverse(damping=damping)
            for block in self._block_diagonal_operator._blocks
        ]

        # Create the inverse block diagonal operator
        inverse_block_diagonal = BlockDiagonalLinearOperator(inverse_blocks)

        return self._from_canonical @ inverse_block_diagonal @ self._to_canonical
