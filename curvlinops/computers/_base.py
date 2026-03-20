"""Base class for KFAC/EKFAC computers.

Provides ``_BaseKFACComputer`` with shared validation, setup logic, and utility
methods. Subclasses implement ``_compute_kronecker_factors`` for their specific
backend (hooks or FX graph tracing).
"""

from collections.abc import Callable, Iterable, MutableMapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from einops import rearrange
from torch import Generator, Tensor, eye
from torch.func import vmap
from torch.linalg import eigh
from torch.nn import (
    BCEWithLogitsLoss,
    Conv2d,
    CrossEntropyLoss,
    Linear,
    Module,
    MSELoss,
    Parameter,
)

from curvlinops._empirical_risk import _EmpiricalRiskMixin
from curvlinops.ggn_utils import make_grad_output_fn
from curvlinops.kfac_utils import FisherType, KFACType

# Type alias for a parameter group: maps roles ("W", "b") to full param names
ParamGroup = dict[str, str]
# Type alias for a parameter group key: tuple of full param names
ParamGroupKey = tuple[str, ...]


class _BaseKFACComputer(_EmpiricalRiskMixin):
    r"""Base class for KFAC computers with shared validation and setup logic.

    Subclasses must implement :meth:`_compute_kronecker_factors`.

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
        model_func: Module
        | Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
        loss_func: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
        params: list[Parameter] | dict[str, Tensor],
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
            model_func: Either an ``nn.Module`` or a callable with signature
                ``(params_dict, X) -> prediction``. Callables are only supported
                by subclasses with ``SUPPORTS_FUNCTIONAL = True``.
            loss_func: The loss function.
            params: The parameters defining the Fisher/GGN that will be approximated
                through KFAC. Either a ``list[Parameter]`` (for ``Module``) or a
                ``dict[str, Tensor]`` (for callable ``model_func``).
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

    def _computation_context(self) -> AbstractContextManager:
        """Return a context manager for the computation.

        By default returns ``nullcontext``. Subclasses can override to
        temporarily modify state during computation (e.g. setting module
        parameters from ``self._params``).

        Returns:
            A context manager.
        """
        return nullcontext()

    def compute(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Compute the Kronecker factors.

        Returns:
            Tuple of ``(input_covariances, gradient_covariances, mapping)`` where the
            first two are dictionaries mapping parameter group keys
            (``tuple[str, ...]``) to covariance matrices and ``mapping`` is a
            list of parameter groups.
        """
        with self._computation_context():
            return self._compute_kronecker_factors()

    def _compute_kronecker_factors(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor], dict[ParamGroupKey, Tensor], list[ParamGroup]
    ]:
        """Compute KFAC's Kronecker factors. Must be implemented by subclasses.

        Returns:
            Tuple of (input_covariances, gradient_covariances, mapping).

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError

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
        grad_output_fn = make_grad_output_fn(loss_func, fisher_type, mc_samples)
        randomness = "different" if fisher_type == FisherType.MC else "same"
        return vmap(
            grad_output_fn, in_dims=(0, 0, None), out_dims=1, randomness=randomness
        )

    def _set_gradient_covariances_to_identity(
        self,
        gradient_covariances: dict[ParamGroupKey, Tensor],
        mapping: list[ParamGroup],
    ) -> None:
        """Set gradient covariances to identity for forward-only KFAC.

        For the FOOF/ISAAC method, the gradient covariance is the identity.
        We set it explicitly for simplicity, though this could be more efficient.

        Args:
            gradient_covariances: Dictionary to populate with identity matrices.
            mapping: List of parameter groups.
        """
        for group in mapping:
            param = self._params[next(iter(group.values()))]
            gradient_covariances[tuple(group.values())] = eye(
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

    @staticmethod
    def _set_or_add_(dictionary: dict[Any, Tensor], key: Any, value: Tensor) -> None:
        """Set or add a value to a dictionary entry (in-place).

        Args:
            dictionary: The dictionary to update.
            key: The key to update.
            value: The value to add.

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


class _EKFACMixin:
    """Mixin for EKFAC computers with shared eigenvalue correction logic.

    Subclasses must implement :meth:`compute_eigenvalue_correction` and inherit
    from a KFAC computer (``HooksKFACComputer`` or ``MakeFxKFACComputer``).
    """

    _SUPPORTED_FISHER_TYPE: tuple[FisherType, ...] = (
        FisherType.TYPE2,
        FisherType.MC,
        FisherType.EMPIRICAL,
    )

    def compute(
        self,
    ) -> tuple[
        dict[ParamGroupKey, Tensor],
        dict[ParamGroupKey, Tensor],
        dict[ParamGroupKey, Tensor],
        list[ParamGroup],
    ]:
        """Compute eigenvalue-corrected Kronecker factors.

        Returns:
            Tuple of ``(input_covariance_eigenvectors, gradient_covariance_eigenvectors,
            corrected_eigenvalues, mapping)`` where the first two are dictionaries
            mapping parameter group keys (``tuple[str, ...]``) to eigenvector
            matrices, the third maps group keys to eigenvalue corrections, and
            ``mapping`` is a list of parameter groups.
        """
        with self._computation_context():
            input_covariances, gradient_covariances, mapping = (
                self._compute_kronecker_factors()
            )
            input_covariances = self._eigenvectors_(input_covariances)
            gradient_covariances = self._eigenvectors_(gradient_covariances)
            corrected_eigenvalues = self.compute_eigenvalue_correction(
                input_covariances, gradient_covariances, mapping
            )
        return input_covariances, gradient_covariances, corrected_eigenvalues, mapping

    @staticmethod
    def _rearrange_for_larger_than_2d_output(
        output: Tensor, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""Reject >2d output for EKFAC.

        EKFAC's individual gradient implementation does not support loss terms
        that depend on each other (i.e., loss terms other than per-data point).

        Args:
            output: The model's prediction.
            y: The labels.

        Returns:
            The unchanged output and target.

        Raises:
            ValueError: If the output is not 2d and y is not 1d/2d.
        """
        if output.ndim != 2 or y.ndim not in {1, 2}:
            raise ValueError(
                "Only 2d output and 1d/2d target are supported for EKFAC. "
                f"Got {output.ndim=} and {y.ndim=}."
            )
        return output, y

    @staticmethod
    def _eigenvectors_(dictionary: dict[Any, Tensor]) -> dict[Any, Tensor]:
        """Replace all matrix values with their eigenvectors (in-place).

        Args:
            dictionary: A dictionary mapping parameter group keys to square matrices.

        Returns:
            The modified dictionary with eigenvectors replacing the original matrices.
        """
        for key, value in dictionary.items():
            dictionary[key] = eigh(value).eigenvectors
        return dictionary

    def compute_eigenvalue_correction(
        self,
        input_covariances_eigenvectors: dict[ParamGroupKey, Tensor],
        gradient_covariances_eigenvectors: dict[ParamGroupKey, Tensor],
        mapping: list[ParamGroup],
    ) -> dict[ParamGroupKey, Tensor]:
        """Compute eigenvalue corrections. Must be implemented by subclasses.

        Args:
            input_covariances_eigenvectors: Input covariance eigenvectors.
            gradient_covariances_eigenvectors: Gradient covariance eigenvectors.
            mapping: List of parameter groups.

        Returns:
            Dictionary mapping parameter group keys to corrected eigenvalues.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError
