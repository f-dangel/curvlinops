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

from collections.abc import Callable, Iterable, MutableMapping

from torch import Tensor
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Module,
    MSELoss,
)

from curvlinops._torch_base import _ChainPyTorchLinearOperator
from curvlinops.blockdiagonal import BlockDiagonalLinearOperator
from curvlinops.computers._base import ParamGroup, _BaseKFACComputer
from curvlinops.computers.kfac_hooks import HooksKFACComputer
from curvlinops.computers.kfac_make_fx import MakeFxKFACComputer
from curvlinops.kfac_utils import (
    FisherType,
    FromCanonicalLinearOperator,
    KFACType,
    ToCanonicalLinearOperator,
)
from curvlinops.kronecker import KroneckerProductLinearOperator


class KFACLinearOperator(_ChainPyTorchLinearOperator):
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

    (see :class:`curvlinops.GGNLinearOperator` with ``mc_samples > 0``).
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
        SELF_ADJOINT: Whether the operator is self-adjoint. ``True`` for KFAC.
    """

    _BACKENDS: dict[str, type] = {
        "hooks": HooksKFACComputer,
        "make_fx": MakeFxKFACComputer,
    }
    SELF_ADJOINT: bool = True

    def __init__(
        self,
        model_func: Module
        | Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
        loss_func: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
        params: dict[str, Tensor],
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
        backend: str = "hooks",
    ):
        """Kronecker-factored approximate curvature (KFAC) proxy of the Fisher/GGN.

        Warning:
            This is an early proto-type with limitations:
                - Only Linear and Conv2d modules are supported.
                - The ``hooks`` backend assumes each module is called exactly
                  once per forward pass. Weight tying (same module called
                  multiple times) will silently produce incorrect results.
                  Use ``backend="make_fx"`` for weight-tied architectures.

        Args:
            model_func: The neural network's forward pass, defining the functional
                relationship ``(params, X) -> prediction``. Either an ``nn.Module``
                (architecture) or a callable ``(params_dict, X) -> prediction``.
                Callables require ``backend="make_fx"``.
            loss_func: The loss function.
            params: The parameter values at which the Fisher/GGN is approximated.
                A dictionary mapping parameter names to tensors.
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
                Defaults to ``True``. Setting this to ``False`` is more efficient
                because gradient covariances are computed once per layer rather than
                separately for weight and bias.
            num_data: Number of data points. If ``None``, it is inferred from the data
                at the cost of one traversal through the data loader.
            batch_size_fn: If the ``X``'s in ``data`` are not ``torch.Tensor``, this
                needs to be specified. The intended behavior is to consume the first
                entry of the iterates from ``data`` and return their batch size.
            backend: The backend to use for computing Kronecker factors.
                ``"hooks"`` uses forward/backward hooks (default).
                ``"make_fx"`` uses FX graph tracing via the IO collector.
                Defaults to ``"hooks"``.

                Note: The ``"make_fx"`` backend incurs a significant one-time
                tracing overhead (seconds for large models) on the first batch.
                The traced function is cached by batch size, so subsequent
                batches of the same size reuse it. However, each distinct batch
                size triggers a re-trace. Use uniform batch sizes in the data
                loader to avoid repeated tracing.

        Raises:
            ValueError: If ``backend`` is not supported.
        """
        if backend not in self._BACKENDS:
            raise ValueError(
                f"Invalid backend: {backend!r}. Supported: {tuple(self._BACKENDS)}."
            )
        computer_cls = self._BACKENDS[backend]
        computer = computer_cls(
            model_func,
            loss_func,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            seed=seed,
            fisher_type=fisher_type,
            mc_samples=mc_samples,
            kfac_approx=kfac_approx,
            num_per_example_loss_terms=num_per_example_loss_terms,
            separate_weight_and_bias=separate_weight_and_bias,
            num_data=num_data,
            batch_size_fn=batch_size_fn,
        )
        # KFAC = P @ K @ PT
        K, mapping = self._compute_canonical_op(computer)
        P, PT = self._build_converters(computer, mapping)
        super().__init__(P, K, PT)

    @staticmethod
    def _compute_canonical_op(
        computer: _BaseKFACComputer,
    ) -> tuple[BlockDiagonalLinearOperator, list[ParamGroup]]:
        """Compute Kronecker factors and assemble the canonical block-diagonal operator.

        Args:
            computer: A KFAC computer instance.

        Returns:
            Tuple of (block diagonal operator in canonical basis, mapping).
        """
        input_covariances, gradient_covariances, mapping = computer.compute()
        factors = []
        for group in mapping:
            group_key = tuple(group.values())
            aaT = input_covariances.get(group_key)
            ggT = gradient_covariances[group_key]
            factors.append([ggT, aaT] if aaT is not None else [ggT])

        # Create Kronecker product linear operators for each block
        blocks = [KroneckerProductLinearOperator(*fs) for fs in factors]

        # KFAC in the canonical basis
        return BlockDiagonalLinearOperator(blocks), mapping

    @staticmethod
    def _build_converters(
        computer: _BaseKFACComputer,
        mapping: list[ParamGroup],
    ) -> tuple[FromCanonicalLinearOperator, ToCanonicalLinearOperator]:
        """Build the canonical space converters.

        Args:
            computer: A KFAC computer instance.
            mapping: List of parameter groups.

        Returns:
            Tuple of ``(from_canonical_op, to_canonical_op)``.
        """
        PT = ToCanonicalLinearOperator(
            {name: p.shape for name, p in computer._params.items()},
            mapping,
            computer.device,
            computer.dtype,
        )
        P = PT.adjoint()
        return P, PT

    def trace(self) -> Tensor:
        """Trace of the KFAC approximation.

        Returns:
            Trace of the KFAC approximation.
        """
        _, K, _ = self
        return K.trace()

    def det(self) -> Tensor:
        """Compute the determinant of the KFAC approximation.

        Returns:
            Determinant of the KFAC approximation.
        """
        _, K, _ = self
        return K.det()

    def logdet(self) -> Tensor:
        """Log determinant of the KFAC approximation.

        More numerically stable than the ``det`` method.

        Returns:
            Log determinant of the KFAC approximation.
        """
        _, K, _ = self
        return K.logdet()

    def frobenius_norm(self) -> Tensor:
        """Frobenius norm of the KFAC approximation.

        Returns:
            Frobenius norm of the KFAC approximation.
        """
        _, K, _ = self
        return K.frobenius_norm()

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
                (Section 6.3). Only supported for one or two factors.
            min_damping: Minimum damping value. Only used if
                ``use_heuristic_damping`` is ``True``.
            use_exact_damping: Whether to use exact damping, i.e. to invert
                :math:`(A \\otimes B) + \\text{damping}\\; \\mathbf{I}`.
            retry_double_precision: Whether to retry Cholesky decomposition used for
                inversion in double precision.

        Returns:
            Inverse of the KFAC approximation as a linear operator ``P @ K^-1 @ PT``.
        """
        P, K, PT = self
        K_inv = BlockDiagonalLinearOperator([
            block.inverse(
                damping=damping,
                use_heuristic_damping=use_heuristic_damping,
                min_damping=min_damping,
                use_exact_damping=use_exact_damping,
                retry_double_precision=retry_double_precision,
            )
            for block in K
        ])
        return _ChainPyTorchLinearOperator(P, K_inv, PT)
