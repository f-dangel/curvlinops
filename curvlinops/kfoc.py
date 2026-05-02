r"""Linear operator for the Frobenius-optimal Kronecker-factored GGN (KFOC)."""

from collections.abc import Callable, Iterable, MutableMapping

from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss

from curvlinops.computers.kfoc_make_fx import MakeFxKFOCComputer
from curvlinops.kfac import KFACLinearOperator
from curvlinops.kfac_utils import FisherType, KFACType


class KFOCLinearOperator(KFACLinearOperator):
    r"""Frobenius-optimal rank-one Kronecker approximation of the GGN.

    Unlike KFAC, which factorizes each per-layer GGN block
    :math:`\mathbf{G}` as a product of per-axis sums, KFOC returns the
    Frobenius-optimal rank-one Kronecker approximation by solving

    .. math::
        \mathbf{S}_1, \mathbf{S}_2 = \arg\min_{\mathbf{S}_1, \mathbf{S}_2}
        \lVert \mathbf{G} - \mathbf{S}_1 \otimes \mathbf{S}_2 \rVert_F,

    via the top singular pair of the block's Van Loan rearrangement
    :math:`\mathcal{R}(\mathbf{G})`. The factors :math:`\mathbf{S}_1`,
    :math:`\mathbf{S}_2` are always symmetric, and PSD unless
    :math:`\mathcal{R}(\mathbf{G})`'s top singular vector is degenerate.

    Scope:
        - Single-batch data only (``len(list(data)) == 1``).
        - :class:`FisherType.TYPE2` only.

    Attributes:
        SELF_ADJOINT: Whether the operator is self-adjoint. ``True`` for KFOC.

    References:
        - Schnaus, D., Lee, J., Triebel, R. (2021). "Kronecker-Factored Optimal
          Curvature." Bayesian Deep Learning Workshop, NeurIPS 2021.
          http://bayesiandeeplearning.org/2021/papers/33.pdf
        - Koroko, A., Anciaux-Sedrakian, A., Gharbia, I. B., Garès, V., Haddou, M.,
          Tchomba, Q. (2022). "Efficient Approximations of the Fisher Matrix in
          Neural Networks using Kronecker Product Singular Value Decomposition."
          arXiv:2201.10285.
    """

    _BACKENDS: dict[str, type] = {"make_fx": MakeFxKFOCComputer}

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
        num_per_example_loss_terms: int | None = None,
        separate_weight_and_bias: bool = True,
        num_data: int | None = None,
        batch_size_fn: Callable[[MutableMapping | Tensor], int] | None = None,
    ):
        """Frobenius-optimal rank-one Kronecker approximation of the GGN.

        Args:
            model_func: Functional model ``(params, X) -> prediction`` or an
                ``nn.Module``.
            loss_func: The loss function.
            params: Parameter dict.
            data: Iterable of a single ``(X, y)`` batch.
            progressbar: Whether to show a progress bar.
            check_deterministic: Whether to run the determinism check on init.
            seed: RNG seed for the determinism check.
            num_per_example_loss_terms: Number of per-example loss terms. See
                :class:`KFACLinearOperator`.
            separate_weight_and_bias: Whether to treat weight and bias as
                separate parameter groups.
            num_data: Total dataset size (here just the single batch size).
            batch_size_fn: Function extracting the batch size from ``X`` for
                dict-style inputs.
        """
        super().__init__(
            model_func,
            loss_func,
            params,
            data,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            seed=seed,
            fisher_type=FisherType.TYPE2,
            mc_samples=1,
            kfac_approx=KFACType.EXPAND,
            num_per_example_loss_terms=num_per_example_loss_terms,
            separate_weight_and_bias=separate_weight_and_bias,
            num_data=num_data,
            batch_size_fn=batch_size_fn,
            backend="make_fx",
        )
