r"""Linear operator for the Frobenius-optimal Kronecker-factored GGN (KFOC).

Unlike KFAC, which factorizes each per-layer GGN block as a product of
per-axis sums, KFOC returns the best Frobenius-norm rank-one Kronecker
approximation of the exact per-layer block, obtained via the top singular
pair of the block's Van Loan rearrangement.

References:
    - Schnaus, D., Lee, J., Triebel, R. (2021). "Kronecker-Factored Optimal
      Curvature." Bayesian Deep Learning Workshop, NeurIPS 2021.
      http://bayesiandeeplearning.org/2021/papers/33.pdf
    - Koroko, A., Anciaux-Sedrakian, A., Gharbia, I. B., Garès, V., Haddou, M.,
      Tchomba, Q. (2022). "Efficient Approximations of the Fisher Matrix in
      Neural Networks using Kronecker Product Singular Value Decomposition."
      arXiv:2201.10285.
"""

from collections.abc import Callable, Iterable, MutableMapping

from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss

from curvlinops.computers.kfoc_make_fx import MakeFxKFOCComputer
from curvlinops.kfac import KFACLinearOperator
from curvlinops.kfac_utils import FisherType, KFACType


class KFOCLinearOperator(KFACLinearOperator):
    r"""Frobenius-optimal rank-one Kronecker approximation of the GGN.

    For each per-layer Gauss-Newton block
    :math:`B_l = \sum_{v, n} \mathrm{vec}(P_{v, n})\,\mathrm{vec}(P_{v, n})^\top`
    with per-sample ``vec(W)`` gradients
    :math:`P_{v, n} = \sum_t g_{v, n, t} a_{n, t}^\top`, KFOC returns the
    Frobenius-optimal rank-one Kronecker approximation

    .. math::
        B_l \approx G^\star \otimes A^\star,

    where :math:`(G^\star, A^\star)` come from the top singular pair of the
    block's Van Loan rearrangement.

    The factors are stored exactly as the SVD reshape produces them — no
    symmetrization, no PSD projection.

    - **Symmetry** holds by construction: :math:`B_l` is symmetric, so
      :math:`\mathcal{R}(B_l)` commutes with the vec-transpose permutation
      and its top singular triplet lives in the symmetric subspace, making
      ``G_star`` and ``A_star`` symmetric to machine precision.
    - **Positive semi-definiteness is not guaranteed**: even for PSD
      :math:`B_l`, the dominant Kronecker direction can correspond to an
      indefinite factor pair. Downstream ``inverse``, ``eigh``, or
      ``logdet`` may fail or return NaNs in such cases unless sufficient
      damping is supplied.

    Scope:
        - Single-batch data only (``len(list(data)) == 1``).
        - :class:`FisherType.TYPE2` only.
        - :class:`KFACType.EXPAND` only.
        - FX backend only (uses :func:`make_compute_kfac_io_batch` with
          ``intermediate_as_batch=False``).

    Attributes:
        SELF_ADJOINT: Whether the operator is self-adjoint. ``True`` for KFOC.
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
