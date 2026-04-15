r"""Linear operator for the Frobenius-optimal Kronecker-factored approximation (KFOC).

KFOC finds per-layer Kronecker factors that minimize the Frobenius-norm error to
the exact per-layer Gauss-Newton block. Compared to KFAC -- which uses sample-based
input and gradient covariances -- KFOC uses the top rank-1 Kronecker approximation
of the per-layer block, obtained via the top singular pair of its Van Loan
rearrangement.

- Schnaus, D., Lee, J., Triebel, R. (2021). Kronecker-Factored Optimal Curvature.
  Bayesian Deep Learning Workshop, NeurIPS 2021.

- Koroko, A., Anciaux-Sedrakian, A., Gharbia, I. B., Garès, V., Haddou, M., Tchomba, Q.
  (2022). Efficient Approximations of the Fisher Matrix in Neural Networks using
  Kronecker Product Singular Value Decomposition. arXiv:2201.10285.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping

from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss

from curvlinops._torch_base import _ChainPyTorchLinearOperator
from curvlinops.computers.kfoc_make_fx import MakeFxKFOCComputer
from curvlinops.kfac import KFACLinearOperator
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _has_single_element


class KFOCLinearOperator(KFACLinearOperator):
    r"""Linear operator for the Frobenius-optimal KFAC (KFOC) approximation of the GGN.

    KFOC approximates each per-layer GGN block
    :math:`B_l = \sum_n (g_n g_n^\top) \otimes (a_n a_n^\top)` by a single
    Kronecker product :math:`G^\star \otimes A^\star` chosen to minimize
    :math:`\|B_l - G\otimes A\|_F`. The optimum is the top singular pair of the
    Van Loan rearrangement :math:`\mathcal{R}(B_l)`, which we obtain via
    :func:`scipy.sparse.linalg.svds`.

    Cited references:

    - Schnaus, D., Lee, J., Triebel, R. (2021). Kronecker-Factored Optimal Curvature.
      Bayesian Deep Learning Workshop, NeurIPS 2021.
      http://bayesiandeeplearning.org/2021/papers/33.pdf

    - Koroko, A., Anciaux-Sedrakian, A., Gharbia, I. B., Garès, V., Haddou, M.,
      Tchomba, Q. (2022). Efficient Approximations of the Fisher Matrix in
      Neural Networks using Kronecker Product Singular Value Decomposition.
      arXiv:2201.10285.

    .. note::
        KFOC supports only a single batch of data, ``FisherType.TYPE2``, and
        the ``KFACType.EXPAND`` weight-sharing variant. Pass the batch as a
        one-element iterable (e.g., ``[(X, y)]``). The per-sample memory cost
        is :math:`O(N \cdot T \cdot V \cdot \sum_\ell (d_\text{in}^{(\ell)}
        + d_\text{out}^{(\ell)}))` where :math:`V` is the number of backward
        passes required by ``FisherType.TYPE2`` (the output dimension of the
        loss).

    .. note::
        Recovered factors are not guaranteed PSD in degenerate or
        near-degenerate spectral cases. Downstream operations like
        :meth:`inverse`, :meth:`logdet`, or eigendecomposition may fail or
        produce ``NaN`` unless sufficient damping is supplied.

    .. note::
        KFOC minimizes the Frobenius error, which does not in general
        translate to small error on spectrum-dependent quantities like
        :meth:`logdet`. A rank-1 Kronecker product has a very constrained
        spectrum, so approximation quality for such quantities is inherently
        limited.
    """

    def __init__(
        self,
        model_func: Module
        | Callable[[dict[str, Tensor], Tensor | MutableMapping], Tensor],
        loss_func: MSELoss | CrossEntropyLoss | BCEWithLogitsLoss,
        params: dict[str, Tensor],
        data: Iterable[tuple[Tensor | MutableMapping, Tensor]],
        progressbar: bool = False,
        check_deterministic: bool = True,
        num_per_example_loss_terms: int | None = None,
        separate_weight_and_bias: bool = True,
        num_data: int | None = None,
        batch_size_fn: Callable[[MutableMapping | Tensor], int] | None = None,
    ):
        """Initialize the KFOC linear operator.

        Args:
            model_func: Neural network forward pass ``(params, X) -> prediction``;
                either an ``nn.Module`` or a functional callable.
            loss_func: Loss function (``MSELoss``, ``CrossEntropyLoss``, or
                ``BCEWithLogitsLoss``).
            params: Parameter values at which the GGN is approximated.
            data: A one-element iterable ``[(X, y)]``.
                ``len(list(data)) == 1`` is enforced.
            progressbar: Show a progress bar. Default: ``False``.
            check_deterministic: Check operator determinism. Default: ``True``.
            num_per_example_loss_terms: Number of per-example loss terms (e.g.,
                tokens in a sequence). Inferred from data if ``None``.
            separate_weight_and_bias: Treat weight and bias as separate factor
                groups. Default: ``True``.
            num_data: Number of data points. Inferred from data if ``None``.
            batch_size_fn: Function to extract batch size from ``X``. Required
                if ``X`` is not a ``torch.Tensor``.
        """
        _has_single_element(data)
        data_list = list(data)

        computer = MakeFxKFOCComputer(
            model_func,
            loss_func,
            params,
            data_list,
            progressbar=progressbar,
            check_deterministic=check_deterministic,
            fisher_type=FisherType.TYPE2,
            mc_samples=1,
            kfac_approx=KFACType.EXPAND,
            num_per_example_loss_terms=num_per_example_loss_terms,
            separate_weight_and_bias=separate_weight_and_bias,
            num_data=num_data,
            batch_size_fn=batch_size_fn,
        )
        K, mapping = KFACLinearOperator._compute_canonical_op(computer)
        P, PT = KFACLinearOperator._build_converters(computer, mapping)
        _ChainPyTorchLinearOperator.__init__(self, P, K, PT)
