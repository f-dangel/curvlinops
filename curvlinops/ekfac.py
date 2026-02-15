"""Contains LinearOperator implementation of EKFAC approximation of the Fisher/GGN."""

from __future__ import annotations

from typing import Tuple

from curvlinops._torch_base import _ChainPyTorchLinearOperator
from curvlinops.blockdiagonal import BlockDiagonalLinearOperator
from curvlinops.computers.ekfac import EKFACComputer
from curvlinops.eigh import EighDecomposedLinearOperator
from curvlinops.kfac import FisherType, KFACLinearOperator
from curvlinops.kronecker import KroneckerProductLinearOperator


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
        _SUPPORTED_FISHER_TYPE: Tuple of supported Fisher types.
    """

    _COMPUTER_CLS = EKFACComputer
    _SUPPORTED_FISHER_TYPE: Tuple[FisherType, ...] = (
        FisherType.TYPE2,
        FisherType.MC,
        FisherType.EMPIRICAL,
    )

    @staticmethod
    def _compute_canonical_op(computer: EKFACComputer) -> BlockDiagonalLinearOperator:
        """Compute EKFAC factors and assemble the canonical block-diagonal operator.

        Args:
            computer: An ``EKFACComputer`` instance.

        Returns:
            Block diagonal linear operator representing EKFAC in canonical basis.
        """
        input_eigvecs, gradient_eigvecs, corrected_eigenvalues, mapping = (
            computer.compute()
        )
        bases = []
        corrections = []
        for mod_name, param_pos in mapping.items():
            Q_a = input_eigvecs.get(mod_name, None)
            Q_g = gradient_eigvecs[mod_name]
            lambdas = corrected_eigenvalues[mod_name]
            if not computer._separate_weight_and_bias and {"weight", "bias"} == set(
                param_pos.keys()
            ):
                bases.append([Q_g, Q_a])
                corrections.append(lambdas)
            else:
                for p_name, p_pos in param_pos.items():
                    bases.append([Q_g, Q_a] if p_name == "weight" else [Q_g])
                    corrections.append(lambdas[p_pos])
        blocks = [
            EighDecomposedLinearOperator(
                correction.flatten(), KroneckerProductLinearOperator(*basis)
            )
            for basis, correction in zip(bases, corrections)
        ]
        return BlockDiagonalLinearOperator(blocks)

    def inverse(self, damping: float = 0.0) -> _ChainPyTorchLinearOperator:
        """Return the inverse of the EKFAC approximation.

        Inverts each eigendecomposed block of the canonical operator
        and returns the result in parameter space.

        Args:
            damping: Damping term added to eigenvalues before inversion.
                Default: ``0.0``.

        Returns:
            Inverse of the EKFAC approximation as a linear operator.
        """
        P, K, PT = self
        K_inv = BlockDiagonalLinearOperator([
            block.inverse(damping=damping) for block in K
        ])
        return _ChainPyTorchLinearOperator(P, K_inv, PT)
