"""Contains LinearOperator implementation of EKFAC approximation of the Fisher/GGN."""

from collections.abc import Sequence

from curvlinops._torch_base import _ChainPyTorchLinearOperator
from curvlinops.blockdiagonal import BlockDiagonalLinearOperator
from curvlinops.computers._base import ParamGroup
from curvlinops.computers.ekfac_hooks import HooksEKFACComputer
from curvlinops.computers.ekfac_make_fx import MakeFxEKFACComputer
from curvlinops.eigh import EighDecomposedLinearOperator
from curvlinops.kfac import KFACLinearOperator
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
    """

    _BACKENDS: dict[str, type] = {
        "hooks": HooksEKFACComputer,
        "make_fx": MakeFxEKFACComputer,
    }

    @staticmethod
    def _compute_canonical_op(
        computer: HooksEKFACComputer | MakeFxEKFACComputer,
    ) -> tuple[BlockDiagonalLinearOperator, list[ParamGroup]]:
        """Compute EKFAC factors and assemble the canonical block-diagonal operator.

        Args:
            computer: An EKFAC computer instance (hooks or FX backend).

        Returns:
            Tuple of (block diagonal operator in canonical basis, mapping).
        """
        input_eigvecs, gradient_eigvecs, corrected_eigenvalues, mapping = (
            computer.compute()
        )
        bases = []
        corrections = []
        for group in mapping:
            group_key = tuple(group.values())
            Q_a = input_eigvecs.get(group_key)
            Q_g = gradient_eigvecs[group_key]
            lambdas = corrected_eigenvalues[group_key]
            bases.append([Q_g, Q_a] if Q_a is not None else [Q_g])
            corrections.append(lambdas)

        # Create Kronecker product linear operators for each block
        blocks = [
            EighDecomposedLinearOperator(
                correction.flatten(), KroneckerProductLinearOperator(*basis)
            )
            for basis, correction in zip(bases, corrections)
        ]
        # EKFAC in the canonical basis
        return BlockDiagonalLinearOperator(blocks), mapping

    def inverse(
        self, damping: float | Sequence[float] = 0.0
    ) -> _ChainPyTorchLinearOperator:
        """Return the inverse of the EKFAC approximation.

        Inverts each eigendecomposed block of the canonical operator
        and returns the result in parameter space.

        Args:
            damping: Damping value applied to each canonical block. If it is a scalar,
                the same value is used for all blocks. If it is a sequence, it must
                contain one damping value per canonical block. Default: ``0.0``.

        Returns:
            Inverse of the EKFAC approximation as a linear operator.
        """
        P, K, PT = self
        damping_per_block = self._broadcast_damping(damping, len(K))
        K_inv = BlockDiagonalLinearOperator([
            block.inverse(damping=block_damping)
            for block, block_damping in zip(K, damping_per_block)
        ])
        return _ChainPyTorchLinearOperator(P, K_inv, PT)
