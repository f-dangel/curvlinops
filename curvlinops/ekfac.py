"""Contains LinearOperator implementation of EKFAC approximation of the Fisher/GGN."""

from curvlinops._torch_base import _ChainPyTorchLinearOperator
from curvlinops.blockdiagonal import BlockDiagonalLinearOperator
from curvlinops.computers.ekfac import EKFACComputer
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
        "hooks": EKFACComputer,
        "make_fx": MakeFxEKFACComputer,
    }

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
        for usage in mapping:
            group_key = tuple(usage.params.values())
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
