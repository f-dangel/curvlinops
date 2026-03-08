"""Contains tests for ``curvlinops/computers/ekfac``."""

from pytest import raises
from torch import float64, manual_seed, rand
from torch.linalg import qr

from curvlinops.computers.ekfac import (
    compute_eigenvalue_correction_linear_weight_sharing,
)
from curvlinops.utils import allclose_report


def test_compute_eigenvalue_correction_linear_weight_sharing():
    """Verifies equivalence of per-example gradient and Gramian approaches."""
    manual_seed(0)
    N, S, D1, D2 = 2, 3, 4, 5
    DT = float64

    # Generate random layer inputs and output gradients
    g = rand(N, S, D1, dtype=DT)
    a = rand(N, S, D2, dtype=DT)

    # Generate random bases
    ggT_eigvecs, _ = qr(rand(D1, D1, dtype=DT))
    aaT_eigvecs, _ = qr(rand(D2, D2, dtype=DT))

    # Verify both strategies yield the same result
    correction_via_gramian = compute_eigenvalue_correction_linear_weight_sharing(
        g, ggT_eigvecs, a, aaT_eigvecs, _force_strategy="gramian"
    )
    correction_via_gradients = compute_eigenvalue_correction_linear_weight_sharing(
        g, ggT_eigvecs, a, aaT_eigvecs, _force_strategy="per_example_gradients"
    )
    assert allclose_report(correction_via_gramian, correction_via_gradients)

    # Test invalid _force_strategy argument raises an error
    with raises(ValueError, match="Invalid _force_strategy"):
        compute_eigenvalue_correction_linear_weight_sharing(
            g, ggT_eigvecs, a, aaT_eigvecs, _force_strategy="invalid_strategy"
        )

    # Test exception is raised if a and aaT_eigvecs do not have the same type
    with raises(ValueError, match=r"Both \(a, aaT_eigvecs\) must be None or Tensor"):
        compute_eigenvalue_correction_linear_weight_sharing(g, ggT_eigvecs, a, None)
    with raises(ValueError, match=r"Both \(a, aaT_eigvecs\) must be None or Tensor"):
        compute_eigenvalue_correction_linear_weight_sharing(
            g, ggT_eigvecs, None, aaT_eigvecs
        )
