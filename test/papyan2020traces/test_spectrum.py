"""Contains tests of ``curvlinops/papyan2020traces/spectrum``."""

from pathlib import Path
from subprocess import CalledProcessError, check_output

from numpy import allclose, array, diag

from curvlinops.papyan2020traces.spectrum import (
    approximate_boundaries,
    approximate_boundaries_abs,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
BASIC_USAGE = PROJECT_ROOT / "docs" / "examples" / "basic_usage"


def test_example_verification_spectral_density():
    """Integration test to check that the verification example is working.

    It is hard to test the spectral density estimation techniques. This test
    uses the verification example from the documentation.
    """
    EXAMPLE_VERIFICATION_SPECTRAL_DENSITY = (
        BASIC_USAGE / "example_verification_spectral_density.py"
    )

    try:
        check_output(f"python {EXAMPLE_VERIFICATION_SPECTRAL_DENSITY}", shell=True)
    except CalledProcessError as e:
        print(e.output)
        raise e


def test_approximate_boundaries():
    """Test spectrum boundary approximation with partially supplied boundaries."""
    A_diag = array([1.0, 2.0, 3.0, 4.0, 5.0])
    A = diag(A_diag)
    lambda_min, lambda_max = A_diag.min(), A_diag.max()

    cases = [
        [(0.0, 10.0), (0.0, 10.0)],
        [(1.5, None), (1.5, lambda_max)],
        [(None, 2.5), (lambda_min, 2.5)],
        [(None, None), (lambda_min, lambda_max)],
        [None, (lambda_min, lambda_max)],
    ]

    for inputs, results in cases:
        output = approximate_boundaries(A, boundaries=inputs)
        assert len(output) == 2
        assert isinstance(output[0], float)
        assert isinstance(output[1], float)
        assert allclose(output, results)


def test_approximate_boundaries_abs():
    """Test abs spectrum boundary approximation with partially supplied boundaries."""
    A_diag = array([-2.0, -1.0, 3.0, 4.0, 5.0])
    A = diag(A_diag)
    lambda_abs_min, lambda_abs_max = abs(A_diag).min(), abs(A_diag).max()

    cases = [
        [(0.0, 10.0), (0.0, 10.0)],
        [(1.5, None), (1.5, lambda_abs_max)],
        [(None, 2.5), (lambda_abs_min, 2.5)],
        [(None, None), (lambda_abs_min, lambda_abs_max)],
        [None, (lambda_abs_min, lambda_abs_max)],
    ]

    for inputs, results in cases:
        output = approximate_boundaries_abs(A, boundaries=inputs)
        assert len(output) == 2
        assert isinstance(output[0], float)
        assert isinstance(output[1], float)
        assert allclose(output, results)
