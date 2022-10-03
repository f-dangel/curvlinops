"""Contains tests of ``curvlinops/papyan2020traces/spectrum``."""

from pathlib import Path
from subprocess import CalledProcessError, check_output

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
