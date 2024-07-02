"""Setup file for CurvLinOps.

Use ``setup.cfg`` for configuration.
"""

import sys
from importlib.metadata import version

from packaging.version import Version
from setuptools import setup

setuptools_version = Version(version("setuptools"))

if setuptools_version < Version("38.3"):
    print(f"Error: version of setuptools is too old (<38.3). Got {setuptools_version}.")
    sys.exit(1)


if __name__ == "__main__":
    setup(use_scm_version=True)
