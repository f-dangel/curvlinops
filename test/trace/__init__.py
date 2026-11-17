"""Test ``curvlinops.trace``."""

DISTRIBUTIONS = ["rademacher", "normal"]
DISTRIBUTION_IDS = [f"distribution={distribution}" for distribution in DISTRIBUTIONS]

NUM_MATVECS = [3, 6]
NUM_MATVEC_IDS = [f"num_matvecs={num_matvecs}" for num_matvecs in NUM_MATVECS]
