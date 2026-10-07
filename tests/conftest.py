"""Shared test data. No real data is needed, so these tests also run on GitHub CI."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def calibrated_scores():
    """Skewed probabilities like a credit model, with labels drawn from them (so calibrated)."""
    rng = np.random.default_rng(0)
    p = rng.beta(1, 10, 30_000)
    y = (rng.random(len(p)) < p).astype(int)
    return y, p
