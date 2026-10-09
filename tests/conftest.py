"""Shared test data. No real data is needed, so these tests also run on GitHub CI."""

import numpy as np
import pytest

from creditrisk.data.synthetic import make_fake_application, make_fake_history  # noqa: F401


@pytest.fixture(scope="session")
def calibrated_scores():
    """Skewed probabilities like a credit model, with labels drawn from them (so calibrated)."""
    rng = np.random.default_rng(0)
    p = rng.beta(1, 10, 30_000)
    y = (rng.random(len(p)) < p).astype(int)
    return y, p


@pytest.fixture(scope="session")
def fake_train():
    return make_fake_application(3_000, with_target=True, seed=1)


@pytest.fixture(scope="session")
def fake_test():
    return make_fake_application(500, with_target=False, seed=2, first_id=500_000)
