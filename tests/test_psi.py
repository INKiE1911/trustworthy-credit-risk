"""Step 16: the population stability index."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.monitoring.psi import psi, psi_table, status


def test_psi_by_hand_for_categories():
    ref = ["a"] * 50 + ["b"] * 50
    new = ["a"] * 80 + ["b"] * 20
    expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert psi(ref, new) == pytest.approx(expected, rel=1e-3)   # empty bins add ~0
    assert psi(ref, ref) == pytest.approx(0.0)
    assert psi(ref, ["c"] * 100) > 1                             # an unseen value is a shift


def test_psi_for_numbers_and_missing_values():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    assert psi(ref, rng.normal(size=20_000)) < 0.01              # same distribution
    assert 0.1 < psi(ref, rng.normal(0.5, 1, 20_000)) < 0.25     # half a standard deviation
    assert psi(ref, rng.normal(1.5, 1, 20_000)) > 0.25
    with_nan = np.where(rng.random(20_000) < 0.3, np.nan, ref)
    assert psi(ref, with_nan) > 0.25                             # 30% newly missing


def test_status_and_table():
    assert [status(v) for v in (0.05, 0.1, 0.25, 0.3)] == ["stable", "watch", "watch", "shift"]
    ref = pd.DataFrame({"x": np.arange(1000.0), "y": ["a", "b"] * 500})
    new = pd.DataFrame({"x": np.arange(1000.0) + 500, "y": ["a", "b"] * 500})
    table = psi_table(ref, new, ["x", "y"])
    assert list(table.index) == ["x", "y"] and list(table["status"]) == ["shift", "stable"]
