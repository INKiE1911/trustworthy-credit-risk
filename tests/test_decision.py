"""Step 10: the money threshold."""

import numpy as np
import pytest

from creditrisk.decision.threshold import break_even_threshold, realised_profit


def test_break_even_example_and_bad_inputs():
    assert break_even_threshold(0.10, 0.50) == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        break_even_threshold(0.0, 0.5)


def test_realised_profit_by_hand():
    y = [0, 1, 0]
    amount = [100.0, 200.0, 50.0]
    approved = [True, True, False]
    assert realised_profit(y, amount, approved, margin=0.1, lgd=0.5) == pytest.approx(10 - 100)


def test_break_even_maximises_profit_when_calibrated(calibrated_scores):
    # with calibrated p, scanning thresholds on real outcomes lands near m / (m + LGD)
    y, p = calibrated_scores
    amount = np.ones(len(p))
    grid = np.linspace(0.01, 0.5, 50)
    profits = [realised_profit(y, amount, p < t, margin=0.1, lgd=0.5) for t in grid]
    assert abs(grid[int(np.argmax(profits))] - 1 / 6) < 0.05
