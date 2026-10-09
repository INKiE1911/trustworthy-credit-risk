"""Step 13: fairness audit and mitigations."""

import numpy as np
import pytest

from creditrisk.decision.threshold import realised_profit
from creditrisk.fairness.audit import age_band, audit, gaps
from creditrisk.fairness.mitigate import group_thresholds, reweighing_weights


@pytest.fixture(scope="module")
def two_groups(calibrated_scores):
    """Group "b" has a higher default rate than group "a" (like men vs women)."""
    y, p = calibrated_scores
    rng = np.random.default_rng(2)
    groups = np.where(rng.random(len(y)) < 0.3 + 0.4 * y, "b", "a")
    amount = rng.uniform(1, 10, len(y))
    return y, p, groups, amount


def test_age_bands_and_audit_by_hand():
    assert age_band([29.9, 30, 59.9, 60]).tolist() == ["<30", "30-45", "45-60", "60+"]
    y = np.array([0, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.3, 0.4, 0.1, 0.1, 0.4])
    table = audit(y, p, ["a", "a", "a", "b", "b", "b"], approved=p < 0.2)
    assert table.loc["a", "approval rate"] == pytest.approx(1 / 3)
    assert table.loc["a", "good customers declined"] == pytest.approx(0.5)
    assert table.loc["b", "defaulters declined"] == pytest.approx(0.5)
    g = gaps(table)
    assert g["demographic parity difference"] == pytest.approx(1 / 3)
    assert g["disparate impact ratio"] == pytest.approx(0.5)


def test_reweighing_makes_group_and_label_independent(two_groups):
    y, _, groups, _ = two_groups
    w = reweighing_weights(y, groups)
    rates = [np.average(y[groups == g], weights=w[groups == g]) for g in ("a", "b")]
    assert rates[0] == pytest.approx(rates[1])
    assert rates[0] == pytest.approx(y.mean())


def test_group_thresholds_trade_profit_for_a_smaller_gap(two_groups):
    y, p, groups, amount = two_groups
    free = group_thresholds(y, p, groups, amount, 0.10, 0.50, max_gap=1.0)
    fair = group_thresholds(y, p, groups, amount, 0.10, 0.50, max_gap=0.005)

    def result(thr):
        approved = p < np.where(groups == "a", thr["a"], thr["b"])
        declined = [1 - approved[(groups == g) & (y == 0)].mean() for g in ("a", "b")]
        return realised_profit(y, amount, approved, 0.10, 0.50), abs(declined[0] - declined[1])

    (profit_free, gap_free), (profit_fair, gap_fair) = result(free), result(fair)
    assert gap_fair <= 0.005 < gap_free
    assert profit_fair < profit_free
