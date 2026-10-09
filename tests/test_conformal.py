"""Step 11: the class-conditional conformal layer."""

import numpy as np
import pytest

from creditrisk.decision.conformal import class_cutoffs, conformal_quantile, coverage, decide


def test_quantile_by_hand():
    scores = np.arange(1, 11) / 10                      # 0.1 ... 1.0
    assert conformal_quantile(scores, 0.2) == pytest.approx(0.9)   # ceil(11 * 0.8) = 9th
    assert conformal_quantile(scores[:3], 0.1) == 1.0   # ceil(4 * 0.9) = 4 > 3: keep every label


def test_decisions_from_sets():
    # q_repay = 0.2, q_default = 0.6: repay in set if p <= 0.2, default in set if p >= 0.4
    p = np.array([0.1, 0.3, 0.5, 0.15])
    assert decide(p, 0.2, 0.6).tolist() == ["approve", "refer", "decline", "approve"]
    assert decide([0.3], 0.5, 0.8).tolist() == ["refer"]       # both labels in the set


def test_guarantee_holds_per_class_but_not_for_one_cutoff(calibrated_scores):
    y, p = calibrated_scores
    rng = np.random.default_rng(1)
    fit = rng.random(len(y)) < 0.5
    q_repay, q_default = class_cutoffs(p[fit], y[fit], alpha=0.10)
    cov = coverage(p[~fit], y[~fit], q_repay, q_default)
    assert cov["repay"] == pytest.approx(0.90, abs=0.02)
    assert cov["default"] == pytest.approx(0.90, abs=0.03)
    # one cut-off for everyone: about 90% overall, but far fewer defaulters covered
    scores = np.where(y[fit] == 1, 1 - p[fit], p[fit])
    q = conformal_quantile(scores, 0.10)
    assert coverage(p[~fit], y[~fit], q, q)["default"] < 0.5
