"""Tests for creditrisk.evaluation.metrics."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from creditrisk.evaluation.metrics import (
    evaluate,
    evaluate_at_threshold,
    expected_calibration_error,
    ks_statistic,
)


def test_matches_sklearn(calibrated_scores):
    y, p = calibrated_scores
    m = evaluate(y, p)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, p))
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)
    assert m["n"] == len(y)


def test_perfect_and_constant_models():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    constant = evaluate(y, np.full(4, 0.5))
    assert constant["roc_auc"] == pytest.approx(0.5)
    assert constant["ks"] == pytest.approx(0.0)


def test_ks_matches_its_definition(calibrated_scores):
    y, p = calibrated_scores
    thresholds = np.quantile(p, np.linspace(0, 1, 401))
    by_hand = max((p[y == 1] >= t).mean() - (p[y == 0] >= t).mean() for t in thresholds)
    assert ks_statistic(y, p) == pytest.approx(by_hand, abs=0.01)


def test_ece_small_when_calibrated_large_when_not(calibrated_scores):
    y, p = calibrated_scores
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, p, strategy="quantile") < 0.015
    overconfident = np.clip(3 * p, 0, 1)
    assert expected_calibration_error(y, overconfident) > 0.05


def test_threshold_metrics_by_hand():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.3])
    r = evaluate_at_threshold(y, p, threshold=0.5)  # flagged: rows 2 and 3
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 2)
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["approval_rate"] == pytest.approx(0.6)
