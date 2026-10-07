"""Tests for creditrisk.evaluation.bootstrap."""

import numpy as np

from creditrisk.evaluation.bootstrap import bootstrap_ci, paired_bootstrap


def test_ci_contains_estimate_and_is_repeatable(calibrated_scores):
    y, p = calibrated_scores
    first = bootstrap_ci(y, p, n_boot=100, seed=1)
    again = bootstrap_ci(y, p, n_boot=100, seed=1)
    assert first == again
    assert first["low"] <= first["estimate"] <= first["high"]


def test_ci_narrower_with_more_data(calibrated_scores):
    y, p = calibrated_scores
    small = bootstrap_ci(y[:2_000], p[:2_000], n_boot=100)
    big = bootstrap_ci(y, p, n_boot=100)
    assert big["high"] - big["low"] < small["high"] - small["low"]


def test_identical_models_have_zero_difference(calibrated_scores):
    y, p = calibrated_scores
    r = paired_bootstrap(y, p, p, n_boot=100)
    assert r["difference"] == 0 and r["low"] == 0 and r["high"] == 0
    assert not r["significant"]


def test_detects_a_clearly_better_model(calibrated_scores):
    y, p = calibrated_scores
    noisy = 0.5 * p + 0.5 * np.random.default_rng(1).random(len(p))  # worse model
    r = paired_bootstrap(y, p, noisy, n_boot=100)
    assert r["difference"] > 0 and r["low"] > 0 and r["significant"]
    assert r["p_value"] < 0.05
