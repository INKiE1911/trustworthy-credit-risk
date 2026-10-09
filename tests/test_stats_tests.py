"""Step 15: McNemar, Holm and the 5x2cv t-test."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from creditrisk.evaluation.stats_tests import five_by_two_cv, holm, mcnemar


def test_mcnemar_counts_only_the_disagreements():
    a = np.array([1, 1, 1, 0, 1, 0], bool)
    b = np.array([1, 0, 0, 0, 1, 1], bool)
    result = mcnemar(a, b)
    assert (result["only_a_right"], result["only_b_right"]) == (2, 1)
    assert result["p_value"] == pytest.approx(1.0)                 # 2 vs 1 is a coin flip
    assert mcnemar([True] * 30, [False] * 30)["p_value"] < 1e-8
    assert mcnemar(a, a)["p_value"] == 1.0                         # no disagreements


def test_holm_by_hand():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])          # running max keeps the order


def test_five_by_two_cv_tells_a_real_model_from_a_dummy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(2_000, 3)))
    y = (X[0] + rng.normal(size=2_000) > 0).astype(int).to_numpy()
    real = five_by_two_cv(LogisticRegression, lambda: DummyClassifier(), X, y)
    assert real["mean_difference"] > 0.2 and real["p_value"] < 0.01
    same = five_by_two_cv(LogisticRegression, LogisticRegression, X, y)
    assert same["mean_difference"] == pytest.approx(0.0)
