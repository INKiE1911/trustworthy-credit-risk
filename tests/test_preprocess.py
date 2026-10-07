"""Tests for creditrisk.models.preprocess."""

import numpy as np
import pandas as pd

from creditrisk.models.preprocess import (
    QuantileClipper,
    make_linear_preprocessor,
    make_tree_preprocessor,
)


def test_clipper_uses_training_quantiles_and_keeps_nan():
    train = np.array([[1.0], [2.0], [3.0], [1000.0], [np.nan]])
    clip = QuantileClipper(lower=0.0, upper=0.75).fit(train)
    out = clip.transform(np.array([[5000.0], [np.nan], [-10.0]]))
    assert out[0, 0] == clip.high_[0] and np.isnan(out[1, 0]) and out[2, 0] == 1.0


def _frame(repeat=1):
    return pd.DataFrame({
        "a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0] * repeat,
        "small": pd.Categorical(["x", "y", None, "x", "y", "x"] * repeat),
        "large": pd.Categorical(["p", "q", "r", "p", None, "q"] * repeat),
    })


def test_linear_preprocessor_handles_nan_and_unseen_values():
    X, y = _frame(repeat=10), np.array([0, 1] * 30)  # target encoding needs 5 per class
    prep = make_linear_preprocessor(["a"], ["small"], ["large"]).fit(X, y)
    new = pd.DataFrame({"a": [np.nan], "small": pd.Categorical(["unseen"]),
                        "large": pd.Categorical(["unseen"])})
    out = prep.transform(new)
    assert out.shape[1] == prep.transform(X).shape[1]
    assert not np.isnan(out).any()


def test_tree_preprocessor_keeps_nan_for_trees():
    X = _frame()
    out = make_tree_preprocessor(["a"], ["small", "large"]).fit(X).transform(X)
    assert np.isnan(out[1, 0])  # numeric NaN passes through
    assert out[2, 1] == -2  # missing text -> -2
