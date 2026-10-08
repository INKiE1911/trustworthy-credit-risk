"""Tests for the WoE scorecard encoder and the LightGBM feature selector."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from creditrisk.models.transformers import (
    DropUninformativeColumns,
    LightGBMSelector,
    WoEEncoder,
    scorecard_points,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 4_000
    risk = rng.normal(size=n)  # drives default
    X = pd.DataFrame({
        "risky_number": np.where(rng.random(n) < 0.1, np.nan, risk + rng.normal(0, 0.5, n)),
        "noise": rng.normal(size=n),
        "city": pd.Categorical(np.where(risk > 1, "A", rng.choice(["B", "C", "D"], n))),
    })
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * risk - 2.5)))).astype(int)
    return X, y


def test_woe_sign_and_iv(toy):
    X, y = toy
    enc = WoEEncoder(n_bins=5, min_iv=0.0).fit(X, y)
    woe = enc.woe_["risky_number"][:5]  # the 5 number bins, low to high
    assert woe[0] > 0 > woe[-1]  # low values are safe (positive WoE), high values risky
    assert enc.iv_["risky_number"] > enc.iv_["noise"]
    assert enc.iv_["city"] > enc.iv_["noise"]
    assert enc.selected_[-1] == "noise"  # ranked by IV


def test_woe_selection_and_unseen_values(toy):
    X, y = toy
    enc = WoEEncoder(min_iv=0.02).fit(X, y)
    assert "noise" not in enc.selected_
    new = pd.DataFrame({"risky_number": [np.nan, 100.0], "noise": [0.0, 0.0],
                        "city": pd.Categorical(["Z", None])})
    out = enc.transform(new)
    assert out.shape == (2, len(enc.selected_)) and np.isfinite(out).all()


def test_points_add_up_to_the_model_score(toy):
    X, y = toy
    pipe = Pipeline([("woe", WoEEncoder()), ("model", LogisticRegression(max_iter=1000))]).fit(X, y)
    table = scorecard_points(pipe, pdo=20, base_score=600, base_odds=50)
    enc = pipe.named_steps["woe"]
    row = X.iloc[[7]]
    total = 0.0
    for col in enc.selected_:
        b = enc._bins(col, row[col])[0]
        total += table[table["feature"] == col].iloc[b]["points"]
    p = pipe.predict_proba(row)[0, 1]
    factor = 20 / np.log(2)
    expected = 600 - factor * np.log(50) + factor * np.log((1 - p) / p)
    assert total == pytest.approx(expected, abs=1e-6)


def test_selector_keeps_the_most_useful_columns(toy):
    X, y = toy
    sel = LightGBMSelector(k=1, n_estimators=30).fit(X, y)
    assert sel.selected_[0] in ("city", "risky_number")  # an informative column comes first
    assert list(sel.transform(X).columns) == sel.selected_
    assert len(LightGBMSelector(k=2, n_estimators=30).fit(X, y).selected_) == 2


def test_drop_uninformative_columns():
    X = pd.DataFrame({"good": [1.0, 2.0, np.nan], "empty": [np.nan] * 3, "const": [5.0] * 3,
                      "text": pd.Categorical(["a", "b", "a"])})
    drop = DropUninformativeColumns().fit(X)
    assert drop.keep_ == ["good", "text"] and drop.dropped_ == ["empty", "const"]
    assert list(drop.transform(X).columns) == ["good", "text"]
