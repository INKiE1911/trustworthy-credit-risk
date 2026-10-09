"""Step 8: the PyTorch MLP. The input preparation is tested always; the network only when
PyTorch is installed (it is optional for the rest of the project)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from creditrisk.models.mlp import TabularInputs, TorchMLPClassifier

needs_torch = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                 reason="PyTorch is not installed")


def make_data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(5)})
    X.loc[rng.random(n) < 0.2, "x0"] = np.nan
    X["empty"] = np.nan
    X["city"] = pd.Categorical(rng.choice(["a", "b", "c", "d", None], size=n),
                               categories=["a", "b", "c", "d", "e"])
    X["job"] = pd.Categorical(rng.choice(["p", "q"], size=n))
    logit = -2.0 + 1.5 * X["x1"].to_numpy() + 1.2 * (X["city"] == "a").to_numpy()
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


def test_inputs_codes_and_numbers():
    X, _ = make_data(400, 0)
    inputs = TabularInputs().fit(X)
    assert inputs.cardinalities == [6, 3]  # categories + 1 slot for missing/unseen
    new = X.head(3).copy()
    new["city"] = pd.Categorical(["e", None, "b"], categories=["a", "b", "c", "d", "e"])
    numbers, codes = inputs.transform(new)
    assert codes[1, 0] == 0  # missing -> slot 0
    assert codes[2, 0] == 2  # "b" is the 2nd category
    assert codes[0, 0] == 5  # "e" is a known category (never seen in this sample, still coded)
    assert numbers.dtype == np.float32 and not np.isnan(numbers).any()
    assert numbers.shape[1] >= 6  # 6 numeric columns + missing flags


@needs_torch
def test_mlp_trains_and_predicts():
    X, y = make_data(3000, 1)
    model = TorchMLPClassifier(hidden=(32, 16), learning_rate=3e-3, batch_size=128,
                               max_epochs=15, patience=4, seed=0, threads=1)
    model.fit(X.iloc[:2400], y[:2400])
    p = model.predict_proba(X.iloc[2400:])[:, 1]
    assert p.shape == (600,) and np.all((p >= 0) & (p <= 1))
    assert roc_auc_score(y[2400:], p) > 0.75
    assert 1 <= model.best_epoch_ <= len(model.history_) <= 15


@needs_torch
def test_mlp_same_seed_same_predictions():
    X, y = make_data(1500, 2)
    settings = dict(hidden=(16,), batch_size=128, max_epochs=3, patience=3, seed=7, threads=1)
    a = TorchMLPClassifier(**settings).fit(X, y).predict_proba(X)
    b = TorchMLPClassifier(**settings).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b, atol=1e-5)
