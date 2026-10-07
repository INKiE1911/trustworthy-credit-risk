"""Tests for creditrisk.evaluation.cv."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from creditrisk.evaluation.cv import results_table, run_cv


def _data(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(X["signal"] * 2 - 2)))).astype(int)
    folds = rng.integers(1, 6, n)
    return X, y.to_numpy(), folds


def test_every_row_gets_an_out_of_fold_prediction(tmp_path):
    X, y, folds = _data()
    ids = np.arange(len(y))
    result = run_cv(LogisticRegression, X, y, folds, name="unit", ids=ids, n_boot=50,
                    log=False, oof_dir=tmp_path)
    assert not np.isnan(result.oof).any()
    assert list(result.fold_metrics.index) == [1, 2, 3, 4, 5]
    assert result.summary["roc_auc"] > 0.75
    s = result.summary
    assert s["roc_auc_low"] <= s["roc_auc"] <= s["roc_auc_high"]
    saved = pd.read_parquet(tmp_path / "unit.parquet")
    assert len(saved) == len(y) and saved["p"].notna().all()


def test_results_table_sorted_best_first():
    X, y, folds = _data()
    good = run_cv(LogisticRegression, X, y, folds, name="good", n_boot=20, log=False)
    bad = run_cv(LogisticRegression, X[["noise"]], y, folds, name="bad", n_boot=20, log=False)
    table = results_table([bad, good])
    assert list(table.index) == ["good", "bad"]
