"""Step 7: imbalance variants, tuning search (with resume) and the saved final model."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.models import tune
from creditrisk.models.data import column_groups
from creditrisk.models.final import load_final_model, save_final_model
from creditrisk.models.imbalance import LightSMOTENC, imbalance_models


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 1200
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
    X.loc[rng.random(n) < 0.2, "x0"] = np.nan  # gaps, like the real data
    X["empty"] = np.nan  # a fully empty column must not break anything
    X["city"] = pd.Categorical(rng.choice(["a", "b", "c", None], size=n))
    X["job"] = pd.Categorical(rng.choice(["p", "q"], size=n))
    logit = -2.6 + 1.2 * X["x1"].to_numpy() + (X["job"] == "p").to_numpy()
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)  # about 10% positives
    folds = np.arange(n) % 3 + 1
    return X, y, folds


def test_every_imbalance_variant_trains_and_predicts(data):
    X, y, _ = data
    makers = imbalance_models(column_groups(X), seed=0)
    assert list(makers) == ["none", "class_weights", "undersampling", "imputed_no_smote", "smote"]
    for name, make in makers.items():
        model = make()
        model.fit(X.iloc[:900], y[:900])
        prob = model.predict_proba(X.iloc[900:])[:, 1]
        assert prob.shape == (300,), name  # the held-out rows are never resampled
        assert np.all((prob >= 0) & (prob <= 1)), name


def test_resampling_changes_only_the_training_rows(data):
    X, y, _ = data
    makers = imbalance_models(column_groups(X), seed=0, ratio=0.5)
    under = makers["undersampling"]().fit(X, y)
    assert under["model"].n_features_in_ == X.shape[1]
    X_res, y_res = under["sample"].fit_resample(X, y)
    assert y_res.sum() == y.sum()  # every defaulter kept
    assert (y_res == 0).sum() == 2 * y.sum()  # 1 defaulter per 2 others
    smote = makers["smote"]()
    Z = smote["prep"].fit_transform(X)
    assert not np.isnan(Z).any() and Z.shape[1] == X.shape[1]  # empty column kept in place
    Z_res, y_smote = smote["sample"].fit_resample(Z, y)
    assert (y_smote == 1).sum() == round(0.5 * (y == 0).sum())  # synthetic defaulters added
    assert (y_smote == 0).sum() == (y == 0).sum()  # nobody removed
    assert np.array_equal(Z_res[: len(Z)], Z)  # the real rows are untouched


def test_light_smotenc_makes_points_between_two_real_defaulters():
    rng = np.random.default_rng(1)
    numbers = rng.normal(size=(60, 3))
    codes = rng.integers(0, 4, size=(60, 2)).astype(float)
    X = np.hstack([numbers, codes])
    y = np.r_[np.ones(10, dtype=int), np.zeros(50, dtype=int)]
    sampler = LightSMOTENC(n_numeric=3, sampling_strategy=1.0, k_neighbors=3, random_state=0)
    X_res, y_res = sampler.fit_resample(X, y)
    assert (y_res == 1).sum() == 50 and len(y_res) == 100
    real = X[y == 1]
    for row in X_res[60:]:
        # some pair of real defaulters (a, b) has the new numbers on the segment a -> b,
        # and the new text codes equal to the codes of a or b
        found = False
        for a in real:
            for b in real:
                d = b[:3] - a[:3]
                if not d.any():
                    continue
                gap = np.dot(row[:3] - a[:3], d) / np.dot(d, d)
                on_segment = -1e-9 <= gap <= 1 + 1e-9 and np.allclose(a[:3] + gap * d, row[:3])
                same_codes = np.array_equal(row[3:], a[3:]) or np.array_equal(row[3:], b[3:])
                found = found or (on_segment and same_codes)
        assert found
    again = LightSMOTENC(n_numeric=3, sampling_strategy=1.0, k_neighbors=3, random_state=0)
    assert np.array_equal(again.fit_resample(X, y)[0], X_res)  # reproducible


def test_class_weights_raise_the_average_probability(data):
    X, y, _ = data
    makers = imbalance_models(column_groups(X), seed=0)
    plain = makers["none"]().fit(X, y).predict_proba(X)[:, 1].mean()
    weighted = makers["class_weights"]().fit(X, y).predict_proba(X)[:, 1].mean()
    assert weighted > 1.5 * plain  # weighting inflates probabilities (calibration damage)


def test_grid_and_random_candidates():
    grid = tune.grid_candidates()
    assert len(grid) == 27 and len({tuple(sorted(g.items())) for g in grid}) == 27
    first, again = tune.random_candidates(20, seed=1), tune.random_candidates(20, seed=1)
    assert first == again  # reproducible
    for params in first:
        for name, (low, high, _, whole) in tune.SPACE.items():
            assert low <= params[name] <= high
            assert isinstance(params[name], int) == whole


def test_holdout_score_uses_early_stopping(data):
    X, y, folds = data
    auc, trees = tune.holdout_score({"learning_rate": 0.1, "num_leaves": 8}, X, y, folds,
                                    max_trees=300, patience=10)
    assert 0.5 < auc <= 1 and 1 <= trees < 300


def test_run_search_saves_and_resumes(tmp_path):
    calls = []

    def fake_score(params):
        calls.append(params)
        return params["num_leaves"] / 1000, 7

    candidates = tune.grid_candidates()[:4]
    tune.run_search("grid", candidates[:2] + candidates[2:], fake_score, tmp_path, verbose=False)
    assert len(calls) == 4
    table = tune.run_search("grid", candidates, fake_score, tmp_path, verbose=False)
    assert len(calls) == 4  # nothing re-run
    assert list(table["trial"]) == [0, 1, 2, 3]
    assert table["best_so_far"].is_monotonic_increasing
    with pytest.raises(ValueError, match="different candidates"):
        tune.run_search("grid", tune.random_candidates(4), fake_score, tmp_path, verbose=False)


def test_run_optuna_resumes(tmp_path):
    calls = []

    def fake_score(params):
        calls.append(params)
        return -abs(np.log(params["learning_rate"] / 0.03)), 11

    first = tune.run_optuna(3, fake_score, tmp_path, startup=2, verbose=False)
    assert len(first) == 3 and len(calls) == 3
    second = tune.run_optuna(5, fake_score, tmp_path, startup=2, verbose=False)
    assert len(second) == 5 and len(calls) == 5  # only the 2 missing trials ran
    assert set(second["params"][0]) == set(tune.SPACE)
    best = tune.best_trial(first.assign(method="a"), second)
    assert best["auc"] == max(second["auc"].max(), first["auc"].max())


def test_final_model_round_trip(data, tmp_path):
    X, y, _ = data
    model = tune.tuned_lightgbm({"num_leaves": 8}, trees=30).fit(X, y)
    save_final_model(model, X.columns, {"cv_roc_auc": 0.5}, model_dir=tmp_path)
    loaded, details = load_final_model(tmp_path)
    assert details["features"] == list(X.columns) and details["cv_roc_auc"] == 0.5
    assert np.allclose(loaded.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1])
    assert (tmp_path / "lightgbm_final.txt").exists()
