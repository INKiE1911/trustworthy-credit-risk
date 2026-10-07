"""Every baseline must train and give valid probabilities on the real column layout."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.data.split import make_splits
from creditrisk.features.application import build_application_features
from creditrisk.models import baselines
from creditrisk.models.data import column_groups, get_xy


@pytest.fixture(scope="module")
def train_data(fake_train, fake_test):
    feats = build_application_features(fake_train, fake_test)
    splits = make_splits(fake_train, seed=0)
    return get_xy(feats, splits, "train")


def test_gender_is_not_a_model_input(train_data):
    X, *_ = train_data
    assert "APP_CODE_GENDER" not in X.columns


@pytest.mark.parametrize("name", ["dummy", "naive_bayes", "logistic_regression",
                                  "decision_tree", "lightgbm"])
def test_baseline_trains_and_predicts(train_data, name):
    X, y, folds, _ = train_data
    groups = column_groups(X)
    makers = {
        "dummy": baselines.dummy,
        "naive_bayes": lambda: baselines.naive_bayes(groups),
        "logistic_regression": lambda: baselines.logistic_regression(groups),
        "decision_tree": lambda: baselines.decision_tree(groups, min_samples_leaf=20),
        "lightgbm": lambda: baselines.lightgbm(n_estimators=30),
    }
    fit = folds != 1
    model = makers[name]().fit(X[fit], y[fit])
    p = model.predict_proba(X[~fit])[:, 1]
    assert len(p) == (~fit).sum()
    assert np.all((p >= 0) & (p <= 1))
    assert not pd.isna(p).any()
