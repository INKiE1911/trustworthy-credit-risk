"""Every Step 6 model must train and give valid probabilities on the real column layout."""

import numpy as np
import pytest

from creditrisk.data.split import make_splits
from creditrisk.features.application import build_application_features
from creditrisk.models.data import column_groups, get_xy
from creditrisk.models.ladder import ladder_models

NAMES = list(ladder_models({"numeric": [], "categorical": [], "cat_low": [], "cat_high": []}))


@pytest.fixture(scope="module")
def data(fake_train, fake_test):
    feats = build_application_features(fake_train, fake_test)
    X, y, folds, _ = get_xy(feats, make_splits(fake_train, seed=0), "train")
    # the real data has an all-empty column; a constant one must not break anything either
    X = X.assign(EMPTY_COLUMN=np.nan, CONSTANT_COLUMN=0.0)
    return X, y, folds


@pytest.mark.parametrize("name", NAMES)
def test_model_trains_and_predicts(data, name):
    X, y, folds = data
    models = ladder_models(column_groups(X), seed=0, fast=True)
    fit = folds != 1
    p = models[name]().fit(X[fit], y[fit]).predict_proba(X[~fit])[:, 1]
    assert len(p) == (~fit).sum()
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
