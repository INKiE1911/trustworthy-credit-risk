"""Tests for creditrisk.data.split (uses a made-up applicant table)."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.data.split import make_splits, summarize


@pytest.fixture(scope="module")
def fake_app():
    rng = np.random.default_rng(0)
    n = 20_000
    app = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100_000, 100_000 + n),
            "TARGET": (rng.random(n) < 0.08).astype("int8"),
            "CODE_GENDER": rng.choice(["F", "M"], n, p=[0.66, 0.34]),
        }
    )
    app.loc[[5, 50, 500, 5000], "CODE_GENDER"] = "XNA"
    app["CODE_GENDER"] = app["CODE_GENDER"].astype("category")
    return app


@pytest.fixture(scope="module")
def splits(fake_app):
    return make_splits(fake_app, seed=42)


def test_xna_rows_dropped_everyone_else_kept_once(fake_app, splits):
    assert len(splits) == len(fake_app) - 4
    assert splits["SK_ID_CURR"].is_unique
    assert not splits["SK_ID_CURR"].isin(fake_app.loc[[5, 50, 500, 5000], "SK_ID_CURR"]).any()


def test_split_sizes(splits):
    shares = splits["split"].value_counts(normalize=True)
    for name, expected in {"train": 0.6, "valid": 0.1, "conformal": 0.1, "test": 0.2}.items():
        assert abs(shares[name] - expected) < 0.002


def test_same_default_rate_in_every_split(splits):
    rates = splits.groupby("split", observed=True)["TARGET"].mean()
    assert (rates - splits["TARGET"].mean()).abs().max() < 0.003


def test_folds_only_in_train_and_balanced(splits):
    train = splits[splits["split"] == "train"]
    assert set(train["fold"]) == {1, 2, 3, 4, 5}
    assert (splits.loc[splits["split"] != "train", "fold"] == -1).all()
    sizes = train["fold"].value_counts()
    assert sizes.max() - sizes.min() <= 2
    fold_rates = train.groupby("fold")["TARGET"].mean()
    assert (fold_rates - train["TARGET"].mean()).abs().max() < 0.005


def test_same_seed_gives_same_split(fake_app, splits):
    pd.testing.assert_frame_equal(make_splits(fake_app, seed=42), splits)


def test_other_seed_gives_other_split(fake_app, splits):
    assert not make_splits(fake_app, seed=7)["split"].equals(splits["split"])


def test_summary_table(fake_app, splits):
    gender = fake_app.set_index("SK_ID_CURR")["CODE_GENDER"].astype(str)
    table = summarize(splits, gender)
    assert list(table.index) == ["train", "valid", "conformal", "test"]
    assert table["rows"].sum() == len(splits)
    assert table["women_%"].between(60, 72).all()
