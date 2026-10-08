"""End-to-end test of the feature build on fake tables that have every real column."""

import numpy as np
import pytest
from conftest import make_fake_application, make_fake_history

from creditrisk.features.build import KEY_COLS, PREFIXES, build_features


@pytest.fixture(scope="module")
def built():
    train = make_fake_application(2_000, True, seed=5)
    test = make_fake_application(300, False, seed=6, first_id=700_000)
    tables = {"application_train": train, "application_test": test,
              **make_fake_history(np.r_[train["SK_ID_CURR"], test["SK_ID_CURR"]], seed=7)}

    def read(name, columns=None):
        table = tables[name]
        return table if columns is None else table[columns]  # fails if a column name is wrong

    return build_features(verbose=False, read=read), train, test


def test_one_row_per_applicant(built):
    features, train, test = built
    assert len(features) == len(train) + len(test)
    assert features["SK_ID_CURR"].is_unique


def test_every_column_has_a_table_prefix(built):
    features, *_ = built
    others = [c for c in features.columns if c not in KEY_COLS]
    assert all(c.startswith(tuple(PREFIXES)) for c in others)
    assert not features.columns.duplicated().any()


def test_history_flags_and_count_filling(built):
    features, *_ = built
    for flag, count in [("BUR_HAS_HISTORY", "BUR_COUNT"), ("PREV_HAS_HISTORY", "PREV_COUNT"),
                        ("INST_HAS_HISTORY", "INST_COUNT"), ("POS_HAS_HISTORY", "POS_LOANS"),
                        ("CC_HAS_HISTORY", "CC_CARDS")]:
        missing = features[flag] == 0
        assert missing.any() and (~missing).any()
        assert (features.loc[missing, count] == 0).all()
        assert (features.loc[~missing, count] > 0).all()


def test_no_infinite_values(built):
    features, *_ = built
    numbers = features.select_dtypes("number").to_numpy(dtype=float)
    assert not np.isinf(numbers).any()
