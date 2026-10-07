"""Tests for creditrisk.features.application (on a made-up table with all 122 real columns)."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.features.application import (
    DOCUMENT_COLS,
    NEAR_CONSTANT,
    build_application_features,
)


@pytest.fixture(scope="module")
def feats(fake_train, fake_test):
    return build_application_features(fake_train, fake_test)


def test_one_row_per_applicant(fake_train, fake_test, feats):
    assert len(feats) == len(fake_train) + len(fake_test)
    assert feats["SK_ID_CURR"].is_unique
    assert feats["is_test"].sum() == len(fake_test)
    assert feats.loc[feats["is_test"] == 1, "TARGET"].isna().all()
    assert feats.loc[feats["is_test"] == 0, "TARGET"].notna().all()


def test_every_feature_has_the_prefix(feats):
    others = [c for c in feats.columns if c not in ("SK_ID_CURR", "TARGET", "is_test")]
    assert all(c.startswith("APP_") for c in others)


def test_365243_becomes_nan_plus_flag(fake_train, fake_test, feats):
    raw = pd.concat([fake_train, fake_test], ignore_index=True)
    anomaly = (raw["DAYS_EMPLOYED"] == 365243).to_numpy()
    assert (feats["APP_DAYS_EMPLOYED_ANOM"].to_numpy() == anomaly).all()
    assert feats.loc[anomaly, "APP_EMPLOYED_YEARS"].isna().all()
    assert feats.loc[~anomaly, "APP_EMPLOYED_YEARS"].ge(0).all()


def test_gender_xna_becomes_missing(feats):
    assert "XNA" not in feats["APP_CODE_GENDER"].cat.categories
    assert feats["APP_CODE_GENDER"].isna().sum() == 4


def test_documents_counted_before_dropping(fake_train, fake_test, feats):
    raw = pd.concat([fake_train, fake_test], ignore_index=True)
    assert (feats["APP_DOC_COUNT"].to_numpy() == raw[DOCUMENT_COLS].sum(axis=1).to_numpy()).all()
    assert not any(f"APP_{c}" in feats.columns for c in NEAR_CONSTANT)


def test_ratios_and_years_by_hand(fake_train, fake_test, feats):
    raw = pd.concat([fake_train, fake_test], ignore_index=True)
    i = 7
    assert feats.loc[i, "APP_CREDIT_TO_INCOME"] == pytest.approx(
        raw.loc[i, "AMT_CREDIT"] / raw.loc[i, "AMT_INCOME_TOTAL"])
    assert feats.loc[i, "APP_AGE_YEARS"] == pytest.approx(-raw.loc[i, "DAYS_BIRTH"] / 365.25)
    assert feats.loc[i, "APP_LOG_INCOME"] == pytest.approx(np.log(raw.loc[i, "AMT_INCOME_TOTAL"]))


def test_no_infinite_values(feats):
    numbers = feats.select_dtypes("number")
    assert not np.isinf(numbers.to_numpy(dtype=float)).any()


def test_ext_product_only_when_all_three_exist(fake_train, fake_test, feats):
    raw = pd.concat([fake_train, fake_test], ignore_index=True)
    all_three = raw[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].notna().all(axis=1)
    assert feats.loc[~all_three, "APP_EXT_PROD"].isna().all()
    assert feats.loc[all_three, "APP_EXT_PROD"].notna().all()
