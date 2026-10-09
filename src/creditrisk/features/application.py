"""Features from the main application table: one row per applicant, columns prefixed APP_.

The cleaning rules come from the Step 2 EDA decisions.
"""

import numpy as np
import pandas as pd

from creditrisk.features.common import safe_divide as _safe_divide

# 18 near-constant columns found in EDA (one value in more than 99% of rows)
NEAR_CONSTANT = ["FLAG_MOBIL", "FLAG_CONT_MOBILE"] + [
    f"FLAG_DOCUMENT_{i}" for i in (2, 4, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
]
DOCUMENT_COLS = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
BUREAU_REQ_COLS = [
    f"AMT_REQ_CREDIT_BUREAU_{s}" for s in ("HOUR", "DAY", "WEEK", "MON", "QRT", "YEAR")
]
EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
DAYS_TO_YEARS = {
    "DAYS_BIRTH": "AGE_YEARS",
    "DAYS_EMPLOYED": "EMPLOYED_YEARS",
    "DAYS_REGISTRATION": "REGISTRATION_YEARS",
    "DAYS_ID_PUBLISH": "ID_PUBLISH_YEARS",
    "DAYS_LAST_PHONE_CHANGE": "PHONE_CHANGE_YEARS",
}
KEY_COLS = ["SK_ID_CURR", "TARGET", "is_test"]
# Kept in the table for fairness audits, but NOT used as a model input by default
SENSITIVE = ["APP_CODE_GENDER"]


def build_application_features(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Clean the application table and add domain features, for train + test together.

    Returns one row per SK_ID_CURR with TARGET (empty for test rows), is_test (0/1)
    and the feature columns, all prefixed APP_.
    """
    df = pd.concat([train.assign(is_test=0), test.assign(is_test=1)], ignore_index=True)
    return application_features_from_raw(df)


def application_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """The same cleaning and features for any rows that already have TARGET and is_test.

    Every step works row by row, so this also serves one applicant at a time: the app uses it
    to recompute the features after a what-if change, with exactly the training code.
    """
    df = df.copy()
    # train and test can have different category lists, so rebuild the text columns
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in text_cols:
        df[col] = df[col].astype("category")

    n_missing = df.drop(columns=KEY_COLS).isna().sum(axis=1)

    # --- cleaning (EDA decisions) ---
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype("int8")
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].where(df["DAYS_EMPLOYED"] != 365243)
    df["CODE_GENDER"] = (
        df["CODE_GENDER"].where(df["CODE_GENDER"] != "XNA").cat.remove_unused_categories()
    )

    # --- counts (documents are counted BEFORE dropping the near-constant flags) ---
    df["DOC_COUNT"] = df[DOCUMENT_COLS].sum(axis=1)
    df["BUREAU_REQ_TOTAL"] = df[BUREAU_REQ_COLS].sum(axis=1, min_count=1)
    df["N_MISSING"] = n_missing

    # --- days -> positive years ---
    years = {new: -df[old] / 365.25 for old, new in DAYS_TO_YEARS.items()}
    df = df.assign(**years).drop(columns=list(DAYS_TO_YEARS))

    # --- external scores ---
    ext = df[EXT_COLS]
    df["EXT_MEAN"] = ext.mean(axis=1)
    df["EXT_MIN"] = ext.min(axis=1)
    df["EXT_MAX"] = ext.max(axis=1)
    df["EXT_STD"] = ext.std(axis=1)
    df["EXT_PROD"] = ext.prod(axis=1, min_count=3)  # only when all three exist
    df["EXT_NA_COUNT"] = ext.isna().sum(axis=1)

    # --- domain ratios ---
    df["CREDIT_TO_INCOME"] = _safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_TO_INCOME"] = _safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_TO_CREDIT"] = _safe_divide(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["GOODS_TO_CREDIT"] = _safe_divide(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["CREDIT_MINUS_GOODS"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    df["INCOME_PER_PERSON"] = _safe_divide(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    df["CHILDREN_RATIO"] = _safe_divide(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])
    df["EMPLOYED_TO_AGE"] = _safe_divide(df["EMPLOYED_YEARS"], df["AGE_YEARS"])
    df["LOG_INCOME"] = np.log(df["AMT_INCOME_TOTAL"])

    df = df.drop(columns=NEAR_CONSTANT)
    df = df.rename(columns={c: f"APP_{c}" for c in df.columns if c not in KEY_COLS})
    df["is_test"] = df["is_test"].astype("int8")
    return df
