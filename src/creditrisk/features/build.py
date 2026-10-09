"""Build the feature table that every model uses (all seven tables).

Run from the project root (again after any change to the feature code):
    python -m creditrisk.features.build

Each history table is loaded with only the columns it needs, summarised per applicant,
joined, and freed before the next one, so memory stays low.
"""

import gc
import time

import pandas as pd

from creditrisk.config import get_path
from creditrisk.data.to_parquet import memory_mb, shrink
from creditrisk.features import bureau, credit_card, installments, pos_cash, previous
from creditrisk.features.application import build_application_features
from creditrisk.features.common import attach, safe_divide

FEATURE_FILE = "features.parquet"
KEY_COLS = ["SK_ID_CURR", "TARGET", "is_test"]
PREFIXES = ["APP_", "BUR_", "BB_", "PREV_", "INST_", "POS_", "CC_"]


def _read_parquet(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(get_path("interim") / f"{name}.parquet", columns=columns)


def add_cross_table_features(features: pd.DataFrame) -> pd.DataFrame:
    """Features combining the application with a history table (named after the history table)."""
    income = features["APP_AMT_INCOME_TOTAL"]
    features["BUR_DEBT_TO_INCOME"] = safe_divide(features["BUR_ACTIVE_DEBT_TOTAL"], income)
    features["BUR_ANNUITY_BURDEN"] = safe_divide(
        features["BUR_ACTIVE_ANNUITY_TOTAL"].fillna(0) + features["APP_AMT_ANNUITY"], income
    )
    features["PREV_CURRENT_TO_MEAN_CREDIT"] = safe_divide(features["APP_AMT_CREDIT"],
                                                          features["PREV_CREDIT_MEAN"])
    return features


def build_features(verbose: bool = True, read=None) -> pd.DataFrame:
    """Build all features for train + test applicants (one row per SK_ID_CURR).

    `read(name, columns)` loads one table; by default from data/interim (tests pass fakes).
    """
    _read = read or _read_parquet
    steps = [
        ("bureau + bureau_balance", "BUR_HAS_HISTORY", bureau.COUNT_COLS,
         lambda: bureau.build_bureau_features(_read("bureau", bureau.BUREAU_COLS),
                                              _read("bureau_balance", bureau.BALANCE_COLS))),
        ("previous_application", "PREV_HAS_HISTORY", previous.COUNT_COLS,
         lambda: previous.build_previous_features(
             _read("previous_application", previous.PREVIOUS_COLS))),
        ("installments_payments", "INST_HAS_HISTORY", installments.COUNT_COLS,
         lambda: installments.build_installments_features(
             _read("installments_payments", installments.INSTALLMENTS_COLS))),
        ("pos_cash_balance", "POS_HAS_HISTORY", pos_cash.COUNT_COLS,
         lambda: pos_cash.build_pos_features(_read("pos_cash_balance", pos_cash.POS_COLS))),
        ("credit_card_balance", "CC_HAS_HISTORY", credit_card.COUNT_COLS,
         lambda: credit_card.build_credit_card_features(
             _read("credit_card_balance", credit_card.CREDIT_CARD_COLS))),
    ]

    start = time.time()
    features = build_application_features(_read("application_train"), _read("application_test"))
    n_rows = len(features)
    if verbose:
        n_app = features.shape[1] - 3
        print(f"{'application':<26} {n_app:>4} features  ({time.time() - start:.0f}s)")
    for label, flag, count_cols, make_part in steps:
        start = time.time()
        part = make_part()
        features = attach(features, part, count_cols, flag)
        if verbose:
            covered = features.loc[features["is_test"] == 0, flag].mean()
            print(f"{label:<26} {part.shape[1] + 1:>4} features  "
                  f"covers {covered:6.1%} of train applicants  ({time.time() - start:.0f}s)")
        del part
        gc.collect()

    features = add_cross_table_features(features)
    useless = [c for c in features.columns
               if c not in KEY_COLS and features[c].nunique(dropna=True) <= 1]
    if useless and verbose:
        print(f"WARNING: columns with no information (empty or constant): {useless}")
    assert len(features) == n_rows and features["SK_ID_CURR"].is_unique
    assert not features.columns.duplicated().any(), "duplicate column names"
    return shrink(features)


def load_features(columns: list[str] | None = None) -> pd.DataFrame:
    """Read the saved feature table."""
    path = get_path("processed") / FEATURE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make features` first.")
    return pd.read_parquet(path, columns=columns)


def main() -> None:
    start = time.time()
    features = build_features()
    features.to_parquet(get_path("processed") / FEATURE_FILE, index=False)
    n_features = len([c for c in features.columns if c not in KEY_COLS])
    n_test = int(features["is_test"].sum())
    per_table = {p.rstrip("_"): sum(c.startswith(p) for c in features.columns) for p in PREFIXES}
    print(f"\nfeatures per table: {per_table}")
    print(
        f"{len(features):,} applicants ({len(features) - n_test:,} train, {n_test:,} test) "
        f"x {n_features} features, {memory_mb(features):.0f} MB in memory, "
        f"{time.time() - start:.0f}s total"
    )
    print(f"saved data/processed/{FEATURE_FILE}")


if __name__ == "__main__":
    main()
