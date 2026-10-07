"""Build the feature table that every model uses.

Run from the project root (again after any change to the feature code):
    python -m creditrisk.features.build

Step 4: application features only. Step 5 adds bureau, previous applications,
installments, POS/cash and credit cards.
"""

import time

import pandas as pd

from creditrisk.config import get_path
from creditrisk.data.to_parquet import memory_mb, shrink
from creditrisk.features.application import build_application_features

FEATURE_FILE = "features.parquet"
KEY_COLS = ["SK_ID_CURR", "TARGET", "is_test"]


def build_features() -> pd.DataFrame:
    """Build all features for train + test applicants (one row per SK_ID_CURR)."""
    interim = get_path("interim")
    train = pd.read_parquet(interim / "application_train.parquet")
    test = pd.read_parquet(interim / "application_test.parquet")
    features = build_application_features(train, test)
    # Step 5: join the history-table features here
    assert features["SK_ID_CURR"].is_unique, "feature table must have one row per applicant"
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
    print(
        f"{len(features):,} applicants ({len(features) - n_test:,} train, {n_test:,} test) "
        f"x {n_features} features, {memory_mb(features):.0f} MB in memory, "
        f"{time.time() - start:.0f}s"
    )
    print(f"saved data/processed/{FEATURE_FILE}")


if __name__ == "__main__":
    main()
