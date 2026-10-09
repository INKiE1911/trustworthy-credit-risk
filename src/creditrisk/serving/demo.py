"""Build the demo bundle used by the app and the API.

It holds N real applicants from the Kaggle test file: their raw application columns
(prefixed RAW__, needed to rebuild features after a what-if change) and their full feature
row. These applicants have no labels, and our locked test split is never touched.

Run: make demo-data   ->  data/processed/demo_applicants.parquet
"""

import argparse

import numpy as np
import pandas as pd

from creditrisk.config import get_path
from creditrisk.features.build import load_features
from creditrisk.serving.service import DEMO_FILE, RAW_PREFIX


def make_demo_bundle(features: pd.DataFrame, raw_test: pd.DataFrame, n: int = 1000,
                     seed: int = 42) -> pd.DataFrame:
    test_ids = features.loc[features["is_test"] == 1, "SK_ID_CURR"].to_numpy()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(test_ids, size=min(n, len(test_ids)), replace=False)
    raw = raw_test[raw_test["SK_ID_CURR"].isin(chosen)]
    raw = raw.rename(columns=lambda c: c if c == "SK_ID_CURR" else RAW_PREFIX + c)
    rows = features[features["SK_ID_CURR"].isin(chosen)]
    bundle = raw.merge(rows, on="SK_ID_CURR", validate="one_to_one")
    return bundle.sort_values("SK_ID_CURR").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000, help="number of applicants")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw_test = pd.read_parquet(get_path("interim") / "application_test.parquet")
    bundle = make_demo_bundle(load_features(), raw_test, n=args.n, seed=args.seed)
    path = get_path("processed") / DEMO_FILE
    bundle.to_parquet(path, index=False)
    print(f"saved {len(bundle):,} demo applicants x {bundle.shape[1]} columns to {path}")


if __name__ == "__main__":
    main()
