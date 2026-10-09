"""Build the demo bundle used by the app and the API.

It holds N applicants: their raw application columns (prefixed RAW__, needed to rebuild
features after a what-if change) and their full feature row.

- Default: real applicants from the Kaggle test file (no labels; our test split is never
  touched). For local use only: the competition rules forbid sharing the data.
- --synthetic: made-up applicants (creditrisk.data.synthetic) run through the same feature
  code. This is the bundle for a public demo.

Run: make demo-data              ->  data/processed/demo_applicants.parquet
     make demo-data-synthetic    ->  demo/demo_synthetic.parquet (small, safe to commit)
"""

import argparse

import numpy as np
import pandas as pd

from creditrisk.config import PROJECT_ROOT, get_path
from creditrisk.data.synthetic import make_fake_application, make_fake_history
from creditrisk.features.build import build_features, load_features
from creditrisk.serving.service import DEMO_FILE, RAW_PREFIX, SYNTHETIC_FILE


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


def make_synthetic_bundle(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Made-up applicants with made-up histories, through the real feature code."""
    train = make_fake_application(200, with_target=True, seed=seed)  # the builder needs one
    test = make_fake_application(n, with_target=False, seed=seed + 1, first_id=900_000)
    tables = {"application_train": train, "application_test": test,
              **make_fake_history(np.r_[train["SK_ID_CURR"], test["SK_ID_CURR"]], seed + 2)}
    features = build_features(
        verbose=False, read=lambda name, columns=None: tables[name][columns or slice(None)])
    return make_demo_bundle(features, test, n=n, seed=seed).copy().assign(DEMO_SYNTHETIC=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000, help="number of applicants")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="made-up applicants")
    args = parser.parse_args()
    if args.synthetic:
        bundle = make_synthetic_bundle(n=min(args.n, 300), seed=args.seed)
        path = PROJECT_ROOT / "demo" / SYNTHETIC_FILE
        path.parent.mkdir(exist_ok=True)
    else:
        raw_test = pd.read_parquet(get_path("interim") / "application_test.parquet")
        bundle = make_demo_bundle(load_features(), raw_test, n=args.n, seed=args.seed)
        path = get_path("processed") / DEMO_FILE
    bundle.to_parquet(path, index=False)
    print(f"saved {len(bundle):,} demo applicants x {bundle.shape[1]} columns to {path}")


if __name__ == "__main__":
    main()
