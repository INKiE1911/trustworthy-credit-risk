"""Build the demo bundle used by the app and the API.

It holds N applicants: their raw application columns (prefixed RAW__, needed to rebuild
features after a what-if change) and their full feature row.

- Default: real applicants from the Kaggle test file (no labels; our test split is never
  touched). For local use only: the competition rules forbid sharing the data.
- --synthetic: made-up applicants (creditrisk.data.synthetic) run through the same feature
  code, plus a stand-in model trained only on made-up applicants (demo/model/). This is what a
  public demo uses: the real model and the real data never leave this machine.

Run: make demo-data              ->  data/processed/demo_applicants.parquet
     make demo-data-synthetic    ->  demo/demo_synthetic.parquet + demo/model/ (safe to commit)
"""

import argparse
import json

import numpy as np
import pandas as pd

from creditrisk.config import PROJECT_ROOT, get_path
from creditrisk.data.synthetic import make_fake_application, make_fake_history
from creditrisk.decision.conformal import class_cutoffs
from creditrisk.decision.threshold import break_even_threshold
from creditrisk.features.application import SENSITIVE
from creditrisk.features.build import KEY_COLS, build_features, load_features
from creditrisk.models import baselines
from creditrisk.models.final import save_final_model
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


def _synthetic_features(n_train: int, n_test: int, seed: int):
    """Made-up applicants with made-up histories, through the real feature code."""
    train = make_fake_application(n_train, with_target=True, seed=seed)
    test = make_fake_application(n_test, with_target=False, seed=seed + 1, first_id=900_000)
    tables = {"application_train": train, "application_test": test,
              **make_fake_history(np.r_[train["SK_ID_CURR"], test["SK_ID_CURR"]], seed + 2)}
    features = build_features(
        verbose=False, read=lambda name, columns=None: tables[name][columns or slice(None)])
    return features, test


def make_synthetic_bundle(n: int = 300, seed: int = 42) -> pd.DataFrame:
    features, test = _synthetic_features(200, n, seed)   # the builder needs a train table too
    return make_demo_bundle(features, test, n=n, seed=seed).copy().assign(DEMO_SYNTHETIC=True)


def make_synthetic_model(model_dir, n: int = 20_000, seed: int = 7) -> dict:
    """A stand-in model for the public demo, trained only on made-up applicants.

    Half of them train it, the other half sets its conformal cut-offs (as Step 11 did with the
    conformal split). Same features, money rule and alpha as the real model.
    """
    features, _ = _synthetic_features(n, 10, seed)
    labelled = features[features["is_test"] == 0].reset_index(drop=True)
    columns = [c for c in labelled.columns if c not in KEY_COLS + SENSITIVE]
    y = labelled["TARGET"].astype(int).to_numpy()
    fit = np.random.default_rng(seed).random(len(y)) < 0.5
    model = baselines.lightgbm(seed=seed, n_estimators=200, n_jobs=2)
    model.fit(labelled.loc[fit, columns], y[fit])
    p = model.predict_proba(labelled.loc[~fit, columns])[:, 1]
    q_repay, q_default = class_cutoffs(p, y[~fit], alpha=0.10)
    description = "Stand-in LightGBM trained on made-up applicants (not the real model)"
    save_final_model(model, columns, {"n_estimators": 200, "demo_model": True,
                                      "description": description}, model_dir=model_dir)
    decision = {"note": description, "margin": 0.10, "lgd": 0.50,
                "threshold": break_even_threshold(0.10, 0.50),
                "conformal": {"alpha": 0.10, "q_repay": q_repay, "q_default": q_default,
                              "approve_below": 1 - q_default, "decline_above": q_repay}}
    (model_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    (model_dir / "lightgbm_final.txt").unlink(missing_ok=True)   # the joblib is enough
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=1000, help="number of applicants")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic", action="store_true", help="made-up applicants")
    args = parser.parse_args()
    if args.synthetic:
        bundle = make_synthetic_bundle(n=min(args.n, 300), seed=args.seed)
        path = PROJECT_ROOT / "demo" / SYNTHETIC_FILE
        cut = make_synthetic_model(PROJECT_ROOT / "demo" / "model")["conformal"]
        print(f"stand-in model saved to demo/model (approve below {cut['approve_below']:.3f}, "
              f"decline above {cut['decline_above']:.3f})")
    else:
        raw_test = pd.read_parquet(get_path("interim") / "application_test.parquet")
        bundle = make_demo_bundle(load_features(), raw_test, n=args.n, seed=args.seed)
        path = get_path("processed") / DEMO_FILE
    bundle.to_parquet(path, index=False)
    print(f"saved {len(bundle):,} demo applicants x {bundle.shape[1]} columns to {path}")


if __name__ == "__main__":
    main()
