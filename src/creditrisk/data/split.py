"""Freeze the train / valid / conformal / test split. Run it once.

Run from the project root:
    python -m creditrisk.data.split           # creates data/processed/splits.parquet
    python -m creditrisk.data.split --force   # recreates it (only if you really must)

Every later step reads this file, so all models are trained and tested on the same rows.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from creditrisk.config import get_path, load_config

SPLIT_FILE = "splits.parquet"
SPLIT_NAMES = ["train", "valid", "conformal", "test"]


def make_splits(
    app: pd.DataFrame,
    test_size: float = 0.20,
    conformal_size: float = 0.10,
    valid_size: float = 0.10,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Split applicants into train / valid / conformal / test, and give train rows a CV fold.

    `app` needs the columns SK_ID_CURR, TARGET and CODE_GENDER.
    Rows with CODE_GENDER == "XNA" are dropped (EDA decision).
    Every split keeps the same default rate (stratified by TARGET).

    Returns one row per applicant: SK_ID_CURR, TARGET, split, fold.
    `fold` is 1..n_folds for train rows and -1 for the other splits.
    """
    df = app.loc[app["CODE_GENDER"] != "XNA", ["SK_ID_CURR", "TARGET"]].reset_index(drop=True)

    rest, test = train_test_split(
        df, test_size=test_size, stratify=df["TARGET"], random_state=seed
    )
    conformal_share = conformal_size / (1 - test_size)
    rest, conformal = train_test_split(
        rest, test_size=conformal_share, stratify=rest["TARGET"], random_state=seed
    )
    valid_share = valid_size / (1 - test_size - conformal_size)
    train, valid = train_test_split(
        rest, test_size=valid_share, stratify=rest["TARGET"], random_state=seed
    )

    parts = [train, valid, conformal, test]
    out = pd.concat(
        [part.assign(split=name) for name, part in zip(SPLIT_NAMES, parts, strict=True)],
        ignore_index=True,
    )

    # 5-fold CV inside the train split, stratified by TARGET
    out["fold"] = -1
    is_train = (out["split"] == "train").to_numpy()
    train_targets = out.loc[is_train, "TARGET"].to_numpy()
    folds = np.empty(is_train.sum(), dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, fold_rows) in enumerate(skf.split(train_targets, train_targets), start=1):
        folds[fold_rows] = k
    out.loc[is_train, "fold"] = folds

    out["split"] = pd.Categorical(out["split"], categories=SPLIT_NAMES)
    out["fold"] = out["fold"].astype("int8")
    return out.sort_values("SK_ID_CURR").reset_index(drop=True)


def load_splits() -> pd.DataFrame:
    """Read the frozen split file."""
    path = get_path("processed") / SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make splits` first.")
    return pd.read_parquet(path)


def summarize(splits: pd.DataFrame, gender: pd.Series | None = None) -> pd.DataFrame:
    """Rows, share, default rate (and % women if `gender` is given) for each split."""
    g = splits.groupby("split", observed=True)
    table = pd.DataFrame(
        {
            "rows": g.size(),
            "share_%": (100 * g.size() / len(splits)).round(1),
            "default_rate_%": (100 * g["TARGET"].mean()).round(2),
        }
    )
    if gender is not None:
        is_female = splits["SK_ID_CURR"].map(gender) == "F"
        table["women_%"] = (100 * is_female.groupby(splits["split"], observed=True).mean()).round(1)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the data split.")
    parser.add_argument("--force", action="store_true", help="recreate an existing split file")
    args = parser.parse_args()

    out_file = get_path("processed") / SPLIT_FILE
    if out_file.exists() and not args.force:
        print(f"{out_file.name} already exists, so the split stays frozen.")
        print("Use --force only if you really must redo it (all results would change).")
        return

    cfg = load_config()
    s = cfg["splits"]
    app = pd.read_parquet(
        get_path("interim") / "application_train.parquet",
        columns=["SK_ID_CURR", "TARGET", "CODE_GENDER"],
    )
    splits = make_splits(
        app, s["test_size"], s["conformal_size"], s["valid_size"], s["n_folds"],
        cfg["project"]["seed"],
    )
    splits.to_parquet(out_file, index=False)

    gender = app.set_index("SK_ID_CURR")["CODE_GENDER"].astype(str)
    print(f"Dropped {len(app) - len(splits)} rows with CODE_GENDER == 'XNA'.\n")
    print(summarize(splits, gender).to_string(), "\n")
    train = splits[splits["split"] == "train"]
    folds = train.groupby("fold")["TARGET"].agg(rows="size", default_rate_pct="mean")
    folds["default_rate_pct"] = (100 * folds["default_rate_pct"]).round(2)
    print("CV folds inside train:")
    print(folds.to_string())
    print(f"\nsaved data/processed/{SPLIT_FILE}")


if __name__ == "__main__":
    main()
