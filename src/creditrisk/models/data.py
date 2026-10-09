"""Get model-ready X, y, folds and ids for one split."""

import pandas as pd

from creditrisk.features.application import KEY_COLS, SENSITIVE


def get_xy(features: pd.DataFrame, splits: pd.DataFrame, split: str = "train",
           include_sensitive: bool = False):
    """Rows of one split joined to their features.

    Returns X (DataFrame), y (0/1 array), folds (array, -1 outside train), ids (SK_ID_CURR).
    Gender is left out of X unless include_sensitive=True (it is used only for fairness checks).
    """
    rows = splits.loc[splits["split"] == split, ["SK_ID_CURR", "fold"]]
    df = rows.merge(features, on="SK_ID_CURR", how="left", validate="one_to_one")
    cols = [c for c in features.columns if c not in KEY_COLS]
    if not include_sensitive:
        cols = [c for c in cols if c not in SENSITIVE]
    return (
        df[cols],
        df["TARGET"].astype(int).to_numpy(),
        df["fold"].to_numpy(),
        df["SK_ID_CURR"].to_numpy(),
    )


def column_groups(X: pd.DataFrame, max_onehot: int = 10) -> dict:
    """Split columns into numeric, small text (one-hot) and large text (target encoding)."""
    categorical = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    numeric = [c for c in X.columns if c not in categorical]
    cat_low = [c for c in categorical if len(X[c].cat.categories) <= max_onehot]
    cat_high = [c for c in categorical if c not in cat_low]
    return {
        "numeric": numeric, "categorical": categorical, "cat_low": cat_low, "cat_high": cat_high
    }
