"""Small helpers shared by the feature modules."""

import numpy as np
import pandas as pd


def safe_divide(a, b) -> pd.Series:
    """a / b, with division by zero turned into NaN instead of inf."""
    return (a / b).replace([np.inf, -np.inf], np.nan)


def attach(base: pd.DataFrame, part: pd.DataFrame, count_cols: list[str],
           flag: str) -> pd.DataFrame:
    """Left-join one table's per-applicant features (indexed by SK_ID_CURR) onto `base`.

    Applicants with no rows in that table get 0 for the count columns, NaN for everything
    else, and 0 in the `flag` column (1 = the applicant appears in that table).
    """
    assert part.index.is_unique, "a feature table must have one row per SK_ID_CURR"
    out = base.join(part, on="SK_ID_CURR")
    out[flag] = base["SK_ID_CURR"].isin(part.index).astype("int8").to_numpy()
    for col in count_cols:
        out[col] = out[col].fillna(0)
    assert len(out) == len(base), "the join must not add or remove rows"
    return out
