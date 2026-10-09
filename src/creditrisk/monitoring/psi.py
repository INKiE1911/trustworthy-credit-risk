"""Step 16: drift monitoring with the population stability index (PSI), the credit-risk standard.

Bin the reference data (deciles for numbers, one bin per category for text, plus a bin for
missing values), then compare the share of each bin in the new data:
    PSI = sum over bins of (new% - ref%) * ln(new% / ref%)
Below 0.1 stable, 0.1-0.25 watch, above 0.25 a significant shift: recompute the conformal
cut-offs (their guarantee assumes new applicants look like the old ones) or retrain.

    from creditrisk.monitoring.psi import psi, psi_table, status
    psi(train_scores, new_scores)          # one number
    psi_table(X_ref, X_new, columns)       # one row per column, with its status
"""

import numpy as np
import pandas as pd

FLOOR = 1e-4  # an empty bin would make ln(0); count it as 0.01% instead


def status(value: float) -> str:
    return "stable" if value < 0.1 else "watch" if value <= 0.25 else "shift"


def _shares(values: pd.Series, bins) -> np.ndarray:
    counts = pd.Series(values).value_counts(dropna=False).reindex(bins, fill_value=0)
    return np.maximum(counts.to_numpy() / max(len(values), 1), FLOOR)


def psi(reference, new, n_bins: int = 10) -> float:
    """PSI of new against reference (numbers: reference quantile bins; text: categories)."""
    ref, cur = pd.Series(reference), pd.Series(new)
    if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(cur):
        edges = np.unique(np.nanquantile(ref.astype(float), np.linspace(0, 1, n_bins + 1)[1:-1]))

        def to_bins(s):  # bin number, or -1 for missing; values beyond the edges go to the ends
            return pd.Series(np.where(s.isna(), -1, np.searchsorted(edges, s, side="right")))

        ref, cur = to_bins(ref), to_bins(cur)
        bins = list(range(-1, len(edges) + 1))
    else:
        ref, cur = ref.astype(object), cur.astype(object)
        known = set(ref.dropna())
        cur = cur.where(cur.isna() | cur.isin(known), "(unseen)")
        bins = list(known) + [np.nan, "(unseen)"]
    p, q = _shares(ref, bins), _shares(cur, bins)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_table(reference: pd.DataFrame, new: pd.DataFrame, columns) -> pd.DataFrame:
    """PSI and status for each column, largest shift first."""
    rows = {c: psi(reference[c], new[c]) for c in columns}
    table = pd.DataFrame({"PSI": rows}).sort_values("PSI", ascending=False)
    table["status"] = table["PSI"].map(status)
    return table
