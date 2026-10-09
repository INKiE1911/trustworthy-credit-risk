"""Step 13: fairness audit, the decision rates per group (gender, age band).

Per group: approval rate, the share of good customers wrongly declined (false positive rate),
the share of defaulters declined (true positive rate), ROC-AUC, and mean predicted vs actual
default rate (calibration in the large).
The gaps compare the best and worst group. The main one is equal opportunity (Hardt et al. 2016):
good customers should be declined at the same rate whatever their group.

    from creditrisk.fairness.audit import age_band, audit, gaps
    table = audit(y, p, groups=age_band(X["APP_AGE_YEARS"]), approved=p < 0.1667)
    gaps(table)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

AGE_BANDS = ([0, 30, 45, 60, np.inf], ["<30", "30-45", "45-60", "60+"])


def age_band(age_years) -> pd.Series:
    """<30, 30-45, 45-60, 60+ (left edge included)."""
    bins, labels = AGE_BANDS
    return pd.cut(pd.Series(age_years, dtype=float), bins, labels=labels, right=False)


def audit(y, p, groups, approved) -> pd.DataFrame:
    """One row per group: p gives ROC-AUC and calibration, approved (bool) the decision rates."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p, dtype=float),
                       "group": pd.Series(groups).array, "approved": np.asarray(approved, bool)})
    rows = {}
    for g, d in df.groupby("group", observed=True, sort=True):
        rows[g] = {"applicants": len(d), "default rate": d["y"].mean(),
                   "mean predicted": d["p"].mean(), "ROC-AUC": roc_auc_score(d["y"], d["p"]),
                   "approval rate": d["approved"].mean(),
                   "good customers declined": 1 - d.loc[d["y"] == 0, "approved"].mean(),
                   "defaulters declined": 1 - d.loc[d["y"] == 1, "approved"].mean()}
    return pd.DataFrame(rows).T


def gaps(table: pd.DataFrame) -> dict:
    """Largest gap between groups for each fairness definition."""
    def spread(col):
        return float(table[col].max() - table[col].min())

    return {"demographic parity difference": spread("approval rate"),
            "equal opportunity gap": spread("good customers declined"),
            "equalized odds difference": max(spread("good customers declined"),
                                             spread("defaulters declined")),
            "disparate impact ratio": float(table["approval rate"].min()
                                            / table["approval rate"].max())}
