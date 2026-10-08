"""Features from installments_payments (payment history on old Home Credit loans). Prefix INST_.

One installment can be paid in several parts (several rows), so rows are first combined per
installment, then summarised per old loan and per applicant. The table's own SK_ID_CURR is
used, so loans that are missing from previous_application are not lost (EDA finding).
"""

import pandas as pd

from creditrisk.features.common import safe_divide

INSTALLMENTS_COLS = ["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION",
                     "NUM_INSTALMENT_NUMBER", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
                     "AMT_INSTALMENT", "AMT_PAYMENT"]
COUNT_COLS = ["INST_COUNT", "INST_MISSED_COUNT", "INST_LOANS", "INST_12M_COUNT"]


def per_installment(inst: pd.DataFrame) -> pd.DataFrame:
    """One row per (applicant, loan, installment number), with lateness and payment amounts."""
    keys = ["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_NUMBER"]
    per = inst.groupby(keys, sort=False).agg(
        due=("AMT_INSTALMENT", "max"),
        paid=("AMT_PAYMENT", "sum"),  # all parts added up (a missed payment adds 0)
        due_day=("DAYS_INSTALMENT", "max"),
        last_pay=("DAYS_ENTRY_PAYMENT", "max"),  # fully paid when the last part arrives
        rows=("AMT_PAYMENT", "size"),
    ).reset_index()
    per["dpd"] = (per["last_pay"] - per["due_day"]).clip(lower=0)  # days late
    per["dbd"] = (per["due_day"] - per["last_pay"]).clip(lower=0)  # days early
    per["late"] = per["dpd"] > 0
    per["missed"] = per["last_pay"].isna()
    per["paid_ratio"] = safe_divide(per["paid"], per["due"]).where(per["due"] > 0)
    per["underpaid"] = (per["due"] - per["paid"]).clip(lower=0)
    per["recent"] = per["due_day"] >= -365
    return per


def build_installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR with INST_ features."""
    per = per_installment(inst[INSTALLMENTS_COLS])
    versions = inst.groupby("SK_ID_PREV")["NUM_INSTALMENT_VERSION"].nunique()
    loans = per.groupby(["SK_ID_CURR", "SK_ID_PREV"]).agg(
        late_share=("late", "mean"), dpd_max=("dpd", "max")
    ).reset_index()
    loans["version_changes"] = loans["SK_ID_PREV"].map(versions) - 1

    g = per.groupby("SK_ID_CURR")
    gl = loans.groupby("SK_ID_CURR")
    recent = per[per["recent"]]
    gr = recent.groupby("SK_ID_CURR")
    return pd.DataFrame({
        "INST_COUNT": g.size(),
        "INST_LATE_SHARE": g["late"].mean(),
        "INST_DPD_MEAN": g["dpd"].mean(),
        "INST_DPD_MAX": g["dpd"].max(),
        "INST_DBD_MEAN": g["dbd"].mean(),
        "INST_PAID_RATIO_MEAN": g["paid_ratio"].mean(),
        "INST_PAID_RATIO_MIN": g["paid_ratio"].min(),
        "INST_UNDERPAID_TOTAL": g["underpaid"].sum(),
        "INST_UNDERPAID_MAX": g["underpaid"].max(),
        "INST_MISSED_COUNT": g["missed"].sum(),
        "INST_SPLIT_PAYMENTS_SHARE": (per["rows"] > 1).groupby(per["SK_ID_CURR"]).mean(),
        "INST_LOANS": gl.size(),
        "INST_LOAN_LATE_SHARE_MEAN": gl["late_share"].mean(),
        "INST_LOAN_DPD_MAX_MEAN": gl["dpd_max"].mean(),
        "INST_VERSION_CHANGES_MAX": gl["version_changes"].max(),
        "INST_12M_COUNT": gr.size(),
        "INST_12M_LATE_SHARE": gr["late"].mean(),
        "INST_12M_DPD_MEAN": gr["dpd"].mean(),
        "INST_12M_DPD_MAX": gr["dpd"].max(),
        "INST_12M_PAID_RATIO_MEAN": gr["paid_ratio"].mean(),
        "INST_12M_UNDERPAID_TOTAL": gr["underpaid"].sum(),
    })
