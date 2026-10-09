"""Features from POS_CASH_balance (monthly status of old shop and cash loans). Prefix POS_.

Summarised per old loan first, then per applicant, using the table's own SK_ID_CURR.
"""

import pandas as pd

POS_COLS = ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT",
            "CNT_INSTALMENT_FUTURE", "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"]
COUNT_COLS = ["POS_LOANS", "POS_DEMAND_COUNT", "POS_TERM_CHANGES_TOTAL", "POS_ACTIVE_LOANS"]


def build_pos_features(pos: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR with POS_ features."""
    p = pos[POS_COLS].assign(
        has_dpd=pos["SK_DPD"] > 0,
        demand=pos["NAME_CONTRACT_STATUS"] == "Demand",
        completed=pos["NAME_CONTRACT_STATUS"] == "Completed",
        dpd_12m=pos["SK_DPD"].where(pos["MONTHS_BALANCE"] >= -12),
    )
    keys = ["SK_ID_CURR", "SK_ID_PREV"]
    g = p.groupby(keys)
    loans = pd.DataFrame({
        "months": g.size(),
        "dpd_max": g["SK_DPD"].max(),
        "dpd_def_max": g["SK_DPD_DEF"].max(),
        "dpd_share": g["has_dpd"].mean(),
        "dpd_12m_max": g["dpd_12m"].max(),
        "ever_completed": g["completed"].max(),
        "demand": g["demand"].max(),
        "term_changes": (g["CNT_INSTALMENT"].nunique() - 1).clip(lower=0),  # all-NaN loan: 0
    })
    latest = p.loc[g["MONTHS_BALANCE"].idxmax()].set_index(keys)  # each loan's latest month
    loans["future_installments"] = latest["CNT_INSTALMENT_FUTURE"]
    loans["active_now"] = latest["NAME_CONTRACT_STATUS"] == "Active"

    gl = loans.reset_index().groupby("SK_ID_CURR")
    return pd.DataFrame({
        "POS_LOANS": gl.size(),
        "POS_MONTHS_MEAN": gl["months"].mean(),
        "POS_DPD_MAX": gl["dpd_max"].max(),
        "POS_DPD_DEF_MAX": gl["dpd_def_max"].max(),
        "POS_DPD_SHARE_MEAN": gl["dpd_share"].mean(),
        "POS_12M_DPD_MAX": gl["dpd_12m_max"].max(),
        "POS_COMPLETED_SHARE": gl["ever_completed"].mean(),
        "POS_DEMAND_COUNT": gl["demand"].sum(),
        "POS_TERM_CHANGES_TOTAL": gl["term_changes"].sum(),
        "POS_ACTIVE_LOANS": gl["active_now"].sum(),
        "POS_FUTURE_INSTALLMENTS_TOTAL": gl["future_installments"].sum(min_count=1),
    })
