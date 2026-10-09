"""Features from credit_card_balance (monthly snapshots of Home Credit cards). Prefix CC_.

Summarised per card first, then per applicant, using the table's own SK_ID_CURR.
"""

import numpy as np
import pandas as pd

CREDIT_CARD_COLS = ["SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE",
                    "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT",
                    "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT", "SK_DPD"]
COUNT_COLS = ["CC_CARDS", "CC_DPD_MONTHS_TOTAL", "CC_LIMIT_CHANGES_TOTAL"]


def build_credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR with CC_ features."""
    limit = cc["AMT_CREDIT_LIMIT_ACTUAL"].where(cc["AMT_CREDIT_LIMIT_ACTUAL"] > 0)  # 0 -> NaN
    minimum = cc["AMT_INST_MIN_REGULARITY"].where(cc["AMT_INST_MIN_REGULARITY"] > 0)
    payment = cc["AMT_PAYMENT_CURRENT"]
    drawn = cc["AMT_DRAWINGS_CURRENT"].clip(lower=0)
    atm = cc["AMT_DRAWINGS_ATM_CURRENT"].clip(lower=0)
    below = (payment.fillna(0) < minimum).astype("float32")  # paid less than the minimum
    c = pd.DataFrame({
        "SK_ID_CURR": cc["SK_ID_CURR"],
        "SK_ID_PREV": cc["SK_ID_PREV"],
        "month": cc["MONTHS_BALANCE"],
        "util": cc["AMT_BALANCE"] / limit,
        "pay_to_min": payment / minimum,
        "below_min": below.where(minimum.notna()),  # no minimum due -> not counted
        "cash_share": (atm / drawn).where(drawn > 0),
        "dpd": cc["SK_DPD"],
        "has_dpd": cc["SK_DPD"] > 0,
        "limit": cc["AMT_CREDIT_LIMIT_ACTUAL"],
        "drawn": drawn,
    })
    c["util_6m"] = c["util"].where(c["month"] >= -6)
    c = c.replace([np.inf, -np.inf], np.nan)

    keys = ["SK_ID_CURR", "SK_ID_PREV"]
    g = c.groupby(keys)
    cards = pd.DataFrame({
        "months": g.size(),
        "util_mean": g["util"].mean(),
        "util_max": g["util"].max(),
        "util_6m": g["util_6m"].mean(),
        "below_min_share": g["below_min"].mean(),
        "pay_to_min_mean": g["pay_to_min"].mean(),
        "cash_share_mean": g["cash_share"].mean(),
        "dpd_max": g["dpd"].max(),
        "dpd_months": g["has_dpd"].sum(),
        "limit_changes": (g["limit"].nunique() - 1).clip(lower=0),  # all-NaN card: 0
        "drawn_total": g["drawn"].sum(),
    })
    latest = c.loc[g["month"].idxmax()].set_index(keys)
    cards["util_latest"] = latest["util"]

    gc = cards.reset_index().groupby("SK_ID_CURR")
    return pd.DataFrame({
        "CC_CARDS": gc.size(),
        "CC_MONTHS_MEAN": gc["months"].mean(),
        "CC_UTIL_MEAN": gc["util_mean"].mean(),
        "CC_UTIL_MAX": gc["util_max"].max(),
        "CC_UTIL_LATEST_MEAN": gc["util_latest"].mean(),
        "CC_UTIL_6M_MEAN": gc["util_6m"].mean(),
        "CC_BELOW_MIN_SHARE_MEAN": gc["below_min_share"].mean(),
        "CC_PAY_TO_MIN_MEAN": gc["pay_to_min_mean"].mean(),
        "CC_CASH_SHARE_MEAN": gc["cash_share_mean"].mean(),
        "CC_DPD_MAX": gc["dpd_max"].max(),
        "CC_DPD_MONTHS_TOTAL": gc["dpd_months"].sum(),
        "CC_LIMIT_CHANGES_TOTAL": gc["limit_changes"].sum(),
        "CC_DRAWN_TOTAL": gc["drawn_total"].sum(),
    })
