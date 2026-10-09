"""Features from previous_application (earlier applications at Home Credit). Prefix PREV_."""

import pandas as pd

from creditrisk.features.common import safe_divide

PREVIOUS_COLS = [
    "SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS", "AMT_APPLICATION",
    "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "RATE_DOWN_PAYMENT", "CNT_PAYMENT",
    "NAME_YIELD_GROUP", "DAYS_DECISION", "NAME_PRODUCT_TYPE", "FLAG_LAST_APPL_PER_CONTRACT",
    "NFLAG_LAST_APPL_IN_DAY", "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION",
]
DAYS_WITH_365243 = ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                    "DAYS_LAST_DUE", "DAYS_TERMINATION"]
COUNT_COLS = ["PREV_COUNT", "PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT", "PREV_CANCELED_COUNT",
              "PREV_UNUSED_COUNT", "PREV_APPS_LAST_YEAR", "PREV_RUNNING_COUNT"]


def build_previous_features(previous: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """One row per SK_ID_CURR with PREV_ features."""
    p = previous[PREVIOUS_COLS].copy()
    if drop_duplicates:  # EDA decision: duplicate applications are not counted twice
        keep = (p["FLAG_LAST_APPL_PER_CONTRACT"] == "Y") & (p["NFLAG_LAST_APPL_IN_DAY"] == 1)
        p = p[keep]
    # In this data a loan that has not ended yet has 365243 in its end-date columns (there are
    # no future dates), so "still running" must be read BEFORE 365243 is turned into NaN.
    not_ended = (p["DAYS_TERMINATION"] == 365243) | (p["DAYS_LAST_DUE"] == 365243)
    for col in DAYS_WITH_365243:
        p[col] = p[col].where(p[col] != 365243)

    status = p["NAME_CONTRACT_STATUS"]
    approved, refused = status == "Approved", status == "Refused"
    future_end = (p["DAYS_TERMINATION"] > 0) | (p["DAYS_LAST_DUE"] > 0)
    running = approved & (not_ended | future_end)
    p = p.assign(
        is_approved=approved,
        is_refused=refused,
        is_canceled=status == "Canceled",
        is_unused=status == "Unused offer",
        is_recent=p["DAYS_DECISION"] >= -365,
        ask_to_given=safe_divide(p["AMT_APPLICATION"], p["AMT_CREDIT"]).where(approved),
        goods_to_credit=safe_divide(p["AMT_GOODS_PRICE"], p["AMT_CREDIT"]).where(approved),
        total_cost=(safe_divide(p["AMT_ANNUITY"] * p["CNT_PAYMENT"], p["AMT_CREDIT"]) - 1)
        .where(approved),
        approved_credit=p["AMT_CREDIT"].where(approved),
        approved_annuity=p["AMT_ANNUITY"].where(approved),
        approved_term=p["CNT_PAYMENT"].where(approved),
        refused_asked=p["AMT_APPLICATION"].where(refused),
        refused_day=p["DAYS_DECISION"].where(refused),
        high_yield=p["NAME_YIELD_GROUP"] == "high",
        x_sell=p["NAME_PRODUCT_TYPE"] == "x-sell",
        cash=p["NAME_CONTRACT_TYPE"] == "Cash loans",
        consumer=p["NAME_CONTRACT_TYPE"] == "Consumer loans",
        revolving=p["NAME_CONTRACT_TYPE"] == "Revolving loans",
        running=running,
        running_credit=p["AMT_CREDIT"].where(running),
    )

    g = p.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "PREV_COUNT": g.size(),
        "PREV_APPROVED_COUNT": g["is_approved"].sum(),
        "PREV_REFUSED_COUNT": g["is_refused"].sum(),
        "PREV_CANCELED_COUNT": g["is_canceled"].sum(),
        "PREV_UNUSED_COUNT": g["is_unused"].sum(),
        "PREV_APPS_LAST_YEAR": g["is_recent"].sum(),
        "PREV_APPROVED_SHARE": g["is_approved"].mean(),
        "PREV_REFUSED_SHARE": g["is_refused"].mean(),
        "PREV_YEARS_SINCE_LAST": -g["DAYS_DECISION"].max() / 365.25,
        "PREV_YEARS_SINCE_FIRST": -g["DAYS_DECISION"].min() / 365.25,
        "PREV_YEARS_SINCE_LAST_REFUSAL": -g["refused_day"].max() / 365.25,
        "PREV_ASK_TO_GIVEN_MEAN": g["ask_to_given"].mean(),
        "PREV_ASK_TO_GIVEN_MAX": g["ask_to_given"].max(),
        "PREV_GOODS_TO_CREDIT_MEAN": g["goods_to_credit"].mean(),
        "PREV_TOTAL_COST_MEAN": g["total_cost"].mean(),
        "PREV_CREDIT_MEAN": g["approved_credit"].mean(),
        "PREV_CREDIT_MAX": g["approved_credit"].max(),
        "PREV_ANNUITY_MEAN": g["approved_annuity"].mean(),
        "PREV_TERM_MEAN": g["approved_term"].mean(),
        "PREV_REFUSED_ASKED_MEAN": g["refused_asked"].mean(),
        "PREV_DOWN_PAYMENT_RATE_MEAN": g["RATE_DOWN_PAYMENT"].mean(),
        "PREV_HIGH_YIELD_SHARE": g["high_yield"].mean(),
        "PREV_XSELL_SHARE": g["x_sell"].mean(),
        "PREV_CASH_SHARE": g["cash"].mean(),
        "PREV_CONSUMER_SHARE": g["consumer"].mean(),
        "PREV_REVOLVING_SHARE": g["revolving"].mean(),
        "PREV_RUNNING_COUNT": g["running"].sum(),
        "PREV_RUNNING_CREDIT_TOTAL": g["running_credit"].sum(min_count=1),
    })
    last = p.loc[g["DAYS_DECISION"].idxmax()].set_index("SK_ID_CURR")  # most recent application
    out["PREV_LAST_REFUSED"] = last["is_refused"].astype("int8")
    out["PREV_LAST_CASH"] = last["cash"].astype("int8")
    return out
