"""Features from bureau (loans at other lenders) and bureau_balance (their monthly status).

bureau_balance is summarised per bureau loan first, joined to bureau, then everything is
summarised per applicant. Prefixes: BUR_ (bureau) and BB_ (bureau_balance).
"""

import numpy as np
import pandas as pd

from creditrisk.features.common import safe_divide

BUREAU_COLS = [
    "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_TYPE", "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "AMT_CREDIT_MAX_OVERDUE",
    "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT",
    "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY",
]
BALANCE_COLS = ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]
LOAN_TYPES = {"Consumer credit": "CONSUMER", "Credit card": "CARD", "Car loan": "CAR",
              "Mortgage": "MORTGAGE", "Microloan": "MICROLOAN"}
COUNT_COLS = (
    ["BUR_COUNT", "BUR_ACTIVE_COUNT", "BUR_CLOSED_COUNT", "BUR_SOLD_OR_BAD_COUNT",
     "BUR_LOANS_LAST_YEAR", "BUR_PROLONG_TOTAL", "BB_LOANS_WITH_HISTORY"]
    + [f"BUR_{label}_COUNT" for label in LOAN_TYPES.values()]
)


def summarise_bureau_balance(bb: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_BUREAU (bureau loan), from its monthly status rows."""
    # work on the category codes (fast and light on memory for 27M rows)
    status = bb["STATUS"].astype("category")
    cats = [str(c) for c in status.cat.categories]
    codes = status.cat.codes.to_numpy()
    level_of_cat = pd.to_numeric(pd.Series(cats), errors="coerce").to_numpy()  # C/X -> NaN
    level = np.where(codes >= 0, level_of_cat[codes], np.nan).astype("float32")

    def is_status(value: str) -> np.ndarray:
        return codes == cats.index(value) if value in cats else np.zeros(len(codes), dtype=bool)

    month = bb["MONTHS_BALANCE"].to_numpy()
    work = pd.DataFrame({
        "id": bb["SK_ID_BUREAU"].to_numpy(),
        "month": month,
        "level": level,  # 0 = on time, 1..5 = more and more late
        "late": level >= 1,
        "known": ~np.isnan(level),
        "closed": is_status("C"),
        "unknown": is_status("X"),
    })
    work["late_12m"] = work["late"] & (month >= -12)

    g = work.groupby("id")
    out = pd.DataFrame({
        "BB_MONTHS": g.size(),
        "BB_HISTORY_MONTHS": -g["month"].min(),
        "BB_WORST_STATUS": g["level"].max(),
        "BB_LATE_MONTHS": g["late"].sum(),
        "BB_LATE_12M": g["late_12m"].sum(),
        "BB_UNKNOWN_SHARE": g["unknown"].mean(),
    })
    known = g["known"].sum()
    out["BB_LATE_SHARE"] = (out["BB_LATE_MONTHS"] / known).where(known > 0)
    latest = work.loc[g["month"].idxmax()].set_index("id")  # each loan's most recent month
    out["BB_LATEST_LATE"] = latest["late"].astype("float32")
    out["BB_LATEST_CLOSED"] = latest["closed"].astype("float32")
    out.index.name = "SK_ID_BUREAU"
    return out


def build_bureau_features(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """One row per SK_ID_CURR with BUR_ and BB_ features."""
    b = bureau[BUREAU_COLS].copy()

    # --- cleaning (EDA decisions) ---
    for col in ("DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT"):
        b[col] = b[col].where(b[col] >= b["DAYS_CREDIT"])  # ending before it started: impossible
    planned_end = b["DAYS_CREDIT_ENDDATE"]
    b["DAYS_CREDIT_ENDDATE"] = planned_end.where(planned_end <= 50 * 365.25)  # > 50 years: error
    for col in ("AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE"):
        b[col] = b[col].clip(lower=0)

    b = b.join(summarise_bureau_balance(bureau_balance), on="SK_ID_BUREAU")

    active = b["CREDIT_ACTIVE"] == "Active"
    closed = b["CREDIT_ACTIVE"] == "Closed"
    b = b.assign(
        is_active=active,
        is_closed=closed,
        is_sold_or_bad=b["CREDIT_ACTIVE"].isin(["Sold", "Bad debt"]),
        is_recent=b["DAYS_CREDIT"] >= -365,
        overdue_now=b["CREDIT_DAY_OVERDUE"] > 0,
        has_bb=b["BB_MONTHS"].notna(),
        active_credit=b["AMT_CREDIT_SUM"].where(active),
        active_debt=b["AMT_CREDIT_SUM_DEBT"].where(active),
        active_annuity=b["AMT_ANNUITY"].where(active),
        active_end=b["DAYS_CREDIT_ENDDATE"].where(active),
        closed_duration=(b["DAYS_ENDDATE_FACT"] - b["DAYS_CREDIT"]).where(closed),
    )
    for name, label in LOAN_TYPES.items():
        b[f"type_{label}"] = b["CREDIT_TYPE"] == name

    g = b.groupby("SK_ID_CURR")
    out = pd.DataFrame({
        "BUR_COUNT": g.size(),
        "BUR_ACTIVE_COUNT": g["is_active"].sum(),
        "BUR_CLOSED_COUNT": g["is_closed"].sum(),
        "BUR_SOLD_OR_BAD_COUNT": g["is_sold_or_bad"].sum(),
        "BUR_LOANS_LAST_YEAR": g["is_recent"].sum(),
        "BUR_OLDEST_LOAN_YEARS": -g["DAYS_CREDIT"].min() / 365.25,
        "BUR_NEWEST_LOAN_YEARS": -g["DAYS_CREDIT"].max() / 365.25,
        "BUR_MEAN_LOAN_AGE_YEARS": -g["DAYS_CREDIT"].mean() / 365.25,
        "BUR_ACTIVE_ENDS_IN_YEARS": g["active_end"].max() / 365.25,
        "BUR_CLOSED_DURATION_MEAN_YEARS": g["closed_duration"].mean() / 365.25,
        "BUR_CREDIT_TOTAL": g["AMT_CREDIT_SUM"].sum(min_count=1),
        "BUR_CREDIT_MAX": g["AMT_CREDIT_SUM"].max(),
        "BUR_ACTIVE_CREDIT_TOTAL": g["active_credit"].sum(min_count=1),
        "BUR_ACTIVE_DEBT_TOTAL": g["active_debt"].sum(min_count=1),
        "BUR_ACTIVE_ANNUITY_TOTAL": g["active_annuity"].sum(min_count=1),
        "BUR_OVERDUE_TOTAL": g["AMT_CREDIT_SUM_OVERDUE"].sum(min_count=1),
        "BUR_MAX_OVERDUE_EVER": g["AMT_CREDIT_MAX_OVERDUE"].max(),
        "BUR_LIMIT_TOTAL": g["AMT_CREDIT_SUM_LIMIT"].sum(min_count=1),
        "BUR_DAYS_OVERDUE_MAX": g["CREDIT_DAY_OVERDUE"].max(),
        "BUR_OVERDUE_NOW_SHARE": g["overdue_now"].mean(),
        "BUR_PROLONG_TOTAL": g["CNT_CREDIT_PROLONG"].sum(),
        **{f"BUR_{label}_COUNT": g[f"type_{label}"].sum() for label in LOAN_TYPES.values()},
        "BB_LOANS_WITH_HISTORY": g["has_bb"].sum(),
        "BB_HISTORY_MONTHS_MAX": g["BB_HISTORY_MONTHS"].max(),
        "BB_WORST_STATUS_MAX": g["BB_WORST_STATUS"].max(),
        "BB_LATE_MONTHS_TOTAL": g["BB_LATE_MONTHS"].sum(min_count=1),
        "BB_LATE_12M_TOTAL": g["BB_LATE_12M"].sum(min_count=1),
        "BB_LATE_SHARE_MEAN": g["BB_LATE_SHARE"].mean(),
        "BB_LATEST_LATE_SHARE": g["BB_LATEST_LATE"].mean(),
        "BB_LATEST_CLOSED_SHARE": g["BB_LATEST_CLOSED"].mean(),
        "BB_UNKNOWN_SHARE_MEAN": g["BB_UNKNOWN_SHARE"].mean(),
    })
    out["BUR_DEBT_RATIO_ACTIVE"] = safe_divide(out["BUR_ACTIVE_DEBT_TOTAL"],
                                               out["BUR_ACTIVE_CREDIT_TOTAL"])
    gap = (g["DAYS_CREDIT"].max() - g["DAYS_CREDIT"].min()) / (out["BUR_COUNT"] - 1)
    out["BUR_AVG_GAP_DAYS"] = gap.where(out["BUR_COUNT"] > 1)
    return out
