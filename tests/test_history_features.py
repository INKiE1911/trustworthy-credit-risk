"""Hand-made mini tables with answers worked out by hand, one test group per history table."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.features.bureau import build_bureau_features, summarise_bureau_balance
from creditrisk.features.common import attach
from creditrisk.features.credit_card import build_credit_card_features
from creditrisk.features.installments import build_installments_features
from creditrisk.features.pos_cash import build_pos_features
from creditrisk.features.previous import build_previous_features

NAN = np.nan


# ---------------------------------------------------------------- bureau + bureau_balance
@pytest.fixture
def bureau_balance():
    # loan 10: months 0, -1, -2 with C, 1, 0 | loan 11: months -1, -20 with X, 5
    return pd.DataFrame({
        "SK_ID_BUREAU": [10, 10, 10, 11, 11],
        "MONTHS_BALANCE": [0, -1, -2, -1, -20],
        "STATUS": pd.Categorical(["C", "1", "0", "X", "5"]),
    })


@pytest.fixture
def bureau():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_BUREAU": [10, 11, 12],
        "CREDIT_ACTIVE": pd.Categorical(["Active", "Closed", "Sold"]),
        "CREDIT_TYPE": pd.Categorical(["Consumer credit", "Credit card", "Mortgage"]),
        "DAYS_CREDIT": [-100, -800, -3000],
        "CREDIT_DAY_OVERDUE": [0, 0, 30],
        "DAYS_CREDIT_ENDDATE": [500.0, -400.0, -5000.0],  # loan 12 "ends" before it started
        "DAYS_ENDDATE_FACT": [NAN, -400.0, NAN],
        "AMT_CREDIT_MAX_OVERDUE": [NAN, 0.0, 100.0],
        "CNT_CREDIT_PROLONG": [0, 1, 0],
        "AMT_CREDIT_SUM": [1000.0, 2000.0, 5000.0],
        "AMT_CREDIT_SUM_DEBT": [500.0, 0.0, -10.0],  # negative debt -> clipped to 0
        "AMT_CREDIT_SUM_LIMIT": [0.0, 300.0, NAN],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 50.0],
        "AMT_ANNUITY": [50.0, NAN, NAN],
    })


def test_bureau_balance_per_loan(bureau_balance):
    bb = summarise_bureau_balance(bureau_balance)
    assert bb.loc[10, "BB_MONTHS"] == 3 and bb.loc[10, "BB_HISTORY_MONTHS"] == 2
    assert bb.loc[10, "BB_WORST_STATUS"] == 1 and bb.loc[11, "BB_WORST_STATUS"] == 5
    assert bb.loc[10, "BB_LATE_SHARE"] == pytest.approx(0.5)  # 1 late of 2 known (C not counted)
    assert bb.loc[11, "BB_LATE_SHARE"] == pytest.approx(1.0)  # X not counted
    assert bb.loc[11, "BB_LATE_12M"] == 0  # the late month is 20 months ago
    assert bb.loc[10, "BB_LATEST_CLOSED"] == 1 and bb.loc[11, "BB_LATEST_CLOSED"] == 0
    assert bb.loc[11, "BB_UNKNOWN_SHARE"] == pytest.approx(0.5)


def test_bureau_per_applicant(bureau, bureau_balance):
    f = build_bureau_features(bureau, bureau_balance)
    assert f.loc[1, "BUR_COUNT"] == 2 and f.loc[1, "BUR_ACTIVE_COUNT"] == 1
    assert f.loc[1, "BUR_CLOSED_COUNT"] == 1 and f.loc[2, "BUR_SOLD_OR_BAD_COUNT"] == 1
    assert f.loc[1, "BUR_LOANS_LAST_YEAR"] == 1
    assert f.loc[1, "BUR_DEBT_RATIO_ACTIVE"] == pytest.approx(0.5)
    assert f.loc[1, "BUR_CLOSED_DURATION_MEAN_YEARS"] == pytest.approx(400 / 365.25)
    assert f.loc[1, "BUR_AVG_GAP_DAYS"] == pytest.approx(700)
    assert np.isnan(f.loc[2, "BUR_AVG_GAP_DAYS"])
    assert f.loc[2, "BUR_DAYS_OVERDUE_MAX"] == 30 and f.loc[2, "BUR_OVERDUE_NOW_SHARE"] == 1
    assert f.loc[1, "BUR_CONSUMER_COUNT"] == 1 and f.loc[2, "BUR_MORTGAGE_COUNT"] == 1
    assert f.loc[1, "BB_LOANS_WITH_HISTORY"] == 2 and f.loc[2, "BB_LOANS_WITH_HISTORY"] == 0
    assert f.loc[1, "BB_WORST_STATUS_MAX"] == 5
    assert f.loc[1, "BB_LATE_SHARE_MEAN"] == pytest.approx(0.75)
    assert f.loc[1, "BB_LATEST_CLOSED_SHARE"] == pytest.approx(0.5)
    assert f.loc[1, "BUR_ACTIVE_ENDS_IN_YEARS"] == pytest.approx(500 / 365.25)


# ---------------------------------------------------------------- previous_application
@pytest.fixture
def previous():
    return pd.DataFrame({
        "SK_ID_PREV": [1, 2, 3, 4],
        "SK_ID_CURR": [1, 1, 1, 2],
        "NAME_CONTRACT_TYPE": pd.Categorical(["Cash loans", "Consumer loans", "Cash loans",
                                              "Revolving loans"]),
        "NAME_CONTRACT_STATUS": pd.Categorical(["Approved", "Refused", "Approved", "Approved"]),
        "AMT_APPLICATION": [1000.0, 500.0, 999.0, 300.0],
        "AMT_CREDIT": [800.0, 0.0, 999.0, 300.0],
        "AMT_ANNUITY": [100.0, NAN, 10.0, 30.0],
        "AMT_GOODS_PRICE": [800.0, 500.0, NAN, NAN],
        "RATE_DOWN_PAYMENT": [0.1, NAN, 0.0, NAN],
        "CNT_PAYMENT": [10.0, 12.0, 12.0, 0.0],
        "NAME_YIELD_GROUP": pd.Categorical(["middle", "high", "middle", "high"]),
        "DAYS_DECISION": [-200, -50, -60, -1000],
        "NAME_PRODUCT_TYPE": pd.Categorical(["walk-in", "x-sell", "walk-in", "x-sell"]),
        "FLAG_LAST_APPL_PER_CONTRACT": pd.Categorical(["Y", "Y", "N", "Y"]),  # row 3: duplicate
        "NFLAG_LAST_APPL_IN_DAY": [1, 1, 1, 1],
        "DAYS_FIRST_DRAWING": [365243.0, NAN, NAN, 365243.0],
        "DAYS_FIRST_DUE": [-170.0, NAN, NAN, -990.0],
        "DAYS_LAST_DUE_1ST_VERSION": [100.0, NAN, NAN, 365243.0],
        "DAYS_LAST_DUE": [100.0, NAN, NAN, -10.0],
        "DAYS_TERMINATION": [365243.0, NAN, NAN, 365243.0],  # the code must not count as "future"
    })


def test_previous_features(previous):
    f = build_previous_features(previous)
    assert f.loc[1, "PREV_COUNT"] == 2  # the duplicate row is dropped
    assert f.loc[1, "PREV_REFUSED_SHARE"] == pytest.approx(0.5)
    assert f.loc[1, "PREV_ASK_TO_GIVEN_MEAN"] == pytest.approx(1000 / 800)  # approved only
    assert f.loc[1, "PREV_TOTAL_COST_MEAN"] == pytest.approx(100 * 10 / 800 - 1)
    assert f.loc[1, "PREV_RUNNING_COUNT"] == 1  # DAYS_LAST_DUE is in the future
    assert f.loc[2, "PREV_RUNNING_COUNT"] == 0  # only 365243 says "future" -> not running
    assert f.loc[1, "PREV_LAST_REFUSED"] == 1  # most recent application was refused
    assert f.loc[1, "PREV_YEARS_SINCE_LAST"] == pytest.approx(50 / 365.25)
    assert f.loc[2, "PREV_HIGH_YIELD_SHARE"] == 1 and f.loc[2, "PREV_REVOLVING_SHARE"] == 1


# ---------------------------------------------------------------- installments_payments
@pytest.fixture
def installments():
    return pd.DataFrame({
        # applicant 1, loan 1: installment 1 paid in two halves (last part 5 days late),
        #                      installment 2 paid 1 day early, calendar version changed
        # applicant 1, loan 2: installment 1 missed (400 days ago)
        # applicant 2, loan 3: installment 1 underpaid and 5 days late
        "SK_ID_PREV": [1, 1, 1, 2, 3],
        "SK_ID_CURR": [1, 1, 1, 1, 2],
        "NUM_INSTALMENT_VERSION": [1.0, 1.0, 2.0, 1.0, 1.0],
        "NUM_INSTALMENT_NUMBER": [1, 1, 2, 1, 1],
        "DAYS_INSTALMENT": [-60.0, -60.0, -30.0, -400.0, -10.0],
        "DAYS_ENTRY_PAYMENT": [-62.0, -55.0, -31.0, NAN, -5.0],
        "AMT_INSTALMENT": [1000.0, 1000.0, 1000.0, 200.0, 300.0],
        "AMT_PAYMENT": [500.0, 500.0, 1000.0, NAN, 100.0],
    })


def test_installments_combines_split_payments(installments):
    f = build_installments_features(installments)
    assert f.loc[1, "INST_COUNT"] == 3  # 3 installments, not 4 rows
    assert f.loc[1, "INST_UNDERPAID_TOTAL"] == pytest.approx(200)  # only the missed one
    assert f.loc[1, "INST_LATE_SHARE"] == pytest.approx(1 / 3)
    assert f.loc[1, "INST_DPD_MAX"] == 5 and f.loc[1, "INST_MISSED_COUNT"] == 1
    assert f.loc[1, "INST_PAID_RATIO_MIN"] == pytest.approx(0.0)
    assert f.loc[1, "INST_SPLIT_PAYMENTS_SHARE"] == pytest.approx(1 / 3)
    assert f.loc[1, "INST_LOANS"] == 2 and f.loc[1, "INST_VERSION_CHANGES_MAX"] == 1
    assert f.loc[1, "INST_12M_COUNT"] == 2 and f.loc[1, "INST_12M_LATE_SHARE"] == pytest.approx(0.5)
    assert f.loc[2, "INST_UNDERPAID_TOTAL"] == pytest.approx(200)
    assert f.loc[2, "INST_PAID_RATIO_MEAN"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------- POS_CASH_balance
def test_pos_features():
    pos = pd.DataFrame({
        "SK_ID_PREV": [1, 1, 1, 2, 2],
        "SK_ID_CURR": [1, 1, 1, 1, 1],
        "MONTHS_BALANCE": [-3, -2, -1, -20, -1],
        "CNT_INSTALMENT": [12.0, 12.0, 12.0, 6.0, 8.0],  # loan 2's length changed
        "CNT_INSTALMENT_FUTURE": [2.0, 1.0, 0.0, 5.0, 3.0],
        "NAME_CONTRACT_STATUS": pd.Categorical(["Active", "Active", "Completed", "Demand",
                                                "Active"]),
        "SK_DPD": [0, 10, 0, 0, 0],
        "SK_DPD_DEF": [0, 10, 0, 0, 0],
    })
    f = build_pos_features(pos)
    assert f.loc[1, "POS_LOANS"] == 2 and f.loc[1, "POS_DPD_MAX"] == 10
    assert f.loc[1, "POS_DPD_SHARE_MEAN"] == pytest.approx((1 / 3 + 0) / 2)
    assert f.loc[1, "POS_12M_DPD_MAX"] == 10
    assert f.loc[1, "POS_COMPLETED_SHARE"] == pytest.approx(0.5)
    assert f.loc[1, "POS_DEMAND_COUNT"] == 1 and f.loc[1, "POS_TERM_CHANGES_TOTAL"] == 1
    assert f.loc[1, "POS_ACTIVE_LOANS"] == 1  # loan 1 is completed in its latest month
    assert f.loc[1, "POS_FUTURE_INSTALLMENTS_TOTAL"] == pytest.approx(3)


# ---------------------------------------------------------------- credit_card_balance
def test_credit_card_features():
    cc = pd.DataFrame({
        "SK_ID_PREV": [1, 1, 2, 3],
        "SK_ID_CURR": [1, 1, 1, 2],
        "MONTHS_BALANCE": [-2, -1, -10, -1],
        "AMT_BALANCE": [500.0, 1000.0, 50.0, 300.0],
        "AMT_CREDIT_LIMIT_ACTUAL": [1000.0, 1000.0, 0.0, 1000.0],  # card 2: limit 0
        "AMT_DRAWINGS_ATM_CURRENT": [100.0, 0.0, NAN, 0.0],
        "AMT_DRAWINGS_CURRENT": [200.0, 0.0, 0.0, 0.0],
        "AMT_INST_MIN_REGULARITY": [50.0, 100.0, 0.0, 0.0],
        "AMT_PAYMENT_CURRENT": [50.0, 20.0, NAN, NAN],
        "SK_DPD": [0, 5, 0, 0],
    })
    f = build_credit_card_features(cc)
    assert f.loc[1, "CC_CARDS"] == 2 and f.loc[1, "CC_UTIL_MAX"] == pytest.approx(1.0)
    assert f.loc[1, "CC_UTIL_LATEST_MEAN"] == pytest.approx(1.0)  # card 2 has no usable limit
    assert f.loc[1, "CC_UTIL_6M_MEAN"] == pytest.approx(0.75)
    assert f.loc[1, "CC_BELOW_MIN_SHARE_MEAN"] == pytest.approx(0.5)
    assert f.loc[1, "CC_PAY_TO_MIN_MEAN"] == pytest.approx(0.6)
    assert f.loc[1, "CC_CASH_SHARE_MEAN"] == pytest.approx(0.5)
    assert f.loc[1, "CC_DPD_MAX"] == 5 and f.loc[1, "CC_DPD_MONTHS_TOTAL"] == 1
    assert f.loc[2, "CC_UTIL_MEAN"] == pytest.approx(0.3)
    assert np.isnan(f.loc[2, "CC_BELOW_MIN_SHARE_MEAN"])  # no minimum was ever due


# ---------------------------------------------------------------- joining
def test_attach_keeps_rows_fills_counts_and_flags():
    base = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "APP_X": [0.1, 0.2, 0.3]})
    part = pd.DataFrame({"T_COUNT": [2, 5], "T_MEAN": [1.5, 2.5]},
                        index=pd.Index([1, 3], name="SK_ID_CURR"))
    out = attach(base, part, count_cols=["T_COUNT"], flag="T_HAS_HISTORY")
    assert len(out) == 3
    assert out["T_COUNT"].tolist() == [2, 0, 5]
    assert np.isnan(out.loc[1, "T_MEAN"])
    assert out["T_HAS_HISTORY"].tolist() == [1, 0, 1]
