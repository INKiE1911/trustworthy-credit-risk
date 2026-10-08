"""Shared test data. No real data is needed, so these tests also run on GitHub CI."""

import numpy as np
import pandas as pd
import pytest

# The 16 text columns of application_train, with realistic values
APP_CATEGORIES = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "CODE_GENDER": ["F", "M"],
    "FLAG_OWN_CAR": ["N", "Y"],
    "FLAG_OWN_REALTY": ["Y", "N"],
    "NAME_TYPE_SUITE": ["Unaccompanied", "Family", "Spouse, partner", "Children", "Other_B"],
    "NAME_INCOME_TYPE": ["Working", "Commercial associate", "State servant", "Unemployed"],
    "NAME_EDUCATION_TYPE": [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary",
    ],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Civil marriage", "Widow"],
    "NAME_HOUSING_TYPE": ["House / apartment", "With parents", "Rented apartment"],
    "OCCUPATION_TYPE": [
        "Laborers",
        "Sales staff",
        "Core staff",
        "Managers",
        "Drivers",
        "High skill tech staff",
        "Accountants",
        "Medicine staff",
        "Security staff",
        "Cooking staff",
        "Cleaning staff",
        "Low-skill Laborers",
    ],
    "WEEKDAY_APPR_PROCESS_START": [
        "MONDAY",
        "TUESDAY",
        "WEDNESDAY",
        "THURSDAY",
        "FRIDAY",
        "SATURDAY",
        "SUNDAY",
    ],
    "ORGANIZATION_TYPE": [f"Business Entity Type {i}" for i in range(1, 4)]
    + [f"Org type {i}" for i in range(1, 30)],
    "FONDKAPREMONT_MODE": ["reg oper account", "org spec account", "not specified"],
    "HOUSETYPE_MODE": ["block of flats", "specific housing", "terraced house"],
    "WALLSMATERIAL_MODE": ["Panel", "Stone, brick", "Block", "Wooden"],
    "EMERGENCYSTATE_MODE": ["No", "Yes"],
}
BUILDING = [
    "APARTMENTS",
    "BASEMENTAREA",
    "YEARS_BEGINEXPLUATATION",
    "YEARS_BUILD",
    "COMMONAREA",
    "ELEVATORS",
    "ENTRANCES",
    "FLOORSMAX",
    "FLOORSMIN",
    "LANDAREA",
    "LIVINGAPARTMENTS",
    "LIVINGAREA",
    "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA",
]


def make_fake_application(
    n: int, with_target: bool = True, seed: int = 0, first_id: int = 100_002
) -> pd.DataFrame:
    """A made-up table with ALL 122 real application_train columns (121 without TARGET)."""
    rng = np.random.default_rng(seed)
    pension = rng.random(n) < 0.18
    nan_some = lambda a, share: np.where(rng.random(n) < share, np.nan, a)  # noqa: E731
    credit = rng.lognormal(13, 0.6, n).round(1)
    fam = rng.integers(1, 6, n).astype(float)
    d = {"SK_ID_CURR": np.arange(first_id, first_id + n)}
    for col, values in APP_CATEGORIES.items():
        d[col] = rng.choice(values, n).astype(object)
    d["NAME_INCOME_TYPE"] = np.where(pension, "Pensioner", d["NAME_INCOME_TYPE"])
    d["ORGANIZATION_TYPE"] = np.where(pension, "XNA", d["ORGANIZATION_TYPE"])
    d["OCCUPATION_TYPE"] = np.where(rng.random(n) < 0.31, None, d["OCCUPATION_TYPE"])
    for col in [
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
    ]:
        d[col] = np.where(rng.random(n) < 0.5, None, d[col])
    d.update(
        {
            "CNT_CHILDREN": np.minimum(fam.astype(int) - 1, rng.poisson(0.4, n)),
            "AMT_INCOME_TOTAL": rng.lognormal(12, 0.5, n).round(1),
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": nan_some((credit / rng.uniform(10, 40, n)).round(1), 0.0001),
            "AMT_GOODS_PRICE": nan_some((credit * rng.uniform(0.8, 1.0, n)).round(1), 0.001),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.0003, 0.07, n),
            "DAYS_BIRTH": -rng.integers(7_500, 25_000, n),
            "DAYS_EMPLOYED": np.where(pension, 365_243, -rng.integers(0, 15_000, n)),
            "DAYS_REGISTRATION": -rng.integers(0, 20_000, n).astype(float),
            "DAYS_ID_PUBLISH": -rng.integers(0, 7_000, n),
            "OWN_CAR_AGE": np.where(d["FLAG_OWN_CAR"] == "Y", rng.integers(0, 30, n), np.nan),
            "FLAG_MOBIL": np.ones(n, dtype=int),
            "FLAG_EMP_PHONE": (~pension).astype(int),
            "FLAG_WORK_PHONE": (rng.random(n) < 0.2).astype(int),
            "FLAG_CONT_MOBILE": (rng.random(n) < 0.998).astype(int),
            "FLAG_PHONE": (rng.random(n) < 0.28).astype(int),
            "FLAG_EMAIL": (rng.random(n) < 0.06).astype(int),
            "CNT_FAM_MEMBERS": nan_some(fam, 0.0001),
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n, p=[0.1, 0.74, 0.16]),
            "HOUR_APPR_PROCESS_START": rng.integers(0, 24, n),
            "EXT_SOURCE_1": nan_some(rng.random(n), 0.56),
            "EXT_SOURCE_2": nan_some(rng.random(n), 0.002),
            "EXT_SOURCE_3": nan_some(rng.random(n), 0.2),
            "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 4_000, n).astype(float),
        }
    )
    d["REGION_RATING_CLIENT_W_CITY"] = d["REGION_RATING_CLIENT"]
    for col in [
        "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    ]:
        d[col] = (rng.random(n) < 0.1).astype(int)
    building_missing = rng.random(n) < 0.6
    for base in BUILDING:
        level = rng.random(n)
        for suffix in ["_AVG", "_MODE", "_MEDI"]:
            d[base + suffix] = np.where(building_missing, np.nan, level + rng.normal(0, 0.01, n))
    d["TOTALAREA_MODE"] = np.where(building_missing, np.nan, rng.random(n))
    obs = nan_some(rng.poisson(1.4, n).astype(float), 0.003)
    d["OBS_30_CNT_SOCIAL_CIRCLE"], d["OBS_60_CNT_SOCIAL_CIRCLE"] = obs, obs
    d["DEF_30_CNT_SOCIAL_CIRCLE"] = np.where(np.isnan(obs), np.nan, rng.poisson(0.15, n))
    d["DEF_60_CNT_SOCIAL_CIRCLE"] = d["DEF_30_CNT_SOCIAL_CIRCLE"]
    for i in range(2, 22):
        share = {3: 0.71, 6: 0.09, 8: 0.08, 5: 0.015}.get(i, 0.001)
        d[f"FLAG_DOCUMENT_{i}"] = (rng.random(n) < share).astype(int)
    req_missing = rng.random(n) < 0.13
    for suffix in ["HOUR", "DAY", "WEEK", "MON", "QRT", "YEAR"]:
        d[f"AMT_REQ_CREDIT_BUREAU_{suffix}"] = np.where(req_missing, np.nan, rng.poisson(0.3, n))
    df = pd.DataFrame(d)
    if with_target:
        ext2 = np.nan_to_num(df["EXT_SOURCE_2"].to_numpy(), nan=0.5)
        ext3 = np.nan_to_num(df["EXT_SOURCE_3"].to_numpy(), nan=0.5)
        logit = (
            -2.4
            - 2.5 * (ext2 - 0.5)
            - 2.0 * (ext3 - 0.5)
            + 0.00004 * (df["DAYS_BIRTH"].to_numpy() + 16_000)
        )
        df.insert(1, "TARGET", (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
        df.loc[rng.choice(n, min(4, n), replace=False), "CODE_GENDER"] = "XNA"
    for col in APP_CATEGORIES:
        df[col] = df[col].astype("category")
    return df


@pytest.fixture(scope="session")
def calibrated_scores():
    """Skewed probabilities like a credit model, with labels drawn from them (so calibrated)."""
    rng = np.random.default_rng(0)
    p = rng.beta(1, 10, 30_000)
    y = (rng.random(len(p)) < p).astype(int)
    return y, p


@pytest.fixture(scope="session")
def fake_train():
    return make_fake_application(3_000, with_target=True, seed=1)


@pytest.fixture(scope="session")
def fake_test():
    return make_fake_application(500, with_target=False, seed=2, first_id=500_000)


def make_fake_history(app_ids, seed: int = 0) -> dict:
    """Made-up history tables with ALL their real columns, linked to the given SK_ID_CURR values."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(app_ids)
    cat = lambda values, n, p=None: pd.Categorical(rng.choice(values, n, p=p))  # noqa: E731

    nb = 3 * len(ids)
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": rng.choice(ids[: int(0.86 * len(ids))], nb),
            "SK_ID_BUREAU": np.arange(5_000_000, 5_000_000 + nb),
            "CREDIT_ACTIVE": cat(
                ["Closed", "Active", "Sold", "Bad debt"], nb, [0.62, 0.37, 0.009, 0.001]
            ),
            "CREDIT_CURRENCY": cat(["currency 1", "currency 2"], nb, [0.999, 0.001]),
            "DAYS_CREDIT": -rng.integers(0, 2_923, nb),
            "CREDIT_DAY_OVERDUE": np.where(rng.random(nb) < 0.02, rng.integers(1, 300, nb), 0),
            "DAYS_CREDIT_ENDDATE": np.where(
                rng.random(nb) < 0.06, np.nan, rng.integers(-3_000, 31_199, nb)
            ),
            "DAYS_ENDDATE_FACT": np.where(
                rng.random(nb) < 0.37, np.nan, -rng.integers(0, 2_900, nb)
            ),
            "AMT_CREDIT_MAX_OVERDUE": np.where(
                rng.random(nb) < 0.65, np.nan, rng.exponential(500, nb)
            ),
            "CNT_CREDIT_PROLONG": (rng.random(nb) < 0.005).astype(int),
            "AMT_CREDIT_SUM": rng.lognormal(12, 1, nb),
            "AMT_CREDIT_SUM_DEBT": np.where(
                rng.random(nb) < 0.15, np.nan, rng.lognormal(10, 2, nb)
            ),
            "AMT_CREDIT_SUM_LIMIT": np.where(
                rng.random(nb) < 0.35, np.nan, rng.exponential(1_000, nb)
            ),
            "AMT_CREDIT_SUM_OVERDUE": np.where(
                rng.random(nb) < 0.01, rng.exponential(1_000, nb), 0.0
            ),
            "CREDIT_TYPE": cat(
                [
                    "Consumer credit",
                    "Credit card",
                    "Car loan",
                    "Mortgage",
                    "Microloan",
                    "Another type of loan",
                ],
                nb,
                [0.73, 0.23, 0.017, 0.01, 0.007, 0.006],
            ),
            "DAYS_CREDIT_UPDATE": -rng.integers(0, 2_000, nb),
            "AMT_ANNUITY": np.where(rng.random(nb) < 0.71, np.nan, rng.lognormal(9, 1, nb)),
        }
    )
    nbb = 6 * nb
    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": rng.integers(5_000_000, 5_000_000 + int(0.5 * nb), nbb),
            "MONTHS_BALANCE": -rng.integers(0, 97, nbb),
            "STATUS": cat(
                list("C0X12345"), nbb, [0.5, 0.275, 0.213, 0.009, 0.0009, 0.0003, 0.0002, 0.0016]
            ),
        }
    )
    npv = 5 * len(ids)
    status = rng.choice(
        ["Approved", "Canceled", "Refused", "Unused offer"], npv, p=[0.62, 0.19, 0.17, 0.02]
    )
    approved = status == "Approved"
    days = lambda share_365243: np.where(  # noqa: E731
        ~approved,
        np.nan,
        np.where(rng.random(npv) < share_365243, 365_243.0, -rng.integers(-500, 2_900, npv)),
    )
    previous = pd.DataFrame(
        {
            "SK_ID_PREV": np.arange(1_000_001, 1_000_001 + npv),
            "SK_ID_CURR": rng.choice(ids[: int(0.95 * len(ids))], npv),
            "NAME_CONTRACT_TYPE": cat(
                ["Cash loans", "Consumer loans", "Revolving loans", "XNA"],
                npv,
                [0.45, 0.44, 0.1098, 0.0002],
            ),
            "AMT_ANNUITY": rng.lognormal(9, 0.8, npv),
            "AMT_APPLICATION": rng.lognormal(11, 1, npv),
            "AMT_CREDIT": np.where(approved, rng.lognormal(11, 1, npv), 0.0),
            "AMT_DOWN_PAYMENT": np.where(
                rng.random(npv) < 0.5, np.nan, rng.exponential(5_000, npv)
            ),
            "AMT_GOODS_PRICE": np.where(rng.random(npv) < 0.23, np.nan, rng.lognormal(11, 1, npv)),
            "WEEKDAY_APPR_PROCESS_START": cat(["MONDAY", "TUESDAY", "FRIDAY"], npv),
            "HOUR_APPR_PROCESS_START": rng.integers(0, 24, npv),
            "FLAG_LAST_APPL_PER_CONTRACT": cat(["Y", "N"], npv, [0.995, 0.005]),
            "NFLAG_LAST_APPL_IN_DAY": (rng.random(npv) < 0.996).astype(int),
            "RATE_DOWN_PAYMENT": np.where(rng.random(npv) < 0.5, np.nan, rng.random(npv) * 0.3),
            "RATE_INTEREST_PRIMARY": np.where(rng.random(npv) < 0.996, np.nan, rng.random(npv)),
            "RATE_INTEREST_PRIVILEGED": np.where(rng.random(npv) < 0.996, np.nan, rng.random(npv)),
            "NAME_CASH_LOAN_PURPOSE": cat(["XAP", "XNA", "Repairs"], npv),
            "NAME_CONTRACT_STATUS": pd.Categorical(status),
            "DAYS_DECISION": -rng.integers(1, 2_923, npv),
            "NAME_PAYMENT_TYPE": cat(["Cash through the bank", "XNA"], npv),
            "CODE_REJECT_REASON": cat(["XAP", "HC", "LIMIT", "SCO"], npv),
            "NAME_TYPE_SUITE": cat(["Unaccompanied", "Family"], npv),
            "NAME_CLIENT_TYPE": cat(["Repeater", "New", "Refreshed"], npv),
            "NAME_GOODS_CATEGORY": cat(["XNA", "Mobile", "Consumer Electronics"], npv),
            "NAME_PORTFOLIO": cat(["POS", "Cash", "XNA", "Cards"], npv),
            "NAME_PRODUCT_TYPE": cat(["XNA", "x-sell", "walk-in"], npv),
            "CHANNEL_TYPE": cat(["Credit and cash offices", "Country-wide", "Stone"], npv),
            "SELLERPLACE_AREA": rng.integers(-1, 2_000, npv),
            "NAME_SELLER_INDUSTRY": cat(["XNA", "Consumer electronics", "Connectivity"], npv),
            "CNT_PAYMENT": np.where(
                rng.random(npv) < 0.22, np.nan, rng.choice([6, 12, 24, 36], npv)
            ),
            "NAME_YIELD_GROUP": cat(["XNA", "middle", "high", "low_normal", "low_action"], npv),
            "PRODUCT_COMBINATION": cat(["Cash", "POS household with interest", "Card Street"], npv),
            "DAYS_FIRST_DRAWING": days(0.9),
            "DAYS_FIRST_DUE": days(0.03),
            "DAYS_LAST_DUE_1ST_VERSION": days(0.06),
            "DAYS_LAST_DUE": days(0.13),
            "DAYS_TERMINATION": days(0.14),
            "NFLAG_INSURED_ON_APPROVAL": np.where(
                approved, (rng.random(npv) < 0.33).astype(float), np.nan
            ),
        }
    )
    loans = previous.loc[approved, ["SK_ID_PREV", "SK_ID_CURR"]].reset_index(drop=True)

    pick = loans.sample(10 * len(ids), replace=True, random_state=seed).reset_index(drop=True)
    n = len(pick)
    due_day = -rng.integers(1, 2_923, n).astype(float)
    installments = pick.assign(
        NUM_INSTALMENT_VERSION=rng.choice([0.0, 1.0, 2.0], n, p=[0.1, 0.8, 0.1]),
        NUM_INSTALMENT_NUMBER=rng.integers(1, 60, n),
        DAYS_INSTALMENT=due_day,
        DAYS_ENTRY_PAYMENT=np.where(
            rng.random(n) < 0.0002, np.nan, due_day + rng.integers(-30, 15, n)
        ),
        AMT_INSTALMENT=rng.lognormal(9, 1, n),
    )
    installments["AMT_PAYMENT"] = np.where(
        installments["DAYS_ENTRY_PAYMENT"].isna(),
        np.nan,
        installments["AMT_INSTALMENT"] * rng.choice([1, 0.5], n, p=[0.9, 0.1]),
    )
    installments = installments[
        [
            "SK_ID_PREV",
            "SK_ID_CURR",
            "NUM_INSTALMENT_VERSION",
            "NUM_INSTALMENT_NUMBER",
            "DAYS_INSTALMENT",
            "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT",
            "AMT_PAYMENT",
        ]
    ]

    pick = loans.sample(8 * len(ids), replace=True, random_state=seed + 1).reset_index(drop=True)
    n = len(pick)
    pos = pick.assign(
        MONTHS_BALANCE=-rng.integers(1, 97, n),
        CNT_INSTALMENT=np.where(rng.random(n) < 0.002, np.nan, rng.choice([6.0, 12.0, 24.0], n)),
        CNT_INSTALMENT_FUTURE=np.where(rng.random(n) < 0.002, np.nan, rng.integers(0, 24, n)),
        NAME_CONTRACT_STATUS=cat(
            [
                "Active",
                "Completed",
                "Signed",
                "Demand",
                "Returned to the store",
                "Approved",
                "Amortized debt",
                "Canceled",
                "XNA",
            ],
            n,
            [0.915, 0.0744, 0.0087, 0.0007, 0.0005, 0.0005, 0.0001, 0.00005, 0.00005],
        ),
        SK_DPD=np.where(rng.random(n) < 0.03, rng.integers(1, 4_000, n), 0),
    )
    pos["SK_DPD_DEF"] = np.where(rng.random(n) < 0.5, pos["SK_DPD"], 0)

    pick = loans.sample(3 * len(ids), replace=True, random_state=seed + 2).reset_index(drop=True)
    n = len(pick)
    limit = rng.choice([0.0, 45_000.0, 135_000.0, 270_000.0], n, p=[0.05, 0.4, 0.4, 0.15])
    drawn = np.where(rng.random(n) < 0.2, np.nan, rng.exponential(5_000, n))
    credit_card = pick.assign(
        MONTHS_BALANCE=-rng.integers(1, 97, n),
        AMT_BALANCE=limit * rng.random(n),
        AMT_CREDIT_LIMIT_ACTUAL=limit,
        AMT_DRAWINGS_ATM_CURRENT=np.where(np.isnan(drawn), np.nan, drawn * rng.random(n)),
        AMT_DRAWINGS_CURRENT=drawn,
        AMT_DRAWINGS_OTHER_CURRENT=np.where(np.isnan(drawn), np.nan, 0.0),
        AMT_DRAWINGS_POS_CURRENT=np.where(np.isnan(drawn), np.nan, drawn * 0.3),
        AMT_INST_MIN_REGULARITY=np.where(rng.random(n) < 0.1, np.nan, limit * 0.05),
        AMT_PAYMENT_CURRENT=np.where(rng.random(n) < 0.2, np.nan, rng.exponential(5_000, n)),
        AMT_PAYMENT_TOTAL_CURRENT=rng.exponential(5_000, n),
        AMT_RECEIVABLE_PRINCIPAL=limit * rng.random(n),
        AMT_RECIVABLE=limit * rng.random(n),
        AMT_TOTAL_RECEIVABLE=limit * rng.random(n),
        CNT_DRAWINGS_ATM_CURRENT=np.where(np.isnan(drawn), np.nan, rng.integers(0, 5, n)),
        CNT_DRAWINGS_CURRENT=rng.integers(0, 10, n),
        CNT_DRAWINGS_OTHER_CURRENT=np.where(np.isnan(drawn), np.nan, 0.0),
        CNT_DRAWINGS_POS_CURRENT=np.where(np.isnan(drawn), np.nan, rng.integers(0, 5, n)),
        CNT_INSTALMENT_MATURE_CUM=np.where(rng.random(n) < 0.2, np.nan, rng.integers(0, 60, n)),
        NAME_CONTRACT_STATUS=cat(["Active", "Completed", "Signed"], n, [0.96, 0.033, 0.007]),
        SK_DPD=np.where(rng.random(n) < 0.04, rng.integers(1, 300, n), 0),
    )
    credit_card["SK_DPD_DEF"] = np.where(rng.random(n) < 0.3, credit_card["SK_DPD"], 0)
    return {
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous_application": previous,
        "installments_payments": installments,
        "pos_cash_balance": pos,
        "credit_card_balance": credit_card,
    }
