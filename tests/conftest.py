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
    "NAME_EDUCATION_TYPE": ["Secondary / secondary special", "Higher education",
                            "Incomplete higher", "Lower secondary"],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Civil marriage", "Widow"],
    "NAME_HOUSING_TYPE": ["House / apartment", "With parents", "Rented apartment"],
    "OCCUPATION_TYPE": ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
                        "High skill tech staff", "Accountants", "Medicine staff", "Security staff",
                        "Cooking staff", "Cleaning staff", "Low-skill Laborers"],
    "WEEKDAY_APPR_PROCESS_START": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY",
                                   "SATURDAY", "SUNDAY"],
    "ORGANIZATION_TYPE": [f"Business Entity Type {i}" for i in range(1, 4)]
    + [f"Org type {i}" for i in range(1, 30)],
    "FONDKAPREMONT_MODE": ["reg oper account", "org spec account", "not specified"],
    "HOUSETYPE_MODE": ["block of flats", "specific housing", "terraced house"],
    "WALLSMATERIAL_MODE": ["Panel", "Stone, brick", "Block", "Wooden"],
    "EMERGENCYSTATE_MODE": ["No", "Yes"],
}
BUILDING = ["APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION", "YEARS_BUILD", "COMMONAREA",
            "ELEVATORS", "ENTRANCES", "FLOORSMAX", "FLOORSMIN", "LANDAREA", "LIVINGAPARTMENTS",
            "LIVINGAREA", "NONLIVINGAPARTMENTS", "NONLIVINGAREA"]


def make_fake_application(n: int, with_target: bool = True, seed: int = 0,
                          first_id: int = 100_002) -> pd.DataFrame:
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
    for col in ["FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE",
                "EMERGENCYSTATE_MODE"]:
        d[col] = np.where(rng.random(n) < 0.5, None, d[col])
    d.update({
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
    })
    d["REGION_RATING_CLIENT_W_CITY"] = d["REGION_RATING_CLIENT"]
    for col in ["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
                "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY",
                "LIVE_CITY_NOT_WORK_CITY"]:
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
        logit = (-2.4 - 2.5 * (ext2 - 0.5) - 2.0 * (ext3 - 0.5)
                 + 0.00004 * (df["DAYS_BIRTH"].to_numpy() + 16_000))
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
