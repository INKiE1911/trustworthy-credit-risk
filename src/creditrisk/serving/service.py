"""Scoring for the API and the app: one applicant in, probability and reasons out.

The model needs 240 features, mostly from loan history, so nobody types them in. The demo
uses real applicants from the Kaggle test file (no labels, and our locked test split is never
touched). Up to 8 key fields can be changed ("what if"). The application features and the 3
cross-table features are then rebuilt with exactly the training code; history features stay.

Reasons come from LightGBM's built-in TreeSHAP (pred_contrib): how much each feature pushes
the log-odds of default up or down for this applicant. They add up exactly to the model's score.
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from creditrisk.config import get_path
from creditrisk.features.application import application_features_from_raw
from creditrisk.features.build import add_cross_table_features
from creditrisk.models.final import load_final_model

RAW_PREFIX = "RAW__"  # raw application columns inside the demo bundle
DEMO_FILE = "demo_applicants.parquet"
TRAIN_DEFAULT_RATE = 0.0807  # 24,825 of 307,511 training loans (EDA)
NOT_EMPLOYED = 365243  # the data's code for "no employment record" (mostly pensioners)
CROSS_TABLE = ["BUR_DEBT_TO_INCOME", "BUR_ANNUITY_BURDEN", "PREV_CURRENT_TO_MEAN_CREDIT"]

# what-if field -> raw application column
FIELDS = {
    "income": "AMT_INCOME_TOTAL",
    "loan_amount": "AMT_CREDIT",
    "annuity": "AMT_ANNUITY",
    "age_years": "DAYS_BIRTH",
    "years_employed": "DAYS_EMPLOYED",
    "ext_source_1": "EXT_SOURCE_1",
    "ext_source_2": "EXT_SOURCE_2",
    "ext_source_3": "EXT_SOURCE_3",
}
DETAILS = {
    "Contract type": "NAME_CONTRACT_TYPE",
    "Education": "NAME_EDUCATION_TYPE",
    "Income type": "NAME_INCOME_TYPE",
    "Family status": "NAME_FAMILY_STATUS",
    "Employer type": "ORGANIZATION_TYPE",
    "Family members": "CNT_FAM_MEMBERS",
    "Goods price": "AMT_GOODS_PRICE",
}
HISTORY = {
    "Credit bureau": "BUR_HAS_HISTORY",
    "Previous applications": "PREV_HAS_HISTORY",
    "Installment payments": "INST_HAS_HISTORY",
    "POS / cash loans": "POS_HAS_HISTORY",
    "Credit cards": "CC_HAS_HISTORY",
}

# plain-language names for features that often show up as reasons; others get a generated name
LABELS = {
    "APP_EXT_MEAN": "External credit scores (average)",
    "APP_EXT_MIN": "External credit scores (lowest)",
    "APP_EXT_MAX": "External credit scores (highest)",
    "APP_EXT_STD": "External credit scores (spread)",
    "APP_EXT_PROD": "External credit scores (product)",
    "APP_EXT_NA_COUNT": "External credit scores missing",
    "APP_EXT_SOURCE_1": "External credit score 1",
    "APP_EXT_SOURCE_2": "External credit score 2",
    "APP_EXT_SOURCE_3": "External credit score 3",
    "APP_ANNUITY_TO_CREDIT": "Annuity ÷ loan amount (repayment speed)",
    "APP_CREDIT_TO_INCOME": "Loan amount ÷ income",
    "APP_ANNUITY_TO_INCOME": "Annuity ÷ income",
    "APP_GOODS_TO_CREDIT": "Goods price ÷ loan amount",
    "APP_CREDIT_MINUS_GOODS": "Loan amount above the goods price",
    "APP_INCOME_PER_PERSON": "Income per family member",
    "APP_EMPLOYED_TO_AGE": "Share of life employed",
    "APP_AGE_YEARS": "Age",
    "APP_EMPLOYED_YEARS": "Years employed",
    "APP_DAYS_EMPLOYED_ANOM": "No employment record",
    "APP_REGISTRATION_YEARS": "Years since registration changed",
    "APP_ID_PUBLISH_YEARS": "Years since ID document changed",
    "APP_PHONE_CHANGE_YEARS": "Years since phone changed",
    "APP_AMT_INCOME_TOTAL": "Income",
    "APP_LOG_INCOME": "Income (log)",
    "APP_AMT_CREDIT": "Loan amount",
    "APP_AMT_ANNUITY": "Loan annuity",
    "APP_AMT_GOODS_PRICE": "Goods price",
    "APP_ORGANIZATION_TYPE": "Employer type",
    "APP_OCCUPATION_TYPE": "Occupation",
    "APP_NAME_EDUCATION_TYPE": "Education",
    "APP_NAME_INCOME_TYPE": "Income type",
    "APP_NAME_FAMILY_STATUS": "Family status",
    "APP_NAME_CONTRACT_TYPE": "Contract type",
    "APP_REGION_RATING_CLIENT": "Region rating",
    "APP_REGION_RATING_CLIENT_W_CITY": "Region rating (with city)",
    "APP_DOC_COUNT": "Documents provided",
    "APP_N_MISSING": "Missing fields in the application",
    "APP_OWN_CAR_AGE": "Car age",
    "BUR_DEBT_TO_INCOME": "Credit bureau debt ÷ income",
    "BUR_ANNUITY_BURDEN": "All repayments ÷ income",
    "PREV_CURRENT_TO_MEAN_CREDIT": "This loan vs past loan sizes",
}
PREFIXES = {
    "APP_": "Application",
    "BUR_": "Credit bureau",
    "BB_": "Bureau monthly status",
    "PREV_": "Previous applications",
    "INST_": "Repayment history",
    "POS_": "POS / cash loans",
    "CC_": "Credit cards",
}
WORDS = {"amt": "amount", "cnt": "count", "dpd": "days past due", "pct": "percent",
         "avg": "average", "num": "number"}


def feature_label(name: str) -> str:
    """A readable name: from LABELS, else built from the prefix and the column name."""
    if name in LABELS:
        return LABELS[name]
    for prefix, table in PREFIXES.items():
        if name.startswith(prefix):
            words = [WORDS.get(w, w) for w in name[len(prefix):].lower().split("_")]
            return f"{table}: {' '.join(words)}"
    return name


def format_value(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return "missing"
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}"
    return str(value)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _number(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _cast_like(new: pd.Series, old: pd.Series) -> pd.Series:
    """Give a rebuilt column the stored column's type (categories, float32, ...)."""
    try:
        return new.astype(old.dtype)
    except (TypeError, ValueError):  # e.g. a NaN in a column stored as integers
        return new.astype("float64")


class UnknownApplicant(KeyError):
    """The applicant id is not in the demo bundle."""


class ImpossibleChange(ValueError):
    """The what-if values contradict each other (e.g. employed longer than possible)."""


class Changes(BaseModel):
    """What-if values. Leave a field out to keep the applicant's own value."""

    model_config = ConfigDict(extra="forbid")
    income: float | None = Field(None, gt=0, le=1e8, description="Total income")
    loan_amount: float | None = Field(None, gt=0, le=1e8, description="Loan amount")
    annuity: float | None = Field(None, gt=0, le=1e7, description="Loan annuity")
    age_years: float | None = Field(None, ge=18, le=100)
    years_employed: float | None = Field(None, ge=0, le=60)
    ext_source_1: float | None = Field(None, ge=0, le=1)
    ext_source_2: float | None = Field(None, ge=0, le=1)
    ext_source_3: float | None = Field(None, ge=0, le=1)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applicant_id: int
    changes: Changes = Field(default_factory=Changes)


class Reason(BaseModel):
    feature: str
    label: str
    value: str
    impact: float = Field(description="Push on the log-odds of default (> 0 raises the risk)")


class ScoreResponse(BaseModel):
    applicant_id: int
    probability: float = Field(description="Predicted probability of default (calibrated, Step 10)")
    times_average: float = Field(description="probability ÷ the average default rate")
    average_default_rate: float
    base_probability: float = Field(description="The model's prediction for a typical applicant")
    raises_risk: list[Reason]
    lowers_risk: list[Reason]
    changed: dict[str, float]
    model: str


class Profile(BaseModel):
    applicant_id: int
    current: dict[str, float | None] = Field(description="The 8 what-if fields as they are now")
    details: dict[str, str]
    history: dict[str, bool] = Field(description="Which history tables have records")


class ScoringService:
    """Loads the final model and the demo bundle once; then scores applicants quickly."""

    def __init__(self, model_dir=None, demo_path=None):
        model_dir = model_dir or os.environ.get("CREDITRISK_MODEL_DIR")
        demo_path = Path(demo_path or os.environ.get("CREDITRISK_DEMO_PATH")
                         or get_path("processed") / DEMO_FILE)
        if not demo_path.exists():
            raise FileNotFoundError(f"{demo_path} not found. Run `make demo-data` first.")
        self.model, self.details = load_final_model(model_dir)
        self.columns = list(self.details["features"])
        bundle = pd.read_parquet(demo_path).set_index("SK_ID_CURR")
        raw_columns = [c for c in bundle.columns if c.startswith(RAW_PREFIX)]
        self.raw = bundle[raw_columns].rename(columns=lambda c: c[len(RAW_PREFIX):])
        self.features = bundle.drop(columns=raw_columns)
        self.applicant_ids = [int(i) for i in bundle.index]
        self.model_name = (f"LightGBM, {self.details.get('n_estimators')} trees, "
                           f"5-fold CV ROC-AUC {self.details.get('cv_roc_auc')}")

    def _rows(self, applicant_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if applicant_id not in self.features.index:
            raise UnknownApplicant(f"Unknown applicant {applicant_id}")
        return self.raw.loc[[applicant_id]].copy(), self.features.loc[[applicant_id]].copy()

    def build_features(self, applicant_id: int, changes: Changes | None = None) -> pd.DataFrame:
        """The model's input row for this applicant, after the what-if changes."""
        raw, row = self._rows(applicant_id)
        changed = (changes or Changes()).model_dump(exclude_none=True)
        for field, value in changed.items():
            if field in ("age_years", "years_employed"):
                value = -round(value * 365.25)  # back to the data's "days before applying"
            raw[FIELDS[field]] = value
        if {"age_years", "years_employed"} & set(changed):  # only check what the user changed
            age = -float(raw["DAYS_BIRTH"].iloc[0]) / 365.25
            employed = float(raw["DAYS_EMPLOYED"].iloc[0])
            if employed != NOT_EMPLOYED and -employed / 365.25 > age - 14:
                raise ImpossibleChange("Years employed cannot be more than age minus 14.")

        rebuilt = application_features_from_raw(
            raw.reset_index().assign(TARGET=np.nan, is_test=1))
        rebuilt.index = row.index
        for col in rebuilt.columns:
            if col.startswith("APP_") and col in row.columns:
                row[col] = _cast_like(rebuilt[col], row[col])
        stored = {col: row[col] for col in CROSS_TABLE if col in row.columns}
        row = add_cross_table_features(row)
        for col, old in stored.items():
            row[col] = _cast_like(row[col], old)
        return row[self.columns]

    def score(self, applicant_id: int, changes: Changes | None = None,
              top: int = 4) -> ScoreResponse:
        changes = changes or Changes()
        X = self.build_features(applicant_id, changes)
        probability = float(self.model.predict_proba(X)[0, 1])
        contributions = np.asarray(self.model.predict(X, pred_contrib=True))[0]
        impacts, base = contributions[:-1], float(contributions[-1])
        order = np.argsort(impacts)
        return ScoreResponse(
            applicant_id=applicant_id,
            probability=round(probability, 6),
            times_average=round(probability / TRAIN_DEFAULT_RATE, 3),
            average_default_rate=TRAIN_DEFAULT_RATE,
            base_probability=round(sigmoid(base), 6),
            raises_risk=[self._reason(X, j, impacts[j]) for j in order[::-1][:top]
                         if impacts[j] > 0],
            lowers_risk=[self._reason(X, j, impacts[j]) for j in order[:top] if impacts[j] < 0],
            changed=changes.model_dump(exclude_none=True),
            model=self.model_name,
        )

    def _reason(self, X: pd.DataFrame, j: int, impact: float) -> Reason:
        name = self.columns[j]
        return Reason(feature=name, label=feature_label(name), value=format_value(X.iloc[0, j]),
                      impact=round(float(impact), 4))

    def profile(self, applicant_id: int) -> Profile:
        raw, row = self._rows(applicant_id)
        r, f = raw.iloc[0], row.iloc[0]
        employed = r["DAYS_EMPLOYED"]
        current = {
            "income": _number(r["AMT_INCOME_TOTAL"]),
            "loan_amount": _number(r["AMT_CREDIT"]),
            "annuity": _number(r["AMT_ANNUITY"]),
            "age_years": round(-float(r["DAYS_BIRTH"]) / 365.25, 1),
            "years_employed": (None if pd.isna(employed) or employed == NOT_EMPLOYED
                               else round(-float(employed) / 365.25, 1)),
            **{f"ext_source_{k}": _number(r[f"EXT_SOURCE_{k}"]) for k in (1, 2, 3)},
        }
        details = {label: format_value(r[col]) for label, col in DETAILS.items() if col in r.index}
        history = {label: bool(f[col]) for label, col in HISTORY.items() if col in f.index}
        return Profile(applicant_id=applicant_id, current=current, details=details,
                       history=history)
