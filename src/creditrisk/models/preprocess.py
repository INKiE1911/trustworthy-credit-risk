"""Preprocessing pipelines. They are fitted inside each CV fold, so nothing leaks."""

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

_SKLEARN_1_9 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 9)


def make_target_encoder(seed: int = 42) -> TargetEncoder:
    """Target encoding with seeded, cross-fitted folds (works on old and new scikit-learn)."""
    if _SKLEARN_1_9:  # 1.9+ wants the seed inside a CV splitter
        return TargetEncoder(target_type="binary",
                             cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="binary", random_state=seed)


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Cap every column at quantiles learned on the training data (e.g. the 117M income)."""

    def __init__(self, lower: float = 0.001, upper: float = 0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        with np.errstate(all="ignore"):
            low = np.nanquantile(X, self.lower, axis=0) if len(X) else np.full(X.shape[1], np.nan)
            high = np.nanquantile(X, self.upper, axis=0) if len(X) else np.full(X.shape[1], np.nan)
        self.low_ = np.where(np.isnan(low), -np.inf, low)  # all-empty column: no capping
        self.high_ = np.where(np.isnan(high), np.inf, high)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)  # NaN stays NaN


def make_linear_preprocessor(numeric, cat_low, cat_high, seed: int = 42) -> ColumnTransformer:
    """For logistic regression, Naive Bayes, KNN, SVM, MLP.

    numbers: cap outliers -> median imputation + missing flags -> standard scaling
    small text columns: one-hot; large text columns: target encoding (cross-fitted)
    """
    numbers = Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    small_text = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    large_text = Pipeline([
        ("target", make_target_encoder(seed)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [("num", numbers, list(numeric)), ("low", small_text, list(cat_low)),
         ("high", large_text, list(cat_high))],
        remainder="drop",
    )


def make_tree_preprocessor(numeric, categorical) -> ColumnTransformer:
    """For scikit-learn trees: numbers pass through (trees handle NaN), text -> integer codes."""
    codes = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                           encoded_missing_value=-2)
    return ColumnTransformer(
        [("num", "passthrough", list(numeric)), ("cat", codes, list(categorical))],
        remainder="drop",
    )


# --- column pickers that look at the data at fit time (used after feature selection) ---
def numeric_columns(X) -> list:
    return [c for c in X.columns if not isinstance(X[c].dtype, pd.CategoricalDtype)]


def small_text_columns(X, max_onehot: int = 10) -> list:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)
            and len(X[c].cat.categories) <= max_onehot]


def large_text_columns(X, max_onehot: int = 10) -> list:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)
            and len(X[c].cat.categories) > max_onehot]


def text_columns(X) -> list:
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def make_linear_preprocessor_auto(seed: int = 42) -> ColumnTransformer:
    """Same as make_linear_preprocessor, but finds the column types itself at fit time.

    Use it after a feature-selection step, when the kept columns are only known during fit.
    """
    numbers = Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    large_text = Pipeline([("target", make_target_encoder(seed)), ("scale", StandardScaler())])
    return ColumnTransformer(
        [("num", numbers, numeric_columns),
         ("low", OneHotEncoder(handle_unknown="ignore", sparse_output=False), small_text_columns),
         ("high", large_text, large_text_columns)],
        remainder="drop",
    )


def make_tree_preprocessor_auto() -> ColumnTransformer:
    """Same as make_tree_preprocessor, but finds the column types itself at fit time."""
    codes = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                           encoded_missing_value=-2)
    return ColumnTransformer(
        [("num", "passthrough", numeric_columns), ("cat", codes, text_columns)],
        remainder="drop",
    )
