"""Custom steps for model pipelines: a feature selector and the WoE encoder for the scorecard."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from creditrisk.models.baselines import default_threads


class LightGBMSelector(TransformerMixin, BaseEstimator):
    """Keep the k columns a quick LightGBM finds most useful.

    It is fitted inside each training fold, so the choice never sees the held-out rows.
    Works on raw features (NaN and text columns allowed).
    """

    def __init__(self, k: int = 60, n_estimators: int = 100, seed: int = 42):
        self.k = k
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X, y):
        model = LGBMClassifier(
            n_estimators=self.n_estimators, learning_rate=0.1, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
            random_state=self.seed, n_jobs=default_threads(), verbose=-1,
        ).fit(X, y)
        gain = pd.Series(model.booster_.feature_importance(importance_type="gain"),
                         index=X.columns)
        self.selected_ = list(gain.sort_values(ascending=False, kind="stable").index[: self.k])
        return self

    def transform(self, X):
        return X[self.selected_]


class WoEEncoder(TransformerMixin, BaseEstimator):
    """Bin every column and replace each value by the weight of evidence (WoE) of its bin.

    numbers: up to n_bins quantile bins, plus one bin for missing values
    text:    one bin per category (categories under min_share are merged into "other"),
             plus one bin for missing values
    WoE of a bin = ln(share of all good customers in it / share of all defaulters in it),
    so a positive WoE means safer than average. iv_ holds each column's information value.
    Only the `max_features` columns with the highest IV (and IV >= min_iv) are kept.
    """

    def __init__(self, n_bins: int = 10, min_share: float = 0.01, min_iv: float = 0.02,
                 max_features: int = 30, smoothing: float = 0.5):
        self.n_bins = n_bins
        self.min_share = min_share
        self.min_iv = min_iv
        self.max_features = max_features
        self.smoothing = smoothing

    # -- binning --------------------------------------------------------------------------
    def _bins(self, col: str, s: pd.Series) -> np.ndarray:
        """Bin number of every value: 0..k-1 for normal bins, k for missing."""
        rule = self.rules_[col]
        missing_bin = len(rule["labels"]) - 1
        if rule["kind"] == "number":
            values = s.to_numpy(dtype=float)
            out = np.searchsorted(rule["cuts"], values, side="right")
            out[np.isnan(values)] = missing_bin
            return out
        mapping = rule["mapping"]
        other = mapping.get("__other__", -1)
        # np.array(...) makes a writable copy (pandas 3 returns read-only arrays)
        out = np.array(s.astype(object).map(lambda v: mapping.get(v, other)), dtype=int)
        out[np.array(pd.isna(s))] = missing_bin
        return out

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y).astype(int)
        good, bad = (y == 0), (y == 1)
        n_good, n_bad = good.sum(), bad.sum()
        s = self.smoothing
        self.rules_, self.woe_, self.iv_ = {}, {}, {}
        for col in X.columns:
            series = X[col]
            is_text = (isinstance(series.dtype, pd.CategoricalDtype)
                       or not pd.api.types.is_numeric_dtype(series))
            if is_text:
                shares = series.value_counts(normalize=True)
                keep = [c for c in shares.index if shares[c] >= self.min_share]
                mapping = {c: i for i, c in enumerate(keep)}
                labels = [str(c) for c in keep]
                if len(keep) < len(shares):
                    mapping["__other__"] = len(keep)
                    labels.append("other")
                self.rules_[col] = {"kind": "text", "mapping": mapping,
                                    "labels": labels + ["missing"]}
            else:
                values = series.to_numpy(dtype=float)
                inner = np.linspace(0, 1, self.n_bins + 1)[1:-1]
                cuts = (np.unique(np.nanquantile(values, inner))
                        if np.isfinite(values).any() else np.array([]))
                edges = [-np.inf, *cuts, np.inf]
                labels = [f"({a:.4g}, {b:.4g}]" for a, b in zip(edges[:-1], edges[1:], strict=True)]
                self.rules_[col] = {"kind": "number", "cuts": cuts, "labels": labels + ["missing"]}
            bins = self._bins(col, series)
            k = len(self.rules_[col]["labels"])
            g = np.bincount(bins[good], minlength=k)
            b = np.bincount(bins[bad], minlength=k)
            share_g = (g + s) / (n_good + s * k)
            share_b = (b + s) / (n_bad + s * k)
            woe = np.log(share_g / share_b)
            woe[(g + b) == 0] = 0.0  # empty bin (e.g. no missing values in training): neutral
            self.woe_[col] = woe
            self.iv_[col] = float(np.sum((share_g - share_b) * woe))
        ranked = sorted(self.iv_, key=self.iv_.get, reverse=True)
        self.selected_ = [c for c in ranked if self.iv_[c] >= self.min_iv][: self.max_features]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.woe_[c][self._bins(c, X[c])] for c in self.selected_])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_, dtype=object)


def scorecard_points(pipeline, pdo: float = 20, base_score: float = 600,
                     base_odds: float = 50) -> pd.DataFrame:
    """Points table of a fitted WoE scorecard (steps "woe" and "model").

    Convention: base_score points at good:bad odds of base_odds:1; every pdo points
    doubles the odds. An applicant's score is the sum of the points of their bins.
    """
    woe, model = pipeline.named_steps["woe"], pipeline.named_steps["model"]
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    beta, beta0 = model.coef_[0], model.intercept_[0]
    m = len(woe.selected_)
    rows = []
    for j, col in enumerate(woe.selected_):
        for label, w in zip(woe.rules_[col]["labels"], woe.woe_[col], strict=True):
            points = -factor * beta[j] * w + (offset - factor * beta0) / m
            rows.append({"feature": col, "bin": label, "woe": w, "points": points,
                         "iv": woe.iv_[col]})
    return pd.DataFrame(rows)


class DropUninformativeColumns(TransformerMixin, BaseEstimator):
    """Drop columns with fewer than 2 different non-missing values in the training data.

    Such columns carry no information, and a completely empty column crashes
    HistGradientBoosting in scikit-learn 1.9.
    """

    def fit(self, X, y=None):
        self.keep_ = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
        self.dropped_ = [c for c in X.columns if c not in self.keep_]
        return self

    def transform(self, X):
        return X[self.keep_]
