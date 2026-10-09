"""Step 7 experiment: ways to handle the rare defaulters (8% of loans).

Every variant uses the same LightGBM settings, so only the imbalance method changes.
Samplers sit inside an imbalanced-learn Pipeline, so they only change the TRAINING fold;
the held-out fold is never resampled (resampling it would make the scores meaningless).
"""

import numpy as np
from imblearn.pipeline import Pipeline as SamplerPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from creditrisk.models import baselines
from creditrisk.models.preprocess import QuantileClipper


def smote_preprocessor(numeric, categorical) -> ColumnTransformer:
    """SMOTE needs complete numbers: clip, fill gaps and scale the numbers, code text as integers.

    Output order: all numeric columns first, then the text-column codes.
    keep_empty_features keeps a fully empty column (filled with 0), so the positions of the
    text columns never shift; the SMOTE step needs those positions.
    """
    num = Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-2)
    return ColumnTransformer([("num", num, list(numeric)), ("cat", cat, list(categorical))])


class LightSMOTENC(BaseEstimator):
    """SMOTE for mixed data (numbers + text codes), written to stay small in memory.

    Each synthetic defaulter is placed at a random point on the line between a real defaulter
    and one of its k nearest defaulter neighbours (neighbours found on the numeric columns).
    Numeric columns are interpolated; text columns are copied from whichever parent the new
    point is closer to, so they are always real categories, never "in-between" codes.

    Why not imbalanced-learn's SMOTENC: it one-hot encodes the WHOLE training fold into a
    sparse matrix, which needs several GB at this data size. This version only works on the
    ~12k defaulters. Expects numeric columns first, then text codes (smote_preprocessor's order).
    """

    def __init__(self, n_numeric: int, sampling_strategy: float = 0.5, k_neighbors: int = 5,
                 random_state=None):
        self.n_numeric = n_numeric
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        minority = X[y == 1]
        n_new = int(round(self.sampling_strategy * np.sum(y == 0))) - len(minority)
        if n_new <= 0 or len(minority) < 2:
            return X, y
        k = min(self.k_neighbors, len(minority) - 1)
        numbers = minority[:, : self.n_numeric]
        neighbours = NearestNeighbors(n_neighbors=k + 1).fit(numbers).kneighbors(
            numbers, return_distance=False)[:, 1:]  # column 0 is the point itself
        rng = np.random.default_rng(self.random_state)
        base = rng.integers(len(minority), size=n_new)
        partner = neighbours[base, rng.integers(k, size=n_new)]
        gap = rng.random(n_new)
        new = minority[base].copy()
        new[:, : self.n_numeric] += gap[:, None] * (numbers[partner] - numbers[base])
        nearer_partner = gap > 0.5
        new[nearer_partner, self.n_numeric:] = minority[partner[nearer_partner], self.n_numeric:]
        return np.vstack([X, new]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def imbalance_models(groups: dict, seed: int = 42, ratio: float = 0.5) -> dict:
    """Name -> function that builds a fresh model.

    ratio: defaulters per non-defaulter AFTER resampling (0.5 = 1 defaulter for every 2 others,
    instead of about 1 for every 11.4 in the real data).

    none              LightGBM as it is (native missing values and categories).
    class_weights     each defaulter counts about 11.4 times in the loss ("balanced").
    undersampling     randomly drop non-defaulters from the training fold.
    imputed_no_smote  the preprocessing SMOTE needs, but no SMOTE: the fair control for SMOTE,
                      because the preprocessing alone may change the score.
    smote             the same preprocessing, then SMOTE-NC creates synthetic defaulters by
                      interpolating between neighbouring defaulters (LightSMOTENC above).
    """
    numeric, categorical = list(groups["numeric"]), list(groups["categorical"])

    def lgbm(**kw):
        return baselines.lightgbm(seed=seed, **kw)

    return {
        "none": lambda: lgbm(),
        "class_weights": lambda: lgbm(class_weight="balanced"),
        "undersampling": lambda: SamplerPipeline([
            ("sample", RandomUnderSampler(sampling_strategy=ratio, random_state=seed)),
            ("model", lgbm()),
        ]),
        "imputed_no_smote": lambda: Pipeline([
            ("prep", smote_preprocessor(numeric, categorical)),
            ("model", lgbm()),
        ]),
        "smote": lambda: SamplerPipeline([
            ("prep", smote_preprocessor(numeric, categorical)),
            ("sample", LightSMOTENC(len(numeric), sampling_strategy=ratio, random_state=seed)),
            ("model", lgbm()),
        ]),
    }
