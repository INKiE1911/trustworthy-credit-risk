"""PCA through the singular value decomposition, from scratch (Bishop §12.1).

Centre the data, X_c = X - mean, and factor X_c = U S Vᵀ. The rows of Vᵀ are the principal
directions, ordered by variance; the variance along direction i is S_i² / (n - 1).
A direction v and -v describe the same line, so the sign is arbitrary: each component is
flipped to make its largest entry (in absolute value) positive.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def fix_signs(rows: np.ndarray) -> np.ndarray:
    """Flip each row so that its largest-magnitude entry is positive."""
    signs = np.sign(rows[np.arange(len(rows)), np.argmax(np.abs(rows), axis=1)])
    signs[signs == 0] = 1.0
    return rows * signs[:, None]


class PCAScratch(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        k = self.n_components or len(S)
        variance = S**2 / (len(X) - 1)
        self.components_ = fix_signs(Vt[:k])
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = variance[:k] / variance.sum()
        self.singular_values_ = S[:k]
        return self

    def transform(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, Z) -> np.ndarray:
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_
