"""Fisher's linear discriminant and the LDA classifier, from scratch (Bishop §4.1.4–4.1.6).

Projection: find directions v that push the class means apart relative to the spread
inside each class, i.e. maximise (vᵀ S_B v) / (vᵀ S_W v) with
    S_W = Σ_k Σ_{i in k} (x_i - m_k)(x_i - m_k)ᵀ      within-class scatter
    S_B = Σ_k n_k (m_k - m)(m_k - m)ᵀ                between-class scatter.
The best directions solve S_B v = λ S_W v (at most K - 1 of them). For 2 classes this is
v ∝ S_W⁻¹ (m_1 - m_0), Fisher's original answer. Solved by whitening: S_W = L Lᵀ (Cholesky),
then an ordinary symmetric eigenproblem for L⁻¹ S_B L⁻ᵀ, then v = L⁻ᵀ u.

Classifier: each class is a Gaussian with the SAME covariance Σ = S_W / n, so
    log p(k | x) = xᵀ Σ⁻¹ m_k - ½ m_kᵀ Σ⁻¹ m_k + log π_k + constant
is linear in x (like scikit-learn's LinearDiscriminantAnalysis with solver="lsqr").

If some features are exact combinations of others (e.g. a full set of one-hot columns),
S_W is singular and the best direction is no longer unique. Reduce with PCA first.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from creditrisk.scratch.pca import fix_signs


class FisherLDA(ClassifierMixin, BaseEstimator):
    def __init__(self, n_components: int | None = None, reg: float = 1e-10):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        n, d = X.shape
        self.priors_ = counts / n
        self.means_ = np.array([X[y == k].mean(axis=0) for k in self.classes_])
        self.xbar_ = X.mean(axis=0)

        centred = X - self.means_[np.searchsorted(self.classes_, y)]
        s_w = centred.T @ centred
        gaps = self.means_ - self.xbar_
        s_b = (gaps * counts[:, None]).T @ gaps

        # Projection directions: whiten with the Cholesky factor of (slightly regularised) S_W
        L = np.linalg.cholesky(s_w + self.reg * np.trace(s_w) / d * np.eye(d))
        L_inv = np.linalg.inv(L)
        A = L_inv @ s_b @ L_inv.T
        values, vectors = np.linalg.eigh((A + A.T) / 2)
        order = np.argsort(values)[::-1]
        k = self.n_components or min(len(self.classes_) - 1, d)
        self.scalings_ = fix_signs((L_inv.T @ vectors[:, order[:k]]).T).T
        top = np.maximum(values[order[: len(self.classes_) - 1]], 0)
        self.explained_variance_ratio_ = top[:k] / top.sum()

        # Classifier: shared covariance; least squares also copes with a singular Σ
        self.covariance_ = s_w / n
        self.coef_ = np.linalg.lstsq(self.covariance_, self.means_.T, rcond=None)[0].T
        self.intercept_ = -0.5 * np.sum(self.means_ * self.coef_, axis=1) + np.log(self.priors_)
        return self

    def transform(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.xbar_) @ self.scalings_

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
