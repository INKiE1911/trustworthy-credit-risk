"""Gaussian mixture model fitted with EM, from scratch (Bishop §9.2–9.4).

Model: p(x) = Σ_k π_k N(x | μ_k, Σ_k).
E-step: responsibilities r_ik = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j),
        computed with logs and log-sum-exp so tiny densities do not underflow.
M-step: N_k = Σ_i r_ik,  π_k = N_k / n,  μ_k = Σ_i r_ik x_i / N_k,
        Σ_k = Σ_i r_ik (x_i - μ_k)(x_i - μ_k)ᵀ / N_k + reg_covar·I.
An EM round never lowers the log-likelihood (Bishop §9.4); the tests check this.
It starts from K-means (this package's KMeansScratch) unless starting values are given.
EM only finds a local optimum, so n_init starts are tried and the best one is kept.
"""

import numpy as np
from sklearn.base import BaseEstimator

from creditrisk.scratch.kmeans import KMeansScratch


def logsumexp_rows(a: np.ndarray) -> np.ndarray:
    top = a.max(axis=1, keepdims=True)
    return (top + np.log(np.exp(a - top).sum(axis=1, keepdims=True)))[:, 0]


class GaussianMixtureScratch(BaseEstimator):
    def __init__(self, n_components: int = 3, n_init: int = 1, max_iter: int = 200,
                 tol: float = 1e-6, reg_covar: float = 1e-6, random_state=None):
        self.n_components = n_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _log_gauss(self, X: np.ndarray) -> np.ndarray:
        """(n, K) matrix of log N(x_i | μ_k, Σ_k), using a Cholesky factor of each Σ_k."""
        n, d = X.shape
        out = np.empty((n, self.n_components))
        for k in range(self.n_components):
            L = np.linalg.cholesky(self.covariances_[k])
            z = np.linalg.solve(L, (X - self.means_[k]).T)
            log_det = 2 * np.log(np.diag(L)).sum()
            out[:, k] = -0.5 * (d * np.log(2 * np.pi) + log_det + (z**2).sum(axis=0))
        return out

    def _e_step(self, X):
        weighted = self._log_gauss(X) + np.log(self.weights_)
        log_norm = logsumexp_rows(weighted)
        return float(log_norm.mean()), np.exp(weighted - log_norm[:, None])

    def _m_step(self, X, resp):
        nk = resp.sum(axis=0) + 10 * np.finfo(float).eps
        self.weights_ = nk / len(X)
        self.means_ = resp.T @ X / nk[:, None]
        covariances = []
        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov = (resp[:, k, None] * diff).T @ diff / nk[k]
            covariances.append(cov + self.reg_covar * np.eye(X.shape[1]))
        self.covariances_ = np.array(covariances)

    def _run_em(self, X):
        self.log_likelihood_history_ = []
        self.converged_ = False
        for n_iter in range(1, self.max_iter + 1):
            self.n_iter_ = n_iter
            log_lik, resp = self._e_step(X)
            self.log_likelihood_history_.append(log_lik)
            self._m_step(X, resp)
            history = self.log_likelihood_history_
            if len(history) > 1 and abs(history[-1] - history[-2]) < self.tol:
                self.converged_ = True
                break

    def fit(self, X, y=None, weights_init=None, means_init=None, covariances_init=None):
        X = np.asarray(X, dtype=float)
        if means_init is not None:
            starts = [(np.asarray(weights_init, dtype=float), np.asarray(means_init, dtype=float),
                       np.asarray(covariances_init, dtype=float))]
        else:
            seeds = np.random.default_rng(self.random_state).integers(0, 2**31 - 1, self.n_init)
            starts = []
            for seed in seeds:
                km = KMeansScratch(self.n_components, n_init=1, random_state=int(seed)).fit(X)
                self._m_step(X, np.eye(self.n_components)[km.labels_])
                starts.append((self.weights_, self.means_, self.covariances_))
        best = None
        for weights, means, covariances in starts:
            self.weights_, self.means_, self.covariances_ = weights, means, covariances
            self._run_em(X)
            final = self.score(X)
            if best is None or final > best[0]:
                best = (final, self.weights_, self.means_, self.covariances_,
                        self.log_likelihood_history_, self.n_iter_, self.converged_)
        (_, self.weights_, self.means_, self.covariances_, self.log_likelihood_history_,
         self.n_iter_, self.converged_) = best
        return self

    def score_samples(self, X) -> np.ndarray:
        """log p(x) for every row."""
        X = np.asarray(X, dtype=float)
        return logsumexp_rows(self._log_gauss(X) + np.log(self.weights_))

    def score(self, X, y=None) -> float:
        """Average log-likelihood per row (higher is better)."""
        return float(self.score_samples(X).mean())

    def predict_proba(self, X) -> np.ndarray:
        return self._e_step(np.asarray(X, dtype=float))[1]

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
