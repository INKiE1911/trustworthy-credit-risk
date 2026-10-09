"""K-means with k-means++ seeding, from scratch (Bishop §9.1; Arthur & Vassilvitskii 2007).

Lloyd's algorithm repeats two steps until the centres stop moving:
    1. give every point to its nearest centre;
    2. move each centre to the mean of its points.
Each step can only lower the inertia (the sum of squared distances to the nearest centre),
so it always converges, but only to a local minimum. Hence several starts (n_init) and a
good seeding: k-means++ picks the first centre at random and every next one with probability
proportional to D(x)², the squared distance to the nearest centre chosen so far.
"""

import numpy as np
from sklearn.base import BaseEstimator


def squared_distances(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """(n, k) matrix of squared Euclidean distances, via |x|² - 2x·c + |c|²."""
    d2 = (X**2).sum(axis=1)[:, None] - 2 * X @ centres.T + (centres**2).sum(axis=1)[None, :]
    return np.maximum(d2, 0.0)


def plus_plus_seeds(X: np.ndarray, n_clusters: int, rng) -> np.ndarray:
    """k-means++: first centre uniform at random, then each next one with probability ∝ D(x)²."""
    X = np.asarray(X, dtype=float)
    centres = [X[rng.integers(len(X))]]
    d2 = ((X - centres[0]) ** 2).sum(axis=1)
    for _ in range(1, n_clusters):
        nxt = X[rng.choice(len(X), p=d2 / d2.sum())]
        centres.append(nxt)
        d2 = np.minimum(d2, ((X - nxt) ** 2).sum(axis=1))
    return np.array(centres)


class KMeansScratch(BaseEstimator):
    def __init__(self, n_clusters: int = 8, n_init: int = 4, max_iter: int = 300,
                 tol: float = 1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _lloyd(self, X: np.ndarray, centres: np.ndarray, tol: float):
        n_iter = 0
        while n_iter < self.max_iter:
            n_iter += 1
            labels = squared_distances(X, centres).argmin(axis=1)
            new = centres.copy()
            for j in range(self.n_clusters):
                members = labels == j
                if members.any():  # an empty cluster keeps its old centre
                    new[j] = X[members].mean(axis=0)
            shift = ((new - centres) ** 2).sum()
            centres = new
            if shift <= tol:
                break
        d2 = squared_distances(X, centres)
        labels = d2.argmin(axis=1)
        return centres, labels, float(d2[np.arange(len(X)), labels].sum()), n_iter

    def fit(self, X, y=None, init=None):
        """init: optional (k, d) starting centres; then only one start is run."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        tol = self.tol * np.mean(np.var(X, axis=0))  # same scale-free tolerance as scikit-learn
        starts = ([np.asarray(init, dtype=float)] if init is not None
                  else [plus_plus_seeds(X, self.n_clusters, rng) for _ in range(self.n_init)])
        best = None
        for centres in starts:
            result = self._lloyd(X, centres.copy(), tol)
            if best is None or result[2] < best[2]:
                best = result
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X) -> np.ndarray:
        return squared_distances(np.asarray(X, dtype=float), self.cluster_centers_).argmin(axis=1)
