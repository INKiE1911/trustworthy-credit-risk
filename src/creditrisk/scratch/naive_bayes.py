"""Gaussian Naive Bayes from scratch (Bishop §4.2; ISLP §4.4.4).

Assumes the features are independent given the class, each with a normal distribution:
    log p(k | x) = log π_k + Σ_j log N(x_j | μ_kj, σ²_kj) - log p(x).
Each variance gets ε = var_smoothing × (largest feature variance) added, exactly like
scikit-learn, so a feature that is constant inside a class cannot divide by zero.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class GaussianNBScratch(ClassifierMixin, BaseEstimator):
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / len(y)
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        self.theta_ = np.array([X[y == k].mean(axis=0) for k in self.classes_])
        self.var_ = np.array([X[y == k].var(axis=0) for k in self.classes_]) + self.epsilon_
        return self

    def joint_log_likelihood(self, X) -> np.ndarray:
        """log π_k + log p(x | k) for every row and class."""
        X = np.asarray(X, dtype=float)
        columns = []
        for k in range(len(self.classes_)):
            columns.append(np.log(self.class_prior_[k])
                           - 0.5 * np.sum(np.log(2 * np.pi * self.var_[k]))
                           - 0.5 * np.sum((X - self.theta_[k]) ** 2 / self.var_[k], axis=1))
        return np.column_stack(columns)

    def predict_proba(self, X) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)  # log-sum-exp trick: avoids exp underflow
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]
