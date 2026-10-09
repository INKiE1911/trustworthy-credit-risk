"""Logistic regression from scratch, plus its Bayesian (Laplace) version. Bishop §4.3–4.5.

Model: p(y=1 | x) = sigmoid(w·x + b).
Fitting minimises
    J(w, b) = Σ_i [log(1 + e^{z_i}) - y_i z_i]  +  ||w||² / (2C),      z_i = w·x_i + b.
The first part is the negative log-likelihood. The second is minus the log of a Gaussian
prior w ~ N(0, C·I), so the minimiser is the MAP estimate: "L2 regularisation = MAP".
b is not penalised (scikit-learn's lbfgs solver does the same), so the two can be compared.

Solver: Newton's method (also called IRLS). With θ = (w, b), X with a final column of ones,
p_i = sigmoid(z_i) and P = diag(1/C, ..., 1/C, 0):
    gradient  g = Xᵀ(p - y) + P θ
    Hessian   H = Xᵀ diag(p(1 - p)) X + P
Each step solves H d = g and moves θ ← θ - d, halving the step if J would go up.
J is convex and smooth, so Newton needs only about 5–10 steps.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(z):
    """1 / (1 + e^-z), written with tanh so it never overflows."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def with_intercept(X) -> np.ndarray:
    """Append a column of ones (for the intercept b)."""
    X = np.asarray(X, dtype=float)
    return np.hstack([X, np.ones((X.shape[0], 1))])


class LogisticRegressionScratch(ClassifierMixin, BaseEstimator):
    def __init__(self, C: float = 1.0, max_iter: int = 100, tol: float = 1e-8):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _objective(Xd, y, theta, penalty) -> float:
        z = Xd @ theta
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(penalty * theta**2))

    @staticmethod
    def _hessian(Xd, theta, penalty) -> np.ndarray:
        p = sigmoid(Xd @ theta)
        return (Xd * (p * (1 - p))[:, None]).T @ Xd + np.diag(penalty)

    def fit(self, X, y):
        Xd, y = with_intercept(X), np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        penalty = np.full(Xd.shape[1], 1.0 / self.C)
        penalty[-1] = 0.0  # the intercept has no prior
        theta = np.zeros(Xd.shape[1])
        current = self._objective(Xd, y, theta, penalty)
        self.objective_history_ = [current]
        for n_iter in range(1, self.max_iter + 1):
            self.n_iter_ = n_iter
            grad = Xd.T @ (sigmoid(Xd @ theta) - y) + penalty * theta
            step = np.linalg.solve(self._hessian(Xd, theta, penalty), grad)
            size = 1.0
            while True:  # backtracking: never accept a step that makes J worse
                candidate = theta - size * step
                value = self._objective(Xd, y, candidate, penalty)
                if value <= current or size < 1e-10:
                    break
                size /= 2
            moved = np.max(np.abs(candidate - theta))
            theta, current = candidate, value
            self.objective_history_.append(current)
            if moved < self.tol:
                break
        self.theta_ = theta
        self.coef_, self.intercept_ = theta[:-1], float(theta[-1])
        self.hessian_ = self._hessian(Xd, theta, penalty)  # used by the Laplace version
        return self

    def decision_function(self, X) -> np.ndarray:
        return with_intercept(X) @ self.theta_

    def predict_proba(self, X) -> np.ndarray:
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class BayesianLogisticRegressionScratch(LogisticRegressionScratch):
    """Laplace approximation (Bishop §4.4–4.5): posterior over θ ≈ N(θ_MAP, S), S = H⁻¹.

    Predictive probability with the probit approximation (Bishop eq. 4.153–4.155):
        mean logit  μ = θ_MAPᵀ x,   variance σ² = xᵀ S x,
        p(y=1 | x) ≈ sigmoid(μ / sqrt(1 + π σ² / 8)).
    An applicant unlike the training data gets a larger σ², so the probability is pulled
    towards 0.5: the model says it is unsure. logit_std() returns σ, the uncertainty
    for each applicant.
    """

    def fit(self, X, y):
        super().fit(X, y)
        self.posterior_cov_ = np.linalg.inv(self.hessian_)
        return self

    def logit_std(self, X) -> np.ndarray:
        Xd = with_intercept(X)
        return np.sqrt(np.maximum(np.sum((Xd @ self.posterior_cov_) * Xd, axis=1), 0.0))

    def predict_proba(self, X) -> np.ndarray:
        mu, var = self.decision_function(X), self.logit_std(X) ** 2
        p = sigmoid(mu / np.sqrt(1.0 + np.pi * var / 8.0))
        return np.column_stack([1 - p, p])

    def predict_proba_map(self, X) -> np.ndarray:
        """The plain MAP probabilities, ignoring the uncertainty."""
        return super().predict_proba(X)

    def sample_proba(self, X, n_samples: int = 4000, seed: int = 0) -> np.ndarray:
        """Monte Carlo average of sigmoid(θᵀx) over posterior samples: checks the probit trick."""
        rng = np.random.default_rng(seed)
        thetas = rng.multivariate_normal(self.theta_, self.posterior_cov_, size=n_samples)
        return sigmoid(with_intercept(X) @ thetas.T).mean(axis=1)
