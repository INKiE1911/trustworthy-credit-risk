"""A neural network with one hidden layer, written out by hand (Bishop §5.1–5.3).

Forward pass:    h = act(X W1 + b1),    p = sigmoid(h W2 + b2)
Loss for a batch of n rows (cross-entropy + L2, scaled like scikit-learn's MLPClassifier):
    L = -(1/n) Σ [y log p + (1 - y) log(1 - p)]  +  (alpha / 2n) (||W1||² + ||W2||²)
Backward pass (the chain rule, from the output back):
    δ2 = (p - y) / n                          sigmoid + cross-entropy simplify to this
    ∂L/∂W2 = hᵀ δ2 + (alpha/n) W2,           ∂L/∂b2 = Σ δ2
    δ1 = (δ2 W2ᵀ) ⊙ act'(X W1 + b1)          tanh' = 1 - h²,  relu' = 1 if z > 0 else 0
    ∂L/∂W1 = Xᵀ δ1 + (alpha/n) W1,           ∂L/∂b1 = Σ δ1
gradient_check() compares these formulas with finite differences, (L(θ+ε) - L(θ-ε)) / 2ε,
for every single weight. Training: mini-batch Adam, with early stopping on a held-back
part of the training rows.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from creditrisk.scratch.logreg import sigmoid


class MLPScratch(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden: int = 32, activation: str = "tanh", alpha: float = 1e-4,
                 learning_rate: float = 1e-3, batch_size: int = 256, max_epochs: int = 100,
                 patience: int = 10, valid_fraction: float = 0.1, random_state=None):
        self.hidden = hidden
        self.activation = activation
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.valid_fraction = valid_fraction
        self.random_state = random_state

    def init_params(self, n_features: int, rng) -> dict:
        """Glorot-uniform weights (keeps the signal size steady through the layers), zero biases."""
        h = self.hidden
        a1, a2 = np.sqrt(6 / (n_features + h)), np.sqrt(6 / (h + 1))
        return {"W1": rng.uniform(-a1, a1, (n_features, h)), "b1": np.zeros(h),
                "W2": rng.uniform(-a2, a2, (h, 1)), "b2": np.zeros(1)}

    def _forward(self, X, params):
        z1 = X @ params["W1"] + params["b1"]
        h = np.tanh(z1) if self.activation == "tanh" else np.maximum(z1, 0.0)
        return z1, h, (h @ params["W2"] + params["b2"])[:, 0]

    def loss_and_grads(self, X, y, params) -> tuple[float, dict]:
        n = len(X)
        z1, h, z2 = self._forward(X, params)
        penalty = self.alpha / (2 * n) * (np.sum(params["W1"] ** 2) + np.sum(params["W2"] ** 2))
        loss = float(np.mean(np.logaddexp(0.0, z2) - y * z2) + penalty)
        d2 = ((sigmoid(z2) - y) / n)[:, None]
        grads = {"W2": h.T @ d2 + self.alpha / n * params["W2"], "b2": d2.sum(axis=0)}
        dh = d2 @ params["W2"].T
        d1 = dh * (1 - h**2) if self.activation == "tanh" else dh * (z1 > 0)
        grads["W1"] = X.T @ d1 + self.alpha / n * params["W1"]
        grads["b1"] = d1.sum(axis=0)
        return loss, grads

    def gradient_check(self, X, y, eps: float = 1e-6, seed: int = 0) -> float:
        """Largest relative error between the backprop and finite-difference gradients."""
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        params = self.init_params(X.shape[1], np.random.default_rng(seed))
        _, grads = self.loss_and_grads(X, y, params)
        worst = 0.0
        for name, values in params.items():
            for idx in np.ndindex(values.shape):
                original = values[idx]
                values[idx] = original + eps
                up = self.loss_and_grads(X, y, params)[0]
                values[idx] = original - eps
                down = self.loss_and_grads(X, y, params)[0]
                values[idx] = original
                numeric, exact = (up - down) / (2 * eps), grads[name][idx]
                worst = max(worst, abs(numeric - exact) / max(abs(numeric) + abs(exact), 1e-12))
        return worst

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(X))
        n_valid = int(round(self.valid_fraction * len(X)))
        valid, train = order[:n_valid], order[n_valid:]
        params = self.init_params(X.shape[1], rng)
        m = {k: np.zeros_like(v) for k, v in params.items()}  # Adam: running mean of gradients
        v = {k: np.zeros_like(v) for k, v in params.items()}  # Adam: running mean of squares
        beta1, beta2, step = 0.9, 0.999, 0
        best_loss, best_params, waited = np.inf, None, 0
        self.history_ = []
        for epoch in range(1, self.max_epochs + 1):
            shuffled = train[rng.permutation(len(train))]
            for start in range(0, len(shuffled), self.batch_size):
                rows = shuffled[start:start + self.batch_size]
                _, grads = self.loss_and_grads(X[rows], y[rows], params)
                step += 1
                for k in params:
                    m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                    v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                    m_hat, v_hat = m[k] / (1 - beta1**step), v[k] / (1 - beta2**step)
                    params[k] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            train_loss = self._log_loss(X[train], y[train], params)
            valid_loss = self._log_loss(X[valid], y[valid], params) if n_valid else train_loss
            self.history_.append(
                {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
            if valid_loss < best_loss - 1e-5:
                best_loss, waited = valid_loss, 0
                best_params = {k: val.copy() for k, val in params.items()}
            else:
                waited += 1
                if n_valid and waited >= self.patience:
                    break
        self.params_ = best_params if best_params is not None else params
        self.classes_ = np.array([0, 1])
        self.n_epochs_ = epoch
        return self

    def _log_loss(self, X, y, params) -> float:
        z2 = self._forward(X, params)[2]
        return float(np.mean(np.logaddexp(0.0, z2) - y * z2))

    def predict_proba(self, X) -> np.ndarray:
        p = sigmoid(self._forward(np.asarray(X, dtype=float), self.params_)[2])
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
