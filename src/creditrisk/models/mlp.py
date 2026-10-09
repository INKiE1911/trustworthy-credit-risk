"""Step 8: a PyTorch MLP for the tabular data, with embeddings for the text columns.

It is wrapped as a scikit-learn classifier (fit / predict_proba), so it runs through the same
CV runner, folds and metrics as every other model.

Inputs (prepared inside each training fold, so nothing leaks):
    numbers -> clip outliers, fill gaps with the median (+ a missing flag), standardise
    text    -> an integer code per category (0 = missing or never seen), then a learned
               embedding: a small vector per category, trained with the network
               ("entity embeddings", Guo & Berkhahn 2016)
Network:  [numbers | embeddings] -> (Linear -> BatchNorm -> ReLU -> Dropout) x layers -> 1 logit
Training: Adam on the cross-entropy loss, in mini-batches. 10% of the training fold is held
back; training stops when its ROC-AUC has not improved for `patience` epochs, and the
weights from the best epoch are kept.

PyTorch is optional for the rest of the project: only this model needs it.
CPU install: pip install torch --index-url https://download.pytorch.org/whl/cpu
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from creditrisk.models.baselines import default_threads
from creditrisk.models.preprocess import QuantileClipper

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

INSTALL_HINT = "pip install torch --index-url https://download.pytorch.org/whl/cpu"


class TabularInputs:
    """Turns the feature DataFrame into the two arrays the network reads."""

    def fit(self, X: pd.DataFrame):
        self.categorical_ = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        self.numeric_ = [c for c in X.columns if c not in self.categorical_]
        self.numbers_ = Pipeline([
            ("clip", QuantileClipper()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                     keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]).fit(X[self.numeric_])
        self.levels_ = {c: list(X[c].cat.categories) for c in self.categorical_}
        return self

    @property
    def cardinalities(self) -> list[int]:
        """Number of embedding rows per text column: its categories + 1 slot for missing."""
        return [len(self.levels_[c]) + 1 for c in self.categorical_]

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numbers = np.ascontiguousarray(self.numbers_.transform(X[self.numeric_]), dtype=np.float32)
        codes = np.zeros((len(X), len(self.categorical_)), dtype=np.int64)
        for j, c in enumerate(self.categorical_):
            # code -1 (missing, or a category not seen in training) + 1 -> slot 0
            known = pd.Categorical(X[c], categories=self.levels_[c])
            codes[:, j] = known.codes.astype(np.int64) + 1
        return numbers, codes


if torch is not None:

    class TabularNet(nn.Module):
        def __init__(self, n_numbers: int, cardinalities: list[int], hidden, dropout: float,
                     max_emb_dim: int):
            super().__init__()
            dims = [min(max_emb_dim, (k + 1) // 2) for k in cardinalities]
            self.embeddings = nn.ModuleList(
                [nn.Embedding(k, d) for k, d in zip(cardinalities, dims, strict=True)])
            layers, width = [], n_numbers + sum(dims)
            for size in hidden:
                layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU(),
                           nn.Dropout(dropout)]
                width = size
            layers.append(nn.Linear(width, 1))
            self.layers = nn.Sequential(*layers)

        def forward(self, numbers, codes):
            parts = [numbers] + [emb(codes[:, j]) for j, emb in enumerate(self.embeddings)]
            return self.layers(torch.cat(parts, dim=1)).squeeze(1)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(256, 128), dropout: float = 0.3, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5, batch_size: int = 1024, max_epochs: int = 40,
                 patience: int = 4, valid_fraction: float = 0.1, max_emb_dim: int = 16,
                 seed: int = 42, threads: int | None = None, verbose: bool = False):
        self.hidden = hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.valid_fraction = valid_fraction
        self.max_emb_dim = max_emb_dim
        self.seed = seed
        self.threads = threads
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y):
        if torch is None:
            raise ImportError(f"PyTorch is not installed. CPU version: {INSTALL_HINT}")
        start_time = time.perf_counter()
        y = np.asarray(y).astype(np.float32)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads or default_threads())
        rng = np.random.default_rng(self.seed)
        train_rows, check_rows = train_test_split(
            np.arange(len(X)), test_size=self.valid_fraction, stratify=y, random_state=self.seed)

        self.inputs_ = TabularInputs().fit(X.iloc[train_rows])
        numbers, codes = self.inputs_.transform(X)
        numbers_t, codes_t, target_t = (torch.from_numpy(numbers), torch.from_numpy(codes),
                                        torch.from_numpy(y))
        self.net_ = TabularNet(numbers.shape[1], self.inputs_.cardinalities, self.hidden,
                               self.dropout, self.max_emb_dim)
        optimiser = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc, best_state, waited = -np.inf, None, 0
        self.history_ = []
        for epoch in range(1, self.max_epochs + 1):
            self.net_.train()
            order = train_rows[rng.permutation(len(train_rows))]
            total, seen = 0.0, 0
            for first in range(0, len(order), self.batch_size):
                batch = order[first:first + self.batch_size]
                if len(batch) < 2:  # BatchNorm needs at least 2 rows
                    continue
                rows = torch.from_numpy(batch)
                optimiser.zero_grad()
                loss = loss_fn(self.net_(numbers_t[rows], codes_t[rows]), target_t[rows])
                loss.backward()
                optimiser.step()
                total += float(loss.item()) * len(batch)
                seen += len(batch)
            check_auc = float(roc_auc_score(
                y[check_rows], self._probabilities(numbers[check_rows], codes[check_rows])))
            self.history_.append({"epoch": epoch, "train_loss": total / max(seen, 1),
                                  "check_auc": check_auc})
            if self.verbose:
                print(f"    epoch {epoch:>2}: train loss {total / max(seen, 1):.4f}, "
                      f"check ROC-AUC {check_auc:.4f}", flush=True)
            if check_auc > best_auc + 1e-4:
                best_auc, waited = check_auc, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmax([h["check_auc"] for h in self.history_])) + 1
        self.classes_ = np.array([0, 1])
        self.fit_seconds_ = time.perf_counter() - start_time
        return self

    def _probabilities(self, numbers: np.ndarray, codes: np.ndarray) -> np.ndarray:
        self.net_.eval()
        out = []
        with torch.no_grad():
            for first in range(0, len(numbers), 8192):
                logits = self.net_(torch.from_numpy(numbers[first:first + 8192]),
                                   torch.from_numpy(codes[first:first + 8192]))
                out.append(torch.sigmoid(logits).numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self._probabilities(*self.inputs_.transform(X)).astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
