"""Step 4 baseline models. Each function returns a fresh, unfitted model."""

from joblib import cpu_count
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from creditrisk.models.preprocess import make_linear_preprocessor, make_tree_preprocessor


def default_threads(max_threads: int = 8) -> int:
    """Threads for LightGBM: the number of PHYSICAL CPU cores, at most 8.

    With n_jobs=-1, LightGBM uses every logical thread. Inside WSL (a virtual machine), with
    hyper-threading or efficiency cores, the threads keep waiting for each other at every tree
    step, and training can become extremely slow or hang. LightGBM's docs recommend
    physical cores, and more than about 8 threads doesn't help on data this size.
    """
    return max(1, min(max_threads, cpu_count(only_physical_cores=True)))


def dummy() -> DummyClassifier:
    """Always predicts the overall default rate: the 'do nothing' baseline."""
    return DummyClassifier(strategy="prior")


def logistic_regression(groups: dict, C: float = 0.1, seed: int = 42) -> Pipeline:
    prep = make_linear_preprocessor(groups["numeric"], groups["cat_low"], groups["cat_high"], seed)
    return Pipeline([("prep", prep), ("model", LogisticRegression(C=C, max_iter=3000))])


def naive_bayes(groups: dict, seed: int = 42) -> Pipeline:
    prep = make_linear_preprocessor(groups["numeric"], groups["cat_low"], groups["cat_high"], seed)
    return Pipeline([("prep", prep), ("model", GaussianNB())])


def decision_tree(groups: dict, max_depth: int = 6, min_samples_leaf: int = 200,
                  seed: int = 42) -> Pipeline:
    prep = make_tree_preprocessor(groups["numeric"], groups["categorical"])
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  random_state=seed)
    return Pipeline([("prep", prep), ("model", tree)])


def lightgbm(seed: int = 42, **overrides) -> LGBMClassifier:
    """LightGBM v0 with sensible fixed settings (tuning comes in Step 7).

    Text columns are used natively (pandas 'category' dtype), NaN is handled natively.
    Override anything, e.g. lightgbm(n_jobs=2) or lightgbm(n_estimators=1000).
    """
    params = dict(
        n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
        random_state=seed, n_jobs=default_threads(), verbose=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)
