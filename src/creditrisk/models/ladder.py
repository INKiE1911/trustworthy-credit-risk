"""Step 6: every syllabus model, ready for the CV runner.

ladder_models(groups) returns {name: function that builds a fresh, unfitted model}.
KNN, the SVMs, AdaBoost and EBM first keep the top features (LightGBMSelector), which is
fitted inside each training fold, so the selection never leaks.
"""

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from creditrisk.models import baselines
from creditrisk.models.baselines import default_threads
from creditrisk.models.preprocess import (
    make_linear_preprocessor,
    make_linear_preprocessor_auto,
    make_tree_preprocessor,
    make_tree_preprocessor_auto,
)
from creditrisk.models.transformers import DropUninformativeColumns, LightGBMSelector, WoEEncoder


def woe_scorecard(seed: int = 42, max_features: int = 30) -> Pipeline:
    """The banking classic: WoE bins -> logistic regression -> points table."""
    return Pipeline([
        ("woe", WoEEncoder(max_features=max_features)),
        ("model", LogisticRegression(C=1.0, max_iter=2000)),
    ])


def naive_bayes_pca(groups: dict, seed: int = 42, n_components: int = 30) -> Pipeline:
    """Naive Bayes on PCA components: PCA removes the correlations NB cannot handle."""
    prep = make_linear_preprocessor(groups["numeric"], groups["cat_low"], groups["cat_high"], seed)
    return Pipeline([("prep", prep), ("pca", PCA(n_components=n_components, random_state=seed)),
                     ("model", GaussianNB())])


def knn(seed: int = 42, k_features: int = 60, n_components: int = 20,
        n_neighbors: int = 200) -> Pipeline:
    """KNN on PCA components of the top features (distances need few, scaled dimensions)."""
    return Pipeline([
        ("select", LightGBMSelector(k=k_features, seed=seed)),
        ("prep", make_linear_preprocessor_auto(seed)),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("model", KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=default_threads())),
    ])


def svm_linear(seed: int = 42, k_features: int = 100, C: float = 0.05) -> Pipeline:
    """Linear SVM; Platt scaling (sigmoid calibration) turns its scores into probabilities."""
    svm = LinearSVC(C=C, dual="auto", max_iter=3000, random_state=seed)
    return Pipeline([
        ("select", LightGBMSelector(k=k_features, seed=seed)),
        ("prep", make_linear_preprocessor_auto(seed)),
        ("model", CalibratedClassifierCV(svm, method="sigmoid", cv=3)),
    ])


def svm_rbf(seed: int = 42, k_features: int = 50, n_components: int = 300,
            C: float = 0.05) -> Pipeline:
    """RBF-kernel SVM, approximated with Nystroem so it scales to all training rows."""
    svm = LinearSVC(C=C, dual="auto", max_iter=3000, random_state=seed)
    return Pipeline([
        ("select", LightGBMSelector(k=k_features, seed=seed)),
        ("prep", make_linear_preprocessor_auto(seed)),
        ("rbf", Nystroem(kernel="rbf", n_components=n_components, random_state=seed)),
        ("model", CalibratedClassifierCV(svm, method="sigmoid", cv=3)),
    ])


def random_forest(groups: dict, seed: int = 42, n_estimators: int = 150,
                  oob: bool = False) -> Pipeline:
    """Bagging: many trees on random half-samples, averaged."""
    forest = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=50, max_features="sqrt", max_samples=0.3,
        oob_score=oob, n_jobs=default_threads(), random_state=seed,
    )
    return Pipeline([("prep", make_tree_preprocessor(groups["numeric"], groups["categorical"])),
                     ("model", forest)])


def adaboost(seed: int = 42, k_features: int = 50, n_estimators: int = 100) -> Pipeline:
    """The first boosting method: decision stumps, each one focusing on earlier mistakes."""
    return Pipeline([
        ("select", LightGBMSelector(k=k_features, seed=seed)),
        ("prep", make_tree_preprocessor_auto()),
        ("impute", SimpleImputer(strategy="median")),
        ("model", AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                                     n_estimators=n_estimators, learning_rate=0.5,
                                     random_state=seed)),
    ])


def hist_gradient_boosting(seed: int = 42, max_iter: int = 500) -> Pipeline:
    """scikit-learn's gradient boosting (same idea as LightGBM).

    Empty or constant columns are dropped first: an all-empty column crashes it in 1.9.
    """
    model = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=1.0, categorical_features="from_dtype", early_stopping=False,
        random_state=seed,
    )
    return Pipeline([("drop", DropUninformativeColumns()), ("model", model)])


def xgboost(seed: int = 42, n_estimators: int = 500) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, tree_method="hist",
        enable_categorical=True, n_jobs=default_threads(), random_state=seed,
    )


def ebm(seed: int = 42, k_features: int = 40) -> Pipeline:
    """Explainable Boosting Machine: a glass-box model on the top features."""
    model = ExplainableBoostingClassifier(interactions=5, outer_bags=4,
                                          n_jobs=default_threads(), random_state=seed)
    return Pipeline([("select", LightGBMSelector(k=k_features, seed=seed)), ("model", model)])


def ladder_models(groups: dict, seed: int = 42, fast: bool = False) -> dict:
    """All Step 6 models as {name: builder}. fast=True shrinks them (for tests only)."""
    n = 20 if fast else None
    return {
        "logistic_regression": lambda: baselines.logistic_regression(groups, seed=seed),
        "woe_scorecard": lambda: woe_scorecard(seed),
        "naive_bayes_raw": lambda: baselines.naive_bayes(groups, seed=seed),
        "naive_bayes_pca": lambda: naive_bayes_pca(groups, seed, n_components=5 if fast else 30),
        "decision_tree": lambda: baselines.decision_tree(groups, max_depth=8,
                                                         min_samples_leaf=100, seed=seed),
        "knn": lambda: knn(seed, n_neighbors=10 if fast else 200),
        "svm_linear": lambda: svm_linear(seed),
        "svm_rbf_nystroem": lambda: svm_rbf(seed, n_components=50 if fast else 300),
        "random_forest": lambda: random_forest(groups, seed, n_estimators=n or 150),
        "adaboost": lambda: adaboost(seed, n_estimators=n or 100),
        "hist_gradient_boosting": lambda: hist_gradient_boosting(seed, max_iter=n or 500),
        "xgboost": lambda: xgboost(seed, n_estimators=n or 500),
        "ebm": lambda: ebm(seed),
    }
