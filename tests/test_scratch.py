"""Step 8: every from-scratch algorithm is checked against scikit-learn (or exact maths)."""

import warnings

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from creditrisk.scratch.gmm import GaussianMixtureScratch
from creditrisk.scratch.kmeans import KMeansScratch
from creditrisk.scratch.lda import FisherLDA
from creditrisk.scratch.logreg import BayesianLogisticRegressionScratch, LogisticRegressionScratch
from creditrisk.scratch.mlp import MLPScratch
from creditrisk.scratch.naive_bayes import GaussianNBScratch
from creditrisk.scratch.pca import PCAScratch


@pytest.fixture(scope="module")
def binary():
    X, y = make_classification(n_samples=1500, n_features=8, n_informative=5, n_redundant=0,
                               weights=[0.85], random_state=0)  # full rank: no exact copies
    return (X - X.mean(0)) / X.std(0), y


def test_logistic_matches_sklearn(binary):
    X, y = binary
    ours = LogisticRegressionScratch(C=0.5).fit(X, y)
    ref = LogisticRegression(C=0.5, tol=1e-12, max_iter=10_000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_[0], atol=1e-5)
    assert np.isclose(ours.intercept_, ref.intercept_[0], atol=1e-5)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-6)
    assert ours.n_iter_ <= 15  # Newton converges in a few steps
    assert np.all(np.diff(ours.objective_history_) <= 1e-9)  # never goes uphill


def test_l2_is_map_with_gaussian_prior(binary):
    X, y = binary
    strong = LogisticRegressionScratch(C=0.01).fit(X, y)
    weak = LogisticRegressionScratch(C=100).fit(X, y)
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(weak.coef_)  # a tighter prior shrinks w


def test_laplace_bayesian_logistic(binary):
    X, y = binary
    model = BayesianLogisticRegressionScratch(C=1.0).fit(X[:300], y[:300])
    rows = X[300:340]
    exact = model.sample_proba(rows, n_samples=20_000)
    assert np.max(np.abs(model.predict_proba(rows)[:, 1] - exact)) < 0.01  # probit trick works
    far = rows * 8  # applicants unlike anything in the training data
    assert np.all(model.logit_std(far) > model.logit_std(rows))
    p_map, p_bayes = model.predict_proba_map(far)[:, 1], model.predict_proba(far)[:, 1]
    assert np.all(np.abs(p_bayes - 0.5) <= np.abs(p_map - 0.5) + 1e-12)  # pulled towards 0.5


def test_gaussian_nb_matches_sklearn(binary):
    X, y = binary
    X = X.copy()
    X[y == 1, 3] = 1.0  # constant inside one class: the smoothing must handle it
    ours, ref = GaussianNBScratch().fit(X, y), GaussianNB().fit(X, y)
    assert np.allclose(ours.var_, ref.var_)
    assert np.allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-9)


def test_pca_matches_sklearn():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 6)) @ rng.normal(size=(6, 6)) + 3
    ours, ref = PCAScratch(4).fit(X), PCA(4, svd_solver="full").fit(X)
    assert np.allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_)
    signs = np.sign(np.sum(ours.components_ * ref.components_, axis=1))
    assert np.allclose(ours.components_, ref.components_ * signs[:, None], atol=1e-8)
    assert np.allclose(np.abs(ours.transform(X)), np.abs(ref.transform(X)), atol=1e-8)
    full = PCAScratch().fit(X)
    assert np.allclose(full.inverse_transform(full.transform(X)), X)  # nothing lost with all


def test_fisher_direction_two_classes(binary):
    X, y = binary
    lda = FisherLDA().fit(X, y)
    m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
    centred = X - np.where(y[:, None] == 1, m1, m0)
    fisher = np.linalg.solve(centred.T @ centred, m1 - m0)  # w ∝ S_W⁻¹ (m1 - m0)
    w = lda.scalings_[:, 0]
    assert abs(w @ fisher) / (np.linalg.norm(w) * np.linalg.norm(fisher)) > 1 - 1e-9


def test_lda_matches_sklearn_three_classes():
    X, y = make_classification(n_samples=900, n_features=6, n_informative=4, n_redundant=0,
                               n_classes=3, n_clusters_per_class=1, random_state=3)
    ours = FisherLDA().fit(X, y)
    probs = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y).predict_proba(X)
    assert np.allclose(ours.predict_proba(X), probs, atol=1e-8)
    projected = LinearDiscriminantAnalysis(solver="eigen").fit(X, y).transform(X)
    for j in range(2):  # same directions, up to scale and sign
        assert abs(np.corrcoef(ours.transform(X)[:, j], projected[:, j])[0, 1]) > 1 - 1e-8


def test_kmeans_same_start_same_answer_as_sklearn():
    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=1.5, random_state=2)
    start = X[[0, 1, 2, 3]]
    ours = KMeansScratch(4).fit(X, init=start)
    ref = KMeans(4, init=start, n_init=1, algorithm="lloyd").fit(X)
    assert np.allclose(np.sort(ours.cluster_centers_, axis=0),
                       np.sort(ref.cluster_centers_, axis=0))
    assert adjusted_rand_score(ours.labels_, ref.labels_) == 1.0
    assert np.isclose(ours.inertia_, ref.inertia_)


def test_kmeans_plus_plus_quality_and_repeatable():
    X, _ = make_blobs(n_samples=800, centers=6, cluster_std=1.0, random_state=5)
    ours = KMeansScratch(6, n_init=4, random_state=0).fit(X)
    ref = KMeans(6, n_init=4, random_state=0).fit(X)
    assert ours.inertia_ <= ref.inertia_ * 1.01
    again = KMeansScratch(6, n_init=4, random_state=0).fit(X)
    assert np.array_equal(again.labels_, ours.labels_)


def test_gmm_em_never_goes_down_and_matches_sklearn():
    X, _ = make_blobs(n_samples=700, centers=3, cluster_std=[1.0, 2.0, 0.6], random_state=4)
    start = KMeansScratch(3, random_state=0).fit(X)
    resp = np.eye(3)[start.labels_]
    weights = resp.mean(0)
    means = resp.T @ X / resp.sum(0)[:, None]
    covs = np.array([np.cov(X[start.labels_ == k].T, bias=True) + 1e-6 * np.eye(2)
                     for k in range(3)])
    ours = GaussianMixtureScratch(3, tol=1e-10, max_iter=500).fit(
        X, weights_init=weights, means_init=means, covariances_init=covs)
    assert np.all(np.diff(ours.log_likelihood_history_) >= -1e-9)  # EM is monotone
    ref = GaussianMixture(3, tol=1e-10, max_iter=500, weights_init=weights, means_init=means,
                          precisions_init=np.linalg.inv(covs)).fit(X)
    order = np.argsort(ours.means_[:, 0])
    assert np.allclose(ours.means_[order], ref.means_[np.argsort(ref.means_[:, 0])], atol=1e-5)
    assert np.isclose(ours.score(X), ref.score(X), atol=1e-7)
    default = GaussianMixtureScratch(3, random_state=0).fit(X)  # its own K-means start
    assert default.converged_ and default.score(X) >= ref.score(X) - 0.05


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_mlp_gradient_check(activation):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(30, 5)), (rng.random(30) < 0.4).astype(float)
    error = MLPScratch(hidden=4, activation=activation, alpha=0.1).gradient_check(X, y)
    assert error < 1e-6


def test_mlp_learns_xor_where_logistic_cannot():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(2000, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    mlp = MLPScratch(hidden=16, learning_rate=0.01, max_epochs=200, random_state=0).fit(X, y)
    linear = LogisticRegressionScratch().fit(X, y)
    assert roc_auc_score(y, mlp.predict_proba(X)[:, 1]) > 0.97
    assert roc_auc_score(y, linear.predict_proba(X)[:, 1]) < 0.6


def test_mlp_close_to_sklearn(binary):
    X, y = binary
    train, test = slice(0, 1000), slice(1000, None)
    # same size, optimiser, batch and number of epochs; no early stopping on either side
    ours = MLPScratch(hidden=16, max_epochs=150, valid_fraction=0.0, random_state=0)
    ours.fit(X[train], y[train])
    ref = MLPClassifier((16,), activation="tanh", alpha=1e-4, batch_size=256, max_iter=150,
                        tol=0.0, n_iter_no_change=1000, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # it runs all 150 epochs on purpose
        ref.fit(X[train], y[train])
    auc_ours = roc_auc_score(y[test], ours.predict_proba(X[test])[:, 1])
    auc_ref = roc_auc_score(y[test], ref.predict_proba(X[test])[:, 1])
    assert abs(auc_ours - auc_ref) < 0.03
