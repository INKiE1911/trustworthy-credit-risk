"""Confidence intervals for any metric, by bootstrap resampling.

    bootstrap_ci(y, p)                  # 95% CI for ROC-AUC
    paired_bootstrap(y, p_model_a, p_model_b)   # is model A really better than B?
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def _resamples(n: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield rng.integers(0, n, n)


def bootstrap_ci(y_true, y_prob, metric=roc_auc_score, n_boot: int = 1000,
                 ci: float = 0.95, seed: int = 42) -> dict:
    """Point estimate plus a percentile confidence interval for metric(y_true, y_prob)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    stats = []
    for idx in _resamples(len(y_true), n_boot, seed):
        yt = y_true[idx]
        if yt.min() == yt.max():  # a resample with only one class has no AUC
            continue
        stats.append(metric(yt, y_prob[idx]))
    low, high = np.quantile(stats, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "estimate": float(metric(y_true, y_prob)),
        "low": float(low),
        "high": float(high),
        "n_boot": len(stats),
    }


def paired_bootstrap(y_true, prob_a, prob_b, metric=roc_auc_score, n_boot: int = 1000,
                     ci: float = 0.95, seed: int = 42) -> dict:
    """CI for metric(A) - metric(B), using the SAME resamples for both models.

    `significant` is True when the interval does not contain 0.
    `p_value` is an approximate two-sided bootstrap p-value.
    """
    y_true, prob_a, prob_b = np.asarray(y_true), np.asarray(prob_a), np.asarray(prob_b)
    diffs = []
    for idx in _resamples(len(y_true), n_boot, seed):
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        diffs.append(metric(yt, prob_a[idx]) - metric(yt, prob_b[idx]))
    diffs = np.asarray(diffs)
    low, high = np.quantile(diffs, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    p_value = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return {
        "difference": float(metric(y_true, prob_a) - metric(y_true, prob_b)),
        "low": float(low),
        "high": float(high),
        "p_value": float(p_value),
        "significant": bool(low > 0 or high < 0),
    }
