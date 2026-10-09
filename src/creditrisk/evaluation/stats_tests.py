"""Step 15: significance tests for comparing models.

- McNemar (exact): do two models make *different* errors at a threshold? Only the applicants
  where exactly one model is right count; under "no difference" each such case is a coin flip.
- Holm: adjusts a family of p-values for running several tests at once (step-down Bonferroni).
- 5x2cv paired t-test (Dietterich 1998): 5 repetitions of 2-fold CV, a t statistic with 5
  degrees of freedom. Retrains the models, so it runs on the train split, never on test.

    from creditrisk.evaluation.stats_tests import five_by_two_cv, holm, mcnemar
    mcnemar(correct_a, correct_b)["p_value"]
    holm([0.01, 0.04, 0.03])                    # [0.03, 0.06, 0.06]
    five_by_two_cv(make_a, make_b, X, y)        # {"t": ..., "p_value": ..., "differences": ...}
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def mcnemar(correct_a, correct_b) -> dict:
    """Exact McNemar test on two boolean arrays (True = that model decided this row right)."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    p = stats.binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {"only_a_right": only_a, "only_b_right": only_b, "p_value": float(p)}


def holm(p_values) -> np.ndarray:
    """Holm-adjusted p-values, in the original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def five_by_two_cv(make_a, make_b, X, y, metric=roc_auc_score, seed: int = 42) -> dict:
    """Dietterich's 5x2cv paired t-test on metric(A) - metric(B). make_* build unfitted models."""
    y = np.asarray(y)
    diffs = np.zeros((5, 2))
    for i in range(5):
        halves = StratifiedKFold(2, shuffle=True, random_state=seed + i).split(X, y)
        for j, (fit, held) in enumerate(halves):
            scores = [metric(y[held], make().fit(X.iloc[fit], y[fit])
                             .predict_proba(X.iloc[held])[:, 1]) for make in (make_a, make_b)]
            diffs[i, j] = scores[0] - scores[1]
    variance = ((diffs - diffs.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    t = diffs[0, 0] / np.sqrt(variance.mean())
    return {"t": float(t), "p_value": float(2 * stats.t.sf(abs(t), df=5)),
            "mean_difference": float(diffs.mean()), "differences": diffs}
