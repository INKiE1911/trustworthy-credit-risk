"""One place to measure any model: ranking, probability quality and decision metrics.

    from creditrisk.evaluation.metrics import evaluate
    evaluate(y_true, y_prob)   # dict with roc_auc, gini, pr_auc, ks, brier, log_loss, ece, ...
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, y_prob) -> float:
    """KS score used in credit scoring: the largest gap between TPR and FPR."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def expected_calibration_error(
    y_true, y_prob, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """Average gap between predicted probability and real default rate, over bins.

    strategy="uniform": equal-width bins on [0, 1] (the usual definition).
    strategy="quantile": bins with the same number of applicants (better for skewed scores).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    bins = np.clip(np.searchsorted(edges, y_prob, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(y_true[in_bin].mean() - y_prob[in_bin].mean())
    return float(ece)


def evaluate(y_true, y_prob) -> dict:
    """All threshold-free metrics for predicted default probabilities."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    auc = roc_auc_score(y_true, y_prob)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "ece": expected_calibration_error(y_true, y_prob),
        "n": int(len(y_true)),
        "default_rate": float(y_true.mean()),
    }


def evaluate_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """Decision metrics when everyone with probability >= threshold is flagged (declined)."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(y_prob, dtype=float) >= threshold
    tp = int(np.sum(flagged & (y_true == 1)))
    fp = int(np.sum(flagged & (y_true == 0)))
    fn = int(np.sum(~flagged & (y_true == 1)))
    tn = int(np.sum(~flagged & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(y_true),
        "approval_rate": float(1 - flagged.mean()),
    }
