"""Step 11: approve / refer / decline with a guarantee (class-conditional "Mondrian" conformal).

The score of an applicant for a label is 1 - (probability the model gave that label):
p for "repay", 1 - p for "default". On a held-out calibration set, each class gets its own
cut-off q_c, so that about 1 - α of the applicants of that class have a score at or below it.
A new applicant's prediction set holds every label whose score is <= that label's cut-off:
    only "repay" -> approve,  only "default" -> decline,  both or neither -> refer to a human.
If new applicants look like the calibration set, at most about α of real defaulters are
auto-approved and at most about α of good customers are auto-declined (Vovk et al.;
Angelopoulos & Bates 2021). One cut-off for everyone would only promise this on average, and
with 8% defaulters it can miss most of them.

    from creditrisk.decision.conformal import class_cutoffs, decide
    q_repay, q_default = class_cutoffs(p_calib, y_calib, alpha=0.10)
    decide(p_new, q_repay, q_default)          # array of "approve" / "refer" / "decline"
"""

import numpy as np


def conformal_quantile(scores, alpha: float) -> float:
    """The ceil((n + 1)(1 - α))-th smallest score; 1.0 (keep every label) if that is past n."""
    scores = np.sort(np.asarray(scores, dtype=float))
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(scores[k - 1]) if k <= len(scores) else 1.0


def class_cutoffs(p, y, alpha: float) -> tuple[float, float]:
    """(q_repay, q_default): one cut-off per class, from the applicants with that true label."""
    p, y = np.asarray(p, dtype=float), np.asarray(y).astype(int)
    return conformal_quantile(p[y == 0], alpha), conformal_quantile(1 - p[y == 1], alpha)


def decide(p, q_repay: float, q_default: float) -> np.ndarray:
    """Decision per applicant from the prediction set {labels whose score <= their cut-off}."""
    p = np.asarray(p, dtype=float)
    repay_in, default_in = p <= q_repay, 1 - p <= q_default
    return np.where(repay_in & ~default_in, "approve",
                    np.where(default_in & ~repay_in, "decline", "refer"))


def coverage(p, y, q_repay: float, q_default: float) -> dict:
    """Share of each class whose prediction set contains its true label (target: >= 1 - α)."""
    p, y = np.asarray(p, dtype=float), np.asarray(y).astype(int)
    return {"repay": float(np.mean(p[y == 0] <= q_repay)),
            "default": float(np.mean(1 - p[y == 1] <= q_default))}
