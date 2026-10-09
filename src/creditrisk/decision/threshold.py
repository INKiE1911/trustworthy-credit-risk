"""Step 10: turn a default probability into a money decision (Bishop §1.5, expected loss).

Approving a loan of size A earns m·A if it is repaid (m = profit margin) and loses LGD·A if it
defaults (LGD = share of the loan lost). The expected profit of approving is
    (1 - p)·m·A - p·LGD·A,
which is positive exactly when p < m / (m + LGD): the break-even threshold. It only makes sense
if p is a real probability, which is why the model is calibrated first.

    from creditrisk.decision.threshold import break_even_threshold, realised_profit
    t = break_even_threshold(margin=0.10, lgd=0.50)        # 0.1667
    realised_profit(y, amount, approved=p < t, margin=0.10, lgd=0.50)
"""

import numpy as np


def break_even_threshold(margin: float, lgd: float) -> float:
    """Approve when p is below this: m / (m + LGD)."""
    if margin <= 0 or lgd <= 0:
        raise ValueError("margin and lgd must be positive")
    return margin / (margin + lgd)


def realised_profit(y_true, amount, approved, margin: float, lgd: float) -> float:
    """Money actually made on the approved loans: +m·A if repaid, -LGD·A if defaulted."""
    y_true = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount, dtype=float)
    approved = np.asarray(approved, dtype=bool)
    gain = np.where(y_true, -lgd * amount, margin * amount)
    return float(gain[approved].sum())
