"""Step 13: two of the three mitigations (the third, Fairlearn's ExponentiatedGradient, is used
directly in notebook 11).

- Before training, reweighing (Kamiran & Calders 2012): weight each (group, label) cell by
  P(group)·P(label) / P(group, label), so that in the weighted data group and label look
  independent.
- After training, one threshold per group (Hardt et al. 2016), chosen for the most profit while
  the gap in good customers declined stays within max_gap. Fairlearn's ThresholdOptimizer does
  the same but only maximises accuracy, which at 8% defaulters means approving almost everyone.

    from creditrisk.fairness.mitigate import group_thresholds, reweighing_weights
    model.fit(X, y, sample_weight=reweighing_weights(y, gender))
    thresholds = group_thresholds(y, p, gender, amount, margin=0.10, lgd=0.50, max_gap=0.01)
"""

import itertools

import numpy as np
import pandas as pd

GRID = np.round(np.arange(0.05, 0.4001, 0.0025), 4)


def reweighing_weights(y, groups) -> np.ndarray:
    """P(group)·P(label) / P(group, label) for each row."""
    df = pd.DataFrame({"g": np.asarray(groups), "y": np.asarray(y)})
    p_g = df["g"].map(df["g"].value_counts(normalize=True))
    p_y = df["y"].map(df["y"].value_counts(normalize=True))
    p_gy = df.groupby(["g", "y"])["y"].transform("size") / len(df)
    return (p_g * p_y / p_gy).to_numpy()


def group_thresholds(y, p, groups, amount, margin: float, lgd: float, max_gap: float,
                     grid=GRID) -> dict:
    """{group: threshold} with the highest profit whose equal-opportunity gap is <= max_gap.

    Profit adds up over groups, so each group's profit and good-customers-declined rate is
    computed once per grid value and only the combinations are searched.
    """
    y, p, amount = np.asarray(y).astype(bool), np.asarray(p, float), np.asarray(amount, float)
    groups = np.asarray(groups)
    gain = np.where(y, -lgd * amount, margin * amount)
    names = sorted(set(groups))
    profit, declined = {}, {}
    for g in names:
        m = groups == g
        approved = p[m][None, :] < grid[:, None]                 # grid x applicants
        profit[g] = approved @ gain[m]
        declined[g] = 1 - approved[:, ~y[m]].mean(axis=1)
    best, best_profit = None, -np.inf
    # shortcut: tries every combination (grid size ** groups), fine for gender; slow past 3 groups
    for idx in itertools.product(range(len(grid)), repeat=len(names)):
        rates = [declined[g][i] for g, i in zip(names, idx, strict=True)]
        total = sum(profit[g][i] for g, i in zip(names, idx, strict=True))
        if max(rates) - min(rates) <= max_gap and total > best_profit:
            best, best_profit = idx, total
    if best is None:
        raise ValueError("no thresholds on the grid meet max_gap")
    return {g: float(grid[i]) for g, i in zip(names, best, strict=True)}
