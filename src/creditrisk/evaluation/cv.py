"""One cross-validation runner for every model, so all comparisons are fair."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from creditrisk.config import get_path
from creditrisk.evaluation.bootstrap import bootstrap_ci
from creditrisk.evaluation.metrics import evaluate
from creditrisk.evaluation.tracking import log_run


@dataclass
class CVResult:
    name: str
    oof: np.ndarray  # out-of-fold default probability for every training row
    fold_metrics: pd.DataFrame
    summary: dict


def run_cv(make_model, X: pd.DataFrame, y, folds, name: str, ids=None, n_boot: int = 200,
           log: bool = True, step: int | None = None, oof_dir=None) -> CVResult:
    """Train a fresh model per fold, predict the held-out fold, and measure everything.

    make_model: function with no arguments that returns a new unfitted model.
    ids: SK_ID_CURR values; if given, the out-of-fold predictions are saved to
         data/processed/oof/<name>.parquet (used later for paired comparisons).
    """
    y = np.asarray(y).astype(int)
    folds = np.asarray(folds)
    oof = np.full(len(y), np.nan)
    rows = []
    start = time.perf_counter()
    for k in np.unique(folds):
        fit_rows, held_out = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        model = make_model()
        model.fit(X.iloc[fit_rows], y[fit_rows])
        oof[held_out] = model.predict_proba(X.iloc[held_out])[:, 1]
        rows.append({"fold": int(k), **evaluate(y[held_out], oof[held_out])})
    seconds = time.perf_counter() - start

    fold_metrics = pd.DataFrame(rows).set_index("fold")
    overall = evaluate(y, oof)
    ci = bootstrap_ci(y, oof, n_boot=n_boot)
    summary = {
        "roc_auc": overall["roc_auc"],
        "roc_auc_low": ci["low"],
        "roc_auc_high": ci["high"],
        "roc_auc_fold_mean": float(fold_metrics["roc_auc"].mean()),
        "roc_auc_fold_std": float(fold_metrics["roc_auc"].std()),
        "pr_auc": overall["pr_auc"],
        "ks": overall["ks"],
        "brier": overall["brier"],
        "ece": overall["ece"],
        "seconds": seconds,
    }
    if log:
        tags = {"step": str(step)} if step is not None else None
        log_run(name, summary, params={"n_rows": len(y), "n_features": X.shape[1]}, tags=tags)
    if ids is not None:
        out_dir = Path(oof_dir) if oof_dir else get_path("processed") / "oof"
        out_dir.mkdir(parents=True, exist_ok=True)
        oof_table = pd.DataFrame(
            {"SK_ID_CURR": np.asarray(ids), "fold": folds, "TARGET": y, "p": oof}
        )
        oof_table.to_parquet(out_dir / f"{name}.parquet", index=False)
    return CVResult(name, oof, fold_metrics, summary)


def results_table(results) -> pd.DataFrame:
    """One row per model, best ROC-AUC first."""
    rows = []
    for r in results:
        s = r.summary
        rows.append({
            "model": r.name,
            "ROC-AUC": round(s["roc_auc"], 4),
            "95% CI": f"{s['roc_auc_low']:.4f} to {s['roc_auc_high']:.4f}",
            "fold mean ± std": f"{s['roc_auc_fold_mean']:.4f} ± {s['roc_auc_fold_std']:.4f}",
            "PR-AUC": round(s["pr_auc"], 4),
            "KS": round(s["ks"], 4),
            "Brier": round(s["brier"], 4),
            "ECE": round(s["ece"], 4),
            "seconds": round(s["seconds"], 1),
        })
    return pd.DataFrame(rows).set_index("model").sort_values("ROC-AUC", ascending=False)
