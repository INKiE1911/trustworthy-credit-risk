"""Step 7 experiment: grid search vs random search vs Bayesian optimisation (Optuna TPE).

Every method gets the same budget (the same number of LightGBM fits) and the same check:
train on folds 2-5, stop adding trees when the score on fold 1 stops improving, and score
fold 1. Each finished trial is saved immediately, so a long search can be stopped and resumed.

Note: fold 1 is used to choose the settings, so the tuned model's later 5-fold CV score is
slightly optimistic. The honest final number comes from the locked test set (Step 15).
"""

import inspect
import itertools
import json
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from creditrisk.models import baselines

# name: (low, high, log scale?, whole number?)
SPACE = {
    "learning_rate": (0.02, 0.1, True, False),  # below 0.02 needs thousands of trees: too slow
    "num_leaves": (16, 128, True, True),
    "min_child_samples": (20, 500, True, True),
    "subsample": (0.5, 1.0, False, False),
    "colsample_bytree": (0.2, 0.8, False, False),
    "reg_lambda": (0.001, 30.0, True, False),
}

# 3 x 3 x 3 = 27 points; the other settings stay at the LightGBM v1 values.
GRID = {
    "learning_rate": [0.02, 0.05, 0.1],
    "num_leaves": [16, 48, 128],
    "min_child_samples": [30, 150, 500],
}


def grid_candidates() -> list[dict]:
    names = list(GRID)
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*GRID.values())]


def random_candidates(n: int, seed: int = 42) -> list[dict]:
    """n settings drawn at random from SPACE (log-uniform where the scale is log)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        params = {}
        for name, (low, high, log, whole) in SPACE.items():
            if log:
                value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                value = float(rng.uniform(low, high))
            params[name] = int(round(value)) if whole else value
        out.append(params)
    return out


def holdout_score(params: dict, X: pd.DataFrame, y, folds, holdout: int = 1, seed: int = 42,
                  max_trees: int = 3000, patience: int = 100) -> tuple[float, int]:
    """Train on every fold except `holdout`, early-stop on it; return (ROC-AUC, trees used)."""
    y, folds = np.asarray(y), np.asarray(folds)
    fit_rows, check_rows = np.flatnonzero(folds != holdout), np.flatnonzero(folds == holdout)
    model = baselines.lightgbm(seed=seed, n_estimators=max_trees, metric="auc", **params)
    X_check, y_check = X.iloc[check_rows], y[check_rows]
    stop = [lightgbm.early_stopping(patience, verbose=False)]
    if "eval_X" in inspect.signature(model.fit).parameters:  # LightGBM 4.7 and later
        model.fit(X.iloc[fit_rows], y[fit_rows], eval_X=(X_check,), eval_y=(y_check,),
                  callbacks=stop)
    else:
        model.fit(X.iloc[fit_rows], y[fit_rows], eval_set=[(X_check, y_check)], callbacks=stop)
    prob = model.predict_proba(X_check)[:, 1]  # uses the best number of trees
    return float(roc_auc_score(y_check, prob)), int(model.best_iteration_)


def _tidy(rows: list[dict], method: str) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    table.insert(0, "method", method)
    table["trial"] = range(len(table))
    table["params"] = [p if isinstance(p, dict) else json.loads(p) for p in table["params"]]
    table["best_so_far"] = table["auc"].cummax()
    return table


def run_search(method: str, candidates: list[dict], score_fn, cache_dir,
               verbose: bool = True) -> pd.DataFrame:
    """Score each candidate once with score_fn(params) -> (auc, trees).

    Finished trials are saved to <cache_dir>/<method>.csv and skipped on the next run.
    """
    path = Path(cache_dir) / f"{method}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    for row in rows:  # the saved trials must belong to this exact list of candidates
        if json.loads(row["params"]) != candidates[int(row["trial"])]:
            raise ValueError(f"{path} was made with different candidates; delete it to restart.")
    done = {int(row["trial"]) for row in rows}
    best = max((r["auc"] for r in rows), default=float("-inf"))
    for trial, params in enumerate(candidates):
        if trial in done:
            continue
        start = time.perf_counter()
        auc, trees = score_fn(params)
        rows.append({"trial": trial, "params": json.dumps(params), "auc": auc, "trees": trees,
                     "seconds": round(time.perf_counter() - start, 1)})
        pd.DataFrame(rows).to_csv(path, index=False)
        best = max(best, auc)
        if verbose:
            print(f"  {method} {trial + 1}/{len(candidates)}: ROC-AUC {auc:.4f} with {trees} trees "
                  f"({rows[-1]['seconds']:.0f}s), best so far {best:.4f}", flush=True)
    rows.sort(key=lambda r: int(r["trial"]))
    return _tidy(rows, method)


def run_optuna(n_trials: int, score_fn, cache_dir, seed: int = 42, startup: int = 8,
               verbose: bool = True) -> pd.DataFrame:
    """Bayesian optimisation with Optuna's TPE sampler, stored in <cache_dir>/optuna.db.

    The first `startup` trials are random; after that TPE proposes settings that look
    promising given all earlier results. Re-running continues until n_trials are finished.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    path = Path(cache_dir).resolve() / "optuna.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="lightgbm_tpe", storage=f"sqlite:///{path}", load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup),
    )

    def finished():
        return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    def objective(trial):
        params = {}
        for name, (low, high, log, whole) in SPACE.items():
            suggest = trial.suggest_int if whole else trial.suggest_float
            params[name] = suggest(name, low, high, log=log)
        start = time.perf_counter()
        auc, trees = score_fn(params)
        seconds = round(time.perf_counter() - start, 1)
        trial.set_user_attr("trees", trees)
        trial.set_user_attr("seconds", seconds)
        if verbose:
            best = max([auc] + [t.value for t in finished()])
            print(f"  tpe {len(finished()) + 1}/{n_trials}: ROC-AUC {auc:.4f} with {trees} trees "
                  f"({seconds:.0f}s), best so far {best:.4f}", flush=True)
        return auc

    remaining = n_trials - len(finished())
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    rows = [{"params": dict(t.params), "auc": t.value, "trees": t.user_attrs["trees"],
             "seconds": t.user_attrs["seconds"]} for t in finished()[:n_trials]]
    return _tidy(rows, "tpe")


def best_trial(*tables: pd.DataFrame) -> pd.Series:
    """The single best trial across all search methods."""
    every = pd.concat(tables, ignore_index=True)
    return every.loc[every["auc"].idxmax()]


def tuned_lightgbm(params: dict, trees: int, seed: int = 42):
    """LightGBM v1 settings, with the tuned values and a fixed number of trees."""
    return baselines.lightgbm(seed=seed, n_estimators=int(trees), **params)
