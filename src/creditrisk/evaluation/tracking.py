"""Log every model run to MLflow in one standard way."""

import math

from creditrisk.config import setup_mlflow


def log_run(
    run_name: str,
    metrics: dict,
    params: dict | None = None,
    tags: dict | None = None,
    tracking_uri: str | None = None,
    experiment: str | None = None,
) -> str:
    """Save one run (metrics, params, tags) to MLflow and return its run id.

    By default it logs to <project root>/mlflow.db. Non-numeric and NaN metrics are skipped.
    `tracking_uri` / `experiment` are only for tests.
    """
    import mlflow  # slow to import, so only when logging

    if tracking_uri is None:
        setup_mlflow()
    else:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment or "tests")

    clean = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(value):
            clean[key] = float(value)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metrics(clean)
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
    return run.info.run_id
