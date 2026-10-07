"""Tests for creditrisk.evaluation.tracking (writes to a temporary MLflow database)."""

import mlflow

from creditrisk.evaluation.tracking import log_run


def test_log_run_saves_metrics_params_and_tags(tmp_path):
    run_id = log_run(
        "unit-test",
        metrics={"roc_auc": 0.75, "broken": float("nan"), "note": "text"},
        params={"model": "dummy"},
        tags={"step": "3"},
        tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        experiment="unit-tests",
    )
    run = mlflow.get_run(run_id)
    assert run.data.metrics == {"roc_auc": 0.75}  # NaN and text are skipped
    assert run.data.params["model"] == "dummy"
    assert run.data.tags["step"] == "3"
