"""Optional helper to query MLflow and display the best run by ROC-AUC."""

from __future__ import annotations

import os

import mlflow


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "customer-churn-dvc-mlflow"
    )
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"Experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=10,
    )
    columns = [
        "tags.mlflow.runName",
        "params.model_type",
        "params.max_depth",
        "metrics.accuracy",
        "metrics.f1_score",
        "metrics.roc_auc",
        "tags.dataset.dvc_hash",
    ]
    available = [column for column in columns if column in runs.columns]
    print(runs[available].to_string(index=False))


if __name__ == "__main__":
    main()
