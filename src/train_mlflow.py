"""Train a customer-churn model and track the run in MLflow with DVC lineage."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from training_utils import (
    build_pipeline,
    calculate_metrics,
    classification_report_dict,
    confusion_matrix_figure,
    feature_importance_frame,
    load_dataset,
)


def optional_int(value: str) -> int | None:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track churn training with DVC and MLflow")
    parser.add_argument("--data-path", type=Path, default=Path("data/customer_churn.csv"))
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=optional_int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=4)
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def run_git_command(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def read_dvc_hash(data_path: Path) -> tuple[str, Path | None]:
    dvc_file = Path(f"{data_path}.dvc")
    if not dvc_file.exists():
        return "not-tracked-by-dvc", None

    metadata = yaml.safe_load(dvc_file.read_text(encoding="utf-8")) or {}
    outputs = metadata.get("outs", [])
    if not outputs:
        return "unknown", dvc_file

    first_output: dict[str, Any] = outputs[0]
    for key in ("md5", "etag", "checksum", "hash"):
        if first_output.get(key):
            return str(first_output[key]), dvc_file
    return "unknown", dvc_file


def configure_mlflow(args: argparse.Namespace) -> tuple[str, str]:
    tracking_uri = (
        args.tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or "http://127.0.0.1:5000"
    )
    experiment_name = (
        args.experiment_name
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
        or "customer-churn-dvc-mlflow"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri, experiment_name


def main() -> None:
    args = parse_args()
    tracking_uri, experiment_name = configure_mlflow(args)

    data, features, target = load_dataset(args.data_path)
    dvc_hash, dvc_file = read_dvc_hash(args.data_path)
    git_commit = run_git_command("rev-parse", "HEAD") or "uncommitted"
    git_status = run_git_command("status", "--porcelain")
    git_dirty = bool(git_status)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    model_parameters: dict[str, Any] = {
        "model_type": args.model_type,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "dataset_rows": len(data),
        "feature_count": features.shape[1],
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }
    if args.model_type == "random_forest":
        model_parameters.update(
            {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth if args.max_depth is not None else "None",
                "min_samples_split": args.min_samples_split,
            }
        )
    else:
        model_parameters["C"] = args.c

    run_name = args.run_name or f"{args.model_type}-dvc-{dvc_hash[:8]}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "project": "customer-churn",
                "course_section": "DVC + MLflow experiment tracking",
                "dataset.dvc_hash": dvc_hash,
                "dataset.path": str(args.data_path),
                "git.commit": git_commit,
                "git.dirty": str(git_dirty).lower(),
            }
        )
        mlflow.log_params(model_parameters)

        dataset = mlflow.data.from_pandas(
            data,
            source=str(args.data_path),
            name=f"customer-churn-{dvc_hash[:8]}",
            targets="churn",
        )
        mlflow.log_input(dataset, context="training")

        pipeline = build_pipeline(
            args.model_type,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            c_value=args.c,
            random_state=args.random_state,
            feature_frame=features,
        )

        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        training_seconds = time.perf_counter() - started

        predictions = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics = calculate_metrics(y_test, predictions, probabilities)
        metrics["training_seconds"] = float(training_seconds)
        mlflow.log_metrics(metrics)

        report = classification_report_dict(y_test, predictions)
        mlflow.log_dict(report, "reports/classification_report.json")

        confusion_figure = confusion_matrix_figure(y_test, predictions)
        mlflow.log_figure(confusion_figure, "plots/confusion_matrix.png")
        plt.close(confusion_figure)

        importance = feature_importance_frame(pipeline)
        if not importance.empty:
            mlflow.log_text(
                importance.head(25).to_csv(index=False),
                "reports/top_feature_importance.csv",
            )

        if dvc_file is not None:
            mlflow.log_artifact(str(dvc_file), artifact_path="dataset_version")
        if Path("requirements.txt").exists():
            mlflow.log_artifact("requirements.txt", artifact_path="environment")
        if Path("params.yaml").exists():
            mlflow.log_artifact("params.yaml", artifact_path="configuration")

        signature = infer_signature(x_test, predictions)
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=x_test.head(5),
        )

        print("\nMLflow run completed successfully")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Experiment: {experiment_name}")
        print(f"Run name: {run_name}")
        print(f"Run ID: {run.info.run_id}")
        print(f"DVC dataset hash: {dvc_hash}")
        print(f"Logged model URI: {model_info.model_uri}")
        print("Metrics:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
