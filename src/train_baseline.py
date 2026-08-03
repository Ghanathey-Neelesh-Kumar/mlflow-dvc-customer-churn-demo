"""Traditional training script before MLflow tracking is introduced."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from training_utils import build_pipeline, calculate_metrics, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn model without MLflow")
    parser.add_argument("--data-path", type=Path, default=Path("data/customer_churn.csv"))
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "random_forest"],
        default="logistic_regression",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=4)
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, features, target = load_dataset(args.data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    pipeline = build_pipeline(
        args.model_type,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        c_value=args.c,
        random_state=args.random_state,
        feature_frame=features,
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = calculate_metrics(y_test, predictions, probabilities)

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    joblib.dump(pipeline, output_dir / "baseline_model.joblib")
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("Training completed without MLflow")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("Model saved to outputs/baseline_model.joblib")
    print("Metrics saved to outputs/baseline_metrics.json")


if __name__ == "__main__":
    main()
