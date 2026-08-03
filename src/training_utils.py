"""Shared training utilities used by the baseline and MLflow scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate it or run 'dvc pull' first."
        )

    data = pd.read_csv(path)
    required_columns = {TARGET_COLUMN, ID_COLUMN}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    features = data.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    target = data[TARGET_COLUMN].astype(int)
    return data, features, target


def build_pipeline(
    model_type: str,
    *,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    c_value: float,
    random_state: int,
    feature_frame: pd.DataFrame,
) -> Pipeline:
    numeric_columns = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = feature_frame.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    if model_type == "logistic_regression":
        estimator: Any = LogisticRegression(
            C=c_value,
            max_iter=1500,
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def classification_report_dict(
    y_true: pd.Series, predictions: np.ndarray
) -> dict[str, Any]:
    return classification_report(
        y_true,
        predictions,
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_figure(y_true: pd.Series, predictions: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Stayed", "Churned"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Customer churn confusion matrix")
    fig.tight_layout()
    return fig


def feature_importance_frame(model_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        column_name = "importance"
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
        column_name = "absolute_coefficient"
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame({"feature": feature_names, column_name: values})
    return frame.sort_values(column_name, ascending=False).reset_index(drop=True)
