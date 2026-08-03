"""Generate a deterministic synthetic customer-churn dataset for the course demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate customer churn training data")
    parser.add_argument("--rows", type=int, default=2000, help="Number of customer records")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/customer_churn.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_dataset(rows: int, seed: int) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("Use at least 100 rows so the train/test split is meaningful.")

    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 73, size=rows)
    monthly_charges = np.round(rng.normal(72, 24, size=rows).clip(18, 125), 2)
    support_calls = rng.poisson(1.8, size=rows).clip(0, 10)
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=rows,
        p=[0.56, 0.25, 0.19],
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "None"],
        size=rows,
        p=[0.35, 0.52, 0.13],
    )
    payment_method = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
        size=rows,
        p=[0.34, 0.25, 0.24, 0.17],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=rows, p=[0.64, 0.36])
    senior_citizen = rng.choice([0, 1], size=rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=rows, p=[0.49, 0.51])
    dependents = rng.choice(["Yes", "No"], size=rows, p=[0.31, 0.69])

    total_charges = monthly_charges * tenure_months + rng.normal(0, 180, size=rows)
    total_charges = np.round(total_charges.clip(0), 2)

    contract_effect = pd.Series(contract_type).map(
        {"Month-to-month": 1.0, "One year": -0.35, "Two year": -0.9}
    ).to_numpy()
    internet_effect = pd.Series(internet_service).map(
        {"Fiber optic": 0.45, "DSL": 0.05, "None": -0.55}
    ).to_numpy()
    payment_effect = pd.Series(payment_method).map(
        {
            "Electronic check": 0.42,
            "Credit card": -0.12,
            "Bank transfer": -0.18,
            "Mailed check": 0.05,
        }
    ).to_numpy()

    logits = (
        -0.85
        - 0.028 * tenure_months
        + 0.017 * (monthly_charges - 70)
        + 0.20 * support_calls
        + contract_effect
        + internet_effect
        + payment_effect
        + 0.32 * (paperless_billing == "Yes")
        + 0.28 * senior_citizen
        - 0.22 * (partner == "Yes")
        - 0.18 * (dependents == "Yes")
        + rng.normal(0, 0.65, size=rows)
    )
    churn_probability = sigmoid(logits)
    churn = rng.binomial(1, churn_probability)

    data = pd.DataFrame(
        {
            "customer_id": [f"CUST-{seed}-{index:05d}" for index in range(1, rows + 1)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "support_calls": support_calls,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "churn": churn,
        }
    )

    # Add a small amount of missing data so the preprocessing pipeline is realistic.
    numeric_missing = rng.choice(rows, size=max(1, rows // 100), replace=False)
    category_missing = rng.choice(rows, size=max(1, rows // 125), replace=False)
    data.loc[numeric_missing, "total_charges"] = np.nan
    data.loc[category_missing, "internet_service"] = np.nan

    return data


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    data = generate_dataset(args.rows, args.seed)
    data.to_csv(args.output, index=False)

    churn_rate = data["churn"].mean()
    print(f"Dataset written to: {args.output}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Churn rate: {churn_rate:.2%}")


if __name__ == "__main__":
    main()
