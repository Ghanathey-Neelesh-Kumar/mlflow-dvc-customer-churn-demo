$ErrorActionPreference = "Stop"

if (-not $env:MLFLOW_TRACKING_URI) {
    $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
}
if (-not $env:MLFLOW_EXPERIMENT_NAME) {
    $env:MLFLOW_EXPERIMENT_NAME = "customer-churn-dvc-mlflow"
}

python src/train_mlflow.py --model-type logistic_regression --c 1.0 --run-name 01-logistic-baseline
python src/train_mlflow.py --model-type random_forest --n-estimators 150 --max-depth 6 --min-samples-split 4 --run-name 02-random-forest-depth-6
python src/train_mlflow.py --model-type random_forest --n-estimators 300 --max-depth 12 --min-samples-split 2 --run-name 03-random-forest-depth-12
