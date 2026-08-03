# Customer Churn Experiment Tracking with DVC and MLflow

This repository supports the Udemy practical lecture **Tracking Machine Learning Experiments with DVC and MLflow**.

The project begins with a traditional customer-churn training script and then adds:

- **Git** for source-code versioning
- **DVC** for dataset versioning and S3-backed data storage
- **MLflow** for parameters, metrics, datasets, tags, artifacts, and models
- **A central MLflow Tracking Server** running on Kubernetes

> In this lecture, “real-time tracking” means the training script sends experiment information to MLflow while the script runs. It does not mean production prediction monitoring.

## Architecture

```text
Python training script
        │
        │ MLflow Tracking API
        ▼
MLflow Tracking Server on Kubernetes
        ├── PostgreSQL: experiments, runs, parameters, metrics, tags
        └── S3 artifact store: models, plots, reports, environment files

Git repository
        ├── Python code and configuration
        └── DVC pointer file ───────────────► DVC remote on S3
```

## Repository structure

```text
mlflow-dvc-customer-churn-demo/
├── starter-data/
│   └── customer_churn.csv
├── data/
│   └── .gitkeep
├── src/
│   ├── generate_data.py
│   ├── show_best_run.py
│   ├── train_baseline.py
│   ├── train_mlflow.py
│   └── training_utils.py
├── scripts/
│   ├── run_experiments.ps1
│   └── run_experiments.sh
├── aws/
│   └── dvc-s3-policy-template.json
├── docs/
│   └── troubleshooting.md
├── .env.example
├── .gitignore
├── params.yaml
├── requirements.txt
└── SECURITY.md
```

The starter dataset is kept separately so students can create `data/customer_churn.csv` and then let DVC track that file without it already being tracked by Git.

## Prerequisites

- Python 3.12 or 3.13
- Git
- DVC with S3 support
- AWS CLI configured with a least-privilege identity
- An S3 bucket or prefix for the DVC remote
- A working MLflow Tracking Server
- `kubectl` access to the Kubernetes cluster

## 1. Clone the repository

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/mlflow-dvc-customer-churn-demo.git
cd mlflow-dvc-customer-churn-demo
```

## 2. Create and activate a virtual environment

### Windows Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify:

```bash
python --version
mlflow --version
dvc version
```

## 3. Prepare the working dataset

Use either option.

### Option A — copy the supplied starter dataset

Git Bash, Linux, or macOS:

```bash
cp starter-data/customer_churn.csv data/customer_churn.csv
```

PowerShell:

```powershell
Copy-Item starter-data\customer_churn.csv data\customer_churn.csv
```

### Option B — generate the dataset

```bash
python src/generate_data.py \
  --rows 2000 \
  --seed 42 \
  --output data/customer_churn.csv
```

## 4. Run the traditional script before MLflow

```bash
python src/train_baseline.py \
  --model-type logistic_regression
```

This prints metrics and saves the model and metrics locally under `outputs/`. It does not create a central experiment history.

## 5. Initialise DVC and track the dataset

The repository is already tracked by Git after cloning, so initialise DVC:

```bash
dvc init
```

Track the working dataset:

```bash
dvc add data/customer_churn.csv
```

DVC creates:

```text
data/customer_churn.csv.dvc
data/.gitignore
```

Commit the DVC metadata:

```bash
git add .dvc data/customer_churn.csv.dvc data/.gitignore
git commit -m "Track customer churn dataset version 1 with DVC"
```

## 6. Configure the DVC S3 remote

Use a separate bucket or prefix from the MLflow artifact store.

```bash
dvc remote add -d storage s3://<DVC_BUCKET_NAME>/dvc-data
dvc remote modify storage region <AWS_REGION>
```

Verify the active AWS identity:

```bash
aws sts get-caller-identity
```

Push the dataset:

```bash
dvc push
```

Commit the DVC configuration:

```bash
git add .dvc/config
git commit -m "Configure DVC S3 remote"
```

Never commit AWS credentials, `.env`, or `.dvc/config.local`.

## 7. Connect the local training script to MLflow

Keep port-forwarding open in Terminal 1:

```bash
kubectl port-forward -n mlflow svc/mlflow-mlflow 5000:5000
```

Test the server:

```bash
curl http://127.0.0.1:5000/health
```

In Terminal 2, set the tracking configuration.

### Git Bash, Linux, or macOS

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=customer-churn-dvc-mlflow
```

### PowerShell

```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME="customer-churn-dvc-mlflow"
```

## 8. Run the first tracked experiment

```bash
python src/train_mlflow.py \
  --model-type logistic_regression \
  --c 1.0 \
  --run-name 01-logistic-baseline
```

The script logs:

- Hyperparameters
- Accuracy, precision, recall, F1 score, ROC-AUC, and training time
- Dataset metadata and MLflow dataset digest
- DVC dataset hash
- Git commit and working-tree status
- Classification report
- Confusion matrix
- Feature importance or coefficients
- DVC pointer file
- Environment and configuration files
- Complete scikit-learn preprocessing and model pipeline
- Model signature and input example

## 9. Run two additional experiments

```bash
python src/train_mlflow.py \
  --model-type random_forest \
  --n-estimators 150 \
  --max-depth 6 \
  --min-samples-split 4 \
  --run-name 02-random-forest-depth-6
```

```bash
python src/train_mlflow.py \
  --model-type random_forest \
  --n-estimators 300 \
  --max-depth 12 \
  --min-samples-split 2 \
  --run-name 03-random-forest-depth-12
```

Alternatively, run all experiments with:

### Git Bash

```bash
bash scripts/run_experiments.sh
```

### PowerShell

```powershell
.\scripts\run_experiments.ps1
```

## 10. Review the runs in MLflow

Open:

```text
http://127.0.0.1:5000
```

Select:

```text
customer-churn-dvc-mlflow
```

Compare:

1. Parameters used for each run
2. Model metrics
3. Dataset digest and DVC hash
4. Git commit and dirty-working-tree status
5. Logged reports and plots
6. The complete logged model pipeline

## 11. Create dataset version 2

Generate a different dataset:

```bash
python src/generate_data.py \
  --rows 2600 \
  --seed 84 \
  --output data/customer_churn.csv
```

Update and push the DVC version:

```bash
dvc add data/customer_churn.csv
dvc push
git add data/customer_churn.csv.dvc
git commit -m "Track customer churn dataset version 2"
```

Run the same model configuration again:

```bash
python src/train_mlflow.py \
  --model-type random_forest \
  --n-estimators 300 \
  --max-depth 12 \
  --min-samples-split 2 \
  --run-name 04-random-forest-data-v2
```

Compare run 3 and run 4. The model parameters remain the same, but the dataset version and resulting metrics change.

## 12. Query the best run

```bash
python src/show_best_run.py
```

This displays the top runs ordered by ROC-AUC.

## Tool responsibilities

| Tool | Responsibility |
|---|---|
| Git | Python code, configuration, and DVC pointer files |
| DVC | Dataset versions stored in the DVC remote |
| MLflow | Experiments, runs, parameters, metrics, datasets, models, tags, and artifacts |
| PostgreSQL | MLflow structured metadata |
| S3 artifact store | MLflow models, reports, plots, and other files |

## Security

Read [SECURITY.md](SECURITY.md) before publishing or recording. Never include credentials in screenshots, scripts, Git history, or course downloads.

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md).
