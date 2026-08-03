# Troubleshooting Guide

## MLflow connection refused

Confirm the pod and port-forward:

```bash
kubectl get pods -n mlflow
kubectl port-forward -n mlflow svc/mlflow-mlflow 5000:5000
curl http://127.0.0.1:5000/health
```

Confirm the environment variable:

```bash
echo $MLFLOW_TRACKING_URI
```

PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI
```

## MLflow logs locally instead of to the server

If a local `mlruns/` directory appears, the tracking URI was not set correctly. Export the URI again before running the script.

## Experiment does not appear

Check:

```bash
echo $MLFLOW_EXPERIMENT_NAME
```

The script defaults to `customer-churn-dvc-mlflow` if no value is provided.

## DVC says the data file is missing

Run:

```bash
dvc pull
```

Then confirm:

```bash
ls data
```

## DVC S3 authentication failure

Verify the AWS identity:

```bash
aws sts get-caller-identity
```

Check the remote:

```bash
dvc remote list
```

For a named AWS profile, configure it locally so it is not committed:

```bash
dvc remote modify --local storage profile <AWS_PROFILE>
```

## DVC hash shows `not-tracked-by-dvc`

Run:

```bash
dvc add data/customer_churn.csv
```

The file `data/customer_churn.csv.dvc` must exist before the experiment starts.

## S3 artifact logging fails

The local training client usually sends artifacts through the MLflow server. Check the MLflow pod logs and its S3 permissions:

```bash
kubectl logs deployment/mlflow-mlflow -n mlflow --tail=200
```

## Password or access key accidentally displayed

Stop recording. Rotate the exposed credential before continuing. Editing it out of the video is not enough if it was shared elsewhere.

## Model logging dependency error

Make sure the client version matches the course server:

```bash
mlflow --version
```

Reinstall from the provided requirements file:

```bash
python -m pip install --upgrade --force-reinstall -r requirements.txt
```

## `ModuleNotFoundError: training_utils`

Run the command from the project root:

```bash
python src/train_mlflow.py ...
```

Do not change into the `src` directory first.

## Dataset metrics are slightly different

The generated data and model are deterministic when the seed and parameters are unchanged. Differences can still occur across significantly different library versions. Use the provided requirements file for the recording.
