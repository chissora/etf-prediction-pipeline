# ETF Prediction Pipeline

Portfolio project for ETF monitoring and prediction using:

* Python
* Machine Learning
* Docker
* GitHub Actions (planned)
* MLOps fundamentals

## Project Overview

The pipeline performs:

1. Downloading market data from Yahoo Finance
2. Feature engineering
3. Model training
4. Model evaluation
5. Prediction report generation

## Run Locally

```bash
python -m src.main
```

## Run with Docker

Build image:

```bash
docker build -t etf-pipeline .
```

Run container:

```bash
docker run etf-pipeline
```

## MLflow Tracking

Training runs are logged to an MLflow experiment named `etf_prediction`.
The pipeline logs the model parameters, evaluation metrics, and trained
scikit-learn model while still writing `artifacts/model.pkl` and
`artifacts/metrics.json`.

Start the MLflow UI with:

```bash
mlflow ui
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000/).

## Project Structure

```text
src/
tests/

data/
artifacts/
reports/
```
