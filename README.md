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

## Project Structure

```text
src/
tests/

data/
artifacts/
reports/
```