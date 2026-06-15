import json
import joblib
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    REPORTS_DIR,
)
from src.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "daily_return",
    "ma20",
]


def load_metrics():
    file_path = ARTIFACTS_DIR / "metrics.json"

    with open(file_path, "r") as file:
        return json.load(file)


def load_features():
    file_path = PROCESSED_DATA_DIR / "features.csv"

    return pd.read_csv(file_path)


def load_model():
    file_path = ARTIFACTS_DIR / "model.pkl"

    return joblib.load(file_path)


def generate_prediction(model, latest_row):
    prediction = model.predict(latest_row)[0]

    probability = model.predict_proba(latest_row)[0][1]

    return prediction, probability


def create_report():
    logger.info("Generating prediction report")

    metrics = load_metrics()

    df = load_features()

    model = load_model()

    latest_row = df[FEATURE_COLUMNS].tail(1)

    prediction, probability = generate_prediction(
        model,
        latest_row
    )

    latest_price = round(
        float(df["Close"].iloc[-1]),
        2
    )

    prediction_label = (
        "UP" if prediction == 1 else "DOWN"
    )

    report = f"""# ETF Prediction Report

## Latest Market Data

- Latest Close Price: {latest_price}

## Prediction

- Direction: {prediction_label}
- Probability: {probability:.2%}

## Model Metrics

- Accuracy: {metrics['accuracy']}
- Precision: {metrics['precision']}
- Recall: {metrics['recall']}
- ROC AUC: {metrics['roc_auc']}
"""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = REPORTS_DIR / "report.md"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    logger.info(
        f"Report saved to {output_path}"
    )


def main():
    create_report()


if __name__ == "__main__":
    main()