import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)

MLFLOW_EXPERIMENT_NAME = "etf_prediction"
MODEL_MAX_ITER = 1000

FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "daily_return",
    "ma20",
]

TARGET_COLUMN = "target"


def load_features():
    file_path = PROCESSED_DATA_DIR / "features.csv"

    logger.info(f"Loading features from {file_path}")

    return pd.read_csv(file_path)


def prepare_data(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    logger.info(f"Train size: {len(X_train)}")
    logger.info(f"Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    logger.info("Training Logistic Regression")

    model = LogisticRegression(
        max_iter=MODEL_MAX_ITER
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(
            accuracy_score(y_test, predictions),
            4
        ),
        "precision": round(
            precision_score(y_test, predictions),
            4
        ),
        "recall": round(
            recall_score(y_test, predictions),
            4
        ),
        "roc_auc": round(
            roc_auc_score(y_test, probabilities),
            4
        ),
    }

    logger.info(f"Metrics: {metrics}")

    return metrics


def save_model(model):
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = ARTIFACTS_DIR / "model.pkl"

    joblib.dump(model, output_path)

    logger.info(f"Model saved to {output_path}")


def save_metrics(metrics):
    output_path = ARTIFACTS_DIR / "metrics.json"

    with open(output_path, "w") as file:
        json.dump(metrics, file, indent=4)

    logger.info(f"Metrics saved to {output_path}")


def log_mlflow_run(model, metrics):
    logger.info("Logging MLflow experiment run")

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", model.max_iter)

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
    )


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_features()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    with mlflow.start_run():
        model = train_model(
            X_train,
            y_train,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        log_mlflow_run(model, metrics)

        save_model(model)

        save_metrics(metrics)

    logger.info("Training and evaluation completed")


if __name__ == "__main__":
    main()
