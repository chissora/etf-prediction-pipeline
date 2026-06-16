import pandas as pd

from sklearn.linear_model import LogisticRegression

from src.train import (
    prepare_data,
    split_data,
    train_model,
)


def test_train_model_returns_logistic_regression():
    df = pd.DataFrame(
        {
            "Close": range(100, 200),
            "Volume": range(1000, 1100),
            "daily_return": [0.01] * 100,
            "ma20": range(90, 190),
            "target": [0, 1] * 50,
        }
    )

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(
        X_train,
        y_train
    )

    assert isinstance(
        model,
        LogisticRegression
    )