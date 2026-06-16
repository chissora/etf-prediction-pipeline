import pandas as pd

from src.features import create_features


def test_create_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "Date": pd.date_range(
                start="2024-01-01",
                periods=30
            ),
            "Close": range(100, 130),
            "Volume": range(1000, 1030),
        }
    )

    result = create_features(df)

    expected_columns = [
        "Date",
        "Close",
        "Volume",
        "daily_return",
        "ma20",
        "target",
    ]

    for column in expected_columns:
        assert column in result.columns