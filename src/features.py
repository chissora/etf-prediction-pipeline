import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)

ETF_FILE = "SXR8_DE.csv"


def load_data() -> pd.DataFrame:
    file_path = RAW_DATA_DIR / ETF_FILE

    logger.info(f"Loading data from {file_path}")

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating features")

    # podstawowe cechy
    df["daily_return"] = df["Close"].pct_change()

    df["ma20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    # target: czy następnego dnia cena będzie wyższa
    df["target"] = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)

    # zostawiamy tylko potrzebne kolumny
    df = df[
        [
            "Date",
            "Close",
            "Volume",
            "daily_return",
            "ma20",
            "target",
        ]
    ]

    df = df.dropna()

    return df


def save_features(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = PROCESSED_DATA_DIR / "features.csv"

    df.to_csv(output_path, index=False)

    logger.info(f"Features saved to {output_path}")


def main():
    logger.info("Starting feature engineering")

    df = load_data()

    features_df = create_features(df)

    save_features(features_df)

    logger.info(
        f"Feature engineering completed ({len(features_df)} rows)"
    )


if __name__ == "__main__":
    main()