import yfinance as yf

from src.config import TICKERS, START_DATE, RAW_DATA_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def sanitize_filename(ticker: str) -> str:
    """
    Convert ticker name into a filesystem-safe filename.
    """
    return (
        ticker.replace("^", "")
        .replace("=", "")
        .replace(".", "_")
    )


def download_ticker(ticker: str):
    """
    Download historical market data for a single ticker.
    """
    logger.info(f"Downloading data for {ticker}")

    data = yf.download(
        ticker,
        start=START_DATE,
        progress=False,
        auto_adjust=True,
        multi_level_index=False
    )

    if data.empty:
        logger.warning(f"No data downloaded for {ticker}")
        return

    filename = f"{sanitize_filename(ticker)}.csv"
    output_path = RAW_DATA_DIR / filename

    data.to_csv(output_path)

    logger.info(
        f"Saved {len(data)} rows to {output_path}"
    )


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting market data download")

    for ticker in TICKERS:
        download_ticker(ticker)

    logger.info("Download completed")


if __name__ == "__main__":
    main()