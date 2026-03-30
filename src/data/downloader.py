"""
Download DJIA stock data from the ARF Data API.
"""
import io
import time
import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARF_API_BASE = "https://ai.1s.xyz/api/data/ohlcv"

# DJIA 30 component stocks available in the ARF Data API
# Some DJIA components are not available; we use the subset that is.
DJIA_TICKERS_AVAILABLE = [
    "AAPL", "AXP", "BA", "CRM", "CVX", "GS", "HD", "INTC",
    "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "UNH", "V", "WMT",
]

# Full DJIA 30 for reference (not all available via API)
DJIA_30_FULL = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
    "JPM", "KO", "MCD", "MRK", "MSFT", "NKE", "PG", "TRV",
    "UNH", "V", "VZ", "WBA", "WMT",
]


def _fetch_ticker(ticker: str, interval: str = "1d", period: str = "max") -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker from the ARF Data API."""
    url = f"{ARF_API_BASE}?ticker={ticker}&interval={interval}&period={period}"
    logger.info(f"Fetching {ticker} from {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["timestamp"])
    df["ticker"] = ticker
    return df


def download_djia_data(
    output_path: str = "data/raw/djia_data.csv",
    start_date: str = "2009-01-01",
    end_date: str = "2022-12-31",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Download DJIA component stock data from the ARF Data API.

    Args:
        output_path: Path to save the combined CSV.
        start_date: Start date for filtering (inclusive).
        end_date: End date for filtering (inclusive).
        tickers: List of tickers to download. Defaults to available DJIA components.

    Returns:
        Combined DataFrame with columns: timestamp, open, high, low, close, volume, ticker
    """
    tickers = tickers or DJIA_TICKERS_AVAILABLE
    all_dfs = []

    for ticker in tickers:
        try:
            df = _fetch_ticker(ticker)
            df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
            all_dfs.append(df)
            logger.info(f"{ticker}: {len(df)} rows")
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")

    if not all_dfs:
        raise RuntimeError("No data fetched. Check API availability.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"Saved {len(combined)} rows to {output_path}")

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_djia_data()
