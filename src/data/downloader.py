"""
Download DJIA stock data from the ARF Data API, with yfinance fallback
for tickers not available on the API.
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
DJIA_TICKERS_ARF = [
    "AAPL", "AXP", "BA", "CRM", "CVX", "GS", "HD", "INTC",
    "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "UNH", "V", "WMT",
]

# Full DJIA 30 components (WBA delisted; 29 available)
DJIA_30_FULL = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ",
    "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV",
    "UNH", "V", "VZ", "WBA", "WMT",
]

# For backwards compat
DJIA_TICKERS_AVAILABLE = DJIA_TICKERS_ARF


def _fetch_ticker_arf(ticker: str, interval: str = "1d", period: str = "max") -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker from the ARF Data API."""
    url = f"{ARF_API_BASE}?ticker={ticker}&interval={interval}&period={period}"
    logger.info(f"Fetching {ticker} from ARF API")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["timestamp"])
    df["ticker"] = ticker
    return df


def _fetch_ticker_yfinance(
    ticker: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker from yfinance (fallback)."""
    import yfinance as yf

    logger.info(f"Fetching {ticker} from yfinance (fallback)")
    yf_ticker = yf.Ticker(ticker)
    hist = yf_ticker.history(start=start_date, end=end_date, auto_adjust=True)

    if hist.empty:
        raise ValueError(f"No data returned from yfinance for {ticker}")

    df = pd.DataFrame({
        "timestamp": hist.index.tz_localize(None),
        "open": hist["Open"].values,
        "high": hist["High"].values,
        "low": hist["Low"].values,
        "close": hist["Close"].values,
        "volume": hist["Volume"].values,
        "ticker": ticker,
    })
    return df


def download_djia_data(
    output_path: str = "data/raw/djia_data.csv",
    start_date: str = "2009-01-01",
    end_date: str = "2022-12-31",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Download DJIA component stock data. Uses the ARF Data API as the primary
    source and falls back to yfinance for tickers not available on the API.

    Args:
        output_path: Path to save the combined CSV.
        start_date: Start date for filtering (inclusive).
        end_date: End date for filtering (inclusive).
        tickers: List of tickers to download. Defaults to full DJIA 30.

    Returns:
        Combined DataFrame with columns: timestamp, open, high, low, close, volume, ticker
    """
    tickers = tickers or DJIA_30_FULL
    all_dfs = []
    arf_failed = []

    # Phase 1: Try ARF API for all tickers
    for ticker in tickers:
        if ticker in DJIA_TICKERS_ARF:
            try:
                df = _fetch_ticker_arf(ticker)
                df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
                all_dfs.append(df)
                logger.info(f"{ticker} (ARF): {len(df)} rows")
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"ARF API failed for {ticker}: {e}")
                arf_failed.append(ticker)
        else:
            arf_failed.append(ticker)

    # Phase 2: yfinance fallback for tickers not available / failed on ARF API
    if arf_failed:
        logger.info(f"Falling back to yfinance for {len(arf_failed)} tickers: {arf_failed}")
        # Add 1 day buffer to end_date for yfinance (exclusive end)
        yf_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        yf_end_str = yf_end.strftime("%Y-%m-%d")
        for ticker in arf_failed:
            try:
                df = _fetch_ticker_yfinance(ticker, start_date, yf_end_str)
                df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]
                all_dfs.append(df)
                logger.info(f"{ticker} (yfinance): {len(df)} rows")
                time.sleep(0.3)
            except Exception as e:
                logger.warning(f"yfinance also failed for {ticker}: {e}")

    if not all_dfs:
        raise RuntimeError("No data fetched. Check API and yfinance availability.")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    n_tickers = combined["ticker"].nunique()
    logger.info(f"Saved {len(combined)} rows ({n_tickers} tickers) to {output_path}")

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_djia_data()
