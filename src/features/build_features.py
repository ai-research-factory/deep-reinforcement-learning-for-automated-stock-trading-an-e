"""
Technical indicator feature engineering for stock data.
"""
import pandas as pd
import numpy as np


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (upper, middle, lower)."""
    middle = series.rolling(window=period, center=False).mean()
    std = series.rolling(window=period, center=False).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to a stock DataFrame.

    Expects columns: timestamp, open, high, low, close, volume, ticker.
    Computes indicators per ticker:
      - RSI(14)
      - MACD(12, 26, 9): macd, macd_signal, macd_hist
      - Bollinger Bands(20, 2): bb_upper, bb_middle, bb_lower
      - Daily return
      - Volume change ratio
      - Close-to-open ratio
      - High-low spread

    Args:
        df: DataFrame with OHLCV data and ticker column.

    Returns:
        DataFrame with added indicator columns. Rows with NaN from warmup are dropped.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    result_dfs = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        close = g["close"]

        # RSI
        g["rsi_14"] = _rsi(close, period=14)

        # MACD
        macd_line, signal_line, histogram = _macd(close, fast=12, slow=26, signal=9)
        g["macd"] = macd_line
        g["macd_signal"] = signal_line
        g["macd_hist"] = histogram

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = _bollinger_bands(close, period=20, std_dev=2.0)
        g["bb_upper"] = bb_upper
        g["bb_middle"] = bb_middle
        g["bb_lower"] = bb_lower

        # Additional features
        g["daily_return"] = close.pct_change()
        g["volume_change"] = g["volume"].pct_change()
        g["close_open_ratio"] = close / g["open"]
        g["high_low_spread"] = (g["high"] - g["low"]) / close

        result_dfs.append(g)

    result = pd.concat(result_dfs, ignore_index=True)

    # Drop rows with NaN from indicator warmup periods
    indicator_cols = ["rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle", "bb_lower"]
    result = result.dropna(subset=indicator_cols).reset_index(drop=True)

    return result


def process_and_save(
    input_path: str = "data/raw/djia_data.csv",
    output_path: str = "data/processed/djia_processed.feather",
) -> pd.DataFrame:
    """Load raw data, add indicators, save as feather."""
    from pathlib import Path

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df = add_technical_indicators(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(output_path)

    return df


if __name__ == "__main__":
    df = process_and_save()
    print(f"Processed {len(df)} rows with columns: {list(df.columns)}")
