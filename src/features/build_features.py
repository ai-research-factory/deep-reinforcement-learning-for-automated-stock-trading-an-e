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


def add_technical_indicators(df: pd.DataFrame, paper_only: bool = False) -> pd.DataFrame:
    """
    Add technical indicators to a stock DataFrame.

    Expects columns: timestamp, open, high, low, close, volume, ticker.

    Paper-specified indicators (always computed):
      - RSI(14)
      - MACD(12, 26, 9): macd, macd_signal, macd_hist
      - Bollinger Bands(20, 2): bb_upper, bb_middle, bb_lower

    Additional features (only when paper_only=False):
      - Daily return
      - Volume change ratio
      - Close-to-open ratio
      - High-low spread

    Args:
        df: DataFrame with OHLCV data and ticker column.
        paper_only: If True, only compute indicators specified in the paper.

    Returns:
        DataFrame with added indicator columns. Rows with NaN from warmup are dropped.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    result_dfs = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy()
        close = g["close"]

        # Paper-specified indicators
        g["rsi_14"] = _rsi(close, period=14)

        macd_line, signal_line, histogram = _macd(close, fast=12, slow=26, signal=9)
        g["macd"] = macd_line
        g["macd_signal"] = signal_line
        g["macd_hist"] = histogram

        bb_upper, bb_middle, bb_lower = _bollinger_bands(close, period=20, std_dev=2.0)
        g["bb_upper"] = bb_upper
        g["bb_middle"] = bb_middle
        g["bb_lower"] = bb_lower

        # Additional features (beyond paper spec)
        if not paper_only:
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


def generate_baseline_data(
    input_path: str = "data/processed/djia_processed.feather",
    output_path: str = "data/processed/baseline_data.feather",
) -> pd.DataFrame:
    """
    Generate baseline strategy data for comparison with RL agents.

    Computes:
      - Equal-weight (1/N) portfolio daily returns
      - Cumulative portfolio value
      - 12-month momentum signal for momentum baseline

    Args:
        input_path: Path to processed feather with OHLCV + indicators.
        output_path: Path to save baseline data.

    Returns:
        DataFrame with daily portfolio-level metrics.
    """
    from pathlib import Path

    df = pd.read_feather(input_path)

    # Compute daily returns per ticker
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    if "daily_return" not in df.columns:
        df["daily_return"] = df.groupby("ticker")["close"].pct_change()

    # Pivot to wide format: one column per ticker
    returns_wide = df.pivot_table(
        index="timestamp", columns="ticker", values="daily_return"
    )
    close_wide = df.pivot_table(
        index="timestamp", columns="ticker", values="close"
    )

    n_tickers = returns_wide.shape[1]

    # 1/N equal-weight portfolio return (simple average)
    baseline = pd.DataFrame(index=returns_wide.index)
    baseline["eq_weight_return"] = returns_wide.mean(axis=1)
    baseline["eq_weight_cumulative"] = (1 + baseline["eq_weight_return"]).cumprod()

    # 12-month (~252 day) momentum: trailing return for each ticker
    mom_12m = close_wide.pct_change(252)
    # Top 50% momentum tickers get equal weight, bottom 50% get zero
    mom_rank = mom_12m.rank(axis=1, pct=True)
    mom_weights = (mom_rank >= 0.5).astype(float)
    mom_weights = mom_weights.div(mom_weights.sum(axis=1), axis=0).fillna(0)
    baseline["momentum_return"] = (returns_wide * mom_weights).sum(axis=1)
    baseline["momentum_cumulative"] = (1 + baseline["momentum_return"]).cumprod()

    # Vol-targeted 1/N: scale by realized vol to target 10% annualized
    sigma_target = 0.10
    rolling_vol = baseline["eq_weight_return"].rolling(63, min_periods=21).std() * np.sqrt(252)
    leverage = (sigma_target / rolling_vol).clip(upper=2.0).fillna(1.0)
    baseline["voltarget_return"] = baseline["eq_weight_return"] * leverage
    baseline["voltarget_cumulative"] = (1 + baseline["voltarget_return"]).cumprod()

    baseline["n_tickers"] = n_tickers
    baseline = baseline.reset_index()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    baseline.to_feather(output_path)

    return baseline


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
    baseline = generate_baseline_data()
    print(f"Baseline data: {len(baseline)} rows")
