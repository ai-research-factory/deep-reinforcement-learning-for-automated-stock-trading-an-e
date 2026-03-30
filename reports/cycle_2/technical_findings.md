# Technical Findings — Cycle 2: Data Pipeline and Feature Engineering

## Implementation Summary

### Data Pipeline (`src/data/downloader.py`)
- Implemented `download_djia_data()` function that fetches OHLCV data from the ARF Data API
- Downloads 17 available DJIA component stocks (out of 30 total) for the period 2009-01-01 to 2022-12-31
- Rate-limited requests (0.5s between tickers) to avoid API overload
- Output: `data/raw/djia_data.csv` with 59,908 rows (17 tickers x 3,524 trading days)

### Feature Engineering (`src/features/build_features.py`)
- Implemented `add_technical_indicators()` function with the following indicators:
  - **RSI(14)**: Relative Strength Index using EWM (Wilder's smoothing)
  - **MACD(12, 26, 9)**: MACD line, signal line, and histogram
  - **Bollinger Bands(20, 2)**: Upper, middle, and lower bands
  - **Daily return**: Close-to-close percentage change
  - **Volume change**: Volume percentage change
  - **Close-open ratio**: Intraday price movement indicator
  - **High-low spread**: Normalized daily price range
- All indicators computed per-ticker with proper grouping
- Warmup rows (33 per ticker, 561 total) dropped to eliminate NaN from indicator initialization
- Output: `data/processed/djia_processed.feather` with 59,347 rows and 18 columns

### Data Integrity
- Zero NaN values in the processed dataset
- No future data leakage confirmed (all rolling windows use `center=False`, all EWMs are backward-looking)
- No duplicate ticker-date combinations
- RSI values bounded within [0, 100]
- Bollinger Band ordering verified: upper >= middle >= lower
- 13/13 tests passing

## Key Observations

1. **Universe Coverage**: 17/30 DJIA stocks available via ARF API (56.7%). Missing stocks documented in `docs/open_questions.md`. This is a known limitation that may reduce diversification benefits in the ensemble strategy.

2. **Data Quality**: All 17 tickers have identical date coverage (3,524 trading days from 2009-01-02 to 2022-12-30), making panel alignment straightforward.

3. **Indicator Parameters**: All match the paper specification — RSI(14), MACD(12,26,9), Bollinger Bands(20,2).

4. **Trading Metrics**: All trading-related metrics (Sharpe, returns, etc.) are set to 0.0 in metrics.json as this phase covers data pipeline only. These will be populated in subsequent phases.

## Files Created/Modified
- `src/data/__init__.py` — Package init
- `src/data/downloader.py` — ARF API data downloader
- `src/features/__init__.py` — Package init
- `src/features/build_features.py` — Technical indicator computation
- `tests/test_data_integrity.py` — 13 tests for data integrity and leakage prevention
- `notebooks/01_data_exploration.ipynb` — Data exploration and visualization notebook
- `reports/cycle_2/preflight.md` — Preflight checks
- `reports/cycle_2/metrics.json` — Metrics (data phase, trading metrics N/A)
- `reports/cycle_2/technical_findings.md` — This file
- `docs/open_questions.md` — Open questions and limitations
