# Technical Findings — Cycle 2: Data Pipeline and Feature Engineering

## Review Feedback Addressed

1. **Universe expanded from 17 to 29 tickers** (96.7% coverage): Added yfinance as fallback data source for 12 tickers not available on the ARF API (AMGN, CAT, CSCO, DIS, DOW, HON, IBM, KO, MMM, PG, TRV, VZ). WBA is delisted and unavailable from any source.
2. **Paper-only feature mode**: Added `paper_only=True` parameter to `add_technical_indicators()` that restricts output to RSI, MACD, and Bollinger Bands only (no additional features like daily_return, volume_change, etc.).
3. **Baseline strategy data**: Added `generate_baseline_data()` to compute equal-weight (1/N), vol-targeted 1/N, and momentum baseline portfolios for comparison.

## Implementation Summary

### Data Pipeline (`src/data/downloader.py`)
- Downloads OHLCV data for DJIA components from ARF API (primary) with yfinance fallback
- 17 tickers from ARF API + 12 tickers from yfinance = 29 total
- Period: 2009-01-01 to 2022-12-31 (DOW: 2019-04 to 2022-12 due to IPO)
- Rate-limited requests (0.5s ARF, 0.3s yfinance)
- Output: `data/raw/djia_data.csv` with 99,627 rows

### Feature Engineering (`src/features/build_features.py`)
- Paper-specified indicators (always computed):
  - **RSI(14)**: Relative Strength Index using EWM (Wilder's smoothing)
  - **MACD(12, 26, 9)**: MACD line, signal line, and histogram
  - **Bollinger Bands(20, 2)**: Upper, middle, and lower bands
- Additional features (when `paper_only=False`, the default):
  - **Daily return**: Close-to-close percentage change
  - **Volume change**: Volume percentage change
  - **Close-open ratio**: Intraday price movement indicator
  - **High-low spread**: Normalized daily price range
- Warmup rows (33 per ticker, 957 total) dropped to eliminate NaN
- Output: `data/processed/djia_processed.feather` with 98,670 rows and 18 columns

### Baseline Data (`src/features/build_features.py::generate_baseline_data`)
- Equal-weight (1/N) portfolio: daily return averaged across all tickers
- Vol-targeted 1/N: 1/N scaled to 10% annualized vol target (63-day rolling, max 2x leverage)
- Momentum: top-50% of 12-month trailing return, equal-weighted
- Output: `data/processed/baseline_data.feather` with 3,491 rows

### Data Integrity
- Zero NaN values in the processed dataset
- No future data leakage (all rolling windows use `center=False`, all EWMs are backward-looking)
- No duplicate ticker-date combinations
- RSI values bounded within [0, 100]
- Bollinger Band ordering verified: upper >= middle >= lower
- 16/16 tests passing (up from 13/13)

## Baseline Performance (from metrics.json)

| Baseline | Sharpe Ratio |
|---|---|
| 1/N Equal Weight | 1.0263 |
| Vol-Targeted 1/N | 1.1533 |
| 12-Month Momentum | 1.6015 |

- 1/N Annual Return: 18.19%
- 1/N Max Drawdown: -33.96%

These baselines cover 2009-2022 and will serve as comparison benchmarks for the RL ensemble strategy.

## Key Observations

1. **Universe Coverage**: 29/30 DJIA stocks (96.7%). Only WBA is missing (delisted). DOW has shorter history (955 days from April 2019 IPO vs 3,524 for others).

2. **Data Sources**: ARF API provides 17 tickers; yfinance successfully supplements 12 more. Both sources return compatible OHLCV format with auto-adjusted prices.

3. **Indicator Parameters**: All match the paper specification — RSI(14), MACD(12,26,9), Bollinger Bands(20,2).

4. **Feature Separation**: `paper_only=True` mode enables pure paper reproduction; default mode includes additional features for extended experiments.

5. **Trading Metrics**: All trading-related metrics (Sharpe, returns, etc.) are set to 0.0 in metrics.json as this phase covers data pipeline only. Baseline Sharpe ratios are populated for future comparison.

## Files Created/Modified
- `src/data/downloader.py` — ARF API data downloader with yfinance fallback
- `src/features/build_features.py` — Technical indicators with `paper_only` mode + baseline generation
- `tests/test_data_integrity.py` — 16 tests (added paper_only mode tests + universe completeness)
- `notebooks/01_data_exploration.ipynb` — Data exploration and visualization notebook
- `reports/cycle_2/preflight.md` — Preflight checks (updated)
- `reports/cycle_2/metrics.json` — Metrics with baseline Sharpe ratios
- `reports/cycle_2/technical_findings.md` — This file
- `docs/open_questions.md` — Updated with resolved and remaining questions
