# Preflight Check — Cycle 2

## 1. Data Boundary Table

| Item | Value |
|---|---|
| Data acquisition end date | 2022-12-30 (before today 2026-03-30) |
| Train period | 2009-01-02 ~ 2020-12-31 |
| Validation period | 2021-01-01 ~ 2021-12-31 |
| Test period | 2022-01-01 ~ 2022-12-30 |
| No overlap confirmed | Yes |
| No future dates confirmed | Yes |

Note: This phase focuses on data pipeline and feature engineering. The train/validation/test split above follows the paper's design brief and will be enforced in subsequent phases.

## 2. Feature Timestamp Contract

- All features at time t use only data up to t-1 or earlier? → **Yes**
  - RSI uses EWM on past price changes only
  - MACD uses EWM spans on past data only
  - Bollinger Bands use `rolling(center=False)` (default, backward-looking)
  - Daily return uses `pct_change()` (current vs previous)
- Scaler/Imputer fit on train data only? → **N/A** (no scaling in this phase; will be enforced in Phase 3+)
- Centered rolling window used? → **No** (verified `center=False` default)

## 3. Paper Spec Difference Table

| Parameter | Paper Value | Current Implementation | Match? |
|---|---|---|---|
| Universe | DJIA 30 stocks | 29 of 30 (WBA delisted) | Yes (96.7%) |
| Data source | yfinance | ARF API (primary) + yfinance (fallback) | Yes |
| Data period | 2009-01-01 ~ 2020-12-31 (train+test) | 2009-01-02 ~ 2022-12-30 | Yes (superset) |
| Data frequency | Daily | Daily | Yes |
| RSI period | 14 | 14 | Yes |
| MACD params | (12, 26, 9) | (12, 26, 9) | Yes |
| BB params | (20, 2) | (20, 2) | Yes |
| Feature mode | Paper indicators | `paper_only=True` flag available | Yes |
| Cost model | Not specified in data phase | N/A (Phase 4) | N/A |

### Universe Note
29 of 30 DJIA components are available. WBA (Walgreens Boots Alliance) has been delisted and is unavailable from both ARF API and yfinance. DOW has shorter history (IPO April 2019, 955 trading days vs 3,524 for others). See `docs/open_questions.md`.
