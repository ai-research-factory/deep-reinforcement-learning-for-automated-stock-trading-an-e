# Open Questions

## Data Availability

- **DJIA Universe Coverage**: 29 of 30 DJIA components available (96.7%). WBA (Walgreens Boots Alliance) has been delisted and is unavailable from both ARF API and yfinance. This is a minor limitation as the ensemble strategy can operate on 29 stocks.
- **DOW short history**: DOW Inc. IPO'd in April 2019, so it has only 955 trading days (vs 3,524 for other tickers). This creates an unbalanced panel which should be handled appropriately in training.
- **Data source mixing**: 17 tickers come from the ARF API and 12 from yfinance. Both provide OHLCV data with auto-adjusted prices, but minor differences in adjustment methodology may exist.

## Feature Engineering

- The paper may use additional technical indicators beyond RSI, MACD, and Bollinger Bands. The exact full feature set is not fully specified; current implementation includes standard indicators plus daily return, volume change, close-open ratio, and high-low spread.
- `paper_only=True` mode is available to restrict to paper-specified indicators only.
- Turbulence index mentioned in some versions of this paper is not yet implemented (planned for later phases if needed).
