# Open Questions

## Data Availability

- **DJIA Universe Coverage**: Only 17 of 30 DJIA components are available via the ARF Data API. Missing: AMGN, CAT, CSCO, DIS, DOW, HON, IBM, KO, PG, TRV, VZ, WBA, MMM. This may affect the ensemble strategy's diversification compared to the paper's full 30-stock universe.
- The paper uses yfinance for data sourcing; we use the ARF Data API per project rules. Data format and adjusted prices may differ slightly.

## Feature Engineering

- The paper may use additional technical indicators beyond RSI, MACD, and Bollinger Bands. The exact full feature set is not fully specified; current implementation includes standard indicators plus daily return, volume change, close-open ratio, and high-low spread.
- Turbulence index mentioned in some versions of this paper is not yet implemented (planned for later phases if needed).
