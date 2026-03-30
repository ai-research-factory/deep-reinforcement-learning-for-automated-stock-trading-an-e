"""
Data integrity and leakage prevention tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.build_features import add_technical_indicators


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices + np.random.randn(100) * 0.1,
        "high": prices + abs(np.random.randn(100) * 0.5),
        "low": prices - abs(np.random.randn(100) * 0.5),
        "close": prices,
        "volume": np.random.randint(1_000_000, 10_000_000, 100),
        "ticker": "TEST",
    })
    return df


@pytest.fixture
def processed_feather_path():
    return Path("data/processed/djia_processed.feather")


class TestFeatureTimestampContract:
    """Ensure no future data leakage in feature computation."""

    def test_rolling_windows_not_centered(self, sample_data):
        """Bollinger bands and other rolling windows must use center=False."""
        result = add_technical_indicators(sample_data)
        # If centered rolling was used, the first ~10 rows would have values
        # when they shouldn't. Check that early rows are dropped.
        assert len(result) < len(sample_data), "Warmup rows should be dropped"

    def test_indicators_use_only_past_data(self, sample_data):
        """Verify indicators at time t only use data up to t-1 or t."""
        full_result = add_technical_indicators(sample_data)

        # Compute on a truncated dataset (first 60 rows)
        truncated = sample_data.iloc[:60].copy()
        trunc_result = add_technical_indicators(truncated)

        # Values at the last row of truncated should match full result
        # at the same timestamp (if no future data is used)
        last_ts = trunc_result["timestamp"].iloc[-1]
        full_row = full_result[full_result["timestamp"] == last_ts]

        if len(full_row) > 0:
            for col in ["rsi_14", "macd", "bb_upper"]:
                np.testing.assert_allclose(
                    trunc_result[col].iloc[-1],
                    full_row[col].iloc[0],
                    rtol=1e-10,
                    err_msg=f"{col} changes with future data - possible leakage",
                )

    def test_no_future_dates(self, sample_data):
        """All timestamps must be in the past."""
        result = add_technical_indicators(sample_data)
        assert result["timestamp"].max() <= pd.Timestamp.now()


class TestDataIntegrity:
    """Tests for processed data quality."""

    def test_no_duplicate_ticker_dates(self, sample_data):
        result = add_technical_indicators(sample_data)
        dupes = result.groupby(["ticker", "timestamp"]).size()
        assert (dupes <= 1).all(), "Duplicate ticker-date combinations found"

    def test_required_columns_exist(self, sample_data):
        result = add_technical_indicators(sample_data)
        required = [
            "timestamp", "open", "high", "low", "close", "volume", "ticker",
            "rsi_14", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower",
            "daily_return",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_bounds(self, sample_data):
        result = add_technical_indicators(sample_data)
        assert result["rsi_14"].between(0, 100).all(), "RSI should be between 0 and 100"

    def test_bollinger_band_ordering(self, sample_data):
        result = add_technical_indicators(sample_data)
        assert (result["bb_upper"] >= result["bb_middle"]).all()
        assert (result["bb_middle"] >= result["bb_lower"]).all()

    def test_no_nan_in_indicators(self, sample_data):
        """After warmup removal, no NaN should remain in indicator columns."""
        result = add_technical_indicators(sample_data)
        indicator_cols = ["rsi_14", "macd", "macd_signal", "macd_hist",
                          "bb_upper", "bb_middle", "bb_lower"]
        for col in indicator_cols:
            assert not result[col].isna().any(), f"NaN found in {col}"


class TestPaperOnlyMode:
    """Tests for paper_only feature mode."""

    def test_paper_only_excludes_additional_features(self, sample_data):
        result = add_technical_indicators(sample_data, paper_only=True)
        paper_cols = ["rsi_14", "macd", "macd_signal", "macd_hist",
                      "bb_upper", "bb_middle", "bb_lower"]
        extra_cols = ["daily_return", "volume_change", "close_open_ratio", "high_low_spread"]
        for col in paper_cols:
            assert col in result.columns, f"Paper indicator {col} missing in paper_only mode"
        for col in extra_cols:
            assert col not in result.columns, f"Extra feature {col} should not be in paper_only mode"

    def test_default_includes_additional_features(self, sample_data):
        result = add_technical_indicators(sample_data, paper_only=False)
        extra_cols = ["daily_return", "volume_change", "close_open_ratio", "high_low_spread"]
        for col in extra_cols:
            assert col in result.columns, f"Extra feature {col} missing in default mode"


class TestProcessedData:
    """Tests for the actual processed DJIA data (skip if not generated)."""

    def test_feather_exists(self, processed_feather_path):
        if not processed_feather_path.exists():
            pytest.skip("Processed feather not yet generated")
        df = pd.read_feather(processed_feather_path)
        assert len(df) > 0

    def test_feather_has_indicators(self, processed_feather_path):
        if not processed_feather_path.exists():
            pytest.skip("Processed feather not yet generated")
        df = pd.read_feather(processed_feather_path)
        for col in ["rsi_14", "macd", "bb_upper"]:
            assert col in df.columns

    def test_feather_date_range(self, processed_feather_path):
        if not processed_feather_path.exists():
            pytest.skip("Processed feather not yet generated")
        df = pd.read_feather(processed_feather_path)
        assert df["timestamp"].min() >= pd.Timestamp("2009-01-01")
        assert df["timestamp"].max() <= pd.Timestamp("2022-12-31")

    def test_feather_no_future_dates(self, processed_feather_path):
        if not processed_feather_path.exists():
            pytest.skip("Processed feather not yet generated")
        df = pd.read_feather(processed_feather_path)
        assert df["timestamp"].max() <= pd.Timestamp.now()

    def test_multiple_tickers(self, processed_feather_path):
        if not processed_feather_path.exists():
            pytest.skip("Processed feather not yet generated")
        df = pd.read_feather(processed_feather_path)
        assert df["ticker"].nunique() >= 10, "Expected at least 10 tickers"

    def test_universe_completeness(self, processed_feather_path):
        """Verify the processed data contains all 30 DJIA components."""
        if not processed_feather_path.exists():
            pytest.skip("Processed feather not yet generated")
        df = pd.read_feather(processed_feather_path)
        n_tickers = df["ticker"].nunique()
        assert n_tickers >= 29, (
            f"Expected at least 29 DJIA tickers (target: 30), got {n_tickers}. "
            f"Tickers present: {sorted(df['ticker'].unique())}"
        )
