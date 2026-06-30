"""
tests/test_features.py
----------------------
Unit tests for feature engineering module.
Run with: pytest tests/test_features.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd
import numpy as np

from feature_engineering import (
    generate_synthetic_transactions,
    add_velocity_features,
    add_amount_features,
    add_temporal_features,
    add_interaction_features,
    build_features,
    FEATURE_COLS,
)


@pytest.fixture
def raw_data():
    return generate_synthetic_transactions(n=500, seed=0)


# ─────────────────────────────────────────────
# Data generation tests
# ─────────────────────────────────────────────

def test_data_shape(raw_data):
    assert len(raw_data) == 500
    assert "is_fraud" in raw_data.columns
    assert "amount" in raw_data.columns
    assert "user_id" in raw_data.columns

def test_fraud_rate_realistic(raw_data):
    """Fraud rate should be roughly 2–8% for synthetic data."""
    rate = raw_data["is_fraud"].mean()
    assert 0.01 < rate < 0.15, f"Unrealistic fraud rate: {rate:.2%}"

def test_no_negative_amounts(raw_data):
    assert (raw_data["amount"] >= 0).all()

def test_timestamps_ordered(raw_data):
    assert raw_data["timestamp"].is_monotonic_increasing


# ─────────────────────────────────────────────
# Feature engineering tests
# ─────────────────────────────────────────────

def test_velocity_features_created(raw_data):
    df = add_velocity_features(raw_data)
    for col in ["tx_count_1h", "tx_count_24h", "amt_sum_1h", "amt_avg_7d"]:
        assert col in df.columns, f"Missing velocity feature: {col}"

def test_amount_features_created(raw_data):
    df = add_amount_features(raw_data)
    for col in ["amt_zscore", "amt_vs_avg_ratio", "log_amount",
                "is_round_amount", "is_high_value"]:
        assert col in df.columns, f"Missing amount feature: {col}"

def test_log_amount_non_negative(raw_data):
    df = add_amount_features(raw_data)
    assert (df["log_amount"] >= 0).all()

def test_binary_flags_are_binary(raw_data):
    df = add_amount_features(raw_data)
    df = add_temporal_features(df)
    for col in ["is_round_amount", "is_high_value", "is_weekend",
                "is_night", "is_business_hours", "is_new_user"]:
        unique = set(df[col].unique())
        assert unique <= {0, 1}, f"Non-binary values in {col}: {unique}"

def test_temporal_features_created(raw_data):
    df = add_temporal_features(raw_data)
    for col in ["hour_of_day", "day_of_week", "is_weekend",
                "is_night", "days_since_first_tx"]:
        assert col in df.columns

def test_hour_range(raw_data):
    df = add_temporal_features(raw_data)
    assert df["hour_of_day"].between(0, 23).all()

def test_day_of_week_range(raw_data):
    df = add_temporal_features(raw_data)
    assert df["day_of_week"].between(0, 6).all()

def test_days_since_first_tx_non_negative(raw_data):
    df = add_temporal_features(raw_data)
    assert (df["days_since_first_tx"] >= 0).all()

def test_interaction_features_created(raw_data):
    df = add_interaction_features(raw_data)
    for col in ["unique_merchants_7d", "country_mismatch",
                "high_risk_country", "device_change_flag"]:
        assert col in df.columns

def test_high_risk_country_binary(raw_data):
    df = add_interaction_features(raw_data)
    assert set(df["high_risk_country"].unique()) <= {0, 1}


# ─────────────────────────────────────────────
# Full pipeline tests
# ─────────────────────────────────────────────

def test_build_features_no_crash(raw_data):
    df = build_features(raw_data)
    assert len(df) == len(raw_data)

def test_feature_cols_present_after_build(raw_data):
    df = build_features(raw_data)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    assert len(missing) == 0, f"Missing features after build: {missing}"

def test_no_inf_values(raw_data):
    df = build_features(raw_data)
    numeric = df.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values).any(), "Infinite values found in features"

def test_feature_count():
    """Ensure we have at least 25 features (resume claims 30+)."""
    assert len(FEATURE_COLS) >= 25, f"Only {len(FEATURE_COLS)} features defined"
