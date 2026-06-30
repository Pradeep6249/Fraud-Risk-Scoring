"""
tests/test_validate.py
----------------------
Unit tests for the validation pipeline.
Run with: pytest tests/test_validate.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np
import pandas as pd

from validate import psi, stability_test, threshold_optimization


# ─────────────────────────────────────────────
# PSI tests
# ─────────────────────────────────────────────

def test_psi_identical_distributions():
    """PSI of identical distributions should be ~0."""
    arr = np.random.normal(0, 1, 1000)
    result = psi(arr, arr.copy())
    assert result < 0.05, f"PSI should be near 0 for identical distributions, got {result}"

def test_psi_very_different_distributions():
    """PSI of completely different distributions should be high."""
    arr1 = np.random.normal(0, 1, 1000)
    arr2 = np.random.normal(10, 1, 1000)
    result = psi(arr1, arr2)
    assert result > 0.25, f"PSI should be high for drifted distributions, got {result}"

def test_psi_returns_float():
    arr = np.random.uniform(0, 1, 500)
    result = psi(arr, arr + np.random.normal(0, 0.1, 500))
    assert isinstance(result, float)

def test_psi_non_negative():
    arr1 = np.random.exponential(1, 500)
    arr2 = np.random.exponential(2, 500)
    result = psi(arr1, arr2)
    assert result >= 0


# ─────────────────────────────────────────────
# Stability test
# ─────────────────────────────────────────────

@pytest.fixture
def mock_scored_df():
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.05, n)
    y_prob = np.where(y_true == 1,
                      np.random.beta(5, 2, n),
                      np.random.beta(1, 5, n))
    return pd.DataFrame({
        "is_fraud":  y_true,
        "xgb_score": y_prob,
        "amt_zscore": np.random.normal(0, 1, n),
        "high_risk_country": np.random.binomial(1, 0.15, n),
    })


def test_stability_test_returns_dict(mock_scored_df):
    result = stability_test(mock_scored_df)
    assert "ks_stat" in result
    assert "p_value" in result
    assert "stable" in result

def test_stability_ks_stat_range(mock_scored_df):
    result = stability_test(mock_scored_df)
    assert 0.0 <= result["ks_stat"] <= 1.0

def test_stability_p_value_range(mock_scored_df):
    result = stability_test(mock_scored_df)
    assert 0.0 <= result["p_value"] <= 1.0

def test_stability_flag_type(mock_scored_df):
    result = stability_test(mock_scored_df)
    assert isinstance(result["stable"], bool)


# ─────────────────────────────────────────────
# Threshold optimization
# ─────────────────────────────────────────────

def test_threshold_optimization_returns_dict(mock_scored_df):
    result = threshold_optimization(mock_scored_df)
    assert "optimal_threshold" in result
    assert "best_f1" in result

def test_optimal_threshold_in_range(mock_scored_df):
    result = threshold_optimization(mock_scored_df)
    assert 0.1 <= result["optimal_threshold"] <= 0.9

def test_best_f1_in_range(mock_scored_df):
    result = threshold_optimization(mock_scored_df)
    assert 0.0 <= result["best_f1"] <= 1.0
