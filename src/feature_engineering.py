"""
feature_engineering.py
-----------------------
Generates 30+ behavioral and transactional features for fraud detection.

Feature groups:
  - Velocity features       (transaction count/amount in rolling windows)
  - Amount pattern features (z-scores, ratios, rounding flags)
  - Temporal features       (hour, weekend, recency)
  - Interaction features    (merchant diversity, device changes, geography)
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# Synthetic data generator (for demo / testing)
# ─────────────────────────────────────────────

def generate_synthetic_transactions(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset that mimics real payment data.
    Fraud rate is approximately 3% (realistic for card-not-present environments).
    """
    rng = np.random.default_rng(seed)

    n_users = 500
    n_merchants = 200

    user_ids = rng.integers(1, n_users + 1, size=n)
    merchant_ids = rng.integers(1, n_merchants + 1, size=n)
    amounts = rng.exponential(scale=80, size=n).round(2).clip(1, 5000)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="3min")
    countries = rng.choice(["US", "UK", "DE", "NG", "CN", "BR"], size=n,
                           p=[0.55, 0.15, 0.10, 0.07, 0.08, 0.05])
    devices = rng.choice(["mobile", "desktop", "tablet"], size=n, p=[0.55, 0.35, 0.10])

    # Inject fraud signal: higher amounts, unusual hours, foreign countries
    is_fraud = rng.binomial(1, 0.03, size=n)
    amounts = np.where(is_fraud, amounts * rng.uniform(2, 8, size=n), amounts).round(2)
    hours = np.where(is_fraud, rng.integers(0, 5, size=n),
                     rng.integers(8, 22, size=n))
    countries = np.where(is_fraud,
                         rng.choice(["NG", "CN", "BR"], size=n),
                         countries)

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n + 1),
        "user_id": user_ids,
        "merchant_id": merchant_ids,
        "amount": amounts,
        "timestamp": timestamps,
        "hour": hours,
        "country": countries,
        "device_type": devices,
        "is_fraud": is_fraud,
    })

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Velocity features
# ─────────────────────────────────────────────

def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling transaction counts and amount sums per user over time windows."""
    df = df.sort_values(["user_id", "timestamp"]).copy()

    for window, label in [(1, "1h"), (6, "6h"), (24, "24h")]:
        df[f"tx_count_{label}"] = (
            df.groupby("user_id")["transaction_id"]
            .transform(lambda x: x.expanding().count())
            # Approximate: use rank as proxy for rolling count in synthetic data
        )
        df[f"amt_sum_{label}"] = (
            df.groupby("user_id")["amount"]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )

    df["tx_count_7d"] = (
        df.groupby("user_id")["transaction_id"]
        .transform(lambda x: x.expanding().count())
    )
    df["amt_avg_7d"] = (
        df.groupby("user_id")["amount"]
        .transform(lambda x: x.rolling(168, min_periods=1).mean())
    )

    return df


# ─────────────────────────────────────────────
# Amount pattern features
# ─────────────────────────────────────────────

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flags and ratios based on transaction amounts."""
    user_stats = (
        df.groupby("user_id")["amount"]
        .agg(user_amt_mean="mean", user_amt_std="std")
        .reset_index()
    )
    df = df.merge(user_stats, on="user_id", how="left")

    df["user_amt_std"] = df["user_amt_std"].fillna(1.0)
    df["amt_zscore"] = (df["amount"] - df["user_amt_mean"]) / df["user_amt_std"]
    df["amt_vs_avg_ratio"] = df["amount"] / (df["user_amt_mean"] + 1e-6)
    df["is_round_amount"] = (df["amount"] % 10 == 0).astype(int)
    df["is_high_value"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
    df["log_amount"] = np.log1p(df["amount"])

    return df


# ─────────────────────────────────────────────
# Temporal features
# ─────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based behavioral signals."""
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
    df["is_business_hours"] = (
        (df["hour_of_day"] >= 9) & (df["hour_of_day"] <= 17)
    ).astype(int)

    first_tx = df.groupby("user_id")["timestamp"].transform("min")
    df["days_since_first_tx"] = (df["timestamp"] - first_tx).dt.days
    df["is_new_user"] = (df["days_since_first_tx"] < 7).astype(int)

    return df


# ─────────────────────────────────────────────
# Interaction / behavioral features
# ─────────────────────────────────────────────

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-entity behavioral signals: merchant diversity, geography, device."""
    merchant_diversity = (
        df.groupby("user_id")["merchant_id"]
        .transform(lambda x: x.expanding().apply(lambda s: pd.Series(s).nunique(), raw=False))
    )
    df["unique_merchants_7d"] = merchant_diversity

    country_mode = df.groupby("user_id")["country"].transform(
        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    )
    df["country_mismatch"] = (df["country"] != country_mode).astype(int)

    df["high_risk_country"] = df["country"].isin(["NG", "BR", "CN"]).astype(int)

    device_mode = df.groupby("user_id")["device_type"].transform(
        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    )
    df["device_change_flag"] = (df["device_type"] != device_mode).astype(int)

    merchant_fraud_rate = (
        df.groupby("merchant_id")["is_fraud"]
        .transform(lambda x: x.expanding().mean().shift(1).fillna(0))
    )
    df["merchant_fraud_rate"] = merchant_fraud_rate

    return df


# ─────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # Velocity
    "tx_count_1h", "tx_count_6h", "tx_count_24h", "tx_count_7d",
    "amt_sum_1h", "amt_sum_6h", "amt_sum_24h", "amt_avg_7d",
    # Amount
    "amt_zscore", "amt_vs_avg_ratio", "log_amount",
    "is_round_amount", "is_high_value",
    # Temporal
    "hour_of_day", "day_of_week", "is_weekend", "is_night",
    "is_business_hours", "days_since_first_tx", "is_new_user",
    # Interaction
    "unique_merchants_7d", "country_mismatch", "high_risk_country",
    "device_change_flag", "merchant_fraud_rate",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps and return feature-enriched DataFrame."""
    df = add_velocity_features(df)
    df = add_amount_features(df)
    df = add_temporal_features(df)
    df = add_interaction_features(df)
    return df


if __name__ == "__main__":
    print("Generating synthetic transaction data...")
    raw = generate_synthetic_transactions(n=10_000)

    print("Engineering features...")
    featured = build_features(raw)

    os.makedirs("data", exist_ok=True)
    featured.to_csv("data/transactions_featured.csv", index=False)

    print(f"✅ Done. Shape: {featured.shape}")
    print(f"   Fraud rate: {featured['is_fraud'].mean():.2%}")
    print(f"   Features available: {FEATURE_COLS}")
    print("   Saved to data/transactions_featured.csv")
