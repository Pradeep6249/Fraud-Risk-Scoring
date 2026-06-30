"""
predict.py
----------
Real-time fraud risk scoring inference.

Usage:
    python src/predict.py                    # scores a random sample transaction
    python src/predict.py --index 42         # scores transaction at row 42

Output:
    - Risk score (0.0 – 1.0)
    - Risk tier  (Low / Medium / High)
    - Top 5 SHAP drivers (requires model + explainer)
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import FEATURE_COLS


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def load_model(path: str = "models/xgboost_fraud_model.pkl"):
    return joblib.load(path)


def risk_tier(score: float) -> str:
    if score < 0.3:  return "🟢 LOW"
    if score < 0.6:  return "🟡 MEDIUM"
    return "🔴 HIGH"


def score_transaction(features: dict, model, threshold: float = 0.5) -> dict:
    """
    Score a single transaction dict.

    Args:
        features:  dict of feature_name → value
        model:     trained XGBoost model
        threshold: decision boundary (default 0.5)

    Returns:
        dict with score, tier, decision, and top drivers
    """
    available_cols = [c for c in FEATURE_COLS if c in features]
    row = pd.DataFrame([features])[available_cols].fillna(0)

    score     = float(model.predict_proba(row)[0, 1])
    decision  = "BLOCK" if score >= threshold else "ALLOW"
    tier      = risk_tier(score)

    # SHAP drivers
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(row)[0]
    shap_dict  = dict(zip(available_cols, shap_vals))
    top_drivers = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "score":       round(score, 4),
        "tier":        tier,
        "decision":    decision,
        "threshold":   threshold,
        "top_drivers": top_drivers,
    }


def print_result(result: dict, features: dict):
    print("\n" + "=" * 50)
    print("  FRAUD RISK SCORE REPORT")
    print("=" * 50)
    print(f"  Score:     {result['score']:.4f}")
    print(f"  Tier:      {result['tier']}")
    print(f"  Decision:  {result['decision']}  (threshold={result['threshold']})")
    print("\n  Top 5 Risk Drivers:")
    for feat, sv in result["top_drivers"]:
        direction = "↑ increases fraud risk" if sv > 0 else "↓ decreases fraud risk"
        val = features.get(feat, "?")
        print(f"    {feat:30s} = {val}  →  SHAP {sv:+.4f}  ({direction})")
    print("=" * 50)


# ─────────────────────────────────────────────
# Demo: score from test data
# ─────────────────────────────────────────────

def demo_score(index: int = None):
    """Load a row from the test set and score it."""
    df    = pd.read_csv("data/test_scored.csv")
    model = load_model()

    if index is None:
        index = np.random.randint(0, len(df))

    row      = df.iloc[index]
    features = row.drop(["is_fraud", "xgb_score"], errors="ignore").to_dict()
    truth    = int(row.get("is_fraud", -1))

    print(f"\n  Transaction index: {index}")
    print(f"  Ground truth:      {'🚨 FRAUD' if truth == 1 else '✅ LEGIT'}")

    result = score_transaction(features, model)
    print_result(result, features)
    return result


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud risk scoring inference")
    parser.add_argument("--index", type=int, default=None,
                        help="Row index in test_scored.csv to score (default: random)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (default: 0.5)")
    args = parser.parse_args()

    demo_score(index=args.index)
