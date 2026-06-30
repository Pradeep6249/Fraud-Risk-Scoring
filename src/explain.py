"""
explain.py
----------
SHAP-based explainability reporting for the fraud risk scoring model.

Generates:
  - Global feature importance (bar + beeswarm)
  - Local explanation for a single prediction
  - Feature importance CSV export
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap

warnings.filterwarnings("ignore")

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────

def load_artifacts():
    scored = pd.read_csv("data/test_scored.csv")
    model  = joblib.load("models/xgboost_fraud_model.pkl")

    feature_cols = [c for c in scored.columns if c not in ["is_fraud", "xgb_score"]]
    X = scored[feature_cols].fillna(0)
    y = scored["is_fraud"]
    return model, X, y


# ─────────────────────────────────────────────
# SHAP values
# ─────────────────────────────────────────────

def compute_shap_values(model, X: pd.DataFrame):
    """Compute SHAP values using TreeExplainer (fast for XGBoost)."""
    print("Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


# ─────────────────────────────────────────────
# 1. Global Feature Importance (Bar)
# ─────────────────────────────────────────────

def plot_global_importance(shap_values, X: pd.DataFrame):
    """Bar chart of mean absolute SHAP values per feature."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance_df["feature"], importance_df["mean_abs_shap"],
                   color="#2563eb")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Global Feature Importance (SHAP)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, importance_df["mean_abs_shap"].max() * 1.15)

    for bar, val in zip(bars, importance_df["mean_abs_shap"]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/06_shap_global_importance.png", dpi=150)
    plt.close()

    # Save CSV
    importance_df_full = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)
    importance_df_full.to_csv(f"{REPORTS_DIR}/feature_importance.csv", index=False)

    print(f"\n[6] Global SHAP Feature Importance")
    print(importance_df_full.head(10).to_string(index=False))
    print(f"    → Saved: reports/06_shap_global_importance.png")
    print(f"    → Saved: reports/feature_importance.csv")

    return importance_df_full


# ─────────────────────────────────────────────
# 2. SHAP Beeswarm (Summary Plot)
# ─────────────────────────────────────────────

def plot_beeswarm(shap_values, X: pd.DataFrame):
    """SHAP summary beeswarm — shows direction and magnitude per feature."""
    # Use top 15 features by importance
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-15:]
    X_top    = X.iloc[:, top_idx]
    sv_top   = shap_values[:, top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        sv_top, X_top,
        plot_type="dot",
        show=False,
        max_display=15,
    )
    plt.title("SHAP Summary — Feature Impact on Fraud Predictions",
              fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/07_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[7] SHAP Beeswarm Summary Plot")
    print(f"    → Saved: reports/07_shap_beeswarm.png")


# ─────────────────────────────────────────────
# 3. Local Explanation (single prediction)
# ─────────────────────────────────────────────

def plot_local_explanation(explainer, shap_values, X: pd.DataFrame, y: pd.Series,
                            idx: int = None):
    """
    Waterfall chart for a single prediction.
    By default picks the highest-scored fraud case for illustration.
    """
    if idx is None:
        # Pick the highest-confidence fraud prediction
        scored = pd.read_csv("data/test_scored.csv")
        fraud_cases = scored[scored["is_fraud"] == 1]
        if len(fraud_cases) > 0:
            idx = fraud_cases["xgb_score"].idxmax()
            # Map to X index
            idx = min(idx, len(X) - 1)
        else:
            idx = 0

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[0]

    shap_vals_local = shap_values[idx]
    feature_vals    = X.iloc[idx]

    # Top 10 contributors
    top_n = 10
    abs_sv = np.abs(shap_vals_local)
    top_idx = np.argsort(abs_sv)[-top_n:]

    features_top = feature_vals.index[top_idx].tolist()
    shap_top     = shap_vals_local[top_idx]
    fval_top     = feature_vals.values[top_idx]

    labels = [f"{f}\n= {v:.2f}" for f, v in zip(features_top, fval_top)]
    colors = ["#dc2626" if s > 0 else "#16a34a" for s in shap_top]

    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, shap_top, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)")
    ax.set_title(
        f"Local Explanation — Transaction #{idx}\n"
        f"(Red = pushes toward fraud | Green = pushes toward legit)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/08_shap_local_explanation.png", dpi=150)
    plt.close()

    print(f"\n[8] Local SHAP Explanation (transaction #{idx})")
    for feat, sv, fv in zip(features_top, shap_top, fval_top):
        direction = "↑ fraud" if sv > 0 else "↓ fraud"
        print(f"    {feat:30s} = {fv:8.3f}  SHAP={sv:+.4f}  ({direction})")
    print(f"    → Saved: reports/08_shap_local_explanation.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("Loading model and scored data...")
    model, X, y = load_artifacts()
    print(f"  Features: {X.shape[1]} | Rows: {X.shape[0]:,}")

    explainer, shap_values = compute_shap_values(model, X)

    importance_df = plot_global_importance(shap_values, X)
    plot_beeswarm(shap_values, X)
    plot_local_explanation(explainer, shap_values, X, y)

    print("\n✅ All SHAP reports saved to /reports")
    print("\nTop 5 fraud risk drivers:")
    print(importance_df.head(5)[["feature", "mean_abs_shap"]].to_string(index=False))


if __name__ == "__main__":
    main()
