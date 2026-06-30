"""
validate.py
-----------
Automated model validation pipeline covering:
  1. Performance metrics      (AUC, Precision-Recall, F1)
  2. Stability testing        (KS statistic across time windows)
  3. Drift detection          (Population Stability Index on features)
  4. Bias & fairness          (Disparate impact across demographic slices)
  5. Threshold optimization   (Precision-recall tradeoff analysis)

All outputs are saved to /reports.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────

def load_artifacts():
    """Load scored test data and trained model."""
    scored = pd.read_csv("data/test_scored.csv")
    model  = joblib.load("models/xgboost_fraud_model.pkl")
    return scored, model


# ─────────────────────────────────────────────
# 1. Performance Report
# ─────────────────────────────────────────────

def performance_report(df: pd.DataFrame) -> dict:
    """Compute and plot AUC-ROC and Precision-Recall curves."""
    y_true = df["is_fraud"]
    y_prob = df["xgb_score"]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)

    # ROC curve
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    # PR curve
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model Performance", fontsize=14, fontweight="bold")

    axes[0].plot(fpr_arr, tpr_arr, color="#2563eb", lw=2, label=f"AUC = {auc:.4f}")
    axes[0].plot([0,1],[0,1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].plot(rec_arr, prec_arr, color="#16a34a", lw=2, label=f"AP = {ap:.4f}")
    axes[1].axhline(y_true.mean(), color="gray", linestyle="--", label="Baseline")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/01_performance_curves.png", dpi=150)
    plt.close()

    print(f"\n[1] Performance Report")
    print(f"    AUC-ROC:           {auc:.4f}")
    print(f"    Average Precision: {ap:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Legit','Fraud'])}")
    print(f"    → Saved: reports/01_performance_curves.png")

    return {"auc": auc, "ap": ap}


# ─────────────────────────────────────────────
# 2. Stability Testing (KS Statistic)
# ─────────────────────────────────────────────

def stability_test(df: pd.DataFrame):
    """
    Split scored data into two halves (simulating time windows)
    and run a KS test on score distributions.
    A KS statistic > 0.2 or p < 0.05 indicates score instability.
    """
    mid = len(df) // 2
    scores_early = df.iloc[:mid]["xgb_score"]
    scores_late  = df.iloc[mid:]["xgb_score"]

    ks_stat, p_value = stats.ks_2samp(scores_early, scores_late)
    stable = ks_stat < 0.2 and p_value > 0.05

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores_early, bins=40, alpha=0.6, label="Window 1 (early)", color="#2563eb")
    ax.hist(scores_late,  bins=40, alpha=0.6, label="Window 2 (late)",  color="#dc2626")
    ax.set_xlabel("Fraud Risk Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Stability — KS={ks_stat:.4f}  p={p_value:.4f}  {'✅ STABLE' if stable else '⚠️ DRIFT'}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/02_stability_ks_test.png", dpi=150)
    plt.close()

    print(f"\n[2] Stability Test")
    print(f"    KS Statistic: {ks_stat:.4f}")
    print(f"    P-value:      {p_value:.4f}")
    print(f"    Result:       {'✅ STABLE' if stable else '⚠️  POTENTIAL DRIFT DETECTED'}")
    print(f"    → Saved: reports/02_stability_ks_test.png")

    return {"ks_stat": ks_stat, "p_value": p_value, "stable": stable}


# ─────────────────────────────────────────────
# 3. Drift Detection (PSI)
# ─────────────────────────────────────────────

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. PSI < 0.1 = stable, 0.1-0.25 = monitor, >0.25 = unstable."""
    eps = 1e-6
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + eps
    actual_pct   = np.histogram(actual,   bins=breakpoints)[0] / len(actual)   + eps

    # Match lengths after unique breakpoints
    min_len = min(len(expected_pct), len(actual_pct))
    expected_pct = expected_pct[:min_len]
    actual_pct   = actual_pct[:min_len]

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def drift_detection(df: pd.DataFrame):
    """Compute PSI on key features between early/late windows."""
    feature_cols = [c for c in df.columns if c not in ["is_fraud", "xgb_score"]]
    mid = len(df) // 2
    early = df.iloc[:mid]
    late  = df.iloc[mid:]

    psi_results = {}
    for col in feature_cols[:10]:  # top 10 features
        try:
            p = psi(early[col].values, late[col].values)
            psi_results[col] = round(p, 4)
        except Exception:
            psi_results[col] = None

    psi_df = pd.DataFrame.from_dict(psi_results, orient="index", columns=["PSI"])
    psi_df = psi_df.dropna().sort_values("PSI", ascending=False)

    def psi_flag(v):
        if v < 0.1:   return "✅ Stable"
        if v < 0.25:  return "⚠️  Monitor"
        return "🚨 Unstable"

    psi_df["Status"] = psi_df["PSI"].apply(psi_flag)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#16a34a" if v < 0.1 else "#f59e0b" if v < 0.25 else "#dc2626"
              for v in psi_df["PSI"]]
    ax.barh(psi_df.index, psi_df["PSI"], color=colors)
    ax.axvline(0.1,  color="#f59e0b", linestyle="--", label="Monitor threshold (0.1)")
    ax.axvline(0.25, color="#dc2626", linestyle="--", label="Unstable threshold (0.25)")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Feature Drift Detection")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/03_drift_psi.png", dpi=150)
    plt.close()

    print(f"\n[3] Drift Detection (PSI)")
    print(psi_df.to_string())
    print(f"    → Saved: reports/03_drift_psi.png")

    return psi_df


# ─────────────────────────────────────────────
# 4. Bias & Fairness Monitoring
# ─────────────────────────────────────────────

def bias_monitoring(df: pd.DataFrame):
    """
    Disparate impact analysis:
    - Flag rate per group (how often each group is flagged as fraud)
    - Disparate Impact Ratio (DIR): minority / majority flag rate
      DIR < 0.8 indicates potential bias (80% rule).
    """
    # Simulate demographic slices using proxies available in features
    df = df.copy()
    df["risk_tier"] = pd.cut(
        df["xgb_score"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )

    # Use high_risk_country as a demographic proxy slice
    if "high_risk_country" in df.columns:
        slice_col = "high_risk_country"
        slice_labels = {0: "Standard Country", 1: "High-Risk Country"}
    else:
        # Fallback: create synthetic slice
        df["slice"] = (df["xgb_score"].rank(pct=True) > 0.5).astype(int)
        slice_col = "slice"
        slice_labels = {0: "Group A", 1: "Group B"}

    threshold = 0.5
    df["flagged"] = (df["xgb_score"] >= threshold).astype(int)

    group_stats = (
        df.groupby(slice_col)
        .agg(
            total=("flagged", "count"),
            flagged_count=("flagged", "sum"),
            fraud_actual=("is_fraud", "sum"),
        )
        .reset_index()
    )
    group_stats["flag_rate"]   = group_stats["flagged_count"] / group_stats["total"]
    group_stats["actual_rate"] = group_stats["fraud_actual"]  / group_stats["total"]
    group_stats["label"]       = group_stats[slice_col].map(slice_labels)

    # Disparate Impact Ratio
    rates = group_stats["flag_rate"].values
    if len(rates) >= 2:
        dir_ratio = rates.min() / rates.max() if rates.max() > 0 else 1.0
    else:
        dir_ratio = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Bias & Fairness Monitoring", fontsize=14, fontweight="bold")

    axes[0].bar(group_stats["label"], group_stats["flag_rate"],
                color=["#2563eb", "#dc2626"])
    axes[0].axhline(0.05, color="gray", linestyle="--", label="5% baseline")
    axes[0].set_ylabel("Flag Rate")
    axes[0].set_title("Flag Rate by Group")
    axes[0].legend()

    axes[1].bar(group_stats["label"], group_stats["actual_rate"],
                color=["#16a34a", "#f59e0b"])
    axes[1].set_ylabel("Actual Fraud Rate")
    axes[1].set_title("Actual Fraud Rate by Group")

    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/04_bias_fairness.png", dpi=150)
    plt.close()

    print(f"\n[4] Bias & Fairness Monitoring")
    print(group_stats[["label", "total", "flag_rate", "actual_rate"]].to_string(index=False))
    print(f"\n    Disparate Impact Ratio: {dir_ratio:.4f}")
    print(f"    Result: {'✅ FAIR (DIR >= 0.8)' if dir_ratio >= 0.8 else '⚠️  POTENTIAL BIAS (DIR < 0.8)'}")
    print(f"    → Saved: reports/04_bias_fairness.png")

    return {"dir_ratio": dir_ratio}


# ─────────────────────────────────────────────
# 5. Threshold Optimization
# ─────────────────────────────────────────────

def threshold_optimization(df: pd.DataFrame):
    """Plot precision, recall, and F1 across thresholds to find optimal cutoff."""
    y_true = df["is_fraud"]
    y_prob = df["xgb_score"]

    thresholds = np.arange(0.1, 0.9, 0.01)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    best_idx = np.argmax(f1s)
    best_t   = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#2563eb")
    ax.plot(thresholds, recalls,    label="Recall",    color="#16a34a")
    ax.plot(thresholds, f1s,        label="F1 Score",  color="#dc2626", lw=2)
    ax.axvline(best_t, color="gray", linestyle="--",
               label=f"Optimal threshold = {best_t:.2f}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Optimization")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/05_threshold_optimization.png", dpi=150)
    plt.close()

    print(f"\n[5] Threshold Optimization")
    print(f"    Optimal threshold: {best_t:.2f}")
    print(f"    F1 at optimal:     {f1s[best_idx]:.4f}")
    print(f"    Precision:         {precisions[best_idx]:.4f}")
    print(f"    Recall:            {recalls[best_idx]:.4f}")
    print(f"    → Saved: reports/05_threshold_optimization.png")

    return {"optimal_threshold": best_t, "best_f1": f1s[best_idx]}


# ─────────────────────────────────────────────
# Summary Report
# ─────────────────────────────────────────────

def save_summary(perf, stability, bias, threshold):
    """Write a text summary of all validation results."""
    lines = [
        "=" * 55,
        "  FRAUD MODEL VALIDATION REPORT",
        "=" * 55,
        "",
        f"  AUC-ROC:                {perf['auc']:.4f}",
        f"  Average Precision:      {perf['ap']:.4f}",
        "",
        f"  KS Stability:           {stability['ks_stat']:.4f}  "
        f"({'STABLE' if stability['stable'] else 'DRIFT DETECTED'})",
        f"  KS P-value:             {stability['p_value']:.4f}",
        "",
        f"  Disparate Impact Ratio: {bias['dir_ratio']:.4f}  "
        f"({'FAIR' if bias['dir_ratio'] >= 0.8 else 'BIAS RISK'})",
        "",
        f"  Optimal Threshold:      {threshold['optimal_threshold']:.2f}",
        f"  F1 at Optimal:          {threshold['best_f1']:.4f}",
        "",
        "  Artifacts saved in /reports:",
        "    01_performance_curves.png",
        "    02_stability_ks_test.png",
        "    03_drift_psi.png",
        "    04_bias_fairness.png",
        "    05_threshold_optimization.png",
        "=" * 55,
    ]
    summary = "\n".join(lines)
    with open(f"{REPORTS_DIR}/validation_summary.txt", "w") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\n✅ Summary saved to reports/validation_summary.txt")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("Loading scored test data...")
    scored, model = load_artifacts()
    print(f"  Rows: {len(scored):,} | Fraud rate: {scored['is_fraud'].mean():.2%}")

    perf       = performance_report(scored)
    stability  = stability_test(scored)
    drift_df   = drift_detection(scored)
    bias       = bias_monitoring(scored)
    threshold  = threshold_optimization(scored)

    save_summary(perf, stability, bias, threshold)


if __name__ == "__main__":
    main()
