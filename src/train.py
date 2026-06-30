"""
train.py
--------
Trains two models:
  1. XGBoost Classifier  — primary production model
  2. Logistic Regression — interpretable regulatory baseline

Handles class imbalance, outputs trained models to /models,
and prints a full evaluation report.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

from feature_engineering import generate_synthetic_transactions, build_features, FEATURE_COLS

RANDOM_STATE = 42
MODEL_DIR = "models"


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_data(path: str = "data/transactions_featured.csv") -> tuple:
    """Load feature-engineered data and return X, y splits."""
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        print("No data file found — generating synthetic data now...")
        raw = generate_synthetic_transactions(n=10_000)
        df = build_features(raw)
        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0)
    y = df["is_fraud"]
    return X, y


# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────

def evaluate(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Print and return evaluation metrics."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  AUC-ROC:              {auc:.4f}")
    print(f"  Average Precision:    {ap:.4f}")
    print(f"  False Positive Rate:  {fpr:.4f}")
    print(f"\n  Confusion Matrix:\n  TN={tn}  FP={fp}\n  FN={fn}  TP={tp}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")

    return {"name": name, "auc": auc, "ap": ap, "fpr": fpr}


# ─────────────────────────────────────────────
# Model 1 — XGBoost
# ─────────────────────────────────────────────

def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    """Train XGBoost with class-imbalance weighting."""
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=fraud_ratio,   # handles imbalance
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        verbosity=0,
        base_score=0.5,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\nXGBoost CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# Model 2 — Logistic Regression
# ─────────────────────────────────────────────

def train_logistic_regression(X_train, y_train) -> Pipeline:
    """Train a scaled Logistic Regression pipeline."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.1,
            random_state=RANDOM_STATE,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\nLogistic Regression CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────

def main():
    print("Loading data...")
    X, y = load_data()
    print(f"  Dataset shape: {X.shape}")
    print(f"  Fraud rate:    {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")

    # Train
    print("\n--- Training XGBoost ---")
    xgb_model = train_xgboost(X_train, y_train)

    print("\n--- Training Logistic Regression ---")
    lr_model = train_logistic_regression(X_train, y_train)

    # Evaluate
    results_xgb = evaluate("XGBoost Classifier", xgb_model, X_test, y_test)
    results_lr  = evaluate("Logistic Regression", lr_model, X_test, y_test)

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost_fraud_model.pkl")
    joblib.dump(lr_model,  f"{MODEL_DIR}/logistic_regression_fraud_model.pkl")

    # Save test set for validation
    test_df = X_test.copy()
    test_df["is_fraud"] = y_test.values
    test_df["xgb_score"] = xgb_model.predict_proba(X_test)[:, 1]
    test_df.to_csv("data/test_scored.csv", index=False)

    print("\n✅ Models saved to /models")
    print("✅ Scored test set saved to data/test_scored.csv")

    # Summary
    print("\n" + "="*50)
    print("  SUMMARY")
    print("="*50)
    print(f"  XGBoost  AUC: {results_xgb['auc']:.4f} | FPR: {results_xgb['fpr']:.4f}")
    print(f"  LR       AUC: {results_lr['auc']:.4f}  | FPR: {results_lr['fpr']:.4f}")
    print("  Primary model: XGBoost (higher AUC)")


if __name__ == "__main__":
    main()
