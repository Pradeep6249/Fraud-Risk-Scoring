# Enterprise Fraud Risk Scoring & Model Validation Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

An end-to-end fraud detection pipeline on synthetic payment transactions: behavioral feature engineering, XGBoost vs. a logistic regression baseline, decision-threshold tuning, and an automated validation suite covering stability, drift, fairness and SHAP explainability.

*Nov 2024*

---

## Results

| | Logistic Regression (baseline) | XGBoost |
|---|---|---|
| **False-positive rate** | 16.9% | **2.2%** |
| **ROC AUC** (5-fold stratified CV) | n/a | **0.96** |

- **Data:** 10K synthetic payment transactions with a 3% fraud rate
- **Features:** 25 engineered features covering spending velocity, amount anomalies, timing and behavior
- **Imbalance:** handled with class weighting (`scale_pos_weight`) and 5-fold stratified cross-validation
- **Threshold:** decision threshold tuned to balance precision and recall
- **Fairness:** the disparate impact check flagged a geography-driven bias risk before deployment

---

## Project Structure

```
Fraud-Risk-Scoring/
├── data/
│   └── README.md                  # Data dictionary & generation instructions
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py     # Synthetic data + 25 engineered features
│   ├── train.py                   # Training pipeline (XGBoost, Logistic Regression)
│   ├── validate.py                # Validation: stability, drift, fairness
│   ├── predict.py                 # Risk scoring inference
│   └── explain.py                 # SHAP explainability reports
├── models/
│   └── README.md                  # Model versioning notes
├── reports/
│   ├── README.md                  # Auto-generated validation outputs land here
│   ├── feature_importance.csv     # Mean |SHAP| per feature
│   └── validation_summary.txt
├── tests/
│   ├── test_features.py
│   └── test_validate.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Tech Stack

- **Modeling:** XGBoost, Logistic Regression, scikit-learn
- **Explainability:** SHAP
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Validation:** scipy (KS test), PSI, evidently (drift detection)
- **Testing:** pytest

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Pradeep6249/Fraud-Risk-Scoring.git
cd Fraud-Risk-Scoring
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data & run the full pipeline
```bash
python src/feature_engineering.py   # Generate 10K transactions and build features
python src/train.py                  # Train and compare models
python src/validate.py               # Run the validation suite
python src/explain.py                # Generate SHAP report
```

### 4. Run the tests
```bash
pip install pytest
pytest tests/
```

---

## Feature Engineering (25 Features)

Features are grouped into four behavioral categories:

| Category | Examples |
|----------|----------|
| **Velocity** | `tx_count_1h`, `tx_count_24h`, `amt_sum_1h` |
| **Amount anomalies** | `amt_zscore`, `amt_vs_avg_ratio`, `is_round_amount` |
| **Timing** | `hour_of_day`, `is_weekend`, `is_night`, `days_since_first_tx` |
| **Behavior** | `country_mismatch`, `device_change_flag`, `merchant_fraud_rate`, `unique_merchants_7d` |

---

## Model Validation Suite

The validation pipeline (`src/validate.py`) runs automatically and checks:

- **Stability testing:** KS test on score distributions
- **Feature drift detection:** Population Stability Index (PSI)
- **Fairness checks:** disparate impact analysis, which flagged a geography-driven bias risk before deployment
- **Performance metrics:** AUC-ROC, precision-recall, F1, threshold optimization
- **SHAP explainability:** global feature importance plus an explanation for every prediction

The framework is covered by unit tests in `tests/`. All outputs are saved to `reports/`.

---

## Model Architecture

Two models are trained and compared:

1. **XGBoost Classifier:** primary model; handles class imbalance via `scale_pos_weight`
2. **Logistic Regression:** interpretable baseline for comparison

Final scoring uses XGBoost with a tuned decision threshold and SHAP explanations attached to each prediction.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Author

**Pradeep Kumar Voruganti**
[LinkedIn](https://www.linkedin.com/in/pradeep-kumar-voruganti) | [Portfolio](https://pradeep6249.github.io)
