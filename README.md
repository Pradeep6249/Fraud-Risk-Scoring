# Enterprise Fraud Risk Scoring & Model Validation Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()

An end-to-end machine learning platform for payment fraud detection, featuring behavioral feature engineering, ensemble modeling, automated model validation, drift detection, bias monitoring, and SHAP-based explainability reporting.

---

## 📊 Results

| Metric | Baseline | This Framework |
|--------|----------|----------------|
| Fraud Detection Rate | ~71% | **+22% improvement** |
| False Positive Rate | — | **−15% reduction** |
| Manual Investigation Time | High | Reduced via automated scoring |

---

## 🗂️ Project Structure

```
fraud-risk-scoring/
├── data/
│   └── README.md                  # Data dictionary & source instructions
├── src/
│   ├── __init__.py
│   ├── feature_engineering.py     # 30+ behavioral & transactional features
│   ├── train.py                   # Model training pipeline (XGBoost, LR)
│   ├── validate.py                # Model validation: stability, drift, bias
│   ├── predict.py                 # Real-time risk scoring inference
│   └── explain.py                 # SHAP explainability reports
├── models/
│   └── README.md                  # Model versioning notes
├── notebooks/
│   └── fraud_risk_scoring_full.ipynb   # End-to-end walkthrough notebook
├── reports/
│   └── README.md                  # Auto-generated validation reports land here
├── tests/
│   ├── test_features.py
│   └── test_validate.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Tech Stack

- **Modeling:** XGBoost, Logistic Regression, scikit-learn
- **Explainability:** SHAP
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Validation:** scipy, evidently (drift detection)
- **Notebook:** Jupyter

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Pradeep6249/fraud-risk-scoring.git
cd fraud-risk-scoring
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data & run the full pipeline
```bash
python src/feature_engineering.py   # Build features
python src/train.py                  # Train models
python src/validate.py               # Run validation suite
python src/explain.py                # Generate SHAP report
```

### 4. Or run the full walkthrough notebook
```bash
jupyter notebook notebooks/fraud_risk_scoring_full.ipynb
```

---

## 🧠 Feature Engineering (30+ Features)

Features are grouped into four behavioral categories:

| Category | Examples |
|----------|----------|
| **Velocity** | `tx_count_1h`, `tx_count_24h`, `amt_sum_1h` |
| **Amount Patterns** | `amt_zscore`, `amt_vs_avg_ratio`, `is_round_amount` |
| **Temporal** | `hour_of_day`, `is_weekend`, `days_since_first_tx` |
| **Interaction** | `unique_merchants_7d`, `device_change_flag`, `country_mismatch` |

---

## ✅ Model Validation Suite

The validation pipeline (`src/validate.py`) runs automatically and checks:

- **Stability Testing** — KS statistic on score distributions across time windows
- **Drift Detection** — Population Stability Index (PSI) on feature distributions
- **Bias & Fairness Monitoring** — Disparate impact analysis across demographic slices
- **Performance Metrics** — AUC-ROC, Precision-Recall, F1, confusion matrix
- **SHAP Explainability** — Global feature importance + local prediction explanations

All outputs are saved to `reports/`.

---

## 📈 Model Architecture

Two models are trained and compared:

1. **XGBoost Classifier** — primary production model; handles imbalanced data via `scale_pos_weight`
2. **Logistic Regression** — interpretable baseline; used for regulatory explainability requirements

Final scoring uses XGBoost with SHAP explanations attached per prediction.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Author

**Pradeep Kumar Voruganti**
[LinkedIn](
www.linkedin.com/in/pradeep-kumar-voruganti) | [Portfolio](https://pkv-signal.vercel.app/)
