# Models Directory

Trained model artifacts are saved here after running `src/train.py`.
Model `.pkl` files are excluded from git (see `.gitignore`).

## Files (auto-generated)

| File | Description |
|------|-------------|
| `xgboost_fraud_model.pkl` | Primary XGBoost classifier |
| `logistic_regression_fraud_model.pkl` | Interpretable LR baseline |

## Regenerating models

```bash
python src/train.py
```

## Model Details

### XGBoost Classifier (Primary)
- `n_estimators`: 300
- `max_depth`: 5
- `learning_rate`: 0.05
- `scale_pos_weight`: auto-computed from class imbalance ratio
- Handles ~3% fraud rate via weighted training

### Logistic Regression (Baseline)
- `class_weight`: balanced
- `C`: 0.1 (regularized)
- Wrapped in StandardScaler pipeline
- Used for regulatory explainability requirements
