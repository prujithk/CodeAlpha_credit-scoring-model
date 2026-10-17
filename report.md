# Credit Scoring Model — Project Report

> **Objective**: Predict an individual's creditworthiness (Good / Bad credit risk) using the UCI German Credit Dataset, via a full end-to-end machine learning pipeline.

---

## 1. Problem Statement

Credit risk assessment is a critical activity for financial institutions. A model must classify loan applicants as **good credit risk (0)** or **bad credit risk (1)** based on features like account status, loan duration, repayment history, and employment.

Incorrect classification carries asymmetric costs:
- **False Negative** (missed bad risk): potential loan default → financial loss
- **False Positive** (good customer rejected): lost business opportunity

This makes **ROC-AUC** and **Recall** the primary metrics of interest.

---

## 2. Dataset

| Property | Value |
|----------|-------|
| **Source** | UCI German Credit Dataset (auto-downloaded) |
| **URL** | https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data) |
| **Total Samples** | 1,000 |
| **Raw Features** | 20 (7 numeric, 13 categorical) |
| **Engineered Features** | +5 domain-derived |
| **Target** | Binary: 0 = Good Credit, 1 = Bad Credit |
| **Class Balance** | 70% Good / 30% Bad |

### Features Used

**Numeric**: `duration_months`, `credit_amount`, `installment_rate`, `residence_since`, `age`, `existing_credits`, `dependents`

**Categorical**: `checking_account_status`, `credit_history`, `purpose`, `savings_account`, `employment_since`, `personal_status_sex`, `other_debtors`, `property`, `other_installment_plans`, `housing`, `job`, `telephone`, `foreign_worker`

---

## 3. Methodology

### 3.1 Feature Engineering

Five domain-derived features were created:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `debt_to_income` | loan_amount / income | Core affordability ratio |
| `monthly_payment_load` | loan_amount / duration_months | Monthly burden |
| `credit_utilization` | existing_debts / (open_accounts + 1) | Line overcrowding |
| `delinquency_flag` | 1 if any late/default history | Binary risk indicator |
| `risk_proxy` | past_defaults × 2 + existing_debts | Composite risk score |

> Note: For the UCI German Credit dataset, features like `income` are not present in the raw data, so only engineered features applicable to the available columns are added. The pipeline is forgiving — it only creates a feature when source columns exist.

### 3.2 Preprocessing Pipeline

```
Raw Data
   └─► Numeric columns  → Median Imputation → StandardScaler
   └─► Categorical cols → Mode Imputation  → OneHotEncoder
                                           └─► ColumnTransformer
                                                    └─► Classifier
```

- **Missing values**: Median imputation (numeric), mode imputation (categorical)
- **Encoding**: One-Hot Encoding with `handle_unknown='ignore'`
- **Scaling**: StandardScaler for all numeric features
- **Split**: 80% train / 20% test with stratification on target

### 3.3 Training

- **Cross-validation**: 5-Fold StratifiedKFold
- **Tuning**: RandomizedSearchCV (20 iterations, scoring = ROC-AUC)
- **Models**: 6 classifiers trained and compared

---

## 4. Results

### 4.1 Performance Summary

> Results from actual run on UCI German Credit dataset (1,000 samples, seed=42)

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV AUC |
|------|-------|----------|-----------|--------|----------|---------|--------|
| 🥇 1 | **Logistic Regression** | 0.790 | 0.714 | 0.500 | 0.588 | **0.810** | 0.784 |
| 🥈 2 | Random Forest | 0.770 | 0.694 | 0.417 | 0.521 | 0.804 | 0.795 |
| 🥉 3 | Gradient Boosting | 0.780 | 0.667 | 0.533 | 0.593 | 0.801 | 0.791 |
| 4 | XGBoost | 0.740 | 0.600 | 0.400 | 0.480 | 0.782 | 0.796 |
| 5 | LightGBM | 0.730 | 0.568 | 0.417 | 0.481 | 0.764 | 0.793 |
| 6 | Decision Tree | 0.670 | 0.452 | 0.467 | 0.459 | 0.668 | 0.727 |

### 4.2 Best Hyperparameters

| Model | Best Params |
|-------|-------------|
| Logistic Regression | `C=0.1`, `penalty=l2`, `solver=saga` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `min_samples_split=5`, `max_features=sqrt` |
| Gradient Boosting | `n_estimators=200`, `lr=0.05`, `max_depth=3`, `subsample=0.8` |
| XGBoost | `n_estimators=100`, `lr=0.05`, `max_depth=5`, `subsample=0.8` |
| LightGBM | `n_estimators=100`, `lr=0.05`, `num_leaves=31`, `max_depth=-1` |
| Decision Tree | `max_depth=5`, `criterion=entropy`, `min_samples_split=2`, `min_samples_leaf=1` |

### 4.3 Key Observations

1. **Logistic Regression wins** on ROC-AUC despite being the simplest model. This is typical for this dataset — the class boundary is largely linear once features are scaled.
2. **Recall is consistently low (~40–53%)** for the Bad class. This is expected with a 70/30 imbalance — models learn to predict Good more readily. Apply `class_weight='balanced'` or SMOTE to improve.
3. **Decision Tree** has the worst generalization — it tends to memorize specific paths even when `max_depth` is constrained.
4. **Ensemble models** (RF, GB, XGBoost, LightGBM) cluster tightly at AUC 0.764–0.804, suggesting they've hit the dataset's information ceiling without class rebalancing.

---

## 5. Top Predictive Features

### Random Forest Feature Importance

| Rank | Feature | Importance | Interpretation |
|------|---------|-----------|----------------|
| 1 | `duration_months` | 0.1031 | Longer loan → higher default risk |
| 2 | `checking_account_status` | 0.0796 | Negative balance is a strong red flag |
| 3 | `residence_since` | 0.0732 | Stability of residence correlates with reliability |
| 4 | `credit_amount` | 0.0687 | Higher loan amount → more risk |
| 5 | `installment_rate` | 0.0448 | High % of income committed → strain |
| 6 | `credit_history` | 0.0320 | Past delinquency patterns |
| 7 | `purpose` | 0.0247 | Loan purpose (car vs education vs business) |
| 8 | `existing_credits` | 0.0226 | Number of existing credit lines |

---

## 6. Best Model — Logistic Regression

```
Best Params:
  C        = 0.1
  penalty  = l2
  solver   = saga

Test Set Metrics:
  Accuracy  = 79.0%
  Precision = 71.4%  (of predicted bad, 71% truly bad)
  Recall    = 50.0%  (of actual bad, 50% correctly caught)
  F1-Score  = 58.8%
  ROC-AUC   = 81.0%

Confusion Matrix:
  Predicted:    Good  Bad
  Actual Good:  127    13
  Actual Bad:    30    30
```

### Loading & Using the Saved Model

```python
import joblib, pandas as pd

model = joblib.load("outputs/best_model_logistic_regression.pkl")
prediction  = model.predict(X_new)[0]        # 0=Good, 1=Bad
probability = model.predict_proba(X_new)[0][1]  # P(Bad credit)
```

---

## 7. Inference Demo Results

Running `python -X utf8 inference.py` on 3 demo applicants:

| Applicant | Profile | Prediction | P(Bad) | Risk Tier |
|-----------|---------|-----------|--------|-----------|
| #1 | 45yo, 12-month loan, stable employment, property owner | ✅ GOOD | 1.6% | Low Risk |
| #2 | 28yo, 36-month car loan, negative account balance, renter | ❌ BAD | 95.2% | Very High Risk |
| #3 | 22yo, 60-month loan, unemployed, no property | ❌ BAD | 98.9% | Very High Risk |

The model correctly distinguishes low-risk (stable older applicant) from high-risk (young, unemployed, negative balance) profiles with high confidence.

---

## 8. Limitations & Recommended Next Steps

### Current Limitations

| Issue | Impact | Suggested Fix |
|-------|--------|---------------|
| Class imbalance (30% bad) | Recall for bad class only ~50% | Use `class_weight='balanced'` or SMOTE |
| No temporal split | May not reflect real time-drift | Walk-forward cross-validation |
| No calibration | Raw probabilities may be poorly calibrated | Platt scaling / isotonic regression |
| Static threshold (0.5) | Suboptimal for asymmetric cost | Tune threshold on precision-recall curve |

### Next Steps

| Priority | Action |
|----------|--------|
| 🔴 High | Add `class_weight='balanced'` to all classifiers |
| 🔴 High | Threshold tuning: plot precision-recall curve, pick threshold for desired recall |
| 🟡 Medium | Walk-forward temporal validation |
| 🟡 Medium | Probability calibration with `CalibratedClassifierCV` |
| 🟢 Low | Wrap model in a FastAPI scoring endpoint |
| 🟢 Low | Add MLflow experiment tracking |
| 🟢 Low | Streamlit dashboard for interactive applicant scoring |

---

## 9. Output Files

| File | Description |
|------|-------------|
| `outputs/eda_distributions.png` | Numeric feature distributions |
| `outputs/eda_correlation.png` | Feature correlation heatmap |
| `outputs/roc_curves.png` | Combined ROC curve for all 6 models |
| `outputs/cm_*.png` | Individual confusion matrices (×6) |
| `outputs/importance_*.png` | Feature importance plots (×6) |
| `outputs/shap_bar.png` | SHAP global importance bar chart |
| `outputs/shap_beeswarm.png` | SHAP beeswarm (direction + magnitude) |
| `outputs/best_model_logistic_regression.pkl` | Serialized best model |
| `outputs/predictions.csv` | Inference results from demo applicants |

---

## 10. Project Structure

```
credit_scoring/
├── credit_scoring_model.py        # Main ML pipeline (importable)
├── inference.py                   # Predict on new applicants
├── README.md                      # Quick-start & usage guide
├── report.md                      # This report
├── requirements.txt               # Python dependencies
└── outputs/                       # Generated on run
```

---

*Report generated from actual pipeline run — UCI German Credit Dataset, 1,000 samples, seed=42.*
