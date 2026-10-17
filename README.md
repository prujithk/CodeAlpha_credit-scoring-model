# 🏦 Credit Scoring Model

A complete, modular **end-to-end Machine Learning pipeline** to predict individual creditworthiness (Good / Bad credit risk) using historical financial data.

---

## 📁 Project Structure

```
credit_scoring/
├── credit_scoring_model.py        # Main pipeline script (importable)
├── inference.py                   # Predict on new applicants
├── report.md                      # Full project report with results
├── requirements.txt               # Python dependencies
└── outputs/                       # Auto-generated on run
    ├── eda_*.png                  # EDA charts
    ├── roc_curves.png             # Combined ROC curve (all models)
    ├── confusion_matrices_all.png # All confusion matrices
    ├── performance_heatmap.png    # Metric heatmap
    ├── importance_*.png           # Feature importance per model
    ├── shap_bar.png               # SHAP global importance
    ├── shap_beeswarm.png          # SHAP beeswarm (direction + magnitude)
    └── best_model_*.pkl           # Serialized best model
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
# Uses UCI German Credit dataset (auto-downloaded)
python -X utf8 credit_scoring_model.py

# Use your own CSV file
python -X utf8 credit_scoring_model.py path/to/your/data.csv
```

### 3. Predict on new applicants
```bash
python -X utf8 inference.py
```

### 4. Interactive Jupyter notebook
```bash
jupyter notebook credit_scoring_notebook.ipynb
```

---

## 🧩 Pipeline Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | **Data Loading** | UCI German Credit (auto-fetch) or custom CSV |
| 2 | **EDA** | Distributions, correlation heatmap, class balance |
| 3 | **Feature Engineering** | 5 domain-derived features (DTI, delinquency flag, etc.) |
| 4 | **Preprocessing** | Imputation + scaling + one-hot encoding via sklearn Pipeline |
| 5 | **Train/Test Split** | 80/20 stratified split |
| 6 | **Model Training** | 6 classifiers × RandomizedSearchCV (ROC-AUC scoring) |
| 7 | **Evaluation** | Precision / Recall / F1 / ROC-AUC + confusion matrices |
| 8 | **Interpretability** | Feature importance + SHAP TreeExplainer |
| 9 | **Model Saving** | Best model serialized via `joblib` |

---

## 📊 Results (UCI German Credit Dataset)

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 🥇 1 | **Logistic Regression** | 0.790 | 0.714 | 0.500 | 0.588 | **0.810** |
| 🥈 2 | Random Forest | 0.770 | 0.694 | 0.417 | 0.521 | 0.804 |
| 🥉 3 | Gradient Boosting | 0.780 | 0.667 | 0.533 | 0.593 | 0.801 |
| 4 | XGBoost | 0.740 | 0.600 | 0.400 | 0.480 | 0.782 |
| 5 | LightGBM | 0.730 | 0.568 | 0.417 | 0.481 | 0.764 |
| 6 | Decision Tree | 0.670 | 0.452 | 0.467 | 0.459 | 0.668 |

### Top Predictive Features
1. `duration_months` — loan tenure (0.103 importance)
2. `checking_account_status` — account balance category
3. `residence_since` — stability of residence
4. `credit_amount` — loan size
5. `installment_rate` — repayment rate as % of disposable income

---

## 🔌 Using as a Module

```python
from credit_scoring_model import run_pipeline, load_dataset, engineer_features

# Run full pipeline on your own data
summary, trained_models = run_pipeline("my_data.csv")

# Or use individual components
df = load_dataset("my_data.csv")
df = engineer_features(df)
```

---

## 🔮 Predict on New Applicants

```python
import joblib, pandas as pd

model = joblib.load("outputs/best_model_logistic_regression.pkl")

applicant = pd.DataFrame([{
    "checking_account_status": "A11",
    "duration_months": 24,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 5000,
    "savings_account": "A61",
    "employment_since": "A73",
    "installment_rate": 3,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "residence_since": 2,
    "property": "A121",
    "age": 35,
    "other_installment_plans": "A143",
    "housing": "A152",
    "existing_credits": 1,
    "job": "A173",
    "dependents": 1,
    "telephone": "A191",
    "foreign_worker": "A201",
}])

prediction  = model.predict(applicant)[0]
probability = model.predict_proba(applicant)[0][1]

print("Risk:", "BAD" if prediction == 1 else "GOOD")
print(f"P(Bad Credit): {probability:.1%}")
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥ 1.5 | Data manipulation |
| numpy | ≥ 1.23 | Numerical computing |
| scikit-learn | ≥ 1.2 | ML pipeline, models, metrics |
| matplotlib | ≥ 3.6 | Plotting |
| seaborn | ≥ 0.12 | Statistical charts |
| xgboost | ≥ 1.7 | XGBoost classifier |
| lightgbm | ≥ 3.3 | LightGBM classifier |
| shap | ≥ 0.41 | Model explainability |
| joblib | ≥ 1.2 | Model serialization |

---

## 📝 Notes

- **Windows users**: run with `python -X utf8 credit_scoring_model.py` to avoid encoding issues
- **Custom CSV**: must include a `target` column (0 = good, 1 = bad)
- **XGBoost / LightGBM / SHAP** are optional — the pipeline skips them gracefully if not installed
- **Reproducibility**: all random seeds fixed at `RANDOM_STATE = 42`
