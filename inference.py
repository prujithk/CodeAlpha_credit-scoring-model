"""
=============================================================================
 inference.py  -  Predict credit risk for new loan applicants
=============================================================================
 Usage:
   python -X utf8 inference.py                    <- uses built-in demo applicants
   python -X utf8 inference.py my_applicants.csv  <- batch predict from CSV

 The script loads the saved best model from outputs/ and prints:
   - Risk label  : GOOD / BAD
   - Probability : P(Bad Credit)
   - Risk tier   : Low / Medium / High / Very High
=============================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
import glob


# =============================================================================
# HELPERS
# =============================================================================

def find_best_model(output_dir: str = "outputs") -> str:
    """Locate the saved .pkl model in the outputs directory."""
    pattern = os.path.join(output_dir, "best_model_*.pkl")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No model found in '{output_dir}/'. "
            "Run credit_scoring_model.py first to train and save a model."
        )
    # If multiple exist, pick the most recently modified
    return max(files, key=os.path.getmtime)


def load_model(model_path: str):
    """Load a joblib-serialized sklearn Pipeline."""
    print(f"[OK] Loading model: {model_path}")
    return joblib.load(model_path)


def risk_tier(probability: float) -> str:
    """Map raw probability to a human-readable risk tier."""
    if probability < 0.25:
        return "Low Risk"
    elif probability < 0.50:
        return "Medium Risk"
    elif probability < 0.75:
        return "High Risk"
    else:
        return "Very High Risk"


def predict(model, applicants: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on a DataFrame of applicants.
    Returns the input DataFrame augmented with prediction columns.
    """
    predictions  = model.predict(applicants)
    probabilities = model.predict_proba(applicants)[:, 1]

    results = applicants.copy()
    results["P(Bad Credit)"] = probabilities.round(4)
    results["Risk Label"]    = ["BAD" if p == 1 else "GOOD" for p in predictions]
    results["Risk Tier"]     = [risk_tier(p) for p in probabilities]
    return results


def print_report(results: pd.DataFrame) -> None:
    """Pretty-print a prediction report to the console."""
    print("\n" + "="*70)
    print("  CREDIT RISK PREDICTION REPORT")
    print("="*70)

    for i, row in results.iterrows():
        prob  = row["P(Bad Credit)"]
        label = row["Risk Label"]
        tier  = row["Risk Tier"]

        # Choose visual indicator
        indicator = "[!!]" if label == "BAD" else "[ v]"

        print(f"\n  Applicant #{i + 1}  {indicator}  {label} CREDIT  |  {tier}")
        print(f"  P(Bad Credit) = {prob:.1%}")
        print("  " + "-"*50)

        # Print key features (exclude prediction columns)
        feature_cols = [c for c in row.index if c not in
                        ["P(Bad Credit)", "Risk Label", "Risk Tier"]]
        for col in feature_cols[:8]:   # show first 8 features
            print(f"    {col:<30s}: {row[col]}")
        if len(feature_cols) > 8:
            print(f"    ... (+{len(feature_cols) - 8} more features)")

    print("\n" + "="*70)

    # Summary statistics
    n_bad  = (results["Risk Label"] == "BAD").sum()
    n_good = (results["Risk Label"] == "GOOD").sum()
    print(f"\n  Summary: {n_good} GOOD  |  {n_bad} BAD  |  {len(results)} total applicants")
    avg_prob = results["P(Bad Credit)"].mean()
    print(f"  Average P(Bad Credit): {avg_prob:.1%}")
    print("="*70 + "\n")


# =============================================================================
# DEMO APPLICANTS  (German Credit format)
# =============================================================================

def get_demo_applicants() -> pd.DataFrame:
    """
    Three realistic demo applicants in German Credit Data format.
    A1x = checking account status codes, A6x = savings, etc.
    Reference: UCI German Credit Data attribute description.
    """
    return pd.DataFrame([
        {
            # --- Low-risk applicant ---
            "checking_account_status": "A14",   # no checking account
            "duration_months":          12,
            "credit_history":           "A34",   # no credits taken / all paid back
            "purpose":                  "A42",   # furniture/equipment
            "credit_amount":            2000,
            "savings_account":          "A65",   # unknown/no savings
            "employment_since":         "A75",   # >= 7 years employed
            "installment_rate":         2,
            "personal_status_sex":      "A93",   # male: single
            "other_debtors":            "A101",  # none
            "residence_since":          4,
            "property":                 "A121",  # real estate
            "age":                      45,
            "other_installment_plans":  "A143",  # none
            "housing":                  "A152",  # own
            "existing_credits":         1,
            "job":                      "A173",  # skilled employee
            "dependents":               1,
            "telephone":                "A192",  # yes
            "foreign_worker":           "A201",  # yes
        },
        {
            # --- Medium-risk applicant ---
            "checking_account_status": "A11",   # < 0 DM
            "duration_months":          36,
            "credit_history":           "A32",   # existing credits paid back duly
            "purpose":                  "A40",   # car (new)
            "credit_amount":            8500,
            "savings_account":          "A61",   # < 100 DM
            "employment_since":         "A72",   # 1 <= ... < 4 years
            "installment_rate":         3,
            "personal_status_sex":      "A92",   # female: divorced/sep/married
            "other_debtors":            "A101",
            "residence_since":          2,
            "property":                 "A123",  # car/other
            "age":                      28,
            "other_installment_plans":  "A141",  # bank
            "housing":                  "A151",  # rent
            "existing_credits":         2,
            "job":                      "A172",  # unskilled - resident
            "dependents":               2,
            "telephone":                "A191",  # none
            "foreign_worker":           "A201",
        },
        {
            # --- High-risk applicant ---
            "checking_account_status": "A11",   # < 0 DM
            "duration_months":          60,
            "credit_history":           "A30",   # no credits taken / all paid back
            "purpose":                  "A49",   # retraining
            "credit_amount":            18000,
            "savings_account":          "A61",   # < 100 DM
            "employment_since":         "A71",   # unemployed
            "installment_rate":         4,
            "personal_status_sex":      "A93",
            "other_debtors":            "A102",  # co-applicant
            "residence_since":          1,
            "property":                 "A124",  # unknown / no property
            "age":                      22,
            "other_installment_plans":  "A141",  # bank
            "housing":                  "A151",  # rent
            "existing_credits":         3,
            "job":                      "A171",  # unemployed / non-resident
            "dependents":               2,
            "telephone":                "A191",
            "foreign_worker":           "A201",
        },
    ])


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "#"*70)
    print("  CREDIT RISK INFERENCE ENGINE")
    print("#"*70)

    # 1. Load model
    model_path = find_best_model(output_dir="outputs")
    model      = load_model(model_path)

    # 2. Load applicant data
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        csv_path = sys.argv[1]
        applicants = pd.read_csv(csv_path)
        # Drop target column if present
        if "target" in applicants.columns:
            applicants = applicants.drop(columns=["target"])
        print(f"[OK] Loaded {len(applicants)} applicant(s) from: {csv_path}")
    else:
        print("[i]  No CSV provided — using 3 built-in demo applicants.")
        applicants = get_demo_applicants()

    # 3. Predict
    results = predict(model, applicants)

    # 4. Print report
    print_report(results)

    # 5. Save results
    out_path = "outputs/predictions.csv"
    results.to_csv(out_path, index=False)
    print(f"[OK] Predictions saved to: {out_path}\n")


if __name__ == "__main__":
    main()
