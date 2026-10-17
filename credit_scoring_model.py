"""
=============================================================================
 CREDIT SCORING MODEL — End-to-End Machine Learning Pipeline
=============================================================================
 Objective  : Predict individual creditworthiness (good / bad credit risk)
 Dataset    : German Credit Data (UCI) or user-supplied CSV
 Author     : Antigravity AI Coding Assistant
 Date       : 2026
=============================================================================
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ─── Data Manipulation ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ─── Visualisation ───────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─── Scikit-learn ────────────────────────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score
)

# ─── Optional: XGBoost / LightGBM ────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# ─── Optional: SHAP ──────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ─── Joblib for model persistence ────────────────────────────────────────────
import joblib


# =============================================================================
# SECTION 1 ─ GLOBAL CONFIG
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE     = 0.20
CV_FOLDS      = 5
OUTPUT_DIR    = "outputs"   # All plots + saved models go here

# Colour palette for consistent charts
PALETTE = {
    "Logistic Regression":   "#4C72B0",
    "Decision Tree":         "#DD8452",
    "Random Forest":         "#55A868",
    "Gradient Boosting":     "#C44E52",
    "XGBoost":               "#8172B2",
    "LightGBM":              "#937860",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# SECTION 2 ─ DATA LOADING
# =============================================================================

def load_german_credit_data() -> pd.DataFrame:
    """
    Load the UCI German Credit dataset.
    Column names follow the standard attribute description from UCI.
    The original target is 1 (good) / 2 (bad); we remap to 0 (good) / 1 (bad).
    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "statlog/german/german.data"
    )
    column_names = [
        "checking_account_status", "duration_months", "credit_history",
        "purpose", "credit_amount", "savings_account", "employment_since",
        "installment_rate", "personal_status_sex", "other_debtors",
        "residence_since", "property", "age", "other_installment_plans",
        "housing", "existing_credits", "job", "dependents", "telephone",
        "foreign_worker", "target"
    ]
    try:
        df = pd.read_csv(url, sep=" ", header=None, names=column_names)
        df["target"] = df["target"].map({1: 0, 2: 1})   # 0=good, 1=bad
        print(f"[✓] Loaded German Credit data from UCI: {df.shape}")
        return df
    except Exception as exc:
        print(f"[!] Could not fetch from URL ({exc}). Generating synthetic data …")
        return generate_synthetic_data(n_samples=1000)


def generate_synthetic_data(n_samples: int = 1000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Create a realistic synthetic credit dataset when the UCI data is unavailable.
    Features mirror common real-world credit bureau attributes.
    """
    rng = np.random.default_rng(seed)

    age              = rng.integers(18, 75, n_samples)
    income           = rng.integers(15_000, 150_000, n_samples)
    loan_amount      = rng.integers(1_000, 50_000, n_samples)
    duration_months  = rng.integers(6, 72, n_samples)
    existing_debts   = rng.integers(0, 5, n_samples)
    payment_history  = rng.choice(["on_time", "late_1-30", "late_30+", "default"],
                                   n_samples, p=[0.60, 0.20, 0.12, 0.08])
    employment_yrs   = rng.uniform(0, 30, n_samples).round(1)
    num_open_accts   = rng.integers(0, 10, n_samples)
    past_defaults    = rng.integers(0, 3, n_samples)
    credit_hist_yrs  = rng.uniform(0, 20, n_samples).round(1)
    housing_status   = rng.choice(["own", "rent", "free"], n_samples, p=[0.50, 0.35, 0.15])
    purpose          = rng.choice(["car", "furniture", "education", "business", "other"],
                                   n_samples)

    # Deterministic-ish target: higher risk for high debt, defaults, short history
    risk_score = (
        (loan_amount / income)
        + (existing_debts * 0.3)
        + (past_defaults * 0.5)
        + (payment_history == "default") * 1.0
        + (payment_history == "late_30+") * 0.5
        - (employment_yrs * 0.02)
        - (credit_hist_yrs * 0.03)
    )
    prob_bad = 1 / (1 + np.exp(-2 * (risk_score - risk_score.mean())))
    target   = rng.binomial(1, prob_bad)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "loan_amount": loan_amount,
        "duration_months": duration_months,
        "existing_debts": existing_debts,
        "payment_history": payment_history,
        "employment_yrs": employment_yrs,
        "num_open_accts": num_open_accts,
        "past_defaults": past_defaults,
        "credit_hist_yrs": credit_hist_yrs,
        "housing_status": housing_status,
        "purpose": purpose,
        "target": target,
    })

    # Inject ~5 % missing values randomly
    for col in ["income", "employment_yrs", "credit_hist_yrs", "num_open_accts"]:
        mask = rng.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan

    print(f"[✓] Generated synthetic credit data: {df.shape}")
    return df


def load_dataset(csv_path: str | None = None) -> pd.DataFrame:
    """
    Entry point for data loading.
    - If csv_path is given, load from that file (expects a 'target' column).
    - Otherwise fetch the UCI German Credit dataset (or fall back to synthetic).
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[✓] Loaded custom CSV: {df.shape}")
        return df
    return load_german_credit_data()


# =============================================================================
# SECTION 3 ─ EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """Print summary statistics and save basic distribution plots."""
    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"\nDataset shape  : {df.shape}")
    print(f"Missing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nTarget distribution:\n{df['target'].value_counts(normalize=True).round(3)}")

    # ── Distribution grid ───────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "target"]
    if not numeric_cols:
        return

    n_cols = 3
    n_rows = -(-len(numeric_cols) // n_cols)      # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")
        axes[i].spines[["top", "right"]].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Correlation heatmap ─────────────────────────────────────────────────
    corr = df[numeric_cols + ["target"]].corr()
    fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols)), max(6, len(numeric_cols) - 1)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, center=0, ax=ax, linewidths=0.5, square=True
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/eda_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[✓] EDA plots saved to '{OUTPUT_DIR}/'")


# =============================================================================
# SECTION 4 ─ PREPROCESSING & FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-derived features for credit risk.
    These are computed from raw columns; the function is forgiving — it only
    creates a feature if the required source columns are present.
    """
    df = df.copy()

    # Debt-to-income ratio
    if {"loan_amount", "income"}.issubset(df.columns):
        df["debt_to_income"] = (df["loan_amount"] / df["income"].replace(0, np.nan)).round(4)

    # Monthly payment load (loan amount spread over tenure)
    if {"loan_amount", "duration_months"}.issubset(df.columns):
        df["monthly_payment_load"] = (
            df["loan_amount"] / df["duration_months"].replace(0, np.nan)
        ).round(2)

    # Credit utilization proxy (existing debts / number of open accounts)
    if {"existing_debts", "num_open_accts"}.issubset(df.columns):
        df["credit_utilization"] = (
            df["existing_debts"] / (df["num_open_accts"] + 1)
        ).round(4)

    # Payment delinquency flag (1 if any late/default history)
    if "payment_history" in df.columns:
        df["delinquency_flag"] = df["payment_history"].apply(
            lambda x: 1 if str(x).lower() in ["late_1-30", "late_30+", "default"] else 0
        )

    # Risk score proxy
    if {"past_defaults", "existing_debts"}.issubset(df.columns):
        df["risk_proxy"] = df["past_defaults"] * 2 + df["existing_debts"]

    print("[✓] Feature engineering complete. New features added where applicable.")
    return df


def build_preprocessor(df: pd.DataFrame, target_col: str = "target"):
    """
    Build a ColumnTransformer that:
      - Imputes + scales numeric columns
      - Imputes + one-hot-encodes categorical columns
    Returns the preprocessor and the (X, y) split.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    numeric_cols     = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"   Numeric features    : {numeric_cols}")
    print(f"   Categorical features: {categorical_cols}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", pd.get_dummies.__class__),        # placeholder — handled below
    ])

    # One-hot encode categoricals manually (sklearn's OHE is more robust)
    from sklearn.preprocessing import OneHotEncoder
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline,     numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ], remainder="drop")

    return preprocessor, X, y


# =============================================================================
# SECTION 5 ─ MODEL DEFINITIONS & HYPERPARAMETER GRIDS
# =============================================================================

def get_models_and_grids() -> dict:
    """
    Returns a dictionary of {model_name: (estimator, param_grid)} pairs.
    XGBoost and LightGBM are added only if available.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "classifier__C":        [0.01, 0.1, 1, 10],
                "classifier__penalty":  ["l1", "l2"],
                "classifier__solver":   ["liblinear", "saga"],
            }
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "classifier__max_depth":        [3, 5, 7, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf":  [1, 2, 4],
                "classifier__criterion":         ["gini", "entropy"],
            }
        ),
        "Random Forest": (
            RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
            {
                "classifier__n_estimators":      [100, 200, 300],
                "classifier__max_depth":         [5, 10, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__max_features":      ["sqrt", "log2"],
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators":  [100, 200],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__max_depth":     [3, 5],
                "classifier__subsample":     [0.8, 1.0],
            }
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
            ),
            {
                "classifier__n_estimators":  [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth":     [3, 5, 7],
                "classifier__subsample":     [0.8, 1.0],
            }
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = (
            LGBMClassifier(
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1
            ),
            {
                "classifier__n_estimators":  [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__num_leaves":    [31, 63],
                "classifier__max_depth":     [-1, 5],
            }
        )

    return models


# =============================================================================
# SECTION 6 ─ TRAINING & CROSS-VALIDATION
# =============================================================================

def train_all_models(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    n_iter_search: int = 20,
) -> dict:
    """
    For every model:
      1. Wrap in a Pipeline (preprocessor + classifier).
      2. Run RandomizedSearchCV for hyperparameter tuning.
      3. Store the best estimator and CV results.
    Returns dict[model_name] -> best_pipeline.
    """
    models_grids = get_models_and_grids()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    trained_models = {}

    print("\n" + "="*60)
    print("  MODEL TRAINING & HYPERPARAMETER TUNING")
    print("="*60)

    for name, (estimator, param_grid) in models_grids.items():
        print(f"\n[→] Training: {name} …")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   estimator),
        ])

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=min(n_iter_search, _count_param_combinations(param_grid)),
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
            return_train_score=True,
        )
        search.fit(X_train, y_train)
        trained_models[name] = search.best_estimator_

        print(f"   Best params : {_format_params(search.best_params_)}")
        print(f"   CV ROC-AUC  : {search.best_score_:.4f}")

    print("\n[✓] All models trained.")
    return trained_models


def _count_param_combinations(param_grid: dict) -> int:
    """Count total combinations in a param grid (for capping n_iter)."""
    from math import prod
    return prod(len(v) for v in param_grid.values())


def _format_params(params: dict) -> str:
    """Pretty-print param dict by stripping 'classifier__' prefix."""
    return str({k.replace("classifier__", ""): v for k, v in params.items()})


# =============================================================================
# SECTION 7 ─ EVALUATION
# =============================================================================

def evaluate_models(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all trained models on the held-out test set.
    Returns a summary DataFrame and saves:
      - One confusion matrix per model
      - Combined ROC curve chart
    """
    print("\n" + "="*60)
    print("  MODEL EVALUATION ON TEST SET")
    print("="*60)

    results = []
    roc_data = {}

    for name, model in trained_models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        prec   = precision_score(y_test, y_pred, zero_division=0)
        rec    = recall_score(y_test, y_pred, zero_division=0)
        f1     = f1_score(y_test, y_pred, zero_division=0)
        auc    = roc_auc_score(y_test, y_proba)
        acc    = accuracy_score(y_test, y_pred)

        results.append({
            "Model":     name,
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1-Score":  round(f1,   4),
            "ROC-AUC":   round(auc,  4),
        })
        roc_data[name] = (y_proba,)

        # ── Confusion matrix ────────────────────────────────────────────────
        cm  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"],
            linewidths=1, linecolor="white"
        )
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual",    fontweight="bold")
        ax.set_title(f"Confusion Matrix — {name}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fname = name.lower().replace(" ", "_")
        plt.savefig(f"{OUTPUT_DIR}/cm_{fname}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── Classification report ────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        print(classification_report(y_test, y_pred, target_names=["Good", "Bad"]))

    # ── Combined ROC curve ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.50)")

    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score   = roc_auc_score(y_test, y_proba)
        colour      = PALETTE.get(name, None)
        ax.plot(fpr, tpr, lw=2, color=colour, label=f"{name} (AUC = {auc_score:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Summary table ────────────────────────────────────────────────────────
    summary = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    summary.index += 1
    print("\n" + "="*60)
    print("  PERFORMANCE SUMMARY (ranked by ROC-AUC)")
    print("="*60)
    print(summary.to_string())
    print(f"\n[✓] Confusion matrices + ROC curve saved to '{OUTPUT_DIR}/'")

    return summary


# =============================================================================
# SECTION 8 ─ FEATURE IMPORTANCE & INTERPRETABILITY
# =============================================================================

def plot_feature_importance(trained_models: dict, preprocessor, X_train: pd.DataFrame) -> None:
    """
    Plot feature importance for tree-based models and coefficients for LR.
    Also attempts SHAP explanation for the best tree-based model.
    """
    print("\n" + "="*60)
    print("  FEATURE IMPORTANCE & INTERPRETABILITY")
    print("="*60)

    # Reconstruct feature names after preprocessing
    feature_names = _get_feature_names(preprocessor, X_train)

    tree_based = ["Random Forest", "Decision Tree", "Gradient Boosting",
                  "XGBoost", "LightGBM"]
    linear     = ["Logistic Regression"]

    for name, model in trained_models.items():
        classifier = model.named_steps["classifier"]

        if name in tree_based and hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            _plot_importances(name, feature_names, importances, top_n=15)

        elif name in linear and hasattr(classifier, "coef_"):
            coef = classifier.coef_[0]
            _plot_importances(name, feature_names, np.abs(coef), top_n=15,
                              label="Coefficient Magnitude")

    # ── SHAP explanation for the best available tree model ──────────────────
    if SHAP_AVAILABLE:
        _explain_with_shap(trained_models, preprocessor, X_train, feature_names)
    else:
        print("\n[i] SHAP not installed — skipping SHAP explanation.")
        print("    Install with: pip install shap")


def _get_feature_names(preprocessor, X: pd.DataFrame) -> list[str]:
    """Extract output feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: use raw column names
        return list(X.columns)


def _plot_importances(
    model_name: str,
    feature_names: list[str],
    scores: np.ndarray,
    top_n: int = 15,
    label: str = "Feature Importance",
) -> None:
    """Bar chart of the top-N most important features."""
    if len(feature_names) != len(scores):
        # Trim / pad to match
        n = min(len(feature_names), len(scores))
        feature_names = feature_names[:n]
        scores        = scores[:n]

    idx  = np.argsort(scores)[-top_n:]
    names = np.array(feature_names)[idx]
    vals  = scores[idx]

    colour = PALETTE.get(model_name, "#4C72B0")
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    bars = ax.barh(names, vals, color=colour, edgecolor="white", alpha=0.88)

    ax.set_xlabel(label, fontsize=10)
    ax.set_title(f"Top {top_n} Features — {model_name}", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()

    fname = model_name.lower().replace(" ", "_")
    plt.savefig(f"{OUTPUT_DIR}/importance_{fname}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [{model_name}] feature importance plot saved.")


def _explain_with_shap(
    trained_models: dict,
    preprocessor,
    X_train: pd.DataFrame,
    feature_names: list[str],
    n_samples: int = 200,
) -> None:
    """
    Compute SHAP values using the modern shap.Explainer API (compatible with
    SHAP >= 0.40) and save two plots:
      - shap_bar.png      : Mean |SHAP| per feature (global importance bar chart)
      - shap_beeswarm.png : Per-sample dot plot showing direction + magnitude
    """
    priority = ["Random Forest", "Gradient Boosting", "XGBoost",
                "LightGBM", "Decision Tree"]

    chosen_name  = None
    chosen_model = None
    for pname in priority:
        if pname in trained_models:
            chosen_name  = pname
            chosen_model = trained_models[pname]
            break

    if chosen_model is None:
        return

    print(f"\n[-] SHAP explanation using: {chosen_name}")

    pre = chosen_model.named_steps["preprocessor"]
    clf = chosen_model.named_steps["classifier"]

    X_sample      = X_train.sample(min(n_samples, len(X_train)), random_state=RANDOM_STATE)
    X_transformed = pre.transform(X_sample)

    # Align feature names to the transformed matrix width
    n_feats       = X_transformed.shape[1]
    aligned_names = (feature_names + [f"feat_{i}" for i in range(n_feats)])[:n_feats]

    try:
        # Use the modern unified Explainer (works with RF, GB, XGB, LGBM, DT)
        explainer  = shap.Explainer(clf, X_transformed, feature_names=aligned_names)
        shap_exp   = explainer(X_transformed, check_additivity=False)

        # For multi-output Explanation objects take the "bad credit" class (index 1)
        if shap_exp.values.ndim == 3:
            shap_exp = shap_exp[:, :, 1]

        # ── Bar chart: mean absolute SHAP value per feature ────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.bar(shap_exp, max_display=15, show=False, ax=ax)
        ax.set_title(f"SHAP Feature Importance (Mean |SHAP|) - {chosen_name}",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   [OK] SHAP bar chart saved to '{OUTPUT_DIR}/shap_bar.png'")

        # ── Beeswarm: direction + magnitude for each sample ────────────────
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.beeswarm(shap_exp, max_display=15, show=False)
        plt.title(f"SHAP Beeswarm - {chosen_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   [OK] SHAP beeswarm saved to '{OUTPUT_DIR}/shap_beeswarm.png'")

    except Exception as exc:
        print(f"   [!] SHAP (modern API) failed ({exc}), falling back to legacy ...")
        try:
            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_transformed)
            shap_vals   = shap_values[1] if isinstance(shap_values, list) else shap_values

            # Manual bar chart from mean absolute values
            mean_abs = np.abs(shap_vals).mean(axis=0)
            top_idx  = np.argsort(mean_abs)[-15:]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(np.array(aligned_names)[top_idx], mean_abs[top_idx],
                    color="#4C72B0", edgecolor="white", alpha=0.88)
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"SHAP Feature Importance - {chosen_name}",
                         fontsize=12, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/shap_bar.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"   [OK] SHAP bar chart (legacy) saved to '{OUTPUT_DIR}/shap_bar.png'")
        except Exception as exc2:
            print(f"   [!] SHAP legacy fallback also failed: {exc2}")


# =============================================================================
# SECTION 9 ─ MODEL PERSISTENCE
# =============================================================================

def save_best_model(trained_models: dict, summary: pd.DataFrame) -> None:
    """Save the best model (by ROC-AUC) to disk using joblib."""
    best_name = summary.iloc[0]["Model"]
    best_model = trained_models[best_name]
    path = f"{OUTPUT_DIR}/best_model_{best_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(best_model, path)
    print(f"\n[✓] Best model '{best_name}' saved to '{path}'")


# =============================================================================
# SECTION 10 ─ MAIN PIPELINE
# =============================================================================

def run_pipeline(csv_path: str | None = None) -> None:
    """
    Orchestrates the full pipeline end-to-end.
    Pass csv_path to use your own dataset instead of UCI German Credit.
    """
    print("\n" + "#"*60)
    print("  CREDIT SCORING MODEL - PIPELINE START")
    print("#"*60)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    df = load_dataset(csv_path)

    # ── 2. EDA ───────────────────────────────────────────────────────────────
    run_eda(df)

    # ── 3. Feature engineering ───────────────────────────────────────────────
    df = engineer_features(df)

    # ── 4. Build preprocessor & split ────────────────────────────────────────
    print("\n[-] Building preprocessor ...")
    preprocessor, X, y = build_preprocessor(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"   Train size: {X_train.shape} | Test size: {X_test.shape}")
    print(f"   Class balance (train) — Bad: {y_train.mean():.2%}")

    # ── 5. Train all models ───────────────────────────────────────────────────
    trained_models = train_all_models(preprocessor, X_train, y_train)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    summary = evaluate_models(trained_models, X_test, y_test)

    # ── 7. Feature importance & SHAP ─────────────────────────────────────────
    plot_feature_importance(trained_models, preprocessor, X_train)

    # ── 8. Save best model ───────────────────────────────────────────────────
    save_best_model(trained_models, summary)

    print("\n" + "#"*60)
    print("  PIPELINE COMPLETE - check 'outputs/' for all artefacts")
    print("#"*60)
    return summary, trained_models


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Optionally pass a CSV path as command-line argument
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(csv_file)
