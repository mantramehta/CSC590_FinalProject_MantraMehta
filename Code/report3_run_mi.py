"""
CSC 590 – Report 3 runner (Mutual Information feature selection)

Runs Logistic Regression, RandomForest, and XGBoost on a given dataset file.
For high-dimensional "challenge" datasets (10,000 features), it:

  - applies Mutual Information (SelectKBest) to select top K features
  - evaluates models with 5-fold CV ROC-AUC
  - saves:
      * metrics_<dataset>.csv
      * roc_<dataset>.png  (overlay of all three models)
      * featimp_RandomForest_<dataset>.png
      * featimp_XGBoost_<dataset>.png
      * mi_scores_<dataset>.csv (MI scores for all features, if FS used)

Usage (from the folder where the .txt files are):
    python report3_run_mi.py 4-wayAdditive_100feat.txt
    python report3_run_mi.py 2-wayEpi_10000feat_with_NA.txt
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# XGBoost (already installed on your system)
try:
    from xgboost import XGBClassifier
except Exception as e:
    print("XGBoost not installed or misconfigured.")
    print("Try: pip install xgboost")
    raise e

RANDOM_STATE = 42


def make_prep(k: int | None):
    """
    Build the preprocessing pipeline:
      - impute missing values
      - (optional) Mutual Information feature selection
      - scale features (helps Logistic Regression; harmless for trees)
    """
    steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]

    if k is not None:
        steps.append(("fs", SelectKBest(score_func=mutual_info_classif, k=k)))

    steps.append(("scaler", StandardScaler(with_mean=False)))
    return Pipeline(steps)


def build_models():
    """Same three models as in Report 2."""
    return {
        "LogReg": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            max_depth=None,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
    }


def run_one(dataset_path: str):
    base = os.path.splitext(os.path.basename(dataset_path))[0]
    os.makedirs("results", exist_ok=True)

    # --- Load data
    df = pd.read_csv(dataset_path, sep="\t")
    X = df.drop(columns=["Class"])
    y = df["Class"]

    n_features = X.shape[1]

    # For challenge datasets (10,000 features), select top 200 by MI.
    # For basic datasets (100 features), we keep all features (no FS).
    if n_features > 1000:
        k = 200
        print(f"[{base}] High-dimensional dataset detected ({n_features} features).")
        print(f"Using Mutual Information to select top {k} features.")
    else:
        k = None
        print(f"[{base}] Basic dataset detected ({n_features} features).")
        print("No feature selection applied (all features used).")

    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []

    # --- Prepare ROC overlay figure (dedicated fig/axes)
    fig_roc, ax_roc = plt.subplots()

    mi_scores_saved = False

    for name, clf in models.items():
        prep = make_prep(k=k)
        pipe = Pipeline([
            ("prep", prep),
            ("clf", clf),
        ])

        # --- Cross-validated ROC-AUC on train
        auc_cv = cross_val_score(
            pipe, Xtr, ytr,
            cv=cv,
            scoring="roc_auc"
        )

        # --- Fit on full train and evaluate on test
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        ypr = pipe.predict_proba(Xte)[:, 1]

        acc = accuracy_score(yte, yhat)
        f1 = f1_score(yte, yhat)
        auc = roc_auc_score(yte, ypr)

        print(
            f"[{base}] {name} | CV AUC: {auc_cv.mean():.3f}±{auc_cv.std():.3f} "
            f"| Test Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}"
        )

        rows.append({
            "dataset": base,
            "model": name,
            "cv_auc_mean": round(float(auc_cv.mean()), 4),
            "cv_auc_std":  round(float(auc_cv.std()), 4),
            "test_acc":    round(float(acc), 4),
            "test_f1":     round(float(f1), 4),
            "test_auc":    round(float(auc), 4),
        })

        # --- Add curve to *shared* ROC overlay fig
        RocCurveDisplay.from_predictions(
            yte, ypr, name=f"{name} (AUC = {auc:.2f})", ax=ax_roc
        )

        # --- Save MI scores once for challenge datasets
        prep_fitted = pipe.named_steps["prep"]
        if (k is not None) and (not mi_scores_saved) and ("fs" in prep_fitted.named_steps):
            fs_step: SelectKBest = prep_fitted.named_steps["fs"]
            mi_scores = pd.Series(fs_step.scores_, index=X.columns)
            mi_scores.sort_values(ascending=False).to_csv(
                f"results/mi_scores_{base}.csv"
            )
            mi_scores_saved = True

        # --- Feature importance plots for RF and XGB (separate fig each time)
        if name in ("RandomForest", "XGBoost"):
            clf_fitted = pipe.named_steps["clf"]

            if (k is not None) and ("fs" in prep_fitted.named_steps):
                fs_step: SelectKBest = prep_fitted.named_steps["fs"]
                selected_idx = fs_step.get_support(indices=True)
                feat_names = X.columns[selected_idx]
            else:
                feat_names = X.columns

            importances = clf_fitted.feature_importances_
            top = pd.Series(importances, index=feat_names).sort_values(
                ascending=False
            ).head(15)

            # new figure ONLY for importance
            fig_imp, ax_imp = plt.subplots()
            top.plot(kind="bar", ax=ax_imp)
            ax_imp.set_ylabel("Importance")
            ax_imp.set_title(f"{name} – Top features – {base}")
            fig_imp.tight_layout()
            fig_imp.savefig(f"results/featimp_{name}_{base}.png", dpi=200)
            plt.close(fig_imp)

    # --- Finalize & save ROC overlay
    ax_roc.set_title(f"ROC – {base}")
    ax_roc.set_xlabel("False Positive Rate (Positive label: 1)")
    ax_roc.set_ylabel("True Positive Rate (Positive label: 1)")
    fig_roc.tight_layout()
    fig_roc.savefig(f"results/roc_{base}.png", dpi=200, bbox_inches="tight")
    plt.close(fig_roc)

    # --- Save metrics table
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(f"results/metrics_{base}.csv", index=False)


def main():
    if len(sys.argv) != 2:
        print("Usage: python report3_run_mi.py <dataset_file>")
        sys.exit(1)
    run_one(sys.argv[1])


if __name__ == "__main__":
    main()
