"""
CSC 590 – Report 2 runner (fixed plotting)
Runs LR (baseline), RandomForest, and XGBoost on a given dataset file and saves:
  - metrics CSV  (CV AUC mean±std, Test Acc/F1/AUC)
  - ROC curve PNG (overlay of LR vs RF vs XGB)
  - Feature-importance PNGs (RF and XGB)

Usage:
  python report2_run.py 4-wayAdditive_100feat.txt
  python report2_run.py 2-wayEpi_100feat.txt
"""

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# XGBoost is a separate package
try:
    from xgboost import XGBClassifier
except Exception as e:
    print("XGBoost not installed. Run: pip install xgboost")
    raise e

RANDOM_STATE = 42


def make_prep():
    # Robust for these SNP tables; OK for high dimension too
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler(with_mean=False)),
    ])


def build_models():
    return {
        "LogReg": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_features="sqrt", max_depth=None,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE
        ),
        # Bonus option:
        # "SVM_RBF": SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=RANDOM_STATE),
    }


def run_one(dataset_path: str):
    base = os.path.splitext(os.path.basename(dataset_path))[0]
    os.makedirs("results", exist_ok=True)

    # Load data
    df = pd.read_csv(dataset_path, sep="\t")
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Train test split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    prep = make_prep()
    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []

    # Dedicated ROC figure and axes so it cannot be cleared by other plots
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))

    for name, clf in models.items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])

        # CV ROC-AUC on train folds
        auc_cv = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc")

        # Fit and evaluate on test
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        ypr = pipe.predict_proba(Xte)[:, 1]

        acc = accuracy_score(yte, yhat)
        f1 = f1_score(yte, yhat)
        auc = roc_auc_score(yte, ypr)

        print(f"[{base}] {name} | CV AUC: {auc_cv.mean():.3f}±{auc_cv.std():.3f} | "
              f"Test Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

        rows.append({
            "dataset": base,
            "model": name,
            "cv_auc_mean": round(float(auc_cv.mean()), 4),
            "cv_auc_std": round(float(auc_cv.std()), 4),
            "test_acc": round(float(acc), 4),
            "test_f1": round(float(f1), 4),
            "test_auc": round(float(auc), 4),
        })

        # Draw ROC for this model on the dedicated axes
        RocCurveDisplay.from_predictions(yte, ypr, name=name, ax=ax_roc)

        # Feature importance plots use their own short-lived figure
        if name in ("RandomForest", "XGBoost"):
            importances = pipe.named_steps["clf"].feature_importances_
            top = pd.Series(importances, index=X.columns).sort_values(ascending=False).head(15)

            fig_imp, ax_imp = plt.subplots(figsize=(10, 4))
            top.plot(kind="bar", ax=ax_imp)
            ax_imp.set_ylabel("Importance")
            ax_imp.set_title(f"{name} – Top features – {base}")
            fig_imp.tight_layout()
            fig_imp.savefig(f"results/featimp_{name}_{base}.png", dpi=200)
            plt.close(fig_imp)  # closes only the importance figure

    # Finalize and save ROC overlay
    ax_roc.set_title(f"ROC – {base}")
    fig_roc.tight_layout()
    fig_roc.savefig(f"results/roc_{base}.png", dpi=200, bbox_inches="tight")
    plt.close(fig_roc)

    # Save metrics
    pd.DataFrame(rows).to_csv(f"results/metrics_{base}.csv", index=False)


def main():
    if len(sys.argv) != 2:
        print("Usage: python report2_run.py <dataset_file>")
        print("Example: python report2_run.py 4-wayAdditive_100feat.txt")
        sys.exit(1)
    run_one(sys.argv[1])


if __name__ == "__main__":
    main()
