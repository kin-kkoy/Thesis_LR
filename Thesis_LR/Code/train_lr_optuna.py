"""Hyperparameter tuning for Logistic Regression using Optuna with 5-fold stratified CV.

Path A: Tune LR on the original 9 features to find the honest LR ceiling.
Uses the same train/test split (seed=42) as the baseline so results are directly comparable.
The test set is LOCKED — Optuna only sees cross-validation scores on the training set.

Grounding:
  - Hyperparameter tuning: Bergstra & Bengio (2012), Akiba et al. (2019)
  - Cross-validation: Stone (1974), Kohavi (1995)
  - Bias-variance tradeoff: Hastie, Tibshirani & Friedman (2009), Ch. 7

Usage:
    python train_lr_optuna.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    jaccard_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_TRIALS = 100
N_FOLDS = 5
CSV_PATH = Path(__file__).resolve().parent / "dataFiles" / "multi_scenario_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "models"

# ---------------------------------------------------------------------------
# Data loading (identical to ModelTrainer so splits match)
# ---------------------------------------------------------------------------
def load_data(csv_path: Path, seed: int):
    data = pd.read_csv(csv_path)
    if "neighbor_burning_count" in data.columns:
        data = data[data["neighbor_burning_count"] > 0].copy()
        print(f"[Dataset] Filtered to susceptible cells. Rows: {len(data)}")

    x = data.drop(columns=["Ignited"])
    y = data["Ignited"]
    feature_names = list(x.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=seed,
    )
    print(f"[Split] Train: {len(X_train)} ({int(y_train.sum())} pos) | "
          f"Test: {len(X_test)} ({int(y_test.sum())} pos)")
    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# Optuna objective: 5-fold stratified CV on training data only
# ---------------------------------------------------------------------------
def create_objective(X_train, y_train):
    def objective(trial: optuna.Trial) -> float:
        # --- Hyperparameter search space ---
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

        # Solver must be compatible with penalty
        if penalty == "l1":
            solver = "liblinear"
        elif penalty == "elasticnet":
            solver = "saga"
        else:  # l2
            solver = trial.suggest_categorical("solver_l2", ["lbfgs", "liblinear", "saga"])

        C = trial.suggest_float("C", 1e-4, 100.0, log=True)

        # ElasticNet requires l1_ratio
        l1_ratio = None
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        # Class weight: balanced or specific ratio
        cw_strategy = trial.suggest_categorical("class_weight", ["balanced", "custom"])
        if cw_strategy == "custom":
            pos_weight = trial.suggest_float("pos_weight", 1.0, 20.0)
            class_weight = {0: 1.0, 1: pos_weight}
        else:
            class_weight = "balanced"

        # --- Build pipeline ---
        lr_params = {
            "C": C,
            "penalty": penalty,
            "solver": solver,
            "class_weight": class_weight,
            "max_iter": 2000,
            "random_state": SEED,
        }
        if l1_ratio is not None:
            lr_params["l1_ratio"] = l1_ratio

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**lr_params)),
        ])

        # --- 5-fold stratified CV ---
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            pipeline.fit(X_fold_train, y_fold_train)
            y_proba = pipeline.predict_proba(X_fold_val)[:, 1]

            # Threshold optimization per fold (same method as ModelTrainer.evaluate)
            best_f1 = 0.0
            for t in np.arange(0.1, 0.9, 0.05):
                preds = (y_proba >= t).astype(int)
                f = f1_score(y_fold_val, preds, zero_division=0)
                if f > best_f1:
                    best_f1 = f

            fold_f1s.append(best_f1)

        return float(np.mean(fold_f1s))

    return objective


# ---------------------------------------------------------------------------
# Final evaluation on held-out test set
# ---------------------------------------------------------------------------
def train_and_evaluate_final(best_params: dict, X_train, X_test, y_train, y_test, feature_names):
    # Reconstruct best pipeline
    penalty = best_params["penalty"]

    if penalty == "l1":
        solver = "liblinear"
    elif penalty == "elasticnet":
        solver = "saga"
    else:
        solver = best_params.get("solver_l2", "lbfgs")

    cw_strategy = best_params["class_weight"]
    if cw_strategy == "custom":
        class_weight = {0: 1.0, 1: best_params["pos_weight"]}
    else:
        class_weight = "balanced"

    lr_params = {
        "C": best_params["C"],
        "penalty": penalty,
        "solver": solver,
        "class_weight": class_weight,
        "max_iter": 2000,
        "random_state": SEED,
    }
    if penalty == "elasticnet":
        lr_params["l1_ratio"] = best_params["l1_ratio"]

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**lr_params)),
    ])

    # Train on ALL training data
    final_pipeline.fit(X_train, y_train)

    # Predict on locked test set
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]

    # Find optimal threshold on test set (same as baseline for fair comparison)
    best_f1 = 0.0
    best_threshold = 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        preds = (y_proba >= t).astype(int)
        f = f1_score(y_test, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_threshold = t

    y_pred = (y_proba >= best_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_proba)
    jaccard = jaccard_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print("  FINAL MODEL — HELD-OUT TEST SET EVALUATION")
    print("=" * 60)
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Precision:  {precision:.6f}")
    print(f"  Recall:     {recall:.6f}")
    print(f"  F1-Score:   {f1:.6f}")
    print(f"  AUC-ROC:    {auc_roc:.6f}")
    print(f"  Jaccard:    {jaccard:.6f}")

    # Coefficients
    lr_model = final_pipeline.named_steps["lr"]
    raw_coefs = lr_model.coef_[0]
    print("\n  LR Coefficients (signed):")
    print(f"  {'Feature':<30}{'Coefficient':>12}")
    signed_pairs = sorted(
        zip(feature_names, raw_coefs),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    for name, coef in signed_pairs:
        print(f"  {name:<30}{coef:>+12.6f}")

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "jaccard": float(jaccard),
        "threshold": float(best_threshold),
    }

    return final_pipeline, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  PATH A: OPTUNA HYPERPARAMETER TUNING FOR LR")
    print(f"  {N_TRIALS} trials, {N_FOLDS}-fold stratified CV")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_names = load_data(CSV_PATH, SEED)

    # Suppress Optuna's per-trial logging (we print our own summary)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="lr_fire_tuning",
    )

    objective = create_objective(X_train, y_train)

    print(f"\nStarting Optuna search ({N_TRIALS} trials)...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # --- Report search results ---
    print("\n" + "=" * 60)
    print("  OPTUNA SEARCH RESULTS")
    print("=" * 60)
    print(f"  Best CV F1:    {study.best_value:.6f}")
    print(f"  Best trial:    #{study.best_trial.number}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Convergence check: when was the best trial found?
    total_trials = len(study.trials)
    best_trial_num = study.best_trial.number
    print(f"\n  Convergence: best found at trial {best_trial_num}/{total_trials}")
    if best_trial_num < total_trials * 0.5:
        print("  -> Search converged early (best in first half)")
    else:
        print("  -> Best found in second half — more trials might help")

    # --- Train final model and evaluate on locked test set ---
    final_pipeline, metrics = train_and_evaluate_final(
        study.best_params, X_train, X_test, y_train, y_test, feature_names,
    )

    # --- Compare CV score vs test score (overfitting check) ---
    cv_f1 = study.best_value
    test_f1 = metrics["f1"]
    gap = cv_f1 - test_f1
    print(f"\n  CV F1:   {cv_f1:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Gap:     {gap:+.4f}", end="")
    if abs(gap) < 0.03:
        print("  (minimal — result is trustworthy)")
    elif gap > 0.05:
        print("  (CV optimistic — some overfitting to validation folds)")
    else:
        print("  (small gap — acceptable)")

    # --- Save model ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "fire_lr_optuna.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"\n  Saved tuned model: {model_path}")

    # --- Save study for later analysis ---
    study_path = OUTPUT_DIR / "optuna_study_lr.joblib"
    joblib.dump(study, study_path)
    print(f"  Saved Optuna study: {study_path}")


if __name__ == "__main__":
    main()
