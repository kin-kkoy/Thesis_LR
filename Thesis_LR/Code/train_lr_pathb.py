"""Path B: Feature engineering + Optuna for Logistic Regression.

Adds 7 physics-informed interaction features to the original 9, then:
  Phase 1: Train baseline LR on 16 features (default hyperparameters)
  Phase 2: Run Optuna with 5-fold CV on 16 features

This gives us a clean comparison:
  - 9 features, defaults  (baseline)         → already done
  - 9 features, Optuna    (Path A)           → already done
  - 16 features, defaults (Path B Phase 1)   → this script
  - 16 features, Optuna   (Path B Phase 2)   → this script

Usage:
    python train_lr_pathb.py
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

from modules.feature_engineering import add_interaction_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_TRIALS = 100
N_FOLDS = 5
CSV_PATH = Path(__file__).resolve().parent / "dataFiles" / "multi_scenario_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "models"


# ---------------------------------------------------------------------------
# Data loading with feature engineering
# ---------------------------------------------------------------------------
def load_data(csv_path: Path, seed: int):
    data = pd.read_csv(csv_path)
    if "neighbor_burning_count" in data.columns:
        data = data[data["neighbor_burning_count"] > 0].copy()
        print(f"[Dataset] Filtered to susceptible cells. Rows: {len(data)}")

    # Separate target before engineering (so Ignited doesn't get used)
    y_full = data["Ignited"]
    x_full = data.drop(columns=["Ignited"])

    # Add interaction features
    x_full = add_interaction_features(x_full)
    feature_names = list(x_full.columns)

    print(f"[Features] {len(feature_names)} total: {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        x_full, y_full, test_size=0.2, stratify=y_full, random_state=seed,
    )
    print(f"[Split] Train: {len(X_train)} ({int(y_train.sum())} pos) | "
          f"Test: {len(X_test)} ({int(y_test.sum())} pos)")
    return X_train, X_test, y_train, y_test, feature_names


# ---------------------------------------------------------------------------
# Evaluate a model on the test set
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, label: str):
    y_proba = model.predict_proba(X_test)[:, 1]

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

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"  Precision:  {precision:.6f}")
    print(f"  Recall:     {recall:.6f}")
    print(f"  F1-Score:   {f1:.6f}")
    print(f"  AUC-ROC:    {auc_roc:.6f}")
    print(f"  Jaccard:    {jaccard:.6f}")

    # Coefficients
    lr_model = model.named_steps["lr"]
    raw_coefs = lr_model.coef_[0]
    feature_names = list(X_test.columns)
    print(f"\n  LR Coefficients (signed):")
    print(f"  {'Feature':<30}{'Coefficient':>12}")
    signed_pairs = sorted(
        zip(feature_names, raw_coefs),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    for name, coef in signed_pairs:
        print(f"  {name:<30}{coef:>+12.6f}")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float(auc_roc),
        "jaccard": float(jaccard),
        "threshold": float(best_threshold),
    }


# ---------------------------------------------------------------------------
# Phase 1: Baseline LR with default hyperparameters on 16 features
# ---------------------------------------------------------------------------
def run_baseline(X_train, X_test, y_train, y_test):
    print("\n" + "#" * 60)
    print("  PHASE 1: BASELINE LR ON 16 ENGINEERED FEATURES")
    print("#" * 60)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )),
    ])
    pipeline.fit(X_train, y_train)

    metrics = evaluate_model(pipeline, X_test, y_test,
                             "PHASE 1: BASELINE (16 features, default hyperparams)")
    return pipeline, metrics


# ---------------------------------------------------------------------------
# Phase 2: Optuna on 16 features
# ---------------------------------------------------------------------------
def create_objective(X_train, y_train):
    def objective(trial: optuna.Trial) -> float:
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

        if penalty == "l1":
            solver = "liblinear"
        elif penalty == "elasticnet":
            solver = "saga"
        else:
            solver = trial.suggest_categorical("solver_l2", ["lbfgs", "liblinear", "saga"])

        C = trial.suggest_float("C", 1e-4, 100.0, log=True)

        l1_ratio = None
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

        cw_strategy = trial.suggest_categorical("class_weight", ["balanced", "custom"])
        if cw_strategy == "custom":
            pos_weight = trial.suggest_float("pos_weight", 1.0, 20.0)
            class_weight = {0: 1.0, 1: pos_weight}
        else:
            class_weight = "balanced"

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

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_f1s = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            pipeline.fit(X_fold_train, y_fold_train)
            y_proba = pipeline.predict_proba(X_fold_val)[:, 1]

            best_f1 = 0.0
            for t in np.arange(0.1, 0.9, 0.05):
                preds = (y_proba >= t).astype(int)
                f = f1_score(y_fold_val, preds, zero_division=0)
                if f > best_f1:
                    best_f1 = f

            fold_f1s.append(best_f1)

        return float(np.mean(fold_f1s))

    return objective


def run_optuna(X_train, X_test, y_train, y_test):
    print("\n" + "#" * 60)
    print("  PHASE 2: OPTUNA ON 16 ENGINEERED FEATURES")
    print(f"  {N_TRIALS} trials, {N_FOLDS}-fold stratified CV")
    print("#" * 60)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="lr_fire_engineered",
    )

    objective = create_objective(X_train, y_train)

    print(f"\nStarting Optuna search ({N_TRIALS} trials)...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n  Best CV F1:    {study.best_value:.6f}")
    print(f"  Best trial:    #{study.best_trial.number}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    best_trial_num = study.best_trial.number
    total_trials = len(study.trials)
    print(f"\n  Convergence: best found at trial {best_trial_num}/{total_trials}")

    # Rebuild best pipeline
    bp = study.best_params
    penalty = bp["penalty"]
    if penalty == "l1":
        solver = "liblinear"
    elif penalty == "elasticnet":
        solver = "saga"
    else:
        solver = bp.get("solver_l2", "lbfgs")

    if bp["class_weight"] == "custom":
        class_weight = {0: 1.0, 1: bp["pos_weight"]}
    else:
        class_weight = "balanced"

    lr_params = {
        "C": bp["C"],
        "penalty": penalty,
        "solver": solver,
        "class_weight": class_weight,
        "max_iter": 2000,
        "random_state": SEED,
    }
    if penalty == "elasticnet":
        lr_params["l1_ratio"] = bp["l1_ratio"]

    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**lr_params)),
    ])
    final_pipeline.fit(X_train, y_train)

    metrics = evaluate_model(final_pipeline, X_test, y_test,
                             "PHASE 2: OPTUNA-TUNED (16 features)")

    # Overfitting check
    cv_f1 = study.best_value
    test_f1 = metrics["f1"]
    gap = cv_f1 - test_f1
    print(f"\n  CV F1:   {cv_f1:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Gap:     {gap:+.4f}", end="")
    if abs(gap) < 0.03:
        print("  (minimal — result is trustworthy)")
    elif gap > 0.05:
        print("  (CV optimistic — some overfitting)")
    else:
        print("  (small gap — acceptable)")

    return final_pipeline, metrics, study


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  PATH B: FEATURE ENGINEERING + OPTUNA FOR LR")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_names = load_data(CSV_PATH, SEED)

    # Phase 1: Baseline on engineered features
    baseline_pipeline, baseline_metrics = run_baseline(X_train, X_test, y_train, y_test)

    # Phase 2: Optuna on engineered features
    optuna_pipeline, optuna_metrics, study = run_optuna(X_train, X_test, y_train, y_test)

    # --- Summary comparison ---
    print("\n" + "=" * 60)
    print("  FULL COMPARISON (all LR variants)")
    print("=" * 60)
    print(f"  {'Variant':<40} {'F1':>8} {'Recall':>8} {'Prec':>8} {'AUC':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    # Previous results (hardcoded from earlier runs for comparison)
    print(f"  {'9feat, defaults (baseline)':<40} {'0.133':>8} {'0.470':>8} {'0.077':>8} {'0.630':>8}")
    print(f"  {'9feat, Optuna (Path A)':<40} {'0.129':>8} {'0.379':>8} {'0.078':>8} {'0.629':>8}")
    print(f"  {'16feat, defaults (Path B phase 1)':<40} {baseline_metrics['f1']:>8.3f} {baseline_metrics['recall']:>8.3f} {baseline_metrics['precision']:>8.3f} {baseline_metrics['auc_roc']:>8.3f}")
    print(f"  {'16feat, Optuna (Path B phase 2)':<40} {optuna_metrics['f1']:>8.3f} {optuna_metrics['recall']:>8.3f} {optuna_metrics['precision']:>8.3f} {optuna_metrics['auc_roc']:>8.3f}")

    # Save models
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_path = OUTPUT_DIR / "fire_lr_engineered_baseline.joblib"
    joblib.dump(baseline_pipeline, baseline_path)
    print(f"\n  Saved baseline model:  {baseline_path}")

    optuna_path = OUTPUT_DIR / "fire_lr_engineered_optuna.joblib"
    joblib.dump(optuna_pipeline, optuna_path)
    print(f"  Saved Optuna model:    {optuna_path}")

    study_path = OUTPUT_DIR / "optuna_study_lr_engineered.joblib"
    joblib.dump(study, study_path)
    print(f"  Saved Optuna study:    {study_path}")


if __name__ == "__main__":
    main()
