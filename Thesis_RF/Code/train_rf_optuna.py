"""Optuna hyperparameter tuning for Random Forest with 5-fold stratified CV.

Uses the 11-feature multi_scenario_dataset.csv and saves the best model to
models/fire_rf_model.joblib.

Usage:
    python train_rf_optuna.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_TRIALS = 100
N_FOLDS = 5
CSV_PATH = Path(__file__).resolve().parent / "dataFiles" / "multi_scenario_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = OUTPUT_DIR / "fire_rf_model.joblib"
STUDY_PATH = OUTPUT_DIR / "optuna_study_rf.joblib"
SUMMARY_PATH = OUTPUT_DIR / "optuna_rf_search_summary.json"
FILTER_SUSCEPTIBLE = True
CLASS_WEIGHT_CHOICES = {
    "cw_1_3": {0: 1.0, 1: 3.0},
    "cw_1_5": {0: 1.0, 1: 5.0},
    "cw_1_7": {0: 1.0, 1: 7.0},
}

FEATURE_NAMES = [
    "slope_risk",
    "proximity_risk",
    "building_presence",
    "material_risk",
    "material_class",
    "wind_speed",
    "wind_sin",
    "wind_cos",
    "neighbor_burning_count",
    "composite_flammability",
    "wind_weighted_score",
]
TARGET_NAME = "Ignited"


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(csv_path)
    if TARGET_NAME not in data.columns:
        raise ValueError("Input CSV must contain an 'Ignited' target column")

    if FILTER_SUSCEPTIBLE and "neighbor_burning_count" in data.columns:
        filtered = data[data["neighbor_burning_count"] > 0].copy()
        if not filtered.empty:
            data = filtered
            print(
                "[Dataset] Filtered to susceptible cells only "
                f"(neighbor_burning_count > 0). Rows remaining: {len(data)}"
            )
        else:
            print(
                "[Dataset] neighbor_burning_count > 0 produced 0 rows. "
                "Using full dataset instead."
            )

    missing = [name for name in FEATURE_NAMES if name not in data.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Dataset is missing required feature columns: {missing_text}")

    x = data.loc[:, FEATURE_NAMES]
    y = data[TARGET_NAME].astype(int)
    return x, y


def create_objective(x: pd.DataFrame, y: pd.Series):
    def objective(trial: optuna.Trial) -> float:
        class_weight_key = trial.suggest_categorical(
            "class_weight", list(CLASS_WEIGHT_CHOICES.keys())
        )
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 10, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "class_weight": CLASS_WEIGHT_CHOICES[class_weight_key],
            "random_state": SEED,
            "n_jobs": -1,
        }

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_scores: list[float] = []

        for train_idx, val_idx in skf.split(x, y):
            x_train = x.iloc[train_idx]
            y_train = y.iloc[train_idx]
            x_val = x.iloc[val_idx]
            y_val = y.iloc[val_idx]

            model = RandomForestClassifier(**params)
            model.fit(x_train, y_train)
            preds = model.predict(x_val)
            fold_scores.append(f1_score(y_val, preds, zero_division=0))

        return float(np.mean(fold_scores))

    return objective


def save_summary(
    summary_path: Path,
    best_params: dict,
    best_score: float,
    n_rows: int,
    pos_rate: float,
) -> None:
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "csv_path": str(CSV_PATH),
        "n_rows": int(n_rows),
        "positive_rate": float(pos_rate),
        "n_trials": int(N_TRIALS),
        "n_folds": int(N_FOLDS),
        "filter_susceptible": bool(FILTER_SUSCEPTIBLE),
        "feature_names": list(FEATURE_NAMES),
        "best_cv_f1": float(best_score),
        "best_params": best_params,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    print("=" * 60)
    print("  OPTUNA HYPERPARAMETER TUNING FOR RANDOM FOREST")
    print(f"  {N_TRIALS} trials, {N_FOLDS}-fold stratified CV")
    print("=" * 60)

    x, y = load_data(CSV_PATH)
    n_rows = int(len(y))
    pos_rate = float(y.mean()) if n_rows else 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="rf_fire_tuning",
    )

    objective = create_objective(x, y)
    print(f"\nStarting Optuna search ({N_TRIALS} trials)...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("  OPTUNA SEARCH RESULTS")
    print("=" * 60)
    print(f"  Best CV F1:    {study.best_value:.6f}")
    print(f"  Best trial:    #{study.best_trial.number}")
    print("  Best params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    final_params = dict(study.best_params)
    if "class_weight" in final_params:
        final_params["class_weight"] = CLASS_WEIGHT_CHOICES[final_params["class_weight"]]
    final_params.update({"random_state": SEED, "n_jobs": -1})
    final_model = RandomForestClassifier(**final_params)
    final_model.fit(x, y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(study, STUDY_PATH)

    save_summary(SUMMARY_PATH, study.best_params, study.best_value, n_rows, pos_rate)

    print(f"\n  Saved tuned model: {MODEL_PATH}")
    print(f"  Saved Optuna study: {STUDY_PATH}")
    print(f"  Saved summary log: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
