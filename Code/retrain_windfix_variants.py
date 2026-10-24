"""Rerun the 8 per-cell LR variants (Table 7) on the wind-fixed dataset.

The original variants were produced by train_lr.py-style ModelTrainer runs,
train_lr_optuna.py (Path A) and train_lr_pathb.py (Path B), all reading
dataFiles/multi_scenario_dataset.csv and saving over models/*.joblib. This
wrapper reuses their objective/evaluation code unchanged (seed 42, 100 TPE
trials, 5-fold stratified CV, 80/20 stratified split, best-of-16 cutoffs on the
test set) and only swaps the dataset. Nothing existing is overwritten: models go
to models/*_windfix.joblib.

The 9-feature variants are the 10-feature CSV minus wind_weighted_score: the
9- and 10-feature datasets came from the same 200 simulations (identical rows
and labels), wind_weighted_score was only recorded alongside. The "old" pass
re-scores the pre-fix CSV to check this reproduces the published Table 7.

Each variant also reports metrics at a fixed 0.5 cutoff (no cutoff picked on
test labels).

Usage:
    python retrain_windfix_variants.py            # old (check) + new
    python retrain_windfix_variants.py --new-only
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_lr_pathb as pathb
from modules.feature_engineering import add_interaction_features

CODE = Path(__file__).resolve().parent
SEED = 42
N_TRIALS = 100
CSVS = {
	"old": CODE / "dataFiles/multi_scenario_dataset.csv",
	"new": CODE / "dataFiles/multi_scenario_dataset_windfix.csv",
}
# (row, label, keep wind_weighted_score, add engineered, tuned, model name)
VARIANTS = [
	(1, "9-feature, default", False, False, False, "fire_lr_9feat_windfix"),
	(2, "9-feature, Optuna", False, False, True, "fire_lr_optuna_windfix"),
	(3, "9 + 7 engineered, default", False, True, False, "fire_lr_engineered_baseline_windfix"),
	(4, "9 + 7 engineered, Optuna", False, True, True, "fire_lr_engineered_optuna_windfix"),
	(5, "10-feature, default", True, False, False, "fire_lr_10feat_windfix"),
	(6, "10-feature, Optuna", True, False, True, "fire_lr_10feat_optuna_windfix"),
	(7, "10 + 7 engineered, default", True, True, False, "fire_lr_10feat_engineered_windfix"),
	(8, "10 + 7 engineered, Optuna", True, True, True, "fire_lr_10feat_engineered_optuna_windfix"),
]


def load(csv: Path, wws: bool, eng: bool):
	data = pd.read_csv(csv)
	data = data[data["neighbor_burning_count"] > 0].copy()
	y = data["Ignited"]
	x = data.drop(columns=["Ignited"])
	if not wws:
		x = x.drop(columns=["wind_weighted_score"])
	if eng:
		x = add_interaction_features(x)
	return train_test_split(x, y, test_size=0.2, stratify=y, random_state=SEED)


def build(params: dict | None) -> Pipeline:
	if params is None:
		lr = dict(C=1.0, penalty="l2", solver="lbfgs", class_weight="balanced")
	else:
		penalty = params["penalty"]
		solver = {"l1": "liblinear", "elasticnet": "saga"}.get(penalty, params.get("solver_l2", "lbfgs"))
		cw = {0: 1.0, 1: params["pos_weight"]} if params["class_weight"] == "custom" else "balanced"
		lr = dict(C=params["C"], penalty=penalty, solver=solver, class_weight=cw)
		if penalty == "elasticnet":
			lr["l1_ratio"] = params["l1_ratio"]
	return Pipeline([
		("scaler", StandardScaler()),
		("lr", LogisticRegression(max_iter=2000, random_state=SEED, **lr)),
	])


def scores(y, proba, cutoff):
	pred = (proba >= cutoff).astype(int)
	return (f1_score(y, pred, zero_division=0), recall_score(y, pred, zero_division=0),
		precision_score(y, pred, zero_division=0))


def run(job):
	tag, (row, label, wws, eng, tuned, name) = job
	X_train, X_test, y_train, y_test = load(CSVS[tag], wws, eng)
	params, cv = None, None
	if tuned:
		optuna.logging.set_verbosity(optuna.logging.WARNING)
		study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
		study.optimize(pathb.create_objective(X_train, y_train), n_trials=N_TRIALS)
		params, cv = study.best_params, study.best_value
	model = build(params).fit(X_train, y_train)
	proba = model.predict_proba(X_test)[:, 1]
	best = max(np.arange(0.1, 0.9, 0.05), key=lambda t: scores(y_test, proba, t)[0])
	if tag == "new":
		joblib.dump(model, CODE / "models" / f"{name}.joblib")
	return dict(set=tag, row=row, variant=label, n_feat=X_train.shape[1],
		auc=roc_auc_score(y_test, proba), best_t=round(float(best), 2),
		**dict(zip(("f1", "recall", "precision"), scores(y_test, proba, best))),
		**dict(zip(("f1_05", "recall_05", "precision_05"), scores(y_test, proba, 0.5))),
		cv_f1=cv, params=params)


def main():
	tags = ["new"] if "--new-only" in sys.argv else ["old", "new"]
	jobs = [(t, v) for t in tags for v in VARIANTS]
	with ProcessPoolExecutor(max_workers=8) as pool:
		results = list(pool.map(run, jobs))
	df = pd.DataFrame(results)
	pd.set_option("display.width", 200)
	cols = ["set", "row", "variant", "n_feat", "f1", "recall", "precision", "auc", "best_t",
		"f1_05", "recall_05", "precision_05", "cv_f1"]
	print(df[cols].round(4).to_string(index=False))
	for r in results:
		if r["params"]:
			print(r["set"], r["row"], r["params"])
	out = CODE / "output/windfix/per_cell_variants.csv"
	df.to_csv(out, index=False)
	print(f"saved {out}")


if __name__ == "__main__":
	main()
