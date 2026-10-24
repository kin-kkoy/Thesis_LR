"""Retrain LR Stages A, B and C on the wind-fixed datasets (2026-10-23).

The original wind kernel was applied with scipy.ndimage.convolve, which flips
it, so every dataset generated since Apr 10 had fire spreading INTO the wind.
Commit bdb619a fixed the code; the datasets were regenerated to
*_windfix.csv. This script retrains with the exact recipe of the original
models (verified: rerunning it on the old CSVs reproduces the saved
coefficients exactly) so only the data changes.

Recipe (same as sandbox_kent/train_lr_stage_c.py):
  StandardScaler + LogisticRegression(C=1.0, l2, lbfgs, class_weight=balanced,
  max_iter=2000, random_state=42), random stratified 80/20 split, seed 42.

Stages:
  A: sandbox CSV without wind_weighted_score (10 features, Kent's schema)
  B: main CSV (10 features, wind_weighted_score, no material_class)
  C: sandbox CSV, all 11 features

Old models and CSVs are never overwritten; new models get a _windfix suffix.

Usage (from Thesis_1/Code with the venv active):
    python retrain_windfix.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CODE = Path(__file__).resolve().parent

STAGES = {
	"A": dict(
		old_csv=CODE / "sandbox_kent/dataFiles/multi_scenario_dataset.csv",
		new_csv=CODE / "sandbox_kent/dataFiles/multi_scenario_dataset_windfix.csv",
		drop=["wind_weighted_score"],
		old_model=CODE / "sandbox_kent/models/fire_lr_stage_a.joblib",
		new_model=CODE / "sandbox_kent/models/fire_lr_stage_a_windfix.joblib",
	),
	"B": dict(
		old_csv=CODE / "dataFiles/multi_scenario_dataset.csv",
		new_csv=CODE / "dataFiles/multi_scenario_dataset_windfix.csv",
		drop=[],
		old_model=CODE / "models/fire_lr_windweighted.joblib",
		new_model=CODE / "models/fire_lr_windweighted_windfix.joblib",
	),
	"C": dict(
		old_csv=CODE / "sandbox_kent/dataFiles/multi_scenario_dataset.csv",
		new_csv=CODE / "sandbox_kent/dataFiles/multi_scenario_dataset_windfix.csv",
		drop=[],
		old_model=CODE / "sandbox_kent/models/fire_lr_stage_c.joblib",
		new_model=CODE / "sandbox_kent/models/fire_lr_stage_c_windfix.joblib",
	),
}


def make_pipeline() -> Pipeline:
	return Pipeline([
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


def load(csv: Path, drop: list[str]) -> tuple[pd.DataFrame, pd.Series]:
	df = pd.read_csv(csv)
	df = df[df["neighbor_burning_count"] > 0]
	y = df["Ignited"].astype(np.int8)
	X = df.drop(columns=["Ignited", *[c for c in drop if c in df.columns]])
	return X, y


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
	"""Per-cell metrics at a fixed 0.5 cutoff (no threshold picked on test labels)."""
	proba = pipeline.predict_proba(X_test)[:, 1]
	pred = (proba >= 0.5).astype(int)
	return {
		"auc": roc_auc_score(y_test, proba),
		"f1": f1_score(y_test, pred, zero_division=0),
		"recall": recall_score(y_test, pred, zero_division=0),
		"precision": precision_score(y_test, pred, zero_division=0),
	}


def main() -> None:
	for stage, s in STAGES.items():
		if s["new_model"].exists():
			raise SystemExit(f"Refusing to overwrite {s['new_model']}")

		results = {}
		coefs = {}
		for label, csv in (("old", s["old_csv"]), ("new", s["new_csv"])):
			X, y = load(csv, s["drop"])
			X_train, X_test, y_train, y_test = train_test_split(
				X, y, test_size=0.2, stratify=y, random_state=SEED
			)
			pipeline = make_pipeline().fit(X_train, y_train)
			results[label] = evaluate(pipeline, X_test, y_test) | {"rows": len(X), "pos": int(y.sum())}
			coefs[label] = dict(zip(X.columns, pipeline.named_steps["lr"].coef_[0]))
			if label == "new":
				joblib.dump(pipeline, s["new_model"])

		# Sanity: the "old" refit must reproduce the saved model exactly.
		saved = joblib.load(s["old_model"]).named_steps["lr"].coef_[0]
		reproduced = np.abs(np.array(list(coefs["old"].values())) - saved).max()

		print(f"\n{'=' * 64}\n  Stage {stage}  ({len(coefs['new'])} features)\n{'=' * 64}")
		print(f"  Old-model reproduction check: max |coef diff| = {reproduced:.1e}")
		print(f"  {'':<12}{'rows':>8}{'pos':>7}{'AUC':>8}{'F1@0.5':>9}{'Rec@0.5':>9}{'Prec@0.5':>10}")
		for label in ("old", "new"):
			r = results[label]
			print(f"  {label:<12}{r['rows']:>8,}{r['pos']:>7,}{r['auc']:>8.4f}{r['f1']:>9.4f}{r['recall']:>9.4f}{r['precision']:>10.4f}")
		print("  Coefficients (old -> new):")
		for name in sorted(coefs["new"], key=lambda n: abs(coefs["new"][n]), reverse=True):
			print(f"    {name:<26}{coefs['old'][name]:>+9.4f} -> {coefs['new'][name]:>+9.4f}")
		print(f"  Saved: {s['new_model'].relative_to(CODE)}")


if __name__ == "__main__":
	main()
