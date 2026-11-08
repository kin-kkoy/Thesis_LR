"""Rebuild Kristian's LR model inside YOUR scikit-learn version.

The .joblib file was saved with scikit-learn 1.9.0; your environment has 1.8.0,
and a model pickled in a newer version may not load in an older one. This
script builds the identical model (same scaler means/scales, same coefficients)
from fire_lr_kent_r2_unweighted_params.json, so no pickle compatibility is
needed.

Optional check: pass your validation role CSV and it should reproduce
AP 0.8331 / ROC AUC 0.7439 / Brier 0.1921 / log loss 0.5643.

Usage (from this folder, with your Thesis venv):
    python rebuild_lr.py
    python rebuild_lr.py --check path/to/development/validation/data.csv
Writes: fire_lr_kent_r2_unweighted_sk<version>.joblib
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent


def build() -> Pipeline:
	p = json.loads((HERE / "fire_lr_kent_r2_unweighted_params.json").read_text())
	names = np.array(p["feature_names"], dtype=object)

	scaler = StandardScaler()
	scaler.mean_ = np.array(p["scaler"]["mean"])
	scaler.var_ = np.array(p["scaler"]["var"])
	scaler.scale_ = np.array(p["scaler"]["scale"])
	scaler.n_samples_seen_ = p["scaler"]["n_samples_seen"]
	scaler.n_features_in_ = len(names)
	scaler.feature_names_in_ = names

	q = p["logistic_regression"]
	lr = LogisticRegression(C=q["C"], solver=q["solver"], max_iter=q["max_iter"], random_state=q["random_state"])
	lr.coef_ = np.array([q["coef"]])
	lr.intercept_ = np.array([q["intercept"]])
	lr.classes_ = np.array(p["classes"])
	lr.n_features_in_ = len(names)
	lr.n_iter_ = np.array([0])

	return Pipeline([("scaler", scaler), ("lr", lr)])


def main() -> None:
	parser = argparse.ArgumentParser(description="Rebuild the LR model from its JSON parameters")
	parser.add_argument("--check", help="Validation role data.csv to verify the rebuilt model")
	args = parser.parse_args()

	model = build()
	out = HERE / f"fire_lr_kent_r2_unweighted_sk{sklearn.__version__}.joblib"
	joblib.dump(model, out)
	print(f"rebuilt with scikit-learn {sklearn.__version__} -> {out.name}")
	print(f"feature_names_in_: {list(model.feature_names_in_)}")
	print(f"classes_: {model.classes_.tolist()}")

	if args.check:
		import pandas as pd
		from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
		df = pd.read_csv(args.check)
		y = df.pop("Ignited").astype(int)
		prob = model.predict_proba(df[list(model.feature_names_in_)])[:, 1]
		print(f"check: AP {average_precision_score(y, prob):.4f}  ROC AUC {roc_auc_score(y, prob):.4f}  "
			f"Brier {brier_score_loss(y, prob):.4f}  log loss {log_loss(y, prob):.4f}  "
			"(expected 0.8331 / 0.7439 / 0.1921 / 0.5643)")


if __name__ == "__main__":
	main()
