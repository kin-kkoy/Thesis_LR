"""Train LR on Kent's shared Set C dataset (candidate-r2-cropped-1024-v1-g364-r1, 2026-11-01).

Kent's RF was fitted on the train role only and scored on the validation role
with probability metrics (his Table 13). This does the same for LR so the two
are comparable. The calibration role is not used, and the final-test role and
master.csv are never read (they stay inside his zip).

Data: dataFiles/kent_r2_package/{train,validation}/data.csv, extracted from the
package and checked against his manifests (sha256 and row counts).

Recipe: our usual StandardScaler + LogisticRegression(C=1.0, l2, lbfgs,
max_iter=2000, random_state=42), all 11 features (Kent's schema = Stage C), in
two class-weight variants:
  balanced  : our recipe for Stages A/B/C
  unweighted: class_weight=None, the choice Kent's RF uses
His teacher labels ~64% of rows as ignited (ours ~5-7%), so "balanced" here
down-weights positives, the reverse of what it did on our data.

Usage (from Thesis_1/Code with the venv active):
    python train_lr_kent_r2.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CODE = Path(__file__).resolve().parent
DATA = CODE / "dataFiles/kent_r2_package"
MODELS = CODE / "sandbox_kent/models"
TARGET = "Ignited"

VARIANTS = {
	"balanced": "balanced",
	"unweighted": None,
}


def make_pipeline(class_weight) -> Pipeline:
	return Pipeline([
		("scaler", StandardScaler()),
		("lr", LogisticRegression(
			C=1.0,
			penalty="l2",
			solver="lbfgs",
			class_weight=class_weight,
			max_iter=2000,
			random_state=SEED,
		)),
	])


def load(role: str) -> tuple[pd.DataFrame, pd.Series]:
	df = pd.read_csv(DATA / role / "data.csv")
	return df.drop(columns=[TARGET]), df[TARGET].astype(int)


def main() -> None:
	X_train, y_train = load("train")
	X_val, y_val = load("validation")
	print(f"train {len(y_train):,} rows ({y_train.mean():.1%} ignited); "
		f"validation {len(y_val):,} rows ({y_val.mean():.1%} ignited); {X_train.shape[1]} features")

	print(f"\n{'variant':<12}{'AP':>9}{'ROC AUC':>9}{'Brier':>9}{'LogLoss':>9}{'mean p':>9}")
	for name, class_weight in VARIANTS.items():
		model = make_pipeline(class_weight).fit(X_train, y_train)
		p = model.predict_proba(X_val)[:, 1]
		print(f"{name:<12}{average_precision_score(y_val, p):>9.4f}{roc_auc_score(y_val, p):>9.4f}"
			f"{brier_score_loss(y_val, p):>9.4f}{log_loss(y_val, p):>9.4f}{p.mean():>9.4f}")
		coefs = dict(zip(X_train.columns, model.named_steps["lr"].coef_[0].round(3)))
		print(f"{'':<12}coefficients (scaled): {coefs}")
		out = MODELS / f"fire_lr_kent_r2_{name}.joblib"
		joblib.dump(model, out)
		print(f"{'':<12}saved {out.relative_to(CODE)}")

	print("\nKent's RF on the same validation role (his Table 13, uncalibrated): "
		"AP 0.8303, ROC AUC 0.7400, Brier 0.1948, LogLoss 0.5716")


if __name__ == "__main__":
	main()
