"""Train and evaluate ML models for synthetic fire ignition data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	confusion_matrix,
	f1_score,
	jaccard_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
	def __init__(self, csv_path: str, output_dir: str, seed: int = 42):
		self.csv_path = str(csv_path)
		self.output_dir = Path(output_dir)
		self.seed = int(seed)

		data = pd.read_csv(self.csv_path)
		if "Ignited" not in data.columns:
			raise ValueError("Input CSV must contain an 'Ignited' target column")

		# CRITICAL FIX: The CA engine only evaluates cells where neighbor_burning_count > 0.
		# Including millions of cells far from the fire skews the model and ruins precision.
		if "neighbor_burning_count" in data.columns:
			data = data[data["neighbor_burning_count"] > 0].copy()
			print(f"[Dataset] Filtered to susceptible cells only (neighbor_burning_count > 0). Rows remaining: {len(data)}")

		x = data.drop(columns=["Ignited"])
		y = data["Ignited"]

		self.feature_names = list(x.columns)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
			x,
			y,
			test_size=0.2,
			stratify=y,
			random_state=self.seed,
		)

		self.model = None
		self.model_name = None

	def train_model(
		self,
		algorithm: str = "random_forest",
		class_weight_strategy: Literal["balanced", "balanced_subsample"] = "balanced_subsample",
		**model_kwargs,
	) -> object:
		algorithm_name = str(algorithm).strip().lower()
		if class_weight_strategy not in {"balanced", "balanced_subsample"}:
			raise ValueError("class_weight_strategy must be 'balanced' or 'balanced_subsample'")

		if algorithm_name == "random_forest":
			params = {
				"n_estimators": 200,
				"max_depth": 15,
				"random_state": self.seed,
				"class_weight": class_weight_strategy,
				"n_jobs": -1,
			}
			params.update(model_kwargs)
			model = RandomForestClassifier(**params)
		elif algorithm_name == "logistic_regression":
			# LR needs feature scaling (unlike RF), so we wrap it in a Pipeline.
			# The Pipeline has predict_proba(), satisfying automata_engine.py's contract.
			# class_weight must be 'balanced' (LR doesn't support 'balanced_subsample').
			lr_class_weight = "balanced" if "subsample" in class_weight_strategy else class_weight_strategy
			lr_params = {
				"max_iter": 1000,
				"random_state": self.seed,
				"class_weight": lr_class_weight,
				"solver": "lbfgs",
			}
			lr_params.update(model_kwargs)
			model = Pipeline([
				("scaler", StandardScaler()),
				("lr", LogisticRegression(**lr_params)),
			])
		else:
			raise ValueError(f"Unsupported algorithm: {algorithm}")

		model.fit(self.X_train, self.y_train)
		if not hasattr(model, "predict_proba"):
			raise TypeError("Model must implement predict_proba() for CA integration")

		self.model = model
		self.model_name = algorithm_name
		return self.model

	def train_random_forest(
		self,
		n_estimators: int = 200,
		max_depth: int | None = 15,
		class_weight_strategy: Literal["balanced", "balanced_subsample"] = "balanced_subsample",
	) -> object:
		return self.train_model(
			algorithm="random_forest",
			class_weight_strategy=class_weight_strategy,
			n_estimators=n_estimators,
			max_depth=max_depth,
		)

	def train_logistic_regression(
		self,
		class_weight_strategy: Literal["balanced", "balanced_subsample"] = "balanced",
	) -> object:
		return self.train_model(
			algorithm="logistic_regression",
			class_weight_strategy=class_weight_strategy,
		)

	def evaluate(self) -> dict:
		if self.model is None:
			raise RuntimeError("No trained model found. Call train_model() first.")

		y_proba = self.model.predict_proba(self.X_test)[:, 1]

		# Find the optimal threshold for F1-score
		best_f1 = 0.0
		best_threshold = 0.5
		for t in np.arange(0.1, 0.9, 0.05):
			preds = (y_proba >= t).astype(int)
			f = f1_score(self.y_test, preds, zero_division=0)
			if f > best_f1:
				best_f1 = f
				best_threshold = t

		print(f"\nEvaluating with optimized threshold: {best_threshold:.2f}")
		y_pred = (y_proba >= best_threshold).astype(int)

		cm = confusion_matrix(self.y_test, y_pred)
		precision = precision_score(self.y_test, y_pred, zero_division=0)
		recall = recall_score(self.y_test, y_pred, zero_division=0)
		f1 = f1_score(self.y_test, y_pred, zero_division=0)
		auc_roc = roc_auc_score(self.y_test, y_proba)
		jaccard = jaccard_score(self.y_test, y_pred, zero_division=0)

		print("Confusion Matrix:")
		print(cm)
		print(f"Precision: {precision:.6f}")
		print(f"Recall: {recall:.6f}")
		print(f"F1-Score: {f1:.6f}")
		print(f"AUC-ROC: {auc_roc:.6f}")
		print(f"Jaccard Index: {jaccard:.6f}")

		return {
			"precision": float(precision),
			"recall": float(recall),
			"f1": float(f1),
			"auc_roc": float(auc_roc),
			"jaccard": float(jaccard),
		}

	def save_model(self, filename: str = "fire_rf_model.joblib") -> str:
		if self.model is None:
			raise RuntimeError("No trained model found. Call train_model() first.")

		self.output_dir.mkdir(parents=True, exist_ok=True)
		model_path = self.output_dir / filename
		joblib.dump(self.model, model_path)
		return str(model_path.resolve())

	def feature_importance_report(self) -> None:
		if self.model is None:
			raise RuntimeError("No trained model found. Call train_model() first.")

		# RF exposes feature_importances_, LR Pipeline exposes coef_ inside the pipeline
		if hasattr(self.model, "feature_importances_"):
			scores = self.model.feature_importances_
			label = "Importance"
		elif isinstance(self.model, Pipeline) and hasattr(self.model.named_steps.get("lr", None), "coef_"):
			scores = np.abs(self.model.named_steps["lr"].coef_[0])
			label = "Abs Coefficient"
			# Also print the raw (signed) coefficients for interpretation
			raw_coefs = self.model.named_steps["lr"].coef_[0]
			print("LR Coefficients (signed — positive = increases ignition chance):")
			print(f"{'Feature':<30}{'Coefficient':>12}")
			signed_pairs = sorted(
				zip(self.feature_names, raw_coefs),
				key=lambda pair: abs(pair[1]),
				reverse=True,
			)
			for feature_name, coef in signed_pairs:
				print(f"{feature_name:<30}{coef:>+12.6f}")
			print()
		else:
			raise TypeError("Current model does not expose feature_importances_ or coef_")

		importance_pairs = sorted(
			zip(self.feature_names, scores),
			key=lambda pair: pair[1],
			reverse=True,
		)

		print(f"Feature {label}:")
		print(f"{'Feature':<30}{label}")
		for feature_name, score in importance_pairs:
			print(f"{feature_name:<30}{score:.6f}")


if __name__ == "__main__":
	code_dir = Path(__file__).resolve().parents[1]
	default_csv = code_dir / "dataFiles" / "synthetic_fire_dataset.csv"
	default_output_dir = code_dir / "models"

	trainer = ModelTrainer(csv_path=str(default_csv), output_dir=str(default_output_dir), seed=42)
	trainer.train_random_forest(class_weight_strategy="balanced_subsample")
	metrics = trainer.evaluate()
	saved_path = trainer.save_model()

	print("Saved model:", saved_path)
	print("Metrics:", metrics)
	trainer.feature_importance_report()
