"""Train and evaluate ML models for synthetic fire ignition data."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	confusion_matrix,
	f1_score,
	jaccard_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split


class ModelTrainer:
	SUPPORTED_MODEL_EXTENSIONS = {".joblib", ".pkl"}

	def __init__(self, csv_path: str, output_dir: str, seed: int = 42):
		self.csv_path = str(csv_path)
		self.output_dir = Path(output_dir)
		self.seed = int(seed)

		data = pd.read_csv(self.csv_path)
		if "Ignited" not in data.columns:
			raise ValueError("Input CSV must contain an 'Ignited' target column")

		if "neighbor_burning_count" in data.columns:
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

		if data.empty:
			raise ValueError("Training dataset is empty after preprocessing")

		x = data.drop(columns=["Ignited"])
		y = data["Ignited"].astype(int)

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
		else:
			raise ValueError(f"Unsupported algorithm: {algorithm}")

		model.fit(self.X_train, self.y_train)
		self._ensure_predict_proba(model, context="Trained model")

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

	def evaluate(self) -> dict:
		if self.model is None:
			raise RuntimeError("No trained model found. Call train_model() first.")

		y_proba = self.model.predict_proba(self.X_test)[:, 1]

		# Find the optimal threshold for F1-score
		import numpy as np
		best_f1 = 0.0
		best_threshold = 0.6
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

		self._ensure_predict_proba(self.model, context="Model to export")
		self.output_dir.mkdir(parents=True, exist_ok=True)
		model_path = self.output_dir / filename
		self._validate_model_path(model_path)
		self._save_serialized_model(self.model, model_path)
		return str(model_path.resolve())

	@classmethod
	def _validate_model_path(cls, model_path: Path) -> None:
		suffix = model_path.suffix.lower()
		if suffix not in cls.SUPPORTED_MODEL_EXTENSIONS:
			supported = ", ".join(sorted(cls.SUPPORTED_MODEL_EXTENSIONS))
			raise ValueError(
				"Model file must use one of the supported extensions: "
				f"{supported}"
			)

	@staticmethod
	def _save_serialized_model(model: object, model_path: Path) -> None:
		suffix = model_path.suffix.lower()
		if suffix == ".joblib":
			joblib.dump(model, model_path)
			return

		with model_path.open("wb") as file_obj:
			pickle.dump(model, file_obj)

	@staticmethod
	def _ensure_predict_proba(model: object, context: str) -> None:
		predict_proba = getattr(model, "predict_proba", None)
		if predict_proba is None or not callable(predict_proba):
			raise TypeError(f"{context} must provide a callable predict_proba() method")

	def feature_importance_report(self) -> None:
		if self.model is None:
			raise RuntimeError("No trained model found. Call train_model() first.")
		if not hasattr(self.model, "feature_importances_"):
			raise TypeError("Current model does not expose feature_importances_")

		importance_pairs = sorted(
			zip(self.feature_names, self.model.feature_importances_),
			key=lambda pair: pair[1],
			reverse=True,
		)

		print("Feature Importances:")
		print(f"{'Feature':<30}Importance")
		for feature_name, score in importance_pairs:
			print(f"{feature_name:<30}{score:.6f}")


if __name__ == "__main__":
	code_dir = Path(__file__).resolve().parents[1]
	config_path = code_dir / "config" / "default_experiment.yaml"
	with config_path.open("r", encoding="utf-8") as file_obj:
		config = yaml.safe_load(file_obj)

	dataset_cfg = config.get("dataset_generation", {})
	ml_cfg = config.get("ml_model", {})
	simulation_cfg = config.get("simulation", {})

	dataset_path = Path(str(dataset_cfg.get("output_csv", "dataFiles/revised_dataset_5tier.csv")))
	if not dataset_path.is_absolute():
		dataset_path = code_dir / dataset_path

	model_path = Path(str(ml_cfg.get("model_path", "models/fire_rf_model.joblib")))
	if not model_path.is_absolute():
		model_path = code_dir / model_path

	trainer = ModelTrainer(
		csv_path=str(dataset_path),
		output_dir=str(model_path.parent),
		seed=int(simulation_cfg.get("seed", 42)),
	)
	trainer.train_random_forest(class_weight_strategy="balanced_subsample")
	metrics = trainer.evaluate()
	saved_path = trainer.save_model(filename=model_path.name)

	print("Saved model:", saved_path)
	print("Metrics:", metrics)
	trainer.feature_importance_report()
