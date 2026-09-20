from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import (
	average_precision_score,
	brier_score_loss,
	confusion_matrix,
	f1_score,
	jaccard_score,
	log_loss,
	precision_score,
	recall_score,
	roc_auc_score,
)


REFERENCE_BURN_MASK_DESCRIPTION = (
	"satellite-derived approximate reference burned-area mask"
)
FINAL_FOOTPRINT_METRIC_SCOPE = (
	"final-footprint agreement only; not temporal ignition, arrival-time, "
	"or probability-calibration evidence"
)


def evaluate_estimator_probabilities(y_true: np.ndarray, y_score: np.ndarray) -> dict:
	"""Evaluate continuous positive-class probabilities, separate from CA masks."""
	true = np.asarray(y_true)
	score = np.asarray(y_score, dtype=np.float64)
	if true.ndim != 1 or score.ndim != 1 or true.shape != score.shape:
		raise ValueError("y_true and y_score must be one-dimensional arrays of equal length")
	if true.size == 0:
		raise ValueError("Probability evaluation requires at least one observation")
	if not np.all(np.isin(true, (0, 1))):
		raise ValueError("y_true must contain only integer binary labels 0 and 1")
	if np.unique(true).size != 2:
		raise ValueError("Probability evaluation requires both target classes")
	if not np.all(np.isfinite(score)) or np.any((score < 0.0) | (score > 1.0)):
		raise ValueError("y_score must contain finite probabilities in [0, 1]")

	return {
		"roc_auc": float(roc_auc_score(true, score)),
		"pr_auc": float(average_precision_score(true, score)),
		"brier_score": float(brier_score_loss(true, score, pos_label=1)),
		"log_loss": float(log_loss(true, score, labels=[0, 1])),
		"observation_count": int(true.size),
	}


def evaluate_binary_spatial_masks(
	y_true_mask: np.ndarray,
	y_pred_mask: np.ndarray,
	evaluation_mask: np.ndarray,
) -> dict:
	"""Evaluate hard CA masks only inside an explicitly supplied spatial domain."""
	true_mask = np.asarray(y_true_mask, dtype=bool)
	pred_mask = np.asarray(y_pred_mask, dtype=bool)
	domain = np.asarray(evaluation_mask, dtype=bool)
	if true_mask.shape != pred_mask.shape or true_mask.shape != domain.shape:
		raise ValueError("Truth, prediction, and evaluation masks must have identical shapes")
	population = int(np.count_nonzero(domain))
	if population == 0:
		raise ValueError("No pixels remain in the valid building evaluation domain")

	y_true = true_mask[domain].astype(np.uint8)
	y_pred = pred_mask[domain].astype(np.uint8)
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
	return {
		"confusion_matrix": cm.tolist(),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
		"jaccard_index": float(jaccard_score(y_true, y_pred, zero_division=0)),
		"evaluation_pixel_count": population,
		"total_pixel_count": int(domain.size),
		"excluded_pixel_count": int(domain.size - population),
	}


def evaluate_final_state_against_ground_truth(
	final_state_path: str,
	ground_truth_path: str,
	building_path: str,
	burned_state_value: int = 5,
	ground_truth_burned_value: int = 1,
	building_value: int = 10,
) -> dict:
	"""Evaluate final CA footprint against an approximate satellite reference.

	The historical function and ``ground_truth_path`` parameter names are retained
	for API compatibility. The referenced raster is not precise temporal ground
	truth and these metrics do not validate ignition timing or RF calibration.
	"""
	final_path = Path(final_state_path)
	truth_path = Path(ground_truth_path)
	buildings_path = Path(building_path)

	if not final_path.exists():
		raise FileNotFoundError(f"Final state raster not found: {final_path}")
	if not truth_path.exists():
		raise FileNotFoundError(
			f"Satellite-derived approximate reference burned-area mask not found: {truth_path}"
		)
	if not buildings_path.exists():
		raise FileNotFoundError(f"Building raster not found: {buildings_path}")

	with (
		rasterio.open(final_path) as pred_src,
		rasterio.open(truth_path) as truth_src,
		rasterio.open(buildings_path) as buildings_src,
	):
		for name, source in (
			("satellite-derived approximate reference burned-area mask", truth_src),
			("buildings", buildings_src),
		):
			if pred_src.crs != source.crs:
				raise ValueError(f"CRS mismatch between final state and {name}")
			if pred_src.shape != source.shape:
				raise ValueError(f"Shape mismatch between final state and {name}")
			if pred_src.transform != source.transform:
				raise ValueError(f"Affine transform mismatch between final state and {name}")

		pred_arr = pred_src.read(1)
		truth_arr = truth_src.read(1)
		buildings_arr = buildings_src.read(1)
		valid_mask = (
			(pred_src.read_masks(1) > 0)
			& (truth_src.read_masks(1) > 0)
			& (buildings_src.read_masks(1) > 0)
			& np.isfinite(pred_arr)
			& np.isfinite(truth_arr)
			& np.isfinite(buildings_arr)
		)

	building_mask = buildings_arr == building_value
	evaluation_mask = valid_mask & building_mask
	metrics = evaluate_binary_spatial_masks(
		truth_arr == ground_truth_burned_value,
		pred_arr == burned_state_value,
		evaluation_mask,
	)
	valid_count = int(np.count_nonzero(valid_mask))
	metrics.update(
		{
			"reference_kind": REFERENCE_BURN_MASK_DESCRIPTION,
			"metric_scope": FINAL_FOOTPRINT_METRIC_SCOPE,
			"burned_state_value": int(burned_state_value),
			"ground_truth_burned_value": int(ground_truth_burned_value),
			"building_value": int(building_value),
			"valid_pixel_count": valid_count,
			"invalid_pixel_count": int(valid_mask.size - valid_count),
			"non_building_valid_pixel_count": int(
				np.count_nonzero(valid_mask & (~building_mask))
			),
		}
	)
	return metrics


def run_default_validation() -> dict:
	"""Compare the final CA footprint with the approximate satellite reference."""
	code_dir = Path(__file__).resolve().parent
	# final_state_path = code_dir / "output" / "final_state.tif"
	# The legacy stack_ground_truth.tif filename denotes an approximate,
	# satellite-derived final burned-area reference, not temporal ground truth.
	# final_state_path = code_dir / "output" / "final_state_baseline.tif"
	final_state_path = code_dir / "output" / "final_sitio_santa_maria_11_params.tif"
	ground_truth_path = code_dir / "processedData" / "raster" / "raster" / "stack_ground_truth.tif"
	building_path = code_dir / "processedData" / "raster" / "raster" / "stack_buildings.tif"
	# ground_truth_path = code_dir / "processedData" / "raster" / "raster" / "stack_ground_truth.tif"

	metrics = evaluate_final_state_against_ground_truth(
		final_state_path=str(final_state_path),
		ground_truth_path=str(ground_truth_path),
		building_path=str(building_path),
		burned_state_value=5,
		ground_truth_burned_value=1,
	)

	print(
		"Final-footprint agreement against stack_ground_truth.tif "
		"(satellite-derived approximate reference burned-area mask):"
	)
	print("Scope: final-footprint agreement only; no temporal or calibration claim")
	print(f"Confusion Matrix: {metrics['confusion_matrix']}")
	print(f"Precision: {metrics['precision']:.6f}")
	print(f"Recall: {metrics['recall']:.6f}")
	print(f"F1 Score: {metrics['f1_score']:.6f}")
	print(f"Jaccard Index: {metrics['jaccard_index']:.6f}")
	print(f"Evaluation building pixels: {metrics['evaluation_pixel_count']}")

	return metrics


if __name__ == "__main__":
	run_default_validation()
