from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import (
	confusion_matrix,
	f1_score,
	jaccard_score,
	precision_score,
	recall_score,
	roc_auc_score,
)

def evaluate_final_state_against_ground_truth(
	final_state_path: str,
	ground_truth_path: str,
	burned_state_value: int = 5,
	ground_truth_burned_value: int = 1,
	target_recall: float = 0.80,
	target_f1: float = 0.80,
) -> dict:
	"""Evaluate simulated final-state burned cells against historical ground truth raster."""
	final_path = Path(final_state_path)
	truth_path = Path(ground_truth_path)

	if not final_path.exists():
		raise FileNotFoundError(f"Final state raster not found: {final_path}")
	if not truth_path.exists():
		raise FileNotFoundError(f"Ground truth raster not found: {truth_path}")

	with rasterio.open(final_path) as pred_src, rasterio.open(truth_path) as truth_src:
		if pred_src.crs != truth_src.crs:
			raise ValueError(
				"CRS mismatch between final state and ground truth: "
				f"{pred_src.crs} != {truth_src.crs}"
			)
		if pred_src.shape != truth_src.shape:
			raise ValueError(
				"Shape mismatch between final state and ground truth: "
				f"{pred_src.shape} != {truth_src.shape}"
			)
		if pred_src.transform != truth_src.transform:
			raise ValueError(
				"Affine transform mismatch between final state and ground truth: "
				f"{pred_src.transform} != {truth_src.transform}"
			)

		pred_arr = pred_src.read(1)
		truth_arr = truth_src.read(1)
		pred_nodata = pred_src.nodata
		truth_nodata = truth_src.nodata

	valid_mask = np.ones(pred_arr.shape, dtype=bool)
	if pred_nodata is not None:
		valid_mask &= pred_arr != pred_nodata
	if truth_nodata is not None:
		valid_mask &= truth_arr != truth_nodata

	if not np.any(valid_mask):
		raise ValueError("No valid overlapping pixels found after nodata masking.")

	y_pred = (pred_arr == burned_state_value) & valid_mask
	y_true = (truth_arr == ground_truth_burned_value) & valid_mask

	y_pred_flat = y_pred.astype(np.uint8).ravel()
	y_true_flat = y_true.astype(np.uint8).ravel()

	cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])
	precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
	recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
	f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
	jaccard = jaccard_score(y_true_flat, y_pred_flat, zero_division=0)

	auc_roc = float("nan")
	if np.unique(y_true_flat).size > 1:
		auc_roc = roc_auc_score(y_true_flat, y_pred_flat)

	return {
		"confusion_matrix": cm.tolist(),
		"precision": float(precision),
		"recall": float(recall),
		"f1_score": float(f1),
		"auc_roc": float(auc_roc),
		"jaccard_index": float(jaccard),
		"target_recall": float(target_recall),
		"target_f1": float(target_f1),
		"recall_meets_target": bool(recall >= target_recall),
		"f1_meets_target": bool(f1 >= target_f1),
		"burned_state_value": int(burned_state_value),
		"ground_truth_burned_value": int(ground_truth_burned_value),
		"valid_pixel_count": int(np.count_nonzero(valid_mask)),
	}


def run_default_validation() -> dict:
	"""Run validation using repository default raster paths."""
	code_dir = Path(__file__).resolve().parent
	# final_state_path = code_dir / "output" / "final_state.tif"
	# For baseline validation, we compare against the "final_state_baseline.tif" which is generated using the same parameters as the "stack_ground_truth.tif" to ensure a fair comparison.
	# final_state_path = code_dir / "output" / "final_state_baseline.tif"
	final_state_path = code_dir / "output" / "final_sitio_santa_maria_11_params.tif"
	ground_truth_path = code_dir / "processedData" / "raster" / "raster" / "stack_ground_truth.tif"
	# ground_truth_path = code_dir / "processedData" / "raster" / "raster" / "stack_ground_truth.tif"

	metrics = evaluate_final_state_against_ground_truth(
		final_state_path=str(final_state_path),
		ground_truth_path=str(ground_truth_path),
		burned_state_value=5,
		ground_truth_burned_value=1,
	)

	print("Validation metrics against stack_ground_truth.tif:")
	print(f"Confusion Matrix: {metrics['confusion_matrix']}")
	print(f"Precision: {metrics['precision']:.6f}")
	print(f"Recall: {metrics['recall']:.6f} (target >= {metrics['target_recall']:.2f})")
	print(f"F1 Score: {metrics['f1_score']:.6f} (target >= {metrics['target_f1']:.2f})")
	print(f"AUC-ROC: {metrics['auc_roc']:.6f}")
	print(f"Jaccard Index: {metrics['jaccard_index']:.6f}")

	return metrics


if __name__ == "__main__":
	run_default_validation()
