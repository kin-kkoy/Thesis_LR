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
PRIMARY_FINAL_FOOTPRINT_DOMAIN = (
	"simulation_environment_valid & prediction_valid & reference_valid & "
	"building_raster_valid & mapped_building"
)
REFERENCE_BURN_EXCLUSION_PRIORITY = (
	"outside_environmental_validity",
	"outside_prediction_validity",
	"outside_building_raster_validity",
	"outside_mapped_buildings",
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
	if np.all(np.isin(score, (0.0, 1.0))):
		raise ValueError(
			"y_score must contain continuous estimator probabilities, not a hard "
			"binary prediction mask"
		)

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


def _require_boolean_mask(
	name: str,
	value: np.ndarray,
	expected_shape: tuple[int, ...],
) -> np.ndarray:
	mask = np.asarray(value)
	if mask.dtype != np.bool_:
		raise ValueError(f"{name} must be a boolean mask")
	if mask.shape != expected_shape:
		raise ValueError(
			f"{name} shape {mask.shape} does not match expected shape {expected_shape}"
		)
	return mask


def evaluate_final_footprint_arrays(
	final_state: np.ndarray,
	reference_burned_mask: np.ndarray,
	simulation_environment_valid_mask: np.ndarray,
	prediction_valid_mask: np.ndarray,
	reference_valid_mask: np.ndarray,
	building_raster_valid_mask: np.ndarray,
	mapped_building_mask: np.ndarray,
	*,
	ignited_state_value: int = 3,
	blazing_state_value: int = 4,
	extinguished_state_value: int = 5,
	termination_reason: str = "inactive",
	include_secondary_whole_reference: bool = False,
) -> dict:
	"""Evaluate a CA footprint on the approved common-valid building domain.

	Reference-burn exclusion marginals can overlap. The separately reported
	priority partition applies the documented order in
	``REFERENCE_BURN_EXCLUSION_PRIORITY`` so its counts reconcile exactly.
	Reference-invalid cells are unknown and are never classified as burned
	exclusions.
	"""
	state = np.asarray(final_state)
	if state.ndim != 2:
		raise ValueError("final_state must be a two-dimensional array")
	shape = state.shape
	reference_burned = _require_boolean_mask(
		"reference_burned_mask", reference_burned_mask, shape
	)
	environment_valid = _require_boolean_mask(
		"simulation_environment_valid_mask",
		simulation_environment_valid_mask,
		shape,
	)
	prediction_valid = _require_boolean_mask(
		"prediction_valid_mask", prediction_valid_mask, shape
	)
	reference_valid = _require_boolean_mask(
		"reference_valid_mask", reference_valid_mask, shape
	)
	building_valid = _require_boolean_mask(
		"building_raster_valid_mask", building_raster_valid_mask, shape
	)
	mapped_building = _require_boolean_mask(
		"mapped_building_mask", mapped_building_mask, shape
	)

	state_values = {
		int(ignited_state_value),
		int(blazing_state_value),
		int(extinguished_state_value),
	}
	if len(state_values) != 3:
		raise ValueError("Ignited, blazing, and extinguished state values must differ")
	if termination_reason not in {"inactive", "timestep_limit"}:
		raise ValueError(
			"termination_reason must be either 'inactive' or 'timestep_limit'"
		)

	active_mask = np.isin(state, (ignited_state_value, blazing_state_value))
	active_cell_count = int(np.count_nonzero(active_mask))
	if termination_reason == "inactive" and active_cell_count:
		raise ValueError(
			"An inactive termination cannot contain IGNITED or BLAZING cells"
		)

	complete_final_footprint = termination_reason == "inactive"
	if complete_final_footprint:
		predicted_footprint = state == extinguished_state_value
		footprint_status = "complete_inactive_final_footprint"
		footprint_state_values = [int(extinguished_state_value)]
	else:
		predicted_footprint = np.isin(
			state,
			(ignited_state_value, blazing_state_value, extinguished_state_value),
		)
		footprint_status = "censored_incomplete_footprint_through_stopping_time"
		footprint_state_values = sorted(state_values)

	primary_domain = (
		environment_valid
		& prediction_valid
		& reference_valid
		& building_valid
		& mapped_building
	)
	metrics = evaluate_binary_spatial_masks(
		reference_burned,
		predicted_footprint,
		primary_domain,
	)

	reference_valid_burned = reference_valid & reference_burned
	included_reference_burned = reference_valid_burned & primary_domain
	marginal_exclusions = {
		"outside_environmental_validity": int(
			np.count_nonzero(reference_valid_burned & (~environment_valid))
		),
		"outside_prediction_validity": int(
			np.count_nonzero(reference_valid_burned & (~prediction_valid))
		),
		"outside_building_raster_validity": int(
			np.count_nonzero(reference_valid_burned & (~building_valid))
		),
		"outside_mapped_buildings": int(
			np.count_nonzero(
				reference_valid_burned & building_valid & (~mapped_building)
			)
		),
	}

	remaining = reference_valid_burned.copy()
	priority_partition = {}
	for category, valid_mask in (
		("outside_environmental_validity", environment_valid),
		("outside_prediction_validity", prediction_valid),
		("outside_building_raster_validity", building_valid),
		("outside_mapped_buildings", mapped_building),
	):
		category_mask = remaining & (~valid_mask)
		priority_partition[category] = int(np.count_nonzero(category_mask))
		remaining &= valid_mask
	priority_partition["included_in_primary_domain"] = int(
		np.count_nonzero(remaining)
	)

	total_reference_valid_burned = int(np.count_nonzero(reference_valid_burned))
	included_reference_burned_count = int(
		np.count_nonzero(included_reference_burned)
	)
	excluded_reference_burned_count = (
		total_reference_valid_burned - included_reference_burned_count
	)
	partition_total = int(sum(priority_partition.values()))
	if partition_total != total_reference_valid_burned:
		raise RuntimeError("Reference-burn exclusion partition does not reconcile")

	metrics.update(
		{
			"reference_kind": REFERENCE_BURN_MASK_DESCRIPTION,
			"metric_scope": FINAL_FOOTPRINT_METRIC_SCOPE,
			"metric_population": "primary_common_valid_mapped_building",
			"primary_domain_definition": PRIMARY_FINAL_FOOTPRINT_DOMAIN,
			"domain_component_pixel_counts": {
				"simulation_environment_valid": int(
					np.count_nonzero(environment_valid)
				),
				"prediction_valid": int(np.count_nonzero(prediction_valid)),
				"reference_valid": int(np.count_nonzero(reference_valid)),
				"building_raster_valid": int(np.count_nonzero(building_valid)),
				"mapped_building": int(np.count_nonzero(mapped_building)),
			},
			"reference_coverage": {
				"reference_valid_burned_total": total_reference_valid_burned,
				"reference_burned_included_in_primary_domain": (
					included_reference_burned_count
				),
				"reference_burned_excluded_from_primary_domain": (
					excluded_reference_burned_count
				),
				"reference_invalid_pixel_count": int(
					np.count_nonzero(~reference_valid)
				),
				"reference_invalid_semantics": (
					"unknown; never counted as reference-burned exclusions"
				),
				"exclusion_marginal_counts": marginal_exclusions,
				"exclusion_marginals_are_additive": False,
				"exclusion_priority_order": list(
					REFERENCE_BURN_EXCLUSION_PRIORITY
				),
				"exclusion_priority_partition": priority_partition,
				"priority_partition_reconciles": True,
			},
			"termination": {
				"reason": termination_reason,
				"active_cell_count": active_cell_count,
				"is_complete_final_footprint": complete_final_footprint,
				"status": footprint_status,
				"predicted_footprint_state_values": footprint_state_values,
			},
		}
	)

	if include_secondary_whole_reference:
		secondary_domain = environment_valid & prediction_valid & reference_valid
		secondary_metrics = evaluate_binary_spatial_masks(
			reference_burned,
			predicted_footprint,
			secondary_domain,
		)
		secondary_metrics.update(
			{
				"role": "secondary_descriptive_only",
				"domain_definition": (
					"simulation_environment_valid & prediction_valid & "
					"reference_valid; building restrictions not applied"
				),
				"claim_limit": (
					"not a primary thesis metric and not evidence of temporal "
					"or probability performance"
				),
			}
		)
		metrics["secondary_descriptive_whole_reference"] = secondary_metrics

	return metrics


def evaluate_final_state_against_ground_truth(
	final_state_path: str,
	ground_truth_path: str,
	building_path: str,
	simulation_environment_valid_mask: np.ndarray,
	burned_state_value: int = 5,
	ground_truth_burned_value: int = 1,
	building_value: int = 10,
	ignited_state_value: int = 3,
	blazing_state_value: int = 4,
	termination_reason: str = "inactive",
	include_secondary_whole_reference: bool = False,
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
		prediction_valid_mask = (
			(pred_src.read_masks(1) > 0) & np.isfinite(pred_arr)
		)
		reference_valid_mask = (
			(truth_src.read_masks(1) > 0) & np.isfinite(truth_arr)
		)
		building_raster_valid_mask = (
			(buildings_src.read_masks(1) > 0) & np.isfinite(buildings_arr)
		)

	metrics = evaluate_final_footprint_arrays(
		final_state=pred_arr,
		reference_burned_mask=truth_arr == ground_truth_burned_value,
		simulation_environment_valid_mask=simulation_environment_valid_mask,
		prediction_valid_mask=prediction_valid_mask,
		reference_valid_mask=reference_valid_mask,
		building_raster_valid_mask=building_raster_valid_mask,
		mapped_building_mask=buildings_arr == building_value,
		ignited_state_value=ignited_state_value,
		blazing_state_value=blazing_state_value,
		extinguished_state_value=burned_state_value,
		termination_reason=termination_reason,
		include_secondary_whole_reference=include_secondary_whole_reference,
	)
	metrics.update(
		{
			"burned_state_value": int(burned_state_value),
			"ground_truth_burned_value": int(ground_truth_burned_value),
			"building_value": int(building_value),
		}
	)
	return metrics


def run_default_validation(
	simulation_environment_valid_mask: np.ndarray | None = None,
) -> dict:
	"""Compare the final CA footprint with the approximate satellite reference."""
	if simulation_environment_valid_mask is None:
		raise ValueError(
			"run_default_validation requires the exact simulation-environment-valid "
			"mask bound to the evaluated run"
		)
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
		simulation_environment_valid_mask=simulation_environment_valid_mask,
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
	raise SystemExit(
		"Validation requires the exact simulation-environment-valid mask bound "
		"to the evaluated run; call run_default_validation(mask) explicitly."
	)
