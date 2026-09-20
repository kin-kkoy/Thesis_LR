"""Assemble per-cell environmental and dynamic features for ML fire-ignition prediction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.ndimage import correlate

from .wind_convention import WindContract


SET_C_SCHEMA_VERSION = "set_c.v1"
TARGET_NAME = "Ignited"
TARGET_VERSION = "ignition_t_plus_1.v1"
POSITIVE_LABEL = 1
OBSERVATION_MANIFEST_VERSION = "set_c_observation_manifest.v1"
COMBINED_MANIFEST_VERSION = "set_c_combined_manifest.v1"
SPLIT_MANIFEST_VERSION = "set_c_split_manifest.v1"
CALIBRATION_SELECTION_VERSION = "set_c_calibration_selection.v1"
MODEL_MANIFEST_VERSION = "set_c_model_manifest.v1"

CANONICAL_FEATURE_NAMES = (
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
)

OBSERVATION_METADATA_NAMES = (
	"set_id",
	"experiment_id",
	"event_id",
	"scenario_id",
	"run_id",
	"seed",
	"timestep_t",
	"cell_row",
	"cell_col",
	"grid_id",
	"duplicate_group_id",
	"spatial_block_id",
)

SPLIT_ASSIGNMENT_KEY_NAMES = (
	"experiment_id",
	"event_id",
	"scenario_id",
	"run_id",
	"timestep_t",
	"grid_id",
	"cell_row",
	"cell_col",
	"duplicate_group_id",
	"spatial_block_id",
)

PROVENANCE_METADATA_NAMES = OBSERVATION_METADATA_NAMES + ("split_role",)

SPLIT_ROLES = ("train", "validation", "calibration", "test")

FEATURE_DTYPES = tuple("float32" for _ in CANONICAL_FEATURE_NAMES)
FEATURE_DEFINITIONS = {
	"slope_risk": "clipped slope-derived risk in [0, 1]",
	"proximity_risk": "finite proximity-derived risk",
	"building_presence": "binary building eligibility at time t",
	"material_risk": "risk mapped from material_class",
	"material_class": "integer building-material class 0..5",
	"wind_speed": "non-negative scenario wind speed in km/h",
	"wind_sin": "sine of normalized meteorological from-bearing",
	"wind_cos": "cosine of normalized meteorological from-bearing",
	"neighbor_burning_count": "integer count of burning Moore neighbors at time t",
	"composite_flammability": "building_presence multiplied by material_risk",
	"wind_weighted_score": "normalized directional burning-neighbor score at time t",
}

MOORE_NEIGHBOR_KERNEL = np.array(
	[[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8
)


def validate_binary_ca_state(state: object, name: str = "state") -> np.ndarray:
	"""Require the approved binary CA representation without recasting it."""
	values = np.asarray(state)
	if values.ndim != 2:
		raise ValueError(f"{name} must be a two-dimensional CA state")
	if values.dtype != np.dtype(np.int8):
		raise TypeError(f"{name} must use np.int8 CA state encoding")
	if not np.all(np.isin(values, (0, 1))):
		raise ValueError(f"{name} must contain only binary values 0 and 1")
	return values


def normalize_binary_ca_state(
	state: object, raster_valid_mask: object, name: str = "state"
) -> np.ndarray:
	"""Normalize raster-invalid cells to zero and validate all declared-valid cells."""
	values = np.asarray(state)
	valid_mask = np.asarray(raster_valid_mask, dtype=bool)
	if values.ndim != 2 or valid_mask.shape != values.shape:
		raise ValueError(f"{name} and its raster-valid mask must be equal 2D shapes")
	valid_values = values[valid_mask]
	integer_like = np.isfinite(valid_values) & np.isclose(
		valid_values, np.rint(valid_values), atol=1e-6
	)
	if not np.all(integer_like) or not np.all(np.isin(np.rint(valid_values), (0, 1))):
		raise ValueError(f"Raster-valid {name} cells must contain only binary values 0 and 1")
	normalized = np.zeros(values.shape, dtype=np.int8)
	normalized[valid_mask] = np.rint(valid_values).astype(np.int8)
	return normalized


def compute_moore_neighbor_count(state_t: object) -> np.ndarray:
	"""Count burning Moore neighbors at t with constant, non-wrapping edges."""
	state = validate_binary_ca_state(state_t, "state_t")
	counts = correlate(
		state,
		MOORE_NEIGHBOR_KERNEL,
		mode="constant",
		cval=0,
		output=np.int16,
	)
	return counts.astype(np.int8)


def validate_feature_columns(columns: Iterable[object]) -> tuple[str, ...]:
	"""Require the exact approved Set C predictor names and order."""
	actual = tuple(str(column) for column in columns)
	if actual != CANONICAL_FEATURE_NAMES:
		raise ValueError(
			"Feature schema mismatch. "
			f"Expected {CANONICAL_FEATURE_NAMES}, got {actual}."
		)
	return actual


def validate_feature_values(features: object) -> np.ndarray:
	"""Validate verified Set C value contracts without inventing proximity bounds."""
	values = np.asarray(features, dtype=np.float32)
	if values.ndim != 2 or values.shape[1] != len(CANONICAL_FEATURE_NAMES):
		raise ValueError("Feature matrix must have one column per canonical Set C feature")
	if not np.all(np.isfinite(values)):
		raise ValueError("Feature matrix contains non-finite values")
	column = {name: values[:, index] for index, name in enumerate(CANONICAL_FEATURE_NAMES)}
	if np.any((column["slope_risk"] < 0) | (column["slope_risk"] > 1)):
		raise ValueError("slope_risk must be within [0, 1]")
	if not np.all(np.isin(column["building_presence"], (0.0, 1.0))):
		raise ValueError("building_presence must be binary")
	material_class = column["material_class"]
	if not np.all(np.isclose(material_class, np.rint(material_class))):
		raise ValueError("material_class must be integer-like")
	if np.any((material_class < 0) | (material_class > 5)):
		raise ValueError("material_class must be within 0..5")
	allowed_risk = FeatureAssembler.MATERIAL_CLASS_TO_RISK[
		np.rint(material_class).astype(np.int8)
	]
	allowed_risk = np.where(column["building_presence"] > 0, allowed_risk, 0.0)
	if not np.allclose(column["material_risk"], allowed_risk, atol=1e-6):
		raise ValueError("material_risk must match material_class")
	if np.any(column["wind_speed"] < 0):
		raise ValueError("wind_speed must be non-negative")
	for name in ("wind_sin", "wind_cos"):
		if np.any((column[name] < -1) | (column[name] > 1)):
			raise ValueError(f"{name} must be within [-1, 1]")
	neighbor_count = column["neighbor_burning_count"]
	if not np.all(np.isclose(neighbor_count, np.rint(neighbor_count))):
		raise ValueError("neighbor_burning_count must be integer-like")
	if np.any((neighbor_count < 0) | (neighbor_count > 8)):
		raise ValueError("neighbor_burning_count must be within 0..8")
	expected_composite = column["building_presence"] * column["material_risk"]
	if not np.allclose(column["composite_flammability"], expected_composite, atol=1e-6):
		raise ValueError("composite_flammability does not match its derivation")
	if np.any(
		(column["wind_weighted_score"] < 0)
		| (column["wind_weighted_score"] > 1)
	):
		raise ValueError("wind_weighted_score must be within [0, 1]")
	return values


def validate_model_feature_schema(model: object) -> tuple[str, ...]:
	"""Fail closed unless a fitted estimator records the exact Set C schema."""
	feature_names = getattr(model, "feature_names_in_", None)
	if feature_names is None:
		raise ValueError("Model must expose feature_names_in_ for schema validation")
	return validate_feature_columns(feature_names)


def positive_class_index(classes: Iterable[object]) -> int:
	"""Return the probability-column index for the approved integer label 1."""
	class_values = np.asarray(list(classes), dtype=object)
	if class_values.ndim != 1:
		raise ValueError("Model classes_ must be one-dimensional")

	matches = [
		index
		for index, value in enumerate(class_values)
		if isinstance(value, (int, np.integer))
		and not isinstance(value, (bool, np.bool_))
		and int(value) == POSITIVE_LABEL
	]
	if len(matches) != 1:
		raise ValueError(
			"Model classes_ must contain the integer positive label 1 exactly once"
		)
	return matches[0]


def predict_positive_probability(model: object, features: object) -> np.ndarray:
	"""Return validated probabilities for integer ignition label 1."""
	validate_model_feature_schema(model)
	classes = getattr(model, "classes_", None)
	if classes is None:
		raise ValueError("Model must expose classes_ for positive-class validation")
	class_values = np.asarray(classes)
	positive_index = positive_class_index(class_values)

	predict_proba = getattr(model, "predict_proba", None)
	if predict_proba is None or not callable(predict_proba):
		raise TypeError("Model must provide a callable predict_proba() method")

	probabilities = np.asarray(predict_proba(features), dtype=np.float64)
	if probabilities.ndim != 2:
		raise ValueError("predict_proba() must return a two-dimensional array")
	if probabilities.shape[1] != class_values.size:
		raise ValueError(
			"predict_proba() column count does not match model classes_"
		)
	if not np.all(np.isfinite(probabilities)):
		raise ValueError("predict_proba() returned non-finite values")
	if np.any((probabilities < 0.0) | (probabilities > 1.0)):
		raise ValueError("predict_proba() returned values outside [0, 1]")
	return probabilities[:, positive_index]


class FeatureAssembler:
	VALID_MATERIAL_CLASSES = np.arange(6, dtype=np.int8)
	# 0: No Building, 1: Pure Lightweight, 2: Lightweight,
	# 3: Half, 4: Mostly Concrete, 5: Concrete
	MATERIAL_CLASS_TO_RISK = np.array([0.0, 0.95, 0.8, 0.48, 0.30, 0.15], dtype=np.float32)

	FEATURE_NAMES = list(CANONICAL_FEATURE_NAMES)

	def __init__(self, environment: dict, wind_config: dict, flammability_weights: dict | None = None):
		self.slope_risk = np.asarray(environment["slope_risk"], dtype=np.float32)
		self.proximity_risk = np.asarray(environment["proximity_risk"], dtype=np.float32)
		self.building_presence = np.asarray(environment["building_presence"], dtype=np.float32)

		if "material_class" not in environment:
			raise ValueError(
				"Missing material_class in environment. "
				"Expected integer classes 0..5 from stack_materials.tif"
			)

		raw_material = np.asarray(environment["material_class"], dtype=np.float32)

		if not np.all(np.isfinite(raw_material)):
			raise ValueError("material_class contains non-finite values")

		material_class = np.rint(raw_material).astype(np.int8)
		is_integer_like = np.isclose(raw_material, material_class.astype(np.float32), atol=1e-6)
		is_valid_class = np.isin(material_class, self.VALID_MATERIAL_CLASSES)
		valid_mask = is_integer_like & is_valid_class
		if not np.all(valid_mask):
			bad_values = np.unique(raw_material[~valid_mask])
			preview = ", ".join(f"{float(v):.3f}" for v in bad_values[:10])
			raise ValueError(
				"stack_materials.tif must contain only integer classes 0..5. "
				f"Found invalid values: {preview}"
			)

		self.material_class = material_class
		self.material_risk = self.MATERIAL_CLASS_TO_RISK[self.material_class]
		# Enforce class 0 semantics as empty/no-building regardless of buildings raster.
		self.building_presence = np.where(self.material_class == 0, 0.0, self.building_presence)
		self.material_risk = np.where(self.building_presence > 0, self.material_risk, 0.0).astype(np.float32)

		self.grid_shape = environment["grid_shape"]
		incoming_burnable = np.asarray(environment["burnable_mask"], dtype=bool)
		if incoming_burnable.shape != self.grid_shape:
			raise ValueError(
				"burnable_mask shape mismatch: "
				f"{incoming_burnable.shape} != {self.grid_shape}"
			)
		self.burnable_mask = incoming_burnable & (self.building_presence > 0)

		self.wind = WindContract.from_config(wind_config)
		self.wind_speed = self.wind.speed_kmh
		self.wind_sin = self.wind.from_sin
		self.wind_cos = self.wind.from_cos
		self.wind_dy, self.wind_dx = self.wind.source_direction_rc
		(
			self.wind_propagation_dr,
			self.wind_propagation_dc,
		) = self.wind.propagation_direction_rc

		self.composite_flammability = (
			self.building_presence * self.material_risk
		).astype(np.float32)

		self.feature_names = list(self.FEATURE_NAMES)

	def assemble_grid_features(
		self, blazing_neighbor_count: np.ndarray, wind_weighted_score: np.ndarray
	) -> np.ndarray:
		if blazing_neighbor_count.shape != self.grid_shape:
			raise ValueError(
				"blazing_neighbor_count shape mismatch: "
				f"{blazing_neighbor_count.shape} != {self.grid_shape}"
			)

		if wind_weighted_score.shape != self.grid_shape:
			raise ValueError(
				"wind_weighted_score shape mismatch: "
				f"{wind_weighted_score.shape} != {self.grid_shape}"
			)

		blazing_neighbor_count = np.asarray(blazing_neighbor_count, dtype=np.float32)
		wind_weighted_score = np.asarray(wind_weighted_score, dtype=np.float32)

		n_cells = int(np.prod(self.grid_shape))
		wind_speed_col = np.full(n_cells, self.wind_speed, dtype=np.float32)
		wind_sin_col = np.full(n_cells, self.wind_sin, dtype=np.float32)
		wind_cos_col = np.full(n_cells, self.wind_cos, dtype=np.float32)

		features = np.column_stack(
			[
				self.slope_risk.ravel(),
				self.proximity_risk.ravel(),
				self.building_presence.ravel(),
				self.material_risk.ravel(),
				self.material_class.ravel(),
				wind_speed_col,
				wind_sin_col,
				wind_cos_col,
				blazing_neighbor_count.ravel(),
				self.composite_flammability.ravel(),
				wind_weighted_score.ravel(),
			]
		).astype(np.float32)

		return validate_feature_values(features)

	def assemble_masked_features(
		self, blazing_neighbor_count: np.ndarray, wind_weighted_score: np.ndarray
	) -> tuple[np.ndarray, np.ndarray]:
		features_full = self.assemble_grid_features(blazing_neighbor_count, wind_weighted_score)
		burnable_flat = self.burnable_mask.ravel().astype(bool)
		mask_indices = np.flatnonzero(burnable_flat)
		features_masked = features_full[burnable_flat]
		return features_masked, mask_indices
