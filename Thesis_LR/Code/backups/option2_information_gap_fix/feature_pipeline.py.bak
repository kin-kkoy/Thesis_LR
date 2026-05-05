"""Assemble per-cell environmental and dynamic features for ML fire-ignition prediction."""

from __future__ import annotations

import numpy as np


class FeatureAssembler:
	FEATURE_NAMES = [
		"slope_risk",
		"proximity_risk",
		"building_presence",
		"material_risk",
		"wind_speed",
		"wind_sin",
		"wind_cos",
		"neighbor_burning_count",
		"composite_flammability",
	]

	def __init__(self, environment: dict, wind_config: dict, flammability_weights: dict | None = None):
		self.slope_risk = np.asarray(environment["slope_risk"], dtype=np.float32)
		self.proximity_risk = np.asarray(environment["proximity_risk"], dtype=np.float32)
		self.building_presence = np.asarray(environment["building_presence"], dtype=np.float32)
		self.material_risk = np.asarray(environment["material_risk"], dtype=np.float32)
		self.burnable_mask = environment["burnable_mask"]
		self.grid_shape = environment["grid_shape"]

		direction_deg = float(wind_config["direction_deg"])
		radians = np.deg2rad(direction_deg)

		self.wind_dx = float(np.sin(radians))
		self.wind_dy = float(-np.cos(radians))

		self.wind_speed = float(wind_config["speed_kmh"])
		self.wind_sin = float(np.sin(radians))
		self.wind_cos = float(np.cos(radians))

		self.composite_flammability = (
			self.building_presence * self.material_risk
		).astype(np.float32)

		self.feature_names = list(self.FEATURE_NAMES)

	def assemble_grid_features(self, blazing_neighbor_count: np.ndarray) -> np.ndarray:
		if blazing_neighbor_count.shape != self.grid_shape:
			raise ValueError(
				"blazing_neighbor_count shape mismatch: "
				f"{blazing_neighbor_count.shape} != {self.grid_shape}"
			)

		blazing_neighbor_count = np.asarray(blazing_neighbor_count, dtype=np.float32)

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
				wind_speed_col,
				wind_sin_col,
				wind_cos_col,
				blazing_neighbor_count.ravel(),
				self.composite_flammability.ravel(),
			]
		).astype(np.float32)

		return features

	def assemble_masked_features(
		self, blazing_neighbor_count: np.ndarray
	) -> tuple[np.ndarray, np.ndarray]:
		features_full = self.assemble_grid_features(blazing_neighbor_count)
		burnable_flat = self.burnable_mask.ravel().astype(bool)
		mask_indices = np.flatnonzero(burnable_flat)
		features_masked = features_full[burnable_flat]
		return features_masked, mask_indices
