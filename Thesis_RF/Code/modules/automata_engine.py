"""Cellular automata fire spread engine with vectorized timestep updates."""

from pathlib import Path

import joblib
import numpy as np

from .feature_pipeline import (
	CANONICAL_FEATURE_NAMES,
	FeatureAssembler,
	compute_moore_neighbor_count,
	positive_class_index,
	predict_positive_probability,
	validate_model_feature_schema,
)
from .wind_convention import WindContract, compute_wind_weighted_score


STATE_NON_BURNABLE = np.int8(1)
STATE_NOT_YET_BURNING = np.int8(2)
STATE_IGNITED = np.int8(3)
STATE_BLAZING = np.int8(4)
STATE_EXTINGUISHED = np.int8(5)

class FireAutomata:
	def __init__(self, environment: dict, config: dict):
		self.environment = environment
		self.config = config

		self.slope_risk = environment["slope_risk"]
		self.proximity_risk = environment["proximity_risk"]
		self.building_presence = np.asarray(environment["building_presence"], dtype=np.float32)
		self.material_class = np.asarray(environment["material_class"], dtype=np.int8)
		self.material_risk = FeatureAssembler.MATERIAL_CLASS_TO_RISK[self.material_class]
		self.grid_shape = environment["grid_shape"]
		incoming_burnable = np.asarray(environment["burnable_mask"], dtype=bool)
		self.nodata_mask = np.asarray(environment["nodata_mask"], dtype=bool)
		for name, array in (
			("building_presence", self.building_presence),
			("burnable_mask", incoming_burnable),
			("nodata_mask", self.nodata_mask),
		):
			if array.shape != self.grid_shape:
				raise ValueError(f"{name} shape mismatch: {array.shape} != {self.grid_shape}")
		self.burnable_mask = (
			incoming_burnable
			& (~self.nodata_mask)
			& (self.building_presence > 0)
		)
		self.transform = environment["transform"]
		self.crs = environment["crs"]

		self.simulation_cfg = config.get("simulation", {})
		self.wind_cfg = config.get("wind", {})
		self.wind = WindContract.from_config(self.wind_cfg)
		self.transition_cfg = config.get("placeholder_transition", {})
		self.ml_cfg = config.get("ml_model", {})
		self._validate_ml_inference_config()
		self.flammability_weights = config.get("flammability_weights", {})
		self.output_cfg = config.get("output", {})

		self.grid = np.full(self.grid_shape, STATE_NOT_YET_BURNING, dtype=np.int8)
		self.grid[~self.burnable_mask] = STATE_NON_BURNABLE
		self.ignition_timers = np.zeros(self.grid_shape, dtype=np.int16)
		self.blazing_timers = np.zeros(self.grid_shape, dtype=np.int16)
		self.t_3_to_4 = int(self.transition_cfg.get("T_3_to_4", 1))
		self.t_4_to_5 = int(self.transition_cfg.get("T_4_to_5", 1))

		self.rng = np.random.default_rng(self.simulation_cfg["seed"])
		self.model = None
		self._ml_enabled = False
		self.timestep = 0

		self._precompute_base_probability()
		self.wind_kernel = self._compute_wind_kernel()
		self.feature_assembler = FeatureAssembler(
			environment,
			self.wind_cfg,
			self.flammability_weights,
		)

	def _precompute_base_probability(self) -> None:
		base_ignition_prob = np.float32(self.transition_cfg.get("base_ignition_prob", 0.0))
		slope_weight = np.float32(self.transition_cfg.get("slope_weight", 0.0))
		building_weight = np.float32(self.transition_cfg.get("building_weight", 0.0))
		proximity_weight = np.float32(self.transition_cfg.get("proximity_weight", 0.0))
		material_weight = np.float32(self.transition_cfg.get("material_weight", 0.0))

		self.p_base = np.clip(
			base_ignition_prob
			+ slope_weight * self.slope_risk
			+ building_weight * self.building_presence
			+ proximity_weight * self.proximity_risk
			+ material_weight * self.material_risk,
			0.0,
			1.0,
		).astype(np.float32)

	def _validate_ml_inference_config(self) -> None:
		mode = self.ml_cfg.get("inference_mode", "stochastic_probability")
		if mode != "stochastic_probability":
			raise ValueError(
				"Only stochastic_probability ML inference is approved for the CA"
			)
		if int(self.ml_cfg.get("positive_label", 1)) != 1:
			raise ValueError("ml_model.positive_label must be integer 1")
		for threshold_name in ("threshold", "inference_threshold"):
			if self.ml_cfg.get(threshold_name) is not None:
				raise ValueError(
					f"ml_model.{threshold_name} must be absent or null for stochastic CA inference"
				)

	def _compute_wind_kernel(self) -> np.ndarray:
		"""Return the centralized from-bearing kernel for blazing neighbors."""
		c_wind = float(self.transition_cfg.get("wind_weight", 0.20))
		return self.wind.directional_kernel(c_wind)

	def set_ignition(self, points: list[tuple[int, int]]) -> None:
		rows, cols = self.grid_shape
		for row, col in points:
			if row < 0 or col < 0 or row >= rows or col >= cols:
				print(f"Warning: Ignition point ({row}, {col}) is out of bounds.")
				continue
			if self.grid[row, col] == STATE_NON_BURNABLE:
				print(f"Warning: Ignition point ({row}, {col}) is non-burnable.")
				continue
			if self.grid[row, col] == STATE_NOT_YET_BURNING:
				self.grid[row, col] = STATE_BLAZING

	def step(self) -> None:
		try:
			wind_weight = float(self.transition_cfg.get("wind_weight", 0.0))

			current_grid = self.grid
			next_grid = current_grid.copy()

			ignited_now = current_grid == STATE_IGNITED
			blazing_now = current_grid == STATE_BLAZING

			self.ignition_timers[ignited_now] += np.int16(1)
			self.blazing_timers[blazing_now] += np.int16(1)

			ignite_to_blazing = ignited_now & (self.ignition_timers >= self.t_3_to_4)
			next_grid[ignite_to_blazing] = STATE_BLAZING
			self.ignition_timers[ignite_to_blazing] = np.int16(0)
			self.blazing_timers[ignite_to_blazing] = np.int16(0)

			blazing_to_extinguished = blazing_now & (self.blazing_timers >= self.t_4_to_5)
			next_grid[blazing_to_extinguished] = STATE_EXTINGUISHED
			self.blazing_timers[blazing_to_extinguished] = np.int16(0)

			blazing_state = blazing_now.astype(np.int8)
			blazing_neighbor_count = compute_moore_neighbor_count(blazing_state)
			wind_weighted_score, _ = compute_wind_weighted_score(
				blazing_state,
				self.wind,
				wind_weight,
			)
			susceptible = (current_grid == STATE_NOT_YET_BURNING) & (blazing_neighbor_count > 0)

			if np.any(susceptible):
				if self.model is not None:
					# The calibrated model probability is the Bernoulli parameter.
					# Wind is already represented in the approved feature schema.
					p_effective = self._predict_with_model(
						blazing_neighbor_count,
						susceptible,
						wind_weighted_score,
					)
				else:
					p_effective = np.clip(
						self.p_base * wind_weighted_score,
						0.0,
						1.0,
					)
				ignition_draw = self.rng.random(self.grid_shape)
				ignite_mask = susceptible & (ignition_draw < p_effective)
				next_grid[ignite_mask] = STATE_IGNITED
				self.ignition_timers[ignite_mask] = np.int16(0)
				self.blazing_timers[ignite_mask] = np.int16(0)

			self.grid = next_grid
			self.timestep += 1
		except Exception as exc:
			checkpoint_path = self._save_emergency_checkpoint()
			raise RuntimeError(
				"Simulation step failed and an emergency checkpoint was saved at "
				f"{checkpoint_path}. Configure simulation.resume_checkpoint to continue."
			) from exc

	def load_model(self, model_path: str) -> None:
		loaded_model = joblib.load(model_path)
		validate_model_feature_schema(loaded_model)
		classes = getattr(loaded_model, "classes_", None)
		if classes is None:
			raise ValueError("Loaded model must expose classes_")
		positive_class_index(classes)
		predict_proba = getattr(loaded_model, "predict_proba", None)
		if predict_proba is None or not callable(predict_proba):
			raise TypeError("Loaded model must provide a callable predict_proba attribute")
		self.model = loaded_model
		self._ml_enabled = True

	def save_checkpoint(self, filepath: str) -> None:
		grid_to_save = self.grid.astype(np.int8, copy=False)
		ignition_timers_to_save = self.ignition_timers.astype(np.int16, copy=False)
		blazing_timers_to_save = self.blazing_timers.astype(np.int16, copy=False)
		np.savez_compressed(
			filepath,
			grid=grid_to_save,
			ignition_timers=ignition_timers_to_save,
			blazing_timers=blazing_timers_to_save,
			timestep=np.int64(self.timestep),
		)

	def _save_emergency_checkpoint(self) -> str:
		output_dir = Path(self.output_cfg.get("output_dir", "output"))
		if not output_dir.is_absolute():
			output_dir = Path(__file__).resolve().parents[1] / output_dir
		output_dir.mkdir(parents=True, exist_ok=True)
		checkpoint_path = output_dir / f"emergency_checkpoint_step_{self.timestep:06d}.npz"
		self.save_checkpoint(str(checkpoint_path))
		return str(checkpoint_path)

	def load_checkpoint(self, filepath: str) -> None:
		with np.load(filepath, allow_pickle=False) as checkpoint:
			if (
				"grid" not in checkpoint
				or "ignition_timers" not in checkpoint
				or "blazing_timers" not in checkpoint
				or "timestep" not in checkpoint
			):
				raise KeyError(
					"Checkpoint must contain 'grid', 'ignition_timers', 'blazing_timers', and 'timestep' arrays"
				)

			loaded_grid = checkpoint["grid"]
			loaded_ignition_timers = checkpoint["ignition_timers"]
			loaded_blazing_timers = checkpoint["blazing_timers"]
			loaded_timestep = checkpoint["timestep"]

		if loaded_grid.shape != self.grid_shape:
			raise ValueError(
				f"Checkpoint grid shape {loaded_grid.shape} does not match expected {self.grid_shape}"
			)
		if loaded_grid.dtype != np.int8:
			raise TypeError(f"Checkpoint grid dtype must be int8, got {loaded_grid.dtype}")
		if loaded_ignition_timers.shape != self.grid_shape:
			raise ValueError(
				"Checkpoint ignition_timers shape "
				f"{loaded_ignition_timers.shape} does not match expected {self.grid_shape}"
			)
		if loaded_blazing_timers.shape != self.grid_shape:
			raise ValueError(
				"Checkpoint blazing_timers shape "
				f"{loaded_blazing_timers.shape} does not match expected {self.grid_shape}"
			)
		if loaded_ignition_timers.dtype != np.int16:
			raise TypeError(
				"Checkpoint ignition_timers dtype must be int16, "
				f"got {loaded_ignition_timers.dtype}"
			)
		if loaded_blazing_timers.dtype != np.int16:
			raise TypeError(
				"Checkpoint blazing_timers dtype must be int16, "
				f"got {loaded_blazing_timers.dtype}"
			)

		self.grid = loaded_grid.copy()
		self.ignition_timers = loaded_ignition_timers.copy()
		self.blazing_timers = loaded_blazing_timers.copy()
		self.timestep = int(np.asarray(loaded_timestep).item())

	def _predict_with_model(
		self,
		blazing_neighbor_count: np.ndarray,
		susceptible_mask: np.ndarray,
		wind_weighted_score: np.ndarray | None = None,
	) -> np.ndarray:
		flat_susceptible = susceptible_mask.ravel().astype(bool)
		susceptible_indices = np.flatnonzero(flat_susceptible)

		ignition_proba_flat = np.zeros(flat_susceptible.shape[0], dtype=np.float32)
		if susceptible_indices.size == 0:
			return ignition_proba_flat.reshape(self.grid_shape)

		if wind_weighted_score is None:
			wind_weighted_score, _ = compute_wind_weighted_score(
				(self.grid == STATE_BLAZING).astype(np.int8),
				self.wind,
				float(self.transition_cfg.get("wind_weight", 0.0)),
			)

		features_full = self.feature_assembler.assemble_grid_features(
			blazing_neighbor_count,
			wind_weighted_score,
		)
		features_full = features_full.astype(np.float32, copy=False)

		chunk_size = int(self.ml_cfg.get("inference_chunk_size", 200_000))
		if chunk_size <= 0:
			chunk_size = 200_000

		validate_model_feature_schema(self.model)
		classes = getattr(self.model, "classes_", None)
		if classes is None:
			raise ValueError("Model must expose classes_")
		positive_class_index(classes)
		import pandas as pd

		chunk_probabilities: list[np.ndarray] = []
		for start in range(0, susceptible_indices.size, chunk_size):
			end = min(start + chunk_size, susceptible_indices.size)
			idx = susceptible_indices[start:end]

			features_chunk = features_full[idx]

			features_input = pd.DataFrame(
				features_chunk,
				columns=CANONICAL_FEATURE_NAMES,
			)
			proba_chunk = predict_positive_probability(self.model, features_input)
			chunk_probabilities.append(proba_chunk.astype(np.float32))

		ignition_proba_flat[susceptible_indices] = np.concatenate(chunk_probabilities)
		return ignition_proba_flat.reshape(self.grid_shape)

	def is_active(self) -> bool:
		return bool(np.any((self.grid == STATE_IGNITED) | (self.grid == STATE_BLAZING)))

	def get_state_counts(self) -> dict[str, int]:
		return {
			"non_burnable": int(np.count_nonzero(self.grid == STATE_NON_BURNABLE)),
			"not_yet_burning": int(np.count_nonzero(self.grid == STATE_NOT_YET_BURNING)),
			"ignited": int(np.count_nonzero(self.grid == STATE_IGNITED)),
			"blazing": int(np.count_nonzero(self.grid == STATE_BLAZING)),
			"extinguished": int(np.count_nonzero(self.grid == STATE_EXTINGUISHED)),
		}

	def get_grid(self) -> np.ndarray:
		return self.grid.copy()
