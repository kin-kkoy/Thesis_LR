"""Cellular automata fire spread engine with vectorized timestep updates."""

import joblib
import numpy as np
from scipy.ndimage import convolve

from .feature_pipeline import FeatureAssembler


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
		self.building_presence = environment["building_presence"]
		self.material_risk = environment["material_risk"]
		self.burnable_mask = environment["burnable_mask"]
		self.nodata_mask = environment["nodata_mask"]
		self.grid_shape = environment["grid_shape"]
		self.transform = environment["transform"]
		self.crs = environment["crs"]

		self.simulation_cfg = config.get("simulation", {})
		self.wind_cfg = config.get("wind", {})
		self.transition_cfg = config.get("placeholder_transition", {})
		self.ml_cfg = config.get("ml_model", {})
		self.flammability_weights = config.get("flammability_weights", {})

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

	def _compute_wind_kernel(self) -> np.ndarray:
		"""Build a 3x3 directional kernel that weights blazing neighbors by wind alignment.

		In real fire: wind pushes flames in a direction. A blazing cell UPWIND of
		a candidate cell contributes MORE to ignition (wind carries fire toward it).
		A blazing cell DOWNWIND contributes LESS (fire would need to travel against wind).

		Grounding: Alexandridis et al. (2011), Gao et al. (2008) — wind speed and
		direction directly modulate spread probability in CA fire models.

		Returns a 3x3 float32 kernel where each cell is the wind-adjusted weight
		for that neighbor position. Center cell is always 0.
		"""
		speed_kmh = float(self.wind_cfg.get("speed_kmh", 0.0))
		direction_deg = float(self.wind_cfg.get("direction_deg", 0.0))
		c_wind = float(self.transition_cfg.get("wind_weight", 0.20))

		# Normalize speed: 10 km/h → factor 1.0, 25 km/h → 2.5
		speed_factor = speed_kmh / 10.0

		# Wind blow direction in grid coords (row increases southward, col increases eastward).
		# direction_deg is compass bearing of where wind COMES FROM.
		# Wind BLOWS TOWARD (direction_deg + 180) degrees compass.
		blow_rad = np.deg2rad((direction_deg + 180.0) % 360.0)
		blow_dr = -np.cos(blow_rad)   # row component (north = -row, south = +row)
		blow_dc = np.sin(blow_rad)    # col component (west = -col, east = +col)
		blow_norm = np.sqrt(blow_dr ** 2 + blow_dc ** 2)
		if blow_norm > 0:
			blow_dr /= blow_norm
			blow_dc /= blow_norm

		kernel = np.zeros((3, 3), dtype=np.float32)
		for dr in (-1, 0, 1):
			for dc in (-1, 0, 1):
				if dr == 0 and dc == 0:
					continue
				# Fire travels FROM neighbor (dr, dc) TO center (0, 0): direction is (-dr, -dc)
				dist = np.sqrt(float(dr * dr + dc * dc))
				travel_r = -dr / dist
				travel_c = -dc / dist

				# Dot product: +1 = fire travels WITH wind, -1 = AGAINST wind
				alignment = travel_r * blow_dr + travel_c * blow_dc

				# Weight: base 1.0, adjusted by wind. Clamped to stay positive.
				weight = max(1.0 + c_wind * speed_factor * alignment, 0.05)
				kernel[dr + 1, dc + 1] = np.float32(weight)

		return kernel

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
		uniform_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)

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

		# Raw neighbor count (uniform) — used for susceptibility check
		blazing_neighbor_count = convolve(
			blazing_now.astype(np.int8),
			uniform_kernel,
			mode="constant",
			cval=0,
		)
		susceptible = (current_grid == STATE_NOT_YET_BURNING) & (blazing_neighbor_count > 0)

		if np.any(susceptible):
			if self.model is not None:
				p_ignite = self._predict_with_model()
			else:
				# Directional wind kernel: neighbors upwind contribute more,
				# downwind contribute less. Replaces the old flat wind_multiplier.
				wind_weighted_score = convolve(
					blazing_now.astype(np.float32),
					self.wind_kernel,
					mode="constant",
					cval=0.0,
				)
				max_kernel_sum = float(self.wind_kernel.sum())
				p_ignite = self.p_base * (wind_weighted_score / max_kernel_sum)

			p_effective = np.clip(p_ignite, 0.0, 1.0)
			ignition_draw = self.rng.random(self.grid_shape)
			ignite_mask = susceptible & (ignition_draw < p_effective)
			next_grid[ignite_mask] = STATE_IGNITED
			self.ignition_timers[ignite_mask] = np.int16(0)
			self.blazing_timers[ignite_mask] = np.int16(0)

		self.grid = next_grid
		self.timestep += 1

	def load_model(self, model_path: str) -> None:
		loaded_model = joblib.load(model_path)
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

	def _predict_with_model(self) -> np.ndarray:
		kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int8)
		blazing_neighbor_count = convolve(
			(self.grid == STATE_BLAZING).astype(np.int8),
			kernel,
			mode="constant",
			cval=0,
		)

		features = self.feature_assembler.assemble_grid_features(blazing_neighbor_count)
		proba = self.model.predict_proba(features)
		ignition_proba_flat = proba[:, 1]
		ignition_proba_grid = ignition_proba_flat.reshape(self.grid_shape).astype(np.float32)
		return ignition_proba_grid

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
