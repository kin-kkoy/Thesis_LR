from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol
import yaml

try:
	from modules.automata_engine import FireAutomata
	from modules.data_loader import EnvironmentManager
except ModuleNotFoundError:
	from Code.modules.automata_engine import FireAutomata
	from Code.modules.data_loader import EnvironmentManager


DEFAULT_FLAMMABILITY_WEIGHTS = {
	"base_weight": 0.5,
	"slope_weight": 0.3,
	"proximity_weight": 0.2,
}
SUPPORTED_MODEL_EXTENSIONS = {".pkl", ".joblib"}


def load_config(config_path: str) -> dict:
	config_file = Path(config_path)
	with config_file.open("r", encoding="utf-8") as file_obj:
		config = yaml.safe_load(file_obj)

	code_dir = Path(__file__).resolve().parent
	raster_dir = config["environment"]["raster_dir"]
	config["environment"]["raster_dir"] = str((code_dir / raster_dir).resolve())
	return config


def apply_pipeline_defaults(config: dict) -> dict:
	normalized_config = dict(config)
	weights_cfg = dict(normalized_config.get("flammability_weights", {}))
	normalized_config["flammability_weights"] = {
		**DEFAULT_FLAMMABILITY_WEIGHTS,
		**weights_cfg,
	}
	return normalized_config


def load_model(automata: FireAutomata, model_path: str) -> None:
	model_file = Path(model_path)
	if model_file.suffix.lower() not in SUPPORTED_MODEL_EXTENSIONS:
		supported = ", ".join(sorted(SUPPORTED_MODEL_EXTENSIONS))
		raise ValueError(f"Model file must have one of these extensions: {supported}")

	if not model_file.is_absolute():
		model_file = Path(__file__).resolve().parent / model_file

	if not model_file.exists():
		raise FileNotFoundError(f"Model file not found: {model_file}")

	# FireAutomata.load_model validates that the loaded object exposes predict_proba().
	automata.load_model(str(model_file))


def _pick_random_ignition_points(automata: FireAutomata, n_points: int = 3) -> list[tuple[int, int]]:
	candidate_mask = automata.burnable_mask & (automata.building_presence == 1)
	candidate_cells = np.argwhere(candidate_mask)
	if candidate_cells.size == 0:
		return []

	count = min(n_points, candidate_cells.shape[0])
	selected_idx = automata.rng.choice(candidate_cells.shape[0], size=count, replace=False)
	selected_cells = candidate_cells[selected_idx]
	return [(int(row), int(col)) for row, col in selected_cells]


def _convert_geographic_ignition_points(
	automata: FireAutomata,
	ignition_points: list,
) -> list[tuple[int, int]]:
	rows, cols = automata.grid_shape
	converted: list[tuple[int, int]] = []

	for point in ignition_points:
		if not isinstance(point, (list, tuple)) or len(point) != 2:
			raise ValueError(
				"Each ignition point must be [x, y] in the raster CRS (EPSG:32651). "
				f"Received: {point}"
			)

		x = float(point[0])
		y = float(point[1])

		# Use rasterio rowcol for deterministic grid indices.
		row_idx, col_idx = rowcol(automata.transform, x, y, op=np.floor)
		row = int(row_idx)
		col = int(col_idx)

		# Cross-check with inverse affine transform to guard conversion drift.
		col_affine, row_affine = ~automata.transform * (x, y)
		row_affine_idx = int(np.floor(row_affine))
		col_affine_idx = int(np.floor(col_affine))
		if row != row_affine_idx or col != col_affine_idx:
			raise ValueError(
				"Ignition coordinate conversion mismatch between rowcol() and inverse affine. "
				f"Input (x={x:.4f}, y={y:.4f}) -> rowcol=({row}, {col}), "
				f"inverse_affine=({row_affine_idx}, {col_affine_idx})."
			)

		if row < 0 or row >= rows or col < 0 or col >= cols:
			raise ValueError(
				"Ignition coordinate is outside the Lapu-Lapu matrix extent. "
				f"Input (x={x:.4f}, y={y:.4f}) maps to (row={row}, col={col}), "
				f"valid row range is [0, {rows - 1}] and col range is [0, {cols - 1}]."
			)

		converted.append((row, col))

	return converted


def run_simulation(config: dict) -> None:
	config = apply_pipeline_defaults(config)
	sim_cfg = config.get("simulation", {})

	env_manager = EnvironmentManager(config["environment"])
	env_manager.load_rasters()
	env_manager.build_masks()
	env_manager.normalize_layers()
	env_manager.summary()

	# FireAutomata passes flammability_weights into FeatureAssembler during setup.
	automata = FireAutomata(env_manager.get_environment(), config)

	# Allow YAML to override the default model path so different model variants
	# (LR, RF, etc.) can be plugged in without renaming files.
	ml_cfg = config.get("ml_model", {})
	configured_path = str(ml_cfg.get("model_path", "")).strip()
	if configured_path:
		model_path = Path(configured_path)
	else:
		model_path = Path("models") / "fire_rf_model.joblib"
	try:
		load_model(automata, str(model_path))
		print(f"Loaded ML model: {model_path}")
	except FileNotFoundError as exc:
		print(f"ML model not found. Expected at '{model_path}'. Aborting simulation gracefully.")
		print(f"Reason: {exc}")
		return

	output_cfg = config.get("output", {})
	output_dir = Path(output_cfg.get("output_dir", "output"))
	if not output_dir.is_absolute():
		output_dir = Path(__file__).resolve().parent / output_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	resume_checkpoint_cfg = str(sim_cfg.get("resume_checkpoint", "")).strip()
	is_resumed = False
	if resume_checkpoint_cfg:
		resume_checkpoint_path = Path(resume_checkpoint_cfg)
		if not resume_checkpoint_path.is_absolute():
			resume_checkpoint_path = Path(__file__).resolve().parent / resume_checkpoint_path
		if not resume_checkpoint_path.exists():
			raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")
		automata.load_checkpoint(str(resume_checkpoint_path))
		is_resumed = True
		print(f"Resumed simulation from checkpoint: {resume_checkpoint_path}")

	if not is_resumed:
		ignition_points = sim_cfg.get("ignition_points", [])
		if ignition_points:
			try:
				ignition_tuples = _convert_geographic_ignition_points(automata, ignition_points)
			except ValueError as exc:
				print("Invalid ignition point configuration. Aborting simulation gracefully.")
				print(f"Reason: {exc}")
				return
		else:
			ignition_tuples = _pick_random_ignition_points(automata, n_points=3)

		if not ignition_tuples:
			raise ValueError("No valid ignition points available from config or random building candidates.")

		automata.set_ignition(ignition_tuples)
		print(f"Ignition points: {ignition_tuples}")

	max_timesteps = int(sim_cfg.get("max_timesteps", 0))
	checkpoint_interval_cfg = sim_cfg.get("checkpoint_interval", 0)
	checkpoint_interval = int(checkpoint_interval_cfg) if checkpoint_interval_cfg is not None else 0
	checkpoint_enabled = checkpoint_interval > 0

	start_step = int(automata.timestep) + 1
	for step in range(start_step, max_timesteps + 1):
		automata.step()

		if step % 10 == 0:
			counts = automata.get_state_counts()
			print(f"Step {step}: {counts}")

		if checkpoint_enabled and (step % checkpoint_interval == 0):
			checkpoint_path = output_dir / f"checkpoint_step_{step:06d}.npz"
			automata.save_checkpoint(str(checkpoint_path))
			print(f"Checkpoint saved: {checkpoint_path}")

		if not automata.is_active():
			print(f"Simulation stopped early at step {step}: no active fire cells remain.")
			break

	final_grid = automata.get_grid().astype(np.int8)
	output_path = output_dir / "final_state.tif"
	with rasterio.open(
		output_path,
		"w",
		driver="GTiff",
		height=final_grid.shape[0],
		width=final_grid.shape[1],
		count=1,
		dtype="int8",
		crs=automata.crs,
		transform=automata.transform,
	) as dst:
		dst.write(final_grid, 1)

	print(f"Final state written to: {output_path}")


if __name__ == "__main__":
	default_config_path = Path(__file__).resolve().parent / "config" / "default_experiment.yaml"
	run_simulation(load_config(str(default_config_path)))
