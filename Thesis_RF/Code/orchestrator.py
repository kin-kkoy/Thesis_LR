from hashlib import sha256
import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import importlib.metadata
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import struct
import subprocess
import sys
import time

import numpy as np
import rasterio
from rasterio.transform import rowcol
import yaml

try:
	from modules.automata_engine import FireAutomata
	from modules.data_loader import EnvironmentManager
	from modules.model_free_teacher import (
		TEACHER_RUNNER_SCHEMA_VERSION,
		TeacherTerminationRecord,
		run_model_free_teacher as run_model_free_teacher_engine,
	)
	from modules.set_c_pilot import (
		APPROVAL_REFERENCE as PILOT_APPROVAL_REFERENCE,
		activate_approved_pilot_config,
		approved_destination,
		preflight_pilot_launch,
		run_set_c_pilot as run_set_c_pilot_engine,
	)
	from modules.transition_observer import TransitionObserver
	from modules.transition_observer import canonical_metadata_hash
	from modules.ca_state import validate_full_state_grid
	from modules.wind_convention import WindContract
	from modules.set_c_pilot import capture_resource_snapshot
	from modules import candidate_dataset_inputs as candidate_inputs
	from modules import candidate_dataset_crops as candidate_crops
except ModuleNotFoundError:
	from Code.modules.automata_engine import FireAutomata
	from Code.modules.data_loader import EnvironmentManager
	from Code.modules.model_free_teacher import (
		TEACHER_RUNNER_SCHEMA_VERSION,
		TeacherTerminationRecord,
		run_model_free_teacher as run_model_free_teacher_engine,
	)
	from Code.modules.set_c_pilot import (
		APPROVAL_REFERENCE as PILOT_APPROVAL_REFERENCE,
		activate_approved_pilot_config,
		approved_destination,
		preflight_pilot_launch,
		run_set_c_pilot as run_set_c_pilot_engine,
	)
	from Code.modules.transition_observer import TransitionObserver
	from Code.modules.transition_observer import canonical_metadata_hash
	from Code.modules.ca_state import validate_full_state_grid
	from Code.modules.wind_convention import WindContract
	from Code.modules.set_c_pilot import capture_resource_snapshot
	from Code.modules import candidate_dataset_inputs as candidate_inputs
	from Code.modules import candidate_dataset_crops as candidate_crops


DEFAULT_FLAMMABILITY_WEIGHTS = {
	"base_weight": 0.5,
	"slope_weight": 0.3,
	"proximity_weight": 0.2,
}

PILOT_SOURCE_RELATIVE_PATHS = (
	"modules/set_c_pilot.py",
	"modules/model_free_teacher.py",
	"modules/set_c_collector.py",
	"modules/automata_engine.py",
	"modules/feature_pipeline.py",
	"modules/data_loader.py",
	"orchestrator.py",
	"run_set_c_pilot.py",
)
SUPPORTED_MODEL_EXTENSIONS = {".pkl", ".joblib"}


def load_config(config_path: str) -> dict:
	config_file = Path(config_path)
	raw = config_file.read_bytes()
	node = yaml.compose(raw, Loader=yaml.SafeLoader)
	if isinstance(node, yaml.MappingNode) and any(key.value == "phase15" for key, _ in node.value):
		return _phase15_yaml(raw)
	config = yaml.safe_load(raw)

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
	return _convert_geographic_ignition_points_for_grid(
		automata.grid_shape,
		automata.transform,
		ignition_points,
	)


def _convert_geographic_ignition_points_for_grid(
	grid_shape: tuple[int, int],
	transform: object,
	ignition_points: list,
) -> list[tuple[int, int]]:
	rows, cols = grid_shape
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
		row_idx, col_idx = rowcol(transform, x, y, op=np.floor)
		row = int(row_idx)
		col = int(col_idx)

		# Cross-check with inverse affine transform to guard conversion drift.
		col_affine, row_affine = ~transform * (x, y)
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


def run_model_free_teacher(
	config: dict,
	*,
	transition_observer: TransitionObserver,
	transition_provenance: dict,
) -> TeacherTerminationRecord:
	"""Run the explicitly enabled teacher without model or artifact operations."""
	teacher_cfg = config.get("model_free_teacher", {})
	if not isinstance(teacher_cfg, dict) or teacher_cfg.get("enabled") is not True:
		raise ValueError("model_free_teacher.enabled must be true for an explicit run")
	if teacher_cfg.get("runner_schema_version") != TEACHER_RUNNER_SCHEMA_VERSION:
		raise ValueError(
			"model_free_teacher.runner_schema_version must be "
			f"{TEACHER_RUNNER_SCHEMA_VERSION!r}"
		)
	if teacher_cfg.get("persistent_output_enabled", False) is not False:
		raise ValueError(
			"model_free_teacher.persistent_output_enabled must remain false"
		)
	if not callable(transition_observer):
		raise TypeError("transition_observer must be callable")
	if transition_provenance is None:
		raise ValueError("transition_provenance is required for teacher mode")

	ignition_points = config.get("simulation", {}).get("ignition_points", [])
	if not ignition_points:
		raise ValueError(
			"Teacher mode requires explicit simulation.ignition_points; "
			"random ignition selection is not authoritative"
		)

	config = apply_pipeline_defaults(config)
	env_manager = EnvironmentManager(config["environment"])
	env_manager.load_rasters()
	env_manager.build_masks()
	env_manager.normalize_layers()
	environment = env_manager.get_environment()

	ignition_tuples = _convert_geographic_ignition_points_for_grid(
		environment["grid_shape"],
		environment["transform"],
		ignition_points,
	)
	return run_model_free_teacher_engine(
		environment,
		config,
		ignition_points=ignition_tuples,
		transition_observer=transition_observer,
		transition_provenance=transition_provenance,
	)


def _sha256_file(path: Path, on_chunk=None) -> str:
	digest = sha256()
	with path.open("rb") as file_obj:
		for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
			digest.update(chunk)
			if on_chunk is not None:
				on_chunk()
	return digest.hexdigest()


def _pilot_source_context(config: dict, config_path: Path) -> dict[str, object]:
	"""Bind an authorized pilot to source, environment inputs, and versions."""
	code_dir = Path(__file__).resolve().parent
	repo_root = code_dir.parents[1]
	source_paths = tuple(
		code_dir / relative_path for relative_path in PILOT_SOURCE_RELATIVE_PATHS
	) + (config_path.resolve(),)
	if any(not path.is_file() for path in source_paths):
		raise FileNotFoundError("Every pilot source/configuration file must exist")
	source_hashes = {
		str(path.resolve().relative_to(repo_root.resolve())).replace(
			"\\", "/"
		): _sha256_file(path)
		for path in source_paths
	}

	environment = config["environment"]
	raster_dir = Path(environment["raster_dir"])
	input_paths = {
		name: raster_dir / environment[key]
		for name, key in (
			("slope", "slope_file"),
			("proximity", "proximity_file"),
			("buildings", "buildings_file"),
			("materials", "materials_file"),
		)
	}
	if any("ground_truth" in path.name.lower() for path in input_paths.values()):
		raise PermissionError("stack_ground_truth.tif is forbidden in the Phase 8 pilot")
	if any(not path.is_file() for path in input_paths.values()):
		raise FileNotFoundError("Every approved environmental raster must exist")
	input_hashes = {
		path.name: _sha256_file(path) for path in input_paths.values()
	}

	git_base = [
		"git",
		"-c",
		f"safe.directory={repo_root.as_posix()}",
		"-C",
		str(repo_root),
	]
	revision = subprocess.run(
		[*git_base, "rev-parse", "HEAD"],
		check=True,
		capture_output=True,
		text=True,
	).stdout.strip()
	dirty = bool(
		subprocess.run(
			[*git_base, "status", "--porcelain"],
			check=True,
			capture_output=True,
			text=True,
		).stdout.strip()
	)
	versions = {}
	for package in ("numpy", "scipy", "rasterio", "PyYAML"):
		versions[package] = importlib.metadata.version(package)
	return {
		"source_revision": revision,
		"source_dirty": dirty,
		"source_hashes": source_hashes,
		"input_hashes": input_hashes,
		"generation_command": (
			".venv\\Scripts\\python.exe -B Thesis_RF\\Code\\run_set_c_pilot.py "
			"--config Thesis_RF\\Code\\config\\default_experiment.yaml "
			"--execute-approved-pilot"
		),
		"environment_record": {
			"python": platform.python_version(),
			"platform": platform.platform(),
			"packages": versions,
		},
	}


def run_set_c_feasibility_pilot(
	config: dict,
	*,
	explicitly_authorized: bool = False,
	config_path: Path | None = None,
) -> dict[str, object]:
	"""Run the one approved model-free pilot without touching the ML path."""
	if explicitly_authorized is not True:
		raise PermissionError(
			"Phase 8 is disabled; the exact --execute-approved-pilot flag is required"
		)
	active_config = activate_approved_pilot_config(config)
	if (
		active_config["phase8_set_c_pilot"]["authorization_reference"]
		!= PILOT_APPROVAL_REFERENCE
	):
		raise PermissionError("Phase 8 owner authorization reference changed")
	destination = approved_destination()
	preflight_pilot_launch(active_config, destination)
	resolved_config_path = (
		Path(config_path).resolve()
		if config_path is not None
		else Path(__file__).resolve().parent / "config" / "default_experiment.yaml"
	)
	source_context = _pilot_source_context(active_config, resolved_config_path)

	env_manager = EnvironmentManager(active_config["environment"])
	env_manager.load_rasters()
	env_manager.build_masks()
	env_manager.normalize_layers()
	return run_set_c_pilot_engine(
		env_manager.get_environment(),
		active_config,
		destination=destination,
		source_context=source_context,
	)


def _run_legacy_simulation(
	config: dict,
	*,
	transition_observer: TransitionObserver | None = None,
	transition_provenance: dict | None = None,
) -> None:
	"""Run the existing simulation with optional source-only observation.

	The observation boundary is disabled unless a callback is supplied. It adds
	no persistence path; all existing simulation output behavior is unchanged.
	"""
	if transition_observer is None and transition_provenance is not None:
		raise ValueError(
			"transition_provenance is accepted only when an observer is enabled"
		)
	if transition_observer is not None and transition_provenance is None:
		raise ValueError(
			"transition_provenance is required when an observer is enabled"
		)
	config = apply_pipeline_defaults(config)
	sim_cfg = config.get("simulation", {})

	env_manager = EnvironmentManager(config["environment"])
	env_manager.load_rasters()
	env_manager.build_masks()
	env_manager.normalize_layers()
	env_manager.summary()

	# FireAutomata passes flammability_weights into FeatureAssembler during setup.
	if transition_observer is None:
		automata = FireAutomata(env_manager.get_environment(), config)
	else:
		automata = FireAutomata(
			env_manager.get_environment(),
			config,
			transition_observer=transition_observer,
			transition_provenance=transition_provenance,
		)

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
	observation_only = transition_observer is not None
	if not observation_only:
		if not output_dir.is_absolute():
			output_dir = Path(__file__).resolve().parent / output_dir
		output_dir.mkdir(parents=True, exist_ok=True)

	resume_checkpoint_cfg = str(sim_cfg.get("resume_checkpoint", "")).strip()
	if observation_only and resume_checkpoint_cfg:
		raise ValueError(
			"Source-only transition observation cannot resume from a checkpoint"
		)
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
	checkpoint_enabled = (not observation_only) and checkpoint_interval > 0

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

	if observation_only:
		print("Source-only transition observation completed without artifact output.")
		return

	final_grid = automata.get_grid().astype(np.int8)
	# output_path = output_dir / "final_state.tif"
	output_path = output_dir / "final_sitio_santa_maria_11_params.tif"
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


PHASE15_CODE_DIR = Path(__file__).resolve().parent
PHASE15_MODEL = (
    "C:/Users/gibgib/Thesis-Cellular-Automata/Thesis_RF/Code/output/phase14/"
    "phase14-setc-rf-61e8f740-r1/calibration/"
    "phase14-cal-25af5c75-428f1e8076844a2d89e3116ce12d285e/base_rf_calibrated.joblib"
)
PHASE15_MODEL_SHA256 = "6e2754feb015c73c7cc860a2879a76c72dd213fb5a9d28baead5bbb1c857a666"
PHASE15_RECONCILIATION = "00809c8e89a1c6e3f147bdabf127d5f71e1da827da0a84f12d5c6a8181ef9072"
PHASE15_SOURCES = tuple("Thesis_RF/Code/" + name for name in (
    "main.py", "orchestrator.py", "modules/__init__.py", "modules/automata_engine.py",
    "modules/ca_state.py", "modules/data_loader.py", "modules/feature_pipeline.py",
    "modules/model_free_teacher.py", "modules/set_c_collector.py", "modules/set_c_pilot.py",
    "modules/transition_observer.py", "modules/wind_convention.py",
    "modules/candidate_dataset_inputs.py", "modules/candidate_dataset_crops.py",
))
PHASE15_CANDIDATE_SCHEMA = "phase15_candidate_r2_rf_ca.v1"
PHASE16_CANDIDATE_SCHEMA = "phase16_candidate_r2_f01_nasa_hr11.v1"
PHASE16_RF_GATE_SCHEMA = "phase16_f01_rf_probability_gate_040.v1"
PHASE16_LR_GATE_SCHEMA = "phase16_f01_lr_probability_gate_040.v1"
PHASE16_AIRBASE_SCHEMA = "phase16_airbase_rf_nocutoff.v1"
PHASE16_AIRBASE_LR_GATE_SCHEMA = "phase16_airbase_lr_probability_gate_040.v1"
PHASE16_AIRBASE_DURATION90_SCHEMA = "phase16_airbase_rf_duration90.v1"
PHASE16_AIRBASE_DURATION90_BASIS = {
    "schema_version": "airbase_duration_sensitivity.v1",
    "reported_start_local": "2025-01-24T13:28:00+08:00",
    "reported_fireout_local": "2025-01-24T14:58:00+08:00",
    "reported_duration_minutes": 90, "assumed_minutes_per_update": 1.0, "update_limit": 90,
    "mapping_status": "ASSUMPTION: owner-approved sensitivity, not calibrated physical time",
    "stopping_interpretation": "Time-horizon truncation; not simulated suppression or fire-out",
}
PHASE16_AIRBASE_GRID_FROM_DEG = 60.826537
PHASE16_AIRBASE_SCENARIO_ID = "phase16-airbase-rf-w10p6-from60p826537-s42-t1-1"
PHASE16_AIRBASE_WINDOW = {
    "incident_id": "airbase_sanggi_20250124", "global_origin": [820, 1989], "shape": [1024, 1024],
}
PHASE16_AIRBASE_IGNITIONS = [[512, 512]]
PHASE16_AIRBASE_WIND_INPUT_BASIS = {
    "schema_version": "phase16_airbase_nasa_hr13_ignition_center.v1",
    "incident": "Owner-identified Sitio Sewage (Sanggi Interior), near Airbase, Lapu-Lapu City",
    "incident_date": "2025-01-24", "owner_reported_local_time": "13:28 Asia/Manila",
    "source_row": "2025,1,24,13,2.95,60.8",
    "source_export_sha256": "746afb4b11d10e61d18717838f178b267d5747962e8f1e27c7791d4a958cc67f",
    "source_point_lat_lon": [10.3143, 123.9681],
    "source_time_basis": {
        "timezone": "NASA Local Solar Time (LST)", "hour": 13,
        "selection": "Owner-prescribed rounded wind matches HR13; hourly average, not an instantaneous observation",
        "qualification": "NASA LST is longitude-based and is not asserted identical to Philippine civil time",
    },
    "speed_basis": {"source_mps": 2.95, "converted_kmh": 10.62, "prescribed_kmh": 10.6},
    "direction_basis": {
        "source_degrees": 60.8, "prescribed_true_from_degrees": 61.0,
        "direction_convention": "meteorological_from", "north_reference": "geographic_true_north",
        "north_reference_status": "INFERRED", "rounding_authority": "owner request",
    },
    "anchor": {"kind": "ignition_pixel_centre", "crs": "EPSG:32651",
               "easting": 606063.2784, "northing": 1140377.2699},
    "anchor_lat_lon": [10.314786435731733, 123.96867370380014],
    "conversion": {
        "formula": "(prescribed true-from - gamma) mod 360", "additional_180_degree_reversal": False,
        "gamma_degrees": 0.17346326940467718, "grid_from_degrees": 60.826537, "decimal_places": 6,
        "method": "Projected symmetric geographic-north tangent using Rasterio/PROJ; latitude delta 0.0001 degrees",
        "spatial_approximation": "One constant ignition-centre bearing over the crop; not physical accuracy",
    },
}
PHASE16_GRID_FROM_DEG = 48.825208
PHASE16_SCENARIO_ID = "phase16-candidate-r2-f01-nasa-hr11-w17p7-from48p825208-s42-t1-1"
PHASE16_LR_SCENARIO_ID = "phase16-candidate-r2-f01-nasa-hr11-lr-w17p7-from48p825208-s42-t1-1"
# Owner-requested comparisons; only this rebuilt, uncalibrated LR release.
PHASE16_LR_MODEL = (
    PHASE15_CODE_DIR / "output/lr_same_kitchen/20261102-r1/fire_lr_kent_r2_unweighted_sk1.8.0.joblib"
).as_posix()
PHASE16_LR_MODEL_SHA256 = "41e6f55d3e07fa27a57b42d7d0be0828c72cf87ff6cc92c3834b0c0dfd8d217b"
# D-122 adopts this conditional input basis; these are supplied values, not a runtime conversion.
PHASE16_WIND_INPUT_BASIS = {
    "schema_version": "phase16_f01_nasa_hr11_ignition_center.v1",
    "source_row": "2023,12,12,11,4.91,49.0",
    "source_export_sha256": "d29f891aacf0aa7932a886d4cebfe37df1a7bcd96823b77e5d37b7fdfa064a90",
    "source_point_lat_lon": [10.327, 123.9749],
    "source_time_basis": {
        "timezone": "LST", "hour": 11, "adopted_bfp_time": "11:31 AM",
        "selection_authority": "owner/partner",
    },
    "source_direction": {
        "degrees": 49.0, "direction_convention": "meteorological_from",
        "direction_units": "degrees_clockwise", "north_reference": "geographic_true_north",
        "north_reference_status": "INFERRED",
    },
    "speed_basis": {"source_mps": 4.91, "converted_kmh": 17.676, "prescribed_kmh": 17.7},
    "anchor": {
        "kind": "ignition_pixel_centre", "crs": "EPSG:32651",
        "easting": 606750.2784, "northing": 1141691.2699,
    },
    "anchor_lat_lon": [10.326650173394397, 123.9749840198514],
    "gamma": {
        "approximate_degrees": 0.174792123614,
        "definition": "clockwise from true north to grid north",
    },
    "conversion": {
        "formula": "(true-from - gamma) mod 360", "additional_180_degree_reversal": False,
    },
    "numerical_specification": {
        "unrounded_grid_from_deg": 48.825207876386, "decimal_places": 6,
        "grid_from_deg": 48.825208,
    },
    "spatial_approximation": {
        "method": "one constant anchor-derived bearing over the crop",
        "incident": "owner-identified Santa Maria F01 development case",
        "query_point": "nearby NASA point adopted by owner",
        "qualification": "no independent historical geolocation or physical-wind validity claim",
    },
    "source_discrepancies": {
        "researcher_reported_screenshot_text": "12/12/2024-1131H",
        "owner_identified_incident_year": 2023, "dswd_approximate_time": "13:25",
    },
    "evidence_references": {
        "source_contract_handback": "C:/Users/gibgib/.codex/attachments/49766430-ab36-4395-8c74-7bc3a5db22c4/Pasted text.txt",
        "geometry_handback": "C:/Users/gibgib/.codex/attachments/d53164bd-be77-4e97-bdc9-07245b8eb129/Pasted text.txt",
        "authority": "docs/ai/DECISIONS.md D-122",
    },
}
PHASE15_CANDIDATE_MODEL = (
    PHASE15_CODE_DIR / "output/candidate-r2-cropped-1024-v1-g364-r1-rf-estimator-r1-sigmoid-r1/rf_sigmoid.joblib"
).as_posix()
PHASE15_CANDIDATE_MODEL_SHA256 = "78828dd4b2fcc66e9cdfd8ecb4a3ccd18c268478f3b5e6a4374c3286e3903e54"
PHASE15_CANDIDATE_MANIFEST_SHA256 = "3141b8554662d703691c06a327f9a6c96c79f20840ce3738c358d4bdb6632afb"
PHASE15_CANDIDATE_WINDOW = candidate_crops.frozen_design()["family_windows"][0]
PHASE15_CANDIDATE_IGNITIONS = [[638, 554]]
PHASE15_ABSENT = ("Thesis_RF/__init__.py", "Thesis_RF/Code/__init__.py")
PHASE15_OUTPUTS = frozenset((
    "start.json", "termination.json", "final_state.tif", "simulation_environment_valid.tif",
    "final_state.tif.partial", "simulation_environment_valid.tif.partial", "termination.json.partial",
))
PHASE15_KEYS = {
    "phase15": "schema_version mode enabled authorization_reference run_id scenario_id phase14_reconciliation_sha256 timestep_unit resources",
    "environment": "raster_dir slope_file proximity_file buildings_file materials_file input_sha256 nodata_value",
    "simulation": "seed ignition_points max_timesteps early_stop checkpoint_interval emergency_checkpoint_enabled resume_checkpoint",
    "placeholder_transition": "T_3_to_4 T_4_to_5 wind_weight",
    "wind": "speed_kmh direction_deg direction_convention direction_units north_reference schema_version calm_representation",
    "flammability_weights": "base_weight slope_weight proximity_weight",
    "ml_model": "enabled model_path model_sha256 positive_label inference_mode inference_chunk_size threshold inference_threshold classification_threshold",
    "rf_contract": "feature_schema_version target_version",
    "output": "output_dir save_every_n_steps save_final_state",
}


class _Phase15Loader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            if key_node.tag != "tag:yaml.org,2002:str":
                raise ValueError("Phase 15 requires string YAML keys; merge keys are unsupported")
            key = self.construct_scalar(key_node)
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=True)
        return result


def _phase15_metadata(value, ancestors=None):
    ancestors = set() if ancestors is None else ancestors
    if isinstance(value, (dict, list)):
        if id(value) in ancestors:
            raise ValueError("Recursive metadata is unsupported")
        ancestors.add(id(value))
        if isinstance(value, dict) and any(type(key) is not str for key in value):
            raise TypeError("Metadata keys must be strings before normalization")
        for item in (value.values() if isinstance(value, dict) else value):
            _phase15_metadata(item, ancestors)
        ancestors.remove(id(value))
    elif value is not None and type(value) not in (str, bool, int, float):
        raise TypeError("Unsupported metadata value")
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("Non-finite metadata")


def _phase15_yaml(raw):
    config = yaml.load(raw, Loader=_Phase15Loader)
    _phase15_metadata(config)
    if not isinstance(config, dict):
        raise ValueError("Phase 15 configuration must be a mapping")
    return config


def _phase15_path(value):
    if not isinstance(value, str) or not value or any(char in value for char in ("~", "$", "%", "\x00")):
        raise ValueError("An explicit path without expansion is required")
    return Path(os.path.abspath(os.path.join(PHASE15_CODE_DIR, value))).as_posix()


def _phase15_digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def _phase15_exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError(f"Exact fields required: {keys}")


def _phase15_number(value, minimum=0, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError("Invalid numeric configuration value")


# D-172: a separate RF-only profile; legacy schemas/caps/allowlists are unchanged.
PHASE16_DSWD_SCHEMA = "phase16_dswd_rf_snapshots.v1"
PHASE16_DSWD_MANIFEST = "config/phase16_dswd_snapshots_v1.yaml"
PHASE16_DSWD_PROTOCOL = {"schema_version": "phase16_dswd_snapshots.v1",
    "enabled": False,
    "execution_authorized": False,
    "protocol_evidence": {"path": "docs/StoppingRule_DSWD_Plan_v3.md",
    "sha256": "04d4d9017fba0e2edf15c7ed4546c3a1c7dd7bf7c12878543821beca59eea511",
    "authority": "D-171/D-172 source preparation only"},
    "seeds": [42, 43, 44, 45, 46],
    "clocks": [{"minutes_per_update": 0.5,
    "role": "sensitivity"}, {"minutes_per_update": 1,
    "role": "primary"}, {"minutes_per_update": 2,
    "role": "sensitivity"}],
    "trajectory_count": 10,
    "planned_outcome_count": 30,
    "side": "RF side",
    "model_policy": {"model_path": "output/candidate-r2-cropped-1024-v1-g364-r1-rf-estimator-r1-sigmoid-r1/rf_sigmoid.joblib",
    "model_sha256": "78828dd4b2fcc66e9cdfd8ecb4a3ccd18c268478f3b5e6a4374c3286e3903e54",
    "manifest_path": "output/candidate-r2-cropped-1024-v1-g364-r1-rf-estimator-r1-sigmoid-r1/calibration_manifest.json",
    "manifest_sha256": "3141b8554662d703691c06a327f9a6c96c79f20840ce3738c358d4bdb6632afb",
    "inference_mode": "stochastic_probability",
    "positive_label": 1,
    "inference_chunk_size": 200000,
    "threshold": None,
    "inference_threshold": None,
    "classification_threshold": None,
    "cutoff": None},
    "input_sha256": {"slope": "dd61a0da016cd9c0d21d957ffedebd7ce871ef94f9084d5d4feb90e7e32b8482",
    "proximity": "1b12936cd8bc78d1e638fd31ddf1241bd753cb040cc9db556c623b8193de14b5",
    "buildings": "da53c3e63b661471b3366fc68249793c6a5a84e0d63bcc96492f60baaf4b83c7",
    "materials": "21f5c98c7936b4be6e368e7c8c0472a3c045b9c85b1abcb0a0e3fc8c1be0336a"},
    "feature_order": ["slope_risk", "proximity_risk", "building_presence", "material_risk", "material_class", "wind_speed", "wind_sin", "wind_cos", "neighbor_burning_count", "composite_flammability", "wind_weighted_score"],
    "fixed_controls": {"timers": [1, 1],
    "wind_weight": 0.2,
    "flammability_weights": {"base_weight": 0.5,
    "slope_weight": 0.3,
    "proximity_weight": 0.2},
    "feature_schema": "set_c.v1",
    "target": "ignition_t_plus_1.v1",
    "probability_policy": "Unchanged calibrated probabilities; no RF cutoff; unchanged full-grid Bernoulli sampling",
    "resume": False,
    "checkpoint_interval": 0},
    "output_schema": "phase16_dswd_snapshot_series.v1",
    "output_policy": {"native_state": "int8 codes 1..5",
    "reached": "uint8, states 3/4/5 -> 1; zero valid, no nodata",
    "capture": "immediately after completed update, before next step",
    "initial_ignition": "reported separately at update 0; never substituted for a requested horizon",
    "inactive": "both ignited and blazing zero; later horizons reference actual absorbing stop",
    "active": "censored timestep_limit; not suppression/fire-out",
    "failure": "all trajectory outcomes incomplete; preserve every published/partial file",
    "counts": "per completed update: new ignitions, cumulative reached, active and five state counts",
    "allowlist": "new profile only; exclusive creation and existing hard-link publication"},
    "alignment_policy": {"crs": "EPSG:32651",
    "parent_shape": [5489, 6896],
    "parent_transform": [3, 0, 598558.7784, 0, -3, 1144374.7699],
    "validity": "native mask != 0 AND finite AND value != -9999 AND value != embedded nodata",
    "building_encoding": {"unmapped": 0,
    "mapped": 10},
    "reference_encoding": {"negative": 0,
    "positive": 1},
    "domain": "environment_valid & prediction_valid & reference_valid & building_valid & mapped_building",
    "mask_identity": "modules.transition_observer.mask_sha256; matching counts alone insufficient",
    "lr_side_confirmation": {"status": "NEEDS EVIDENCE",
    "contract_reference": None,
    "domain_mask_sha256": None}},
    "reporting": {"group_by": ["incident", "assumed_minutes_per_update"],
    "required_seeds": [42, 43, 44, 45, 46],
    "dispersion": "sample SD, n-1; five seeds per incident/clock",
    "missing_or_failed": "incomplete group; no silent seed removal or partial mean",
    "horizons_are_independent": False,
    "pool_incidents": False,
    "selection": "no best seed/step, reference-area stopping or tuning"},
    "incidents": {"airbase": {"name": "Sitio Sangi Interior, Brgy. Pajo",
    "date": "2025-01-24",
    "duration": {"start_local": "2025-01-24T13:32:00+08:00",
    "end_local": "2025-01-24T14:58:00+08:00",
    "minutes": 86,
    "endpoint": "fire out",
    "approximate": True,
    "source": "DSWD DROMIC Report #3, 03 February 2025, p.1",
    "url": "https://dromic.dswd.gov.ph/wp-content/uploads/2025/02/DSWD-DROMIC-Report-3-on-the-Fire-Incident-in-Brgy.-Pajo-Lapu-Lapu-City-as-of-03-February-2025-6PM.pdf",
    "separate_bfp_time": "13:28; owner-confirmed linkage; historical H90 unchanged"},
    "horizons": [172, 86, 43],
    "max_updates": 172,
    "crop_origin": [820, 1989],
    "shape": [1024, 1024],
    "ignition": [[512, 512]],
    "crop_transform": [3, 0, 604525.7784, 0, -3, 1141914.7699],
    "wind": {"speed_kmh": 10.6,
    "grid_from": 60.826537,
    "basis_symbol": "PHASE16_AIRBASE_WIND_INPUT_BASIS",
    "qualification": "Retain historical HR13/LST, INFERRED true north, ignition-centre/six-decimal constant bearing"},
    "reference_path": "dataFiles/stack_ground_truth_airbase.tif",
    "reference_sha256": "1627ac903354f029572fa9abc9c61088909b38a5357c09228361f9021d2f52db",
    "reference_metadata": {"dtype": "uint8",
    "nodata": None,
    "mask_flags": ["all_valid"],
    "status": "D-167 historical record; not freshly inspected"},
    "expected_domain_cells": 215553,
    "expected_reference_positive_total": 281,
    "expected_reference_positive_included": 178,
    "rf_domain_mask_sha256": None,
    "rf_domain_mask_status": "NEEDS EVIDENCE",
    "qualification": "Development-used; spatial overlap/prior-use/independence and reference provenance qualified"},
    "santa_maria": {"name": "Sitio Sta. Maria, Brgy. Pusok",
    "date": "2023-12-12",
    "duration": {"start_local": "2023-12-12T13:25:00+08:00",
    "end_local": "2023-12-12T16:57:00+08:00",
    "minutes": 212,
    "endpoint": "fire out",
    "approximate": True,
    "source": "DSWD DROMIC Terminal Report, 21 April 2024, p.1",
    "url": "https://dromic.dswd.gov.ph/wp-content/uploads/2023/12/DSWD-DROMIC-Terminal-Report-on-the-Fire-Incident-in-Brgy.-Pusok-Lapu-Lapu-City-Cebu-21-April-2024-6AM.pdf",
    "historical_basis": "NASA HR11/BFP 11:31 retained; screenshot reports 2024, owner identifies 2023; no wind-hour revision"},
    "horizons": [424, 212, 106],
    "max_updates": 424,
    "crop_origin": [256, 2176],
    "shape": [1024, 1024],
    "ignition": [[638, 554]],
    "crop_transform": [3, 0, 605086.7784, 0, -3, 1143606.7699],
    "wind": {"speed_kmh": 17.7,
    "grid_from": 48.825208,
    "basis_symbol": "PHASE16_WIND_INPUT_BASIS",
    "qualification": "Conditional INFERRED NASA true north; ignition-centre/six-decimal constant bearing; no extra 180 reversal"},
    "reference_path": "processedData/raster/raster/stack_ground_truth.tif",
    "reference_sha256": "fcf83513ab223072299e33696cd4ec2265d83ee330e1957ba1eb42feeb3a3880",
    "reference_metadata": {"status": "NEEDS EVIDENCE",
    "dtype": None,
    "nodata": None,
    "mask_flags": None},
    "expected_domain_cells": 135992,
    "expected_reference_positive_total": 3312,
    "expected_reference_positive_included": 1910,
    "rf_domain_mask_sha256": None,
    "rf_domain_mask_status": "NEEDS EVIDENCE",
    "qualification": "F01 calibration/development; reference provenance and physical clock qualified"}}}


PHASE16_DSWD_SHARED_SCHEMA = "phase16_dswd_shared_ca_snapshots.v1"
PHASE16_DSWD_SHARED_MANIFEST = "config/phase16_dswd_snapshots_v2.yaml"
# Frozen source contract; no file is opened at import time.
PHASE16_DSWD_SHARED_PROTOCOL = deepcopy(PHASE16_DSWD_PROTOCOL)
PHASE16_DSWD_SHARED_PROTOCOL.update(
    schema_version=PHASE16_DSWD_SHARED_SCHEMA, design="shared_rf_ca_v1",
    protocol_evidence={"path": "docs/StoppingRule_DSWD_Plan_v4.md", "sha256": "f4d828b63e4ca812d3e679ae1397f1b8a8e191019b358b648fe7749fd3632b14", "authority": "D-200/D-201 source preparation only"},
    trajectory_count=20, planned_outcome_count=60, reporting_group_count=12,
    side="Both frozen releases in RF-side CA")
PHASE16_DSWD_SHARED_PROTOCOL["model_policies"] = {
    "rf": PHASE16_DSWD_SHARED_PROTOCOL.pop("model_policy"),
    "lr": {"model_path": "output/lr_same_kitchen/20261102-r1/fire_lr_kent_r2_unweighted_sk1.8.0.joblib", "model_sha256": "41e6f55d3e07fa27a57b42d7d0be0828c72cf87ff6cc92c3834b0c0dfd8d217b", "manifest_path": None, "manifest_sha256": None, "inference_mode": "stochastic_probability_gate", "positive_label": 1, "inference_chunk_size": 200000, "threshold": None, "inference_threshold": None, "classification_threshold": None, "cutoff": 0.4}}
PHASE16_DSWD_SHARED_PROTOCOL["fixed_controls"]["probability_policy"] = "RF calibrated no cutoff; LR float32 p<0.40 -> 0, p>=0.40 unchanged; existing Bernoulli draw"
PHASE16_DSWD_SHARED_PROTOCOL["reporting"].update(
    group_by=["design", "incident", "arm", "assumed_minutes_per_update"], pool_models=False,
    dispersion="sample SD, n-1; five seeds per incident/arm/clock")
PHASE16_DSWD_SHARED_PROTOCOL["alignment_policy"]["shared_pairing"] = "Verified counterpart records with exact common inputs, geometry, actual ignition, wind, masks, reference and domain; missing counterpart pending; not cross-simulator acceptance"
PHASE16_DSWD_SHARED_PROTOCOL["incidents"]["santa_maria"].update(
    ignition=[[636, 554]], ignition_amendment={"authority": "D-200/D-201", "global_row_col": [[892, 2730]], "historical_local_row_col": [[638, 554]], "historical_global_row_col": [[894, 2730]], "source_point_epsg32651": [606748.78, 1141695.77], "rationale": "Owner-adopted reported affine-floor; original geography/precision qualified", "receiving_eligibility": "NEEDS VERIFICATION", "inherited_wind_anchor_epsg32651": [606750.2784, 1141691.2699], "qualification": "Original conversion anchor retained; not converted at892. Historical comparisons change ignition and duration."})


def _phase16_is_shared(config):
    profile = config.get("phase15") if isinstance(config, dict) else None
    return isinstance(profile, dict) and profile.get("schema_version") == PHASE16_DSWD_SHARED_SCHEMA


def _phase16_shared_basis(incident, arm):
    protocol = PHASE16_DSWD_SHARED_PROTOCOL
    if (not isinstance(incident, str) or not isinstance(arm, str)
            or incident not in protocol["incidents"] or arm not in protocol["model_policies"]):
        raise ValueError("Unknown shared-CA incident or model arm")
    item = protocol["incidents"][incident]
    return {"protocol_schema_version": protocol["schema_version"], "design": protocol["design"],
            "incident": incident, "arm": arm, "model_policy": deepcopy(protocol["model_policies"][arm]),
            "ignition_local_row_col": deepcopy(item["ignition"]),
            "ignition_global_row_col": [[r + item["crop_origin"][0], c + item["crop_origin"][1]]
                                        for r, c in item["ignition"]],
            "duration_basis": deepcopy(item["duration"]),
            "horizons": [{"requested_update": horizon, "assumed_minutes_per_update": clock["minutes_per_update"],
                          "clock_role": clock["role"], "clock_status": "ASSUMPTION"}
                         for clock, horizon in zip(protocol["clocks"], item["horizons"])]}


def _phase16_shared_normalize(config):
    _phase15_metadata(config)
    basis = config.get("snapshot_basis", {})
    if not isinstance(basis, dict):
        raise ValueError("Shared-CA basis must be a mapping")
    incident, arm = basis.get("incident"), basis.get("arm")
    expected = _phase16_shared_basis(incident, arm)
    if _phase15_json_bytes(basis) != _phase15_json_bytes(expected):
        raise ValueError("Shared-CA design, arm, policy, ignition or clock basis differs")
    protocol = PHASE16_DSWD_SHARED_PROTOCOL
    item, sim, p = protocol["incidents"][incident], config["simulation"], config["phase15"]
    seed = sim["seed"]
    run_id = f"p16-dswd-shared-v1-{incident.replace('_', '-')}-{arm}-s{seed}-r1"
    # D-245: fixed replacement attempts, never a generic revision override.
    recovery = {"p16-dswd-shared-v1-airbase-rf-s42-r2"} | {
        f"p16-dswd-shared-v1-airbase-lr-s{value}-r2" for value in range(42, 47)}
    proposed = run_id[:-2] + "r2"
    if proposed in recovery and p["run_id"] == proposed:
        run_id = proposed
    recovery_r3 = "p16-dswd-shared-v1-airbase-rf-s42-r3"
    if incident == "airbase" and arm == "rf" and seed == 42 and p["run_id"] == recovery_r3:
        run_id = recovery_r3
    scenario = run_id + "-snapshots"
    if (type(seed) is not int or seed not in protocol["seeds"]
            or type(sim["max_timesteps"]) is not int or sim["max_timesteps"] != item["max_updates"]
            or _phase15_json_bytes(sim["ignition_points"]) != _phase15_json_bytes(item["ignition"])
            or p["run_id"] != run_id or p["scenario_id"] != scenario
            or config["environment"]["input_sha256"] != protocol["input_sha256"]
            or config["ml_model"]["inference_chunk_size"] != 200000):
        raise ValueError("Shared-CA trajectory identity or frozen controls differ")
    ml, policy = config["ml_model"], expected["model_policy"]
    if (_phase15_path(ml["model_path"]) != _phase15_path(policy["model_path"])
            or ml["model_sha256"] != policy["model_sha256"]
            or ml["calibration_manifest_sha256"] != policy["manifest_sha256"]
            or ml["inference_mode"] != policy["inference_mode"]
            or (None if ml["calibration_manifest_path"] is None else _phase15_path(ml["calibration_manifest_path"]))
               != (None if policy["manifest_path"] is None else _phase15_path(policy["manifest_path"]))
            or (arm == "lr" and (type(ml.get("proba_threshold")) not in (int, float)
                                or ml["proba_threshold"] != policy["cutoff"]))):
        raise ValueError("Shared-CA frozen model/manifest/probability policy differs")
    base = deepcopy(config)
    del base["snapshot_basis"]
    lr, airbase = arm == "lr", incident == "airbase"
    schema = ((PHASE16_AIRBASE_LR_GATE_SCHEMA if airbase else PHASE16_LR_GATE_SCHEMA) if lr
              else (PHASE16_AIRBASE_SCHEMA if airbase else PHASE16_CANDIDATE_SCHEMA))
    old_scenario = PHASE16_AIRBASE_SCENARIO_ID if airbase else PHASE16_LR_SCENARIO_ID if lr else PHASE16_SCENARIO_ID
    if airbase and lr:
        old_scenario = old_scenario.replace("-rf-", "-lr-")
    old_scenario = old_scenario.replace("-s42-", f"-s{seed}-") + ("-gate040" if lr else "")
    base["phase15"].update(schema_version=schema, scenario_id=old_scenario)
    base["simulation"]["max_timesteps"] = 300
    # Validate the actual shared seed, never substitute the historical seed.
    active = _phase15_normalize_base(base, shared_incident=incident)
    active["phase15"].update(schema_version=PHASE16_DSWD_SHARED_SCHEMA, scenario_id=scenario)
    active["simulation"]["max_timesteps"] = item["max_updates"]
    active["snapshot_basis"] = deepcopy(expected)
    return active


def _phase16_is_snapshot(config):
    profile = config.get("phase15") if isinstance(config, dict) else None
    return isinstance(profile, dict) and profile.get("schema_version") in (PHASE16_DSWD_SCHEMA, PHASE16_DSWD_SHARED_SCHEMA)


def _phase16_snapshot_basis(incident):
    item = PHASE16_DSWD_PROTOCOL["incidents"][incident]
    return {"protocol_schema_version": PHASE16_DSWD_PROTOCOL["schema_version"],
            "incident": incident, "duration_basis": deepcopy(item["duration"]),
            "horizons": [{"requested_update": horizon,
                          "assumed_minutes_per_update": clock["minutes_per_update"],
                          "clock_role": clock["role"], "clock_status": "ASSUMPTION"}
                         for clock, horizon in zip(PHASE16_DSWD_PROTOCOL["clocks"], item["horizons"])]}


def _phase16_snapshot_manifest(active=None):
    shared = _phase16_is_shared(active)
    protocol = PHASE16_DSWD_SHARED_PROTOCOL if shared else PHASE16_DSWD_PROTOCOL
    path = PHASE15_CODE_DIR / (PHASE16_DSWD_SHARED_MANIFEST if shared else PHASE16_DSWD_MANIFEST)
    _phase15_no_reparse(path)
    raw = path.read_bytes()
    expected = {"dswd_snapshots": protocol}
    if _phase15_json_bytes(_phase15_yaml(raw)) != _phase15_json_bytes(expected):
        raise ValueError("Frozen disabled DSWD protocol manifest changed")
    identity = {"path": path.as_posix(), "sha256": sha256(raw).hexdigest()}
    if shared:
        document = PHASE15_CODE_DIR.parents[1] / protocol["protocol_evidence"]["path"]
        _phase15_no_reparse(document)
        if _sha256_file(document) != protocol["protocol_evidence"]["sha256"]:
            raise ValueError("Shared-CA v4 protocol document changed")
        identity["protocol_evidence"] = deepcopy(protocol["protocol_evidence"])
    return identity


def _phase16_snapshot_normalize(config):
    _phase15_metadata(config)
    basis = config.get("snapshot_basis", {})
    if not isinstance(basis, dict):
        raise ValueError("DSWD snapshot basis must be a mapping")
    incident = basis.get("incident")
    if incident not in PHASE16_DSWD_PROTOCOL["incidents"]:
        raise ValueError("Unknown DSWD incident")
    if _phase15_json_bytes(basis) != _phase15_json_bytes(_phase16_snapshot_basis(incident)):
        raise ValueError("DSWD horizons/clock/source basis must match the frozen protocol")
    item, sim, p = PHASE16_DSWD_PROTOCOL["incidents"][incident], config["simulation"], config["phase15"]
    seed = sim["seed"]
    expected_run = f"p16-dswd-{incident.replace('_', '-')}-rf-s{seed}-r1"
    base_schema = PHASE16_AIRBASE_SCHEMA if incident == "airbase" else PHASE16_CANDIDATE_SCHEMA
    base_scenario = (PHASE16_AIRBASE_SCENARIO_ID if incident == "airbase" else PHASE16_SCENARIO_ID).replace("-s42-", f"-s{seed}-")
    if (type(seed) is not int or seed not in PHASE16_DSWD_PROTOCOL["seeds"]
            or sim["max_timesteps"] != item["max_updates"] or type(sim["max_timesteps"]) is not int
            or p["run_id"] != expected_run or p["scenario_id"] != base_scenario + "-dswd-snap-v1"
            or config["environment"]["input_sha256"] != PHASE16_DSWD_PROTOCOL["input_sha256"]
            or config["ml_model"]["model_sha256"] != PHASE15_CANDIDATE_MODEL_SHA256
            or config["ml_model"]["calibration_manifest_sha256"] != PHASE15_CANDIDATE_MANIFEST_SHA256
            or config["ml_model"]["inference_chunk_size"] != 200000
            or config["candidate_inputs"]["window"]["global_origin"] != item["crop_origin"]
            or config["candidate_inputs"]["window"]["shape"] != item["shape"]
            or sim["ignition_points"] != item["ignition"]):
        raise ValueError("DSWD trajectory identity, maximum horizon or frozen controls changed")
    base = deepcopy(config)
    del base["snapshot_basis"]
    base["phase15"].update(schema_version=base_schema, scenario_id=base_scenario)
    base["simulation"]["max_timesteps"] = 300
    active = _phase15_normalize(base)
    active["phase15"].update(schema_version=PHASE16_DSWD_SCHEMA,
                             scenario_id=p["scenario_id"])
    active["simulation"]["max_timesteps"] = item["max_updates"]
    active["snapshot_basis"] = deepcopy(basis)
    return active


def _phase16_snapshot_outputs(basis):
    names = set(PHASE15_OUTPUTS)
    for row in basis["horizons"]:
        for kind in ("state", "reached"):
            name = f"snapshot_h{row['requested_update']:04d}_{kind}.tif"
            names.update((name, name + ".partial"))
    return frozenset(names)


class _Phase16Snapshots:
    """Post-step observation only; never calls an estimator or random generator."""

    def __init__(self, basis):
        self.basis = deepcopy(basis)
        horizons = basis["horizons"]
        if (not isinstance(horizons, list) or not horizons
                or any(type(r.get("requested_update")) is not int or r["requested_update"] <= 0
                       or type(r.get("assumed_minutes_per_update")) not in (int, float)
                       or not math.isfinite(r["assumed_minutes_per_update"])
                       or r["assumed_minutes_per_update"] <= 0 for r in horizons)
                or len({r["requested_update"] for r in horizons}) != len(horizons)):
            raise ValueError("Distinct positive integer post-update horizons and finite clocks required")
        self.rows = [dict(deepcopy(row), status="incomplete", output_valid=False,
                         duration_basis=deepcopy(basis.get("duration_basis")),
                         actual_capture_update=None, actual_stop_update=None,
                         absorbing_carry_forward=False, outputs=[]) for row in horizons]
        if basis.get("design") == PHASE16_DSWD_SHARED_PROTOCOL["design"]:
            for row in self.rows:
                row.update({key: deepcopy(basis[key]) for key in
                            ("design", "arm", "model_policy", "ignition_local_row_col", "ignition_global_row_col")})
        self.curve, self.completed, self.counts = [], 0, None
        self.failed_step = None
        self.failure_stage = "snapshot_failure"

    def fail(self):
        # A failed trajectory is never promoted to three successful horizon outcomes.
        for row in self.rows:
            row.update(status="incomplete", output_valid=False,
                       trajectory_stop_update=self.completed, actual_stop_update=self.completed)

    def record(self):
        return {"schema_version": "phase16_dswd_snapshot_series.v1",
                "basis": deepcopy(self.basis), "completed_updates": self.completed,
                "failed_step": self.failed_step,
                "initial": deepcopy(getattr(self, "initial", None)),
                "per_update": deepcopy(self.curve), "outcomes": deepcopy(self.rows)}

    def run(self, automata, check, capture):
        def observe(new_ignitions):
            state = validate_full_state_grid(automata.get_grid())
            counts = {name: int(np.count_nonzero(state == code)) for code, name in
                      enumerate(("non_burnable", "not_yet_burning", "ignited", "blazing", "extinguished"), 1)}
            reached = np.isin(state, (3, 4, 5))
            self.counts = counts
            return state, {"completed_update": self.completed, "new_ignition_count": new_ignitions,
                           "cumulative_reached_count": int(np.count_nonzero(reached)),
                           "active_cell_count": counts["ignited"] + counts["blazing"],
                           "state_counts": counts}
        def emit(state, observation, inactive):
            for row in self.rows:
                if row["actual_capture_update"] is not None:
                    continue
                if row["requested_update"] != self.completed and not inactive:
                    continue
                row.update(actual_capture_update=self.completed,
                           horizon_outcome_update=self.completed,
                           absorbing_carry_forward=self.completed < row["requested_update"],
                           state_counts=deepcopy(observation["state_counts"]),
                           cumulative_reached_count=observation["cumulative_reached_count"],
                           active_cell_count=observation["active_cell_count"],
                           termination_reason="inactive" if inactive else "timestep_limit",
                           censoring=not inactive)
                native = state.copy()
                reached = np.isin(native, (3, 4, 5)).astype(np.uint8)
                native.setflags(write=False)
                reached.setflags(write=False)
                self.failure_stage = "output_failure"
                check("before_snapshot")
                row["outputs"] = capture(deepcopy(row), native, reached)
                check("after_snapshot")
                row.update(status="complete" if inactive else "censored", output_valid=True)
        try:
            state, observation = observe(0)
            self.initial = dict(deepcopy(observation),
                                initial_ignition_count=observation["active_cell_count"],
                                initial_reached_count=observation["cumulative_reached_count"])
            self.curve.append(deepcopy(observation))
            maximum = max(row["requested_update"] for row in self.rows)
            while True:
                inactive = observation["active_cell_count"] == 0
                emit(state, observation, inactive)
                if inactive or self.completed == maximum:
                    for row in self.rows:
                        row["trajectory_stop_update"] = self.completed
                        row["actual_stop_update"] = self.completed
                    return ("inactive" if inactive else "timestep_limit"), self.completed, self.counts
                self.failure_stage = "step_failure"
                check("before_step")
                susceptible_before = (state == 2)
                reached_before = np.isin(state, (3, 4, 5))
                self.failed_step = self.completed + 1
                automata.step()
                self.completed += 1
                self.failed_step = None
                state, observation = observe(0)
                reached_after = np.isin(state, (3, 4, 5))
                if np.any(reached_before & ~reached_after):
                    raise ValueError("Reached-state membership must be cumulative")
                observation["new_ignition_count"] = int(np.count_nonzero(susceptible_before & reached_after))
                self.curve.append(deepcopy(observation))
                check("after_step")
        except BaseException:
            self.fail()
            raise


def _phase15_is_candidate(config):
    profile = config.get("phase15")
    return isinstance(profile, dict) and profile.get("schema_version") in (
        PHASE15_CANDIDATE_SCHEMA, PHASE16_CANDIDATE_SCHEMA, PHASE16_RF_GATE_SCHEMA, PHASE16_LR_GATE_SCHEMA,
        PHASE16_AIRBASE_SCHEMA, PHASE16_AIRBASE_DURATION90_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA,
        PHASE16_DSWD_SCHEMA, PHASE16_DSWD_SHARED_SCHEMA,

    )


def _phase15_candidate_contract(active, *, shared_incident=None):
    c, sim, ml = active["candidate_inputs"], active["simulation"], active["ml_model"]
    schema = active["phase15"]["schema_version"]
    duration90 = schema == PHASE16_AIRBASE_DURATION90_SCHEMA
    airbase = schema in (PHASE16_AIRBASE_SCHEMA, PHASE16_AIRBASE_DURATION90_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA)
    gated = schema in (PHASE16_RF_GATE_SCHEMA, PHASE16_LR_GATE_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA)
    successor = schema in (PHASE16_CANDIDATE_SCHEMA, PHASE16_RF_GATE_SCHEMA, PHASE16_LR_GATE_SCHEMA, PHASE16_AIRBASE_SCHEMA, PHASE16_AIRBASE_DURATION90_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA)
    lr_comparison = schema in (PHASE16_CANDIDATE_SCHEMA, PHASE16_LR_GATE_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA) and ml["model_sha256"] == PHASE16_LR_MODEL_SHA256
    if schema in (PHASE16_LR_GATE_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA) and not lr_comparison:
        raise ValueError("LR gate requires the exact rebuilt LR release")
    bearing = PHASE16_AIRBASE_GRID_FROM_DEG if airbase else PHASE16_GRID_FROM_DEG if successor else 315.0
    allowed_seeds = range(42, 47) if successor else (42,)
    if successor:
        scenario = PHASE16_AIRBASE_SCENARIO_ID if airbase else PHASE16_LR_SCENARIO_ID if lr_comparison else PHASE16_SCENARIO_ID
        if airbase and lr_comparison:
            scenario = scenario.replace("-rf-", "-lr-")
        scenario = scenario.replace("-s42-", f"-s{sim['seed']}-")
        if gated:
            scenario += "-gate040"
        if duration90:
            scenario += "-h90-dt1m"
        # Canonical JSON retains scalar types, unlike dict equality (False == 0, 49 == 49.0).
        basis = PHASE16_AIRBASE_WIND_INPUT_BASIS if airbase else PHASE16_WIND_INPUT_BASIS
        if (_phase15_json_bytes(active["wind_input_basis"]) != _phase15_json_bytes(basis)
                or active["phase15"]["scenario_id"] != scenario):
            raise ValueError("Phase 16 requires the adopted wind input basis and scenario identity")
    if duration90 and _phase15_json_bytes(active["duration_basis"]) != _phase15_json_bytes(PHASE16_AIRBASE_DURATION90_BASIS):
        raise ValueError("Duration sensitivity requires the exact owner-approved assumption and horizon")
    _phase15_exact(c, "interpretation_policy mask_seed_policy expected_grid window reader_capability")
    if (c["interpretation_policy"] != candidate_inputs.POLICY_ID
            or c["mask_seed_policy"] != candidate_inputs.MASK_POLICY_ID
            or canonical_metadata_hash(c["window"]) != canonical_metadata_hash(PHASE16_AIRBASE_WINDOW if airbase else PHASE15_CANDIDATE_WINDOW)):
        raise ValueError("Candidate requires the frozen incident window and input policies")
    if shared_incident is not None and shared_incident != ("airbase" if airbase else "santa_maria"):
        raise ValueError("Shared-CA incident and base policy disagree")
    ignitions = (PHASE16_DSWD_SHARED_PROTOCOL["incidents"][shared_incident]["ignition"]
                 if shared_incident is not None else
                 PHASE16_AIRBASE_IGNITIONS if airbase else PHASE15_CANDIDATE_IGNITIONS)
    if (sim["ignition_points"] != ignitions
            or any(type(v) is not int for point in sim["ignition_points"] for v in point)
            or sim["seed"] not in allowed_seeds or sim["max_timesteps"] != (90 if duration90 else 300)
            or active["placeholder_transition"] != {"T_3_to_4": 1, "T_4_to_5": 1, "wind_weight": 0.20}
            or active["wind"]["speed_kmh"] != (10.6 if airbase else 17.7) or active["wind"]["direction_deg"] != bearing
            or active["flammability_weights"] != DEFAULT_FLAMMABILITY_WEIGHTS
            or active["environment"]["nodata_value"] != -9999):
        raise ValueError("Candidate engineering case may not substitute seeds, wind, timers or weights")
    grid = c["expected_grid"]
    _phase15_exact(grid, "shape crs transform")
    if (grid["shape"] != list(candidate_crops.PARENT_SHAPE)
            or any(type(v) is not int for v in grid["shape"])
            or grid["crs"] != "EPSG:32651"
            or not isinstance(grid["transform"], list) or len(grid["transform"]) != 9
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in grid["transform"])
            or grid["transform"][0:2] != [3.0, 0.0]
            or grid["transform"][3:5] != [0.0, -3.0]
            or grid["transform"][6:] != [0.0, 0.0, 1.0]):
        raise ValueError("Candidate parent geometry is unresolved or unsupported")
    capability = c["reader_capability"]
    if (not isinstance(capability, dict) or capability.get("policy_id") != candidate_inputs.POLICY_ID
            or capability.get("runtime") != {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__}
            or capability.get("controls_verified") is not True
            or capability.get("windowed_reads_verified") is not True
            or capability.get("environment_options") != candidate_inputs.ENV_OPTIONS
            or capability.get("open_options") != candidate_inputs.OPEN_OPTIONS):
        raise ValueError("Scoped candidate window-reader evidence is unresolved")
    candidate_inputs.require_text(capability.get("verification_reference"))
    candidate_inputs.require_text(capability.get("window_verification_reference"))
    if lr_comparison:
        if (ml["model_path"] != _phase15_path(PHASE16_LR_MODEL)
                or ml["calibration_manifest_path"] is not None
                or ml["calibration_manifest_sha256"] is not None):
            raise ValueError("LR comparison requires the exact rebuilt release and no calibration manifest")
        return
    if (ml["model_path"] != _phase15_path(PHASE15_CANDIDATE_MODEL)
            or ml["model_sha256"] != PHASE15_CANDIDATE_MODEL_SHA256
            or ml["calibration_manifest_sha256"] != PHASE15_CANDIDATE_MANIFEST_SHA256):
        raise ValueError("Accepted candidate calibrated release changed")
    ml["calibration_manifest_path"] = _phase15_path(ml["calibration_manifest_path"])
    if ml["calibration_manifest_path"] != (Path(ml["model_path"]).parent / "calibration_manifest.json").as_posix():
        raise ValueError("Candidate calibration manifest must be the accepted sibling")


def _phase15_normalize(config):
    if _phase16_is_shared(config):
        return _phase16_shared_normalize(config)
    if _phase16_is_snapshot(config):
        return _phase16_snapshot_normalize(config)
    return _phase15_normalize_base(config)


def _phase15_normalize_base(config, *, shared_incident=None):
    _phase15_metadata(config)
    candidate = _phase15_is_candidate(config)
    gated = candidate and config["phase15"]["schema_version"] in (PHASE16_RF_GATE_SCHEMA, PHASE16_LR_GATE_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA)
    duration90 = candidate and config["phase15"]["schema_version"] == PHASE16_AIRBASE_DURATION90_SCHEMA
    airbase = candidate and config["phase15"]["schema_version"] in (PHASE16_AIRBASE_SCHEMA, PHASE16_AIRBASE_DURATION90_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA)
    successor = candidate and config["phase15"]["schema_version"] in (PHASE16_CANDIDATE_SCHEMA, PHASE16_RF_GATE_SCHEMA, PHASE16_LR_GATE_SCHEMA, PHASE16_AIRBASE_SCHEMA, PHASE16_AIRBASE_DURATION90_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA)
    keys_by_section = dict(PHASE15_KEYS)
    if candidate:
        keys_by_section["phase15"] = keys_by_section["phase15"].replace(" phase14_reconciliation_sha256", "")
        keys_by_section["ml_model"] += " calibration_manifest_path calibration_manifest_sha256"
        keys_by_section["candidate_inputs"] = "interpretation_policy mask_seed_policy expected_grid window reader_capability"
    if successor:
        keys_by_section["wind_input_basis"] = " ".join(PHASE16_AIRBASE_WIND_INPUT_BASIS if airbase else PHASE16_WIND_INPUT_BASIS)
    if gated:
        keys_by_section["ml_model"] += " proba_threshold"
    if duration90:
        keys_by_section["duration_basis"] = " ".join(PHASE16_AIRBASE_DURATION90_BASIS)
    _phase15_exact(config, " ".join(keys_by_section))
    active = deepcopy(config)
    for section, keys in keys_by_section.items():
        _phase15_exact(active[section], keys)
    p, sim, ml = active["phase15"], active["simulation"], active["ml_model"]
    schema = p["schema_version"] if candidate else "phase15_setc_rf_ca.v1"
    if (p["schema_version"], p["mode"], p["timestep_unit"]) != (schema, "pilot", "simulation_step") or p["enabled"] is not False:
        raise ValueError("Phase 15 disk configuration must remain disabled")
    for key in ("authorization_reference", "scenario_id", "run_id"):
        if not isinstance(p[key], str) or not p[key].strip():
            raise ValueError(f"phase15.{key} is required before activation")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", p["run_id"]) is None or p["run_id"].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("Unsafe run_id")
    if not candidate and p["phase14_reconciliation_sha256"] != PHASE15_RECONCILIATION:
        raise ValueError("Accepted reconciliation provenance changed")
    for key in ("threshold", "inference_threshold", "classification_threshold"):
        if ml[key] is not None:
            raise ValueError("Classification thresholds must remain null")
    if sim["resume_checkpoint"] is not None or sim["emergency_checkpoint_enabled"] is not False or sim["early_stop"] is not True:
        raise ValueError("Pilot requires early stopping and no checkpoint/resume")
    for value in (sim["checkpoint_interval"], active["output"]["save_every_n_steps"]):
        if type(value) is not int or value != 0:
            raise ValueError("Periodic writes must be disabled")
    mode = "stochastic_probability_gate" if gated else "stochastic_probability"
    if ml["enabled"] is not True or type(ml["positive_label"]) is not int or ml["positive_label"] != 1 or ml["inference_mode"] != mode:
        raise ValueError("Approved stochastic model interface required")
    if gated and (type(ml["proba_threshold"]) not in (int, float) or ml["proba_threshold"] != 0.40):
        raise ValueError("Exploratory gate requires exact probability cutoff 0.40")
    if active["output"]["save_final_state"] is not True or active["rf_contract"] != {"feature_schema_version": "set_c.v1", "target_version": "ignition_t_plus_1.v1"}:
        raise ValueError("Unsupported output or feature contract")
    _phase15_number(sim["seed"], integer=True)
    _phase15_number(sim["max_timesteps"], 1, integer=True)
    _phase15_number(ml["inference_chunk_size"], 1, integer=True)
    for key in ("T_3_to_4", "T_4_to_5"):
        _phase15_number(active["placeholder_transition"][key], 1, integer=True)
        if active["placeholder_transition"][key] > 32767:
            raise ValueError("Timer exceeds int16 storage")
    _phase15_number(active["placeholder_transition"]["wind_weight"])
    for value in active["flammability_weights"].values():
        _phase15_number(value)
    _phase15_number(active["wind"]["speed_kmh"])
    direction = active["wind"]["direction_deg"]
    if type(direction) not in (int, float) or not math.isfinite(direction):
        raise ValueError("Finite wind bearing required")
    if successor and direction != (PHASE16_AIRBASE_GRID_FROM_DEG if airbase else PHASE16_GRID_FROM_DEG):
        raise ValueError("Phase 16 requires the exact prescribed raw grid-from bearing")
    active["wind"] = WindContract.from_config(active["wind"]).to_config()
    points = sim["ignition_points"]
    if not isinstance(points, list) or not points:
        raise ValueError("Explicit ignition points required")
    for point in points:
        if not isinstance(point, list) or len(point) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
            raise ValueError("Ignitions must be finite [x,y] pairs")
    resources = p["resources"]
    _phase15_exact(resources, "enforcement max_process_rss_bytes min_system_available_bytes min_free_output_bytes max_wall_seconds sample_interval_seconds")
    if resources["enforcement"] != "between_steps":
        raise ValueError("Only between_steps enforcement is implemented")
    for key in ("max_process_rss_bytes", "min_system_available_bytes", "min_free_output_bytes"):
        _phase15_number(resources[key], 1, integer=True)
    for key in ("max_wall_seconds", "sample_interval_seconds"):
        _phase15_number(resources[key])
        if resources[key] == 0:
            raise ValueError("Positive timing limits required")
    env = active["environment"]
    if type(env["nodata_value"]) not in (int, float) or not math.isfinite(env["nodata_value"]):
        raise ValueError("Finite nodata sentinel required")
    _phase15_exact(env["input_sha256"], "slope proximity buildings materials")
    env["raster_dir"] = _phase15_path(env["raster_dir"])
    for name in env["input_sha256"]:
        filename = env[name + "_file"]
        if not isinstance(filename, str) or re.fullmatch(r"[^/\\:]+\.tif", filename) is None or "ground_truth" in filename.lower():
            raise ValueError("Explicit environmental TIFF basename required")
        _phase15_digest(env["input_sha256"][name])
    if len({env[name + "_file"] for name in env["input_sha256"]}) != 4:
        raise ValueError("Environmental input paths must be distinct")
    ml["model_path"] = _phase15_path(ml["model_path"])
    if not candidate and (ml["model_path"] != _phase15_path(PHASE15_MODEL) or ml["model_sha256"] != PHASE15_MODEL_SHA256):
        raise ValueError("Accepted model binding changed")
    _phase15_digest(ml["model_sha256"])
    root = _phase15_path(active["output"]["output_dir"])
    if root != _phase15_path("output/phase15"):
        raise ValueError("Only the Phase 15 output root is permitted")
    active["output"]["output_dir"] = root
    if candidate:
        _phase15_candidate_contract(active, shared_incident=shared_incident)
    active = apply_pipeline_defaults(active)
    active["phase15"]["enabled"] = True
    return active


def _phase15_no_reparse(path):
    path = Path(path)
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise PermissionError(f"Reparse path is forbidden: {part}")


def _phase15_source():
    repo = PHASE15_CODE_DIR.parents[1]
    for name in PHASE15_ABSENT:
        if os.path.lexists(repo / name):
            raise ValueError("Unexpected package initializer; source inventory review required")
    files = {}
    for name in PHASE15_SOURCES:
        _phase15_no_reparse(repo / name)
        files[name] = _sha256_file(repo / name)
    base = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    def git(*args):
        return subprocess.run(base + list(args), check=True, capture_output=True, text=True,
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()
    return {"repository_root": repo.as_posix(), "head": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
            "files": files, "absent_paths": list(PHASE15_ABSENT)}


def _phase15_runtime():
    return {"executable": Path(sys.executable).resolve().as_posix(),
            "implementation": platform.python_implementation(), "python_version": sys.version,
            "cache_tag": sys.implementation.cache_tag, "pointer_bits": struct.calcsize("P") * 8,
            "platform": platform.platform(), "packages": {name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "rasterio", "scikit-learn", "joblib", "PyYAML")}}


def _phase15_binding(active, config_path, disk_sha256):
    p, env = active["phase15"], active["environment"]
    candidate = _phase15_is_candidate(active)
    result = {"schema_version": "phase15_execution_binding.v1", "approval_reference": p["authorization_reference"],
            "configuration": {"path": _phase15_path(str(config_path)), "disk_sha256": disk_sha256,
                              "activated_effective_sha256": canonical_metadata_hash(active)},
            "source": _phase15_source(), "runtime": _phase15_runtime(),
            "model": {"path": active["ml_model"]["model_path"], "sha256": active["ml_model"]["model_sha256"]},
            "inputs": {name: {"path": (Path(env["raster_dir"]) / env[name + "_file"]).as_posix(), "sha256": digest}
                       for name, digest in env["input_sha256"].items()},
            "run": {"run_id": p["run_id"], "scenario_id": p["scenario_id"],
                    "destination": (Path(active["output"]["output_dir"]) / p["run_id"]).as_posix()}}
    if candidate:
        ml = active["ml_model"]
        if ml["calibration_manifest_path"] is not None:
            result["model"]["manifest"] = {"path": ml["calibration_manifest_path"],
                                           "sha256": ml["calibration_manifest_sha256"]}
        result["candidate_inputs"] = deepcopy(active["candidate_inputs"])
        result["evidence_scope"] = {"family": "F01", "use": "calibration_development",
                                   "final_test_eligible": False, "threshold": None}
        if p["schema_version"] in (PHASE16_AIRBASE_SCHEMA, PHASE16_AIRBASE_DURATION90_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA):
            result["evidence_scope"] = {
                "incident": "airbase_sanggi_20250124", "use": "second_incident_exploratory",
                "final_test_eligible": False, "threshold": None,
                "qualification": "Spatial overlap/prior use and independence unverified; not held-out validation",
                "reference_accessed_by_simulation": False,
            }
            if p["schema_version"] == PHASE16_AIRBASE_DURATION90_SCHEMA:
                result["evidence_scope"]["duration_basis"] = deepcopy(active["duration_basis"])
        if p["schema_version"] in (PHASE16_RF_GATE_SCHEMA, PHASE16_LR_GATE_SCHEMA, PHASE16_AIRBASE_LR_GATE_SCHEMA):
            result["evidence_scope"]["probability_gate"] = {
                "cutoff": 0.40, "rule": "p < cutoff -> 0; p >= cutoff unchanged",
                "probability_dtype": "float32", "sampling": "unchanged full-grid Bernoulli",
                "status": "exploratory_threshold_sensitivity",
            }
    else:
        result["phase14_reconciliation_sha256"] = p["phase14_reconciliation_sha256"]
    if _phase16_is_snapshot(active):
        result["snapshot_basis"] = deepcopy(active["snapshot_basis"])
        result["protocol_manifest"] = (_phase16_snapshot_manifest(active) if _phase16_is_shared(active)
                                       else _phase16_snapshot_manifest())
        source_path = PHASE15_CODE_DIR / "dswd_snapshot_results.py"
        _phase15_no_reparse(source_path)
        result["source"]["files"]["Thesis_RF/Code/dswd_snapshot_results.py"] = _sha256_file(source_path)
        result["evidence_scope"] = {
            "incident": active["snapshot_basis"]["incident"], "use": "development_duration_sensitivity",
            "final_test_eligible": False, "threshold": None,
            "reference_accessed_by_simulation": False,
            "clock_status": "ASSUMPTION; not physical calibration or simulated suppression",
            "production_acceptance": ("PENDING: verified shared-arm counterpart required" if _phase16_is_shared(active)
                                      else "BLOCKED: RF/LR spatial domain parity NEEDS EVIDENCE")}
        if _phase16_is_shared(active):
            guard_path = PHASE15_CODE_DIR / "dswd_inference_guard.py"
            _phase15_no_reparse(guard_path)
            result["source"]["files"]["Thesis_RF/Code/dswd_inference_guard.py"] = _sha256_file(guard_path)
            caller_path = PHASE15_CODE_DIR / "run_dswd_shared_ca_v1.ps1"
            _phase15_no_reparse(caller_path)
            result["source"]["files"]["Thesis_RF/Code/run_dswd_shared_ca_v1.ps1"] = _sha256_file(caller_path)
            result["evidence_scope"].update(
                design=active["snapshot_basis"]["design"], arm=active["snapshot_basis"]["arm"],
                actual_ignition=deepcopy(active["simulation"]["ignition_points"]),
                probability_policy=deepcopy(active["snapshot_basis"]["model_policy"]),
                inherited_wind_anchor_distinct_from_new_ignition=True)
    return result


class _Phase15Stop(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


class _Phase15Resources:
    def __init__(self, policy, destination, started):
        self.policy, self.destination, self.started = policy, destination, started
        self.samples, self.violations = [], []
        self.last_end = None

    def check(self, boundary, force=True):
        begin = time.monotonic()
        if not force and self.last_end is not None and begin - self.last_end < self.policy["sample_interval_seconds"]:
            if begin - self.started >= self.policy["max_wall_seconds"]:
                self.violations = ["wall_time_limit"]
                raise _Phase15Stop("wall_time_limit")
            return
        sample = {"boundary": boundary, "monotonic_start": begin, "monotonic_end": None,
                  "elapsed_seconds": None, "valid": False,
                  "gap_seconds": None if self.last_end is None else begin - self.last_end,
                  "captured_at_utc": None}
        self.samples.append(sample)
        try:
            snapshot = capture_resource_snapshot(self.destination)
            for name, minimum in (("process_rss_bytes", 1), ("available_memory_bytes", 0), ("free_disk_bytes", 0)):
                value = getattr(snapshot, name)
                if type(value) is not int or value < minimum:
                    raise ValueError(f"Invalid resource measurement: {name}")
                sample[name] = value
            stamp = snapshot.captured_at_utc
            if not isinstance(stamp, str) or datetime.fromisoformat(stamp).utcoffset() != timedelta(0):
                raise ValueError("Resource timestamp must be aware UTC")
            sample.update(captured_at_utc=stamp, valid=True)
        except Exception as exc:
            sample["error"] = f"{type(exc).__name__}: {exc}"
        end = time.monotonic()
        sample.update(monotonic_end=end, elapsed_seconds=end - self.started)
        self.last_end = end
        wall = end - self.started >= self.policy["max_wall_seconds"]
        self.violations = ["wall_time_limit"] if wall else []
        if not sample["valid"]:
            self.violations.insert(0, "resource_measurement_failure")
            raise _Phase15Stop("resource_measurement_failure")
        resource = []
        for key, breach in (
            ("process_rss_bytes", sample["process_rss_bytes"] > self.policy["max_process_rss_bytes"]),
            ("available_memory_bytes", sample["available_memory_bytes"] < self.policy["min_system_available_bytes"]),
            ("free_disk_bytes", sample["free_disk_bytes"] < self.policy["min_free_output_bytes"]),
        ):
            if breach:
                resource.append(key)
        self.violations = resource + self.violations
        if resource or wall:
            raise _Phase15Stop("resource_limit" if resource else "wall_time_limit")

    def summary(self):
        valid = [sample for sample in self.samples if sample["valid"]]
        return {"attempted_count": len(self.samples), "valid_count": len(valid),
                "missing_count": len(self.samples) - len(valid), "samples": deepcopy(self.samples),
                "max_gap_seconds": max((s["gap_seconds"] or 0 for s in self.samples), default=None),
                "sampled_max_rss_bytes": max((s["process_rss_bytes"] for s in valid), default=None),
                "sampled_min_available_bytes": min((s["available_memory_bytes"] for s in valid), default=None),
                "sampled_min_free_disk_bytes": min((s["free_disk_bytes"] for s in valid), default=None)}


def _phase15_json_bytes(value):
    _phase15_metadata(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _phase15_write_json(path, record):
    payload = _phase15_json_bytes(record)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if path.read_bytes() != payload:
        raise OSError("Record round trip failed")
    return sha256(payload).hexdigest()


def _phase15_finalize(partial, final):
    # A hard link refuses an existing destination on both supported platforms; retain staging evidence.
    os.link(partial, final)


def _phase15_publish(destination, record):
    _phase15_write_json(destination / "termination.json.partial", record)
    _phase15_finalize(destination / "termination.json.partial", destination / "termination.json")


def _phase15_load_model(automata, identity, monitor):
    path = Path(identity["path"])
    manifest_observed = None
    if "manifest" in identity:
        manifest = identity["manifest"]
        manifest_path = Path(manifest["path"])
        _phase15_no_reparse(manifest_path)
        monitor.check("before_calibration_manifest_hash")
        digest = _sha256_file(manifest_path, lambda: monitor.check("manifest_hash_chunk", force=False))
        if digest != manifest["sha256"]:
            raise ValueError("Calibration manifest SHA-256 mismatch before deserialization")
        manifest_observed = {**manifest, "observed_sha256": digest}
        monitor.check("after_calibration_manifest_hash")
    monitor.check("before_model_hash")
    _phase15_no_reparse(path)
    with path.open("rb") as stream, BytesIO() as buffer:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            buffer.write(chunk)
            monitor.check("model_read_chunk", force=False)
        payload = buffer.getvalue()
    digest = sha256(payload).hexdigest()
    if digest != identity["sha256"]:
        raise ValueError("Model SHA-256 mismatch before deserialization")
    monitor.check("after_model_hash")
    monitor.check("before_model_load")
    # Never reopen the path: substitution cannot change the bytes passed to the generic loader.
    automata.load_model(BytesIO(payload))
    monitor.check("after_model_load")
    observed = {**identity, "observed_sha256": digest, "interface_validated": True}
    if manifest_observed is not None:
        observed["manifest"] = manifest_observed
    return observed


def _phase15_candidate_environment(active, identities, monitor):
    c = active["candidate_inputs"]
    for identity in identities.values():
        _phase15_no_reparse(Path(identity["path"]))
    monitor.check("before_candidate_input_preparation")
    window = c["window"]
    prepared = candidate_inputs.read_inputs(
        {name: item["path"] for name, item in identities.items()},
        {name: item["sha256"] for name, item in identities.items()},
        expected_grid=c["expected_grid"], configured_nodata=active["environment"]["nodata_value"],
        capability=c["reader_capability"], window=window["global_origin"] + window["shape"])
    if (prepared.global_origin != tuple(window["global_origin"])
            or prepared.environment["grid_shape"] != tuple(window["shape"])
            or prepared.full_grid_shape != tuple(c["expected_grid"]["shape"])):
        raise ValueError("Candidate preparation does not match the bound crop")
    points = [[r + prepared.global_origin[0], col + prepared.global_origin[1]]
              for r, col in active["simulation"]["ignition_points"]]
    resolved = candidate_inputs.validate_seeds(points, prepared, "grid_row_col")
    # Atomic seed validation precedes construction and every set_ignition call.
    monitor.check("after_candidate_input_preparation")
    environment = prepared.environment
    return environment, {
        "inputs": {name: {**item, "observed_sha256": prepared.input_hashes[name]}
                   for name, item in identities.items()},
        "grid": {"shape": list(environment["grid_shape"]), "crs": environment["crs"].to_wkt(),
                 "transform": list(environment["transform"])[:6]},
        "parent_grid": deepcopy(c["expected_grid"]), "window": deepcopy(window),
        "interpretation": deepcopy(prepared.interpretation), "mask_identities": prepared.mask_identities(),
        "converted_ignitions": [list(point) for point in resolved],
        "rng_policy": "prescribed_seed_crop_shaped_draw.v1",
    }


def _phase15_environment(active, identities, monitor):
    observed = {}
    sentinel = active["environment"]["nodata_value"]
    for name, identity in identities.items():
        path = Path(identity["path"])
        _phase15_no_reparse(path)
        for suffix in (".msk", ".aux.xml", ".ovr"):
            if os.path.lexists(str(path) + suffix):
                raise ValueError("Companion raster files require a contract amendment")
        monitor.check("before_input_hash_" + name)
        digest = _sha256_file(path, lambda: monitor.check("input_hash_chunk", force=False))
        if digest != identity["sha256"]:
            raise ValueError(f"Input SHA-256 mismatch: {name}")
        observed[name] = {**identity, "observed_sha256": digest}
        monitor.check("after_input_hash_" + name)
        if name in ("materials", "buildings"):
            with rasterio.open(path) as src:
                # Validate native values before the unchanged loader narrows them to float32.
                categorical = src.read(1)
            invalid = categorical == sentinel
            if name == "materials":
                invalid |= categorical == -9999
            allowed = range(6) if name == "materials" else (0, 10)
            if np.any(~invalid & ~np.isin(categorical, allowed)):
                raise ValueError(f"Invalid source {name} values at original precision")
            del categorical, invalid
            monitor.check("after_source_domain_" + name)
    paths = [Path(item["path"]) for item in identities.values()]
    if any(os.path.samefile(left, right) for i, left in enumerate(paths) for right in paths[i + 1:]):
        raise ValueError("Environmental inputs alias one another")
    manager = EnvironmentManager(active["environment"])
    monitor.check("before_environment_load")
    manager.load_rasters()
    monitor.check("after_environment_load")
    for name, identity in identities.items():
        raw = getattr(manager, name + "_raw")
        invalid = raw == sentinel
        if name in ("slope", "materials"):
            invalid |= raw == -9999
        if np.any(~np.isfinite(raw) & ~invalid):
            raise ValueError(f"Non-finite usable {name} values")
        with rasterio.open(identity["path"]) as src:
            if src.count != 1 or src.driver != "GTiff" or src.crs is None or not src.crs.is_projected:
                raise ValueError("Single-band projected environmental GeoTIFF required")
            if src.transform.b != 0 or src.transform.d != 0 or src.transform.a <= 0 or src.transform.e >= 0:
                raise ValueError("Wind requires a north-up projected grid")
            if len(src.files) != 1 or (src.nodata is not None and src.nodata not in (sentinel, -9999 if name == "slope" else sentinel)):
                raise ValueError("Unsupported raster nodata/companion semantics")
            if np.any((src.read_masks(1) == 0) & ~invalid):
                raise ValueError("Native invalid mask is not represented by loader semantics")
        monitor.check("after_input_domain_" + name)
    material = manager.materials_raw
    material_present = (material != sentinel) & (material != -9999) & (material > 0)
    if not np.array_equal(manager.buildings_raw == 10, material_present):
        raise ValueError("Full-grid building/material identities disagree")
    monitor.check("before_environment_prepare")
    manager.build_masks()
    manager.normalize_layers()
    for name, identity in identities.items():
        if _sha256_file(Path(identity["path"]), lambda: monitor.check("input_recheck_chunk", force=False)) != identity["sha256"]:
            raise ValueError("Input changed during loading")
    monitor.check("after_environment_prepare")
    environment = manager.get_environment()
    geometry = {"shape": list(environment["grid_shape"]), "crs": environment["crs"].to_wkt(),
                "transform": list(environment["transform"])[:6]}
    return environment, {"inputs": observed, "grid": geometry}


def _phase15_raster(destination, name, array, automata, monitor, *, allowed_outputs=None):
    allowed_outputs = PHASE15_OUTPUTS if allowed_outputs is None else allowed_outputs
    if name not in allowed_outputs or name + ".partial" not in allowed_outputs:
        raise ValueError("Raster filename is outside the approved output policy")
    partial, final = destination / (name + ".partial"), destination / name
    monitor.check("before_write_" + name)
    with partial.open("xb"):
        pass
    with rasterio.open(partial, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
                       count=1, dtype=array.dtype.name, crs=automata.crs, transform=automata.transform) as dst:
        dst.write(array, 1)
    monitor.check("after_write_" + name)
    monitor.check("before_verify_" + name)
    with rasterio.open(partial) as src:
        if (src.shape != array.shape or src.count != 1 or src.dtypes != (array.dtype.name,)
                or src.crs != automata.crs or src.transform != automata.transform or src.nodata is not None
                or len(src.files) != 1 or not np.array_equal(src.read(1), array)):
            raise ValueError("Final raster round trip failed")
    monitor.check("after_verify_" + name)
    monitor.check("before_finalize_" + name)
    _phase15_finalize(partial, final)
    monitor.check("after_finalize_" + name)
    digest = _sha256_file(final, lambda: monitor.check("output_hash_chunk", force=False))
    if not {path.name for path in destination.iterdir()} <= allowed_outputs:
        raise ValueError("Unexpected output/sidecar")
    return {"filename": name, "bytes": final.stat().st_size, "sha256": digest,
            "verified": True, "shape": list(array.shape), "dtype": array.dtype.name,
            "crs": automata.crs.to_wkt(), "transform": list(automata.transform)[:6]}


def _run_phase15(config, *, explicitly_authorized, approval_reference, expected_binding_sha256,
                 config_path, clock, invocation):
    """Only the owner or explicitly owner-designated reviewer authorizes this binding; assessment alone does not."""
    started, started_at = clock
    if explicitly_authorized is not True or config_path is None:
        raise PermissionError("Phase 15 requires explicit owner authorization and config_path")
    _phase15_digest(expected_binding_sha256)
    config_path = Path(config_path).resolve()
    raw = config_path.read_bytes()
    active = _phase15_normalize(_phase15_yaml(raw))
    if canonical_metadata_hash(_phase15_normalize(config)) != canonical_metadata_hash(active):
        raise ValueError("Direct-call and disk configuration disagree")
    if approval_reference != active["phase15"]["authorization_reference"]:
        raise PermissionError("Approval reference mismatch")
    binding = _phase15_binding(active, config_path, sha256(raw).hexdigest())
    if canonical_metadata_hash(binding) != expected_binding_sha256:
        raise PermissionError("Independent binding SHA-256 mismatch")
    destination = Path(binding["run"]["destination"])
    _phase15_no_reparse(destination.parent)
    if not destination.parent.is_dir() or os.path.lexists(destination):
        raise FileExistsError("Approved output root must exist and run directory must be absent")
    monitor = _Phase15Resources(active["phase15"]["resources"], destination, started)
    monitor.check("preallocation")
    monitor.check("before_allocation")
    destination.mkdir()
    start_hash, completed, counts, failed_step = None, None, None, None
    observed, outputs = {}, []
    snapshots = None
    snapshot_profile = _phase16_is_snapshot(active)
    allowed_outputs = (_phase16_snapshot_outputs(active["snapshot_basis"])
                       if snapshot_profile else PHASE15_OUTPUTS)
    reason, publication_attempted = "output_failure", False

    def termination(stop, error=None):
        success = error is None and stop in ("inactive", "timestep_limit")
        elapsed = time.monotonic() - started
        return {"schema_version": "phase15_run_termination.v1", "run_id": binding["run"]["run_id"],
                "start_record_sha256": start_hash, "binding_sha256": expected_binding_sha256,
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": ("complete" if stop == "inactive" else "censored") if success else
                          ("stopped" if stop in ("resource_limit", "wall_time_limit") else "cancelled" if stop == "cancelled" else "failed"),
                "termination_reason": stop, "exit_code": 0 if success else 1,
                "last_completed_timestep": completed, "failed_step": failed_step,
                "state_counts": counts, "active_cell_count": None if counts is None else counts["ignited"] + counts["blazing"],
                "decision_elapsed_seconds": elapsed,
                "clock_policy": {"boundary": "phase15_entry", "termination_publication_excluded": True,
                                 "limits_are_cooperative": True},
                "resource_observations": monitor.summary(), "violations": list(monitor.violations),
                "observed_bindings": observed, "output_valid": success, "outputs": outputs,
                **({"snapshot_series": snapshots.record()} if snapshots is not None else {}),
                "error": None if error is None else {"type": type(error).__name__, "message": str(error)}}

    try:
        start = {"schema_version": "phase15_run_start.v1", **binding["run"],
                 "started_at_utc": started_at, "clock_boundary": "phase15_entry",
                 "authorization_reference": approval_reference, "invocation": invocation,
                 "binding": {"payload": binding, "expected_sha256": expected_binding_sha256,
                             "calculated_sha256": canonical_metadata_hash(binding)},
                 "configuration": {**binding["configuration"], "effective": active},
                 "source": binding["source"], "runtime": binding["runtime"],
                 "model": {**binding["model"], "observed_sha256": None, "interface_validated": None},
                 "inputs": {key: {**value, "observed_sha256": None} for key, value in binding["inputs"].items()},
                 "scenario": {"seed": active["simulation"]["seed"], "wind": active["wind"],
                              "ignition_points": active["simulation"]["ignition_points"], "converted_ignitions": None,
                              "grid": None, "timing": active["placeholder_transition"],
                              "max_timesteps": active["simulation"]["max_timesteps"]},
                 "resource_policy": monitor.policy, "initial_resource_observations": monitor.summary(),
                 "output_policy": {"allowlist": sorted(allowed_outputs), "resume": False}}
        start_hash = _phase15_write_json(destination / "start.json", start)
        monitor.check("after_start_record")
        reason = "preflight_failure"
        # Disable auxiliary discovery/writes; this contract permits self-contained GeoTIFFs only.
        with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            candidate = _phase15_is_candidate(active)
            if candidate:
                environment, observed = _phase15_candidate_environment(active, binding["inputs"], monitor)
            else:
                environment, observed = _phase15_environment(active, binding["inputs"], monitor)
            automata = FireAutomata(environment, active)
            observed["model"] = _phase15_load_model(automata, binding["model"], monitor)
            points = ([tuple(point) for point in observed["converted_ignitions"]] if candidate else
                      _convert_geographic_ignition_points(automata, active["simulation"]["ignition_points"]))
            if len(set(points)) != len(points) or any(not automata.burnable_mask[row, col] for row, col in points):
                raise ValueError("Every ignition must be a distinct valid mapped-building cell")
            automata.set_ignition(points)
            observed["converted_ignitions"] = [list(point) for point in points]
            completed, counts = 0, automata.get_state_counts()
            reason = "reporting_failure"
            print(f"Phase 15 started: {destination.name}")
            if snapshot_profile:
                snapshots = _Phase16Snapshots(active["snapshot_basis"])
                for row in snapshots.rows:
                    row.update(execution_binding_sha256=expected_binding_sha256,
                               configuration_sha256=binding["configuration"]["activated_effective_sha256"],
                               start_record_sha256=start_hash, run_id=active["phase15"]["run_id"],
                               seed=active["simulation"]["seed"], incident=active["snapshot_basis"]["incident"])
                def capture(row, native, reached):
                    captured = []
                    for kind, array in (("state", native), ("reached", reached)):
                        name = f"snapshot_h{row['requested_update']:04d}_{kind}.tif"
                        identity = _phase15_raster(destination, name, array, automata, monitor,
                                                   allowed_outputs=allowed_outputs)
                        outputs.append(identity)
                        captured.append(identity)
                    return captured
                try:
                    reason, completed, counts = snapshots.run(automata, monitor.check, capture)
                finally:
                    completed, counts = snapshots.completed, snapshots.counts
                    failed_step = snapshots.failed_step
            else:
                for step in range(1, active["simulation"]["max_timesteps"] + 1):
                    monitor.check("before_step")
                    reason, failed_step = "step_failure", step
                    automata.step()
                    completed, counts, failed_step = step, None, None
                    counts = automata.get_state_counts()
                    monitor.check("after_step")
                    if counts["ignited"] + counts["blazing"] == 0:
                        reason = "inactive"
                        break
                    reason = "timestep_limit"
            reason_at_stop = reason
            reason = "output_failure"
            state = validate_full_state_grid(automata.get_grid())
            outputs.append(_phase15_raster(destination, "final_state.tif", state, automata, monitor,
                                           allowed_outputs=allowed_outputs))
            valid = automata.simulation_valid_mask
            if valid.dtype != np.dtype(bool) or valid.shape != state.shape:
                raise ValueError("Exact environmental-validity mask required")
            mask = valid.astype(np.uint8)
            outputs.append(_phase15_raster(destination, "simulation_environment_valid.tif", mask, automata, monitor,
                                           allowed_outputs=allowed_outputs))
            observed["simulation_environment_valid_counts"] = {"true": int(np.count_nonzero(valid)),
                                                               "false": int(valid.size - np.count_nonzero(valid))}
            monitor.check("final_completion_decision")
            # This is the last limit decision; synchronous termination publication is a soft-bound tail.
            record = termination(reason_at_stop)
            record["decision_elapsed_seconds"] = time.monotonic() - started
            record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            if record["decision_elapsed_seconds"] >= monitor.policy["max_wall_seconds"]:
                monitor.violations = ["wall_time_limit"]
                raise _Phase15Stop("wall_time_limit")
            publication_attempted = True
            _phase15_publish(destination, record)
    except BaseException as exc:
        if snapshots is not None:
            snapshots.fail()
            if reason != "output_failure":
                reason = snapshots.failure_stage
        stop = "cancelled" if isinstance(exc, KeyboardInterrupt) else exc.reason if isinstance(exc, _Phase15Stop) else reason
        if not publication_attempted:
            try:
                publication_attempted = True
                _phase15_publish(destination, termination(stop, exc))
            except BaseException as report_error:
                if hasattr(exc, "add_note"):
                    exc.add_note(f"Termination reporting also failed: {type(report_error).__name__}: {report_error}")
        raise
    try:
        print(f"Phase 15 committed: {destination.name} ({record['termination_reason']})")
    except BaseException as exc:
        raise RuntimeError(f"post_commit_reporting_failure: committed run {destination}; owner review required") from exc


def run_simulation(config: dict, *, transition_observer=None, transition_provenance=None,
                   explicitly_authorized=False, approval_reference=None, expected_binding_sha256=None,
                   config_path=None) -> None:
    clock = (time.monotonic(), datetime.now(timezone.utc).isoformat())
    if "dswd_snapshots" in config:
        raise PermissionError("Disabled DSWD campaign manifest is preparation only, not an execution configuration")
    if ("candidate_inputs" in config or "wind_input_basis" in config) and not _phase15_is_candidate(config):
        raise ValueError("Candidate-marked configuration requires the candidate Phase 15 profile")
    if "phase15" in config:
        if transition_observer is not None or transition_provenance is not None:
            raise ValueError("Phase 15 does not collect transition observations")
        return _run_phase15(config, explicitly_authorized=explicitly_authorized,
                            approval_reference=approval_reference, expected_binding_sha256=expected_binding_sha256,
                            config_path=config_path, clock=clock,
                            invocation={"kind": "direct", "interpreter": sys.executable, "argv": None})
    if explicitly_authorized or approval_reference is not None or expected_binding_sha256 is not None or config_path is not None:
        raise PermissionError("Phase 15 controls cannot select a legacy run")
    return _run_legacy_simulation(config, transition_observer=transition_observer,
                                  transition_provenance=transition_provenance)


def simulation_main(argv: list[str] | None = None) -> int:
    clock = (time.monotonic(), datetime.now(timezone.utc).isoformat())
    parser = argparse.ArgumentParser(description="Manual RF simulation; no automatic spatial validation")
    parser.add_argument("--config", default=str(PHASE15_CODE_DIR / "config/default_experiment.yaml"))
    parser.add_argument("--execute-approved-phase15-pilot", action="store_true",
                        help="Requires owner or explicitly owner-designated reviewer authorization; assessment is insufficient")
    parser.add_argument("--approval-reference")
    parser.add_argument("--expected-binding-sha256")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if "dswd_snapshots" in config:
            raise PermissionError("Disabled DSWD campaign manifest cannot dispatch a trajectory")
        if "phase15" in config:
            _run_phase15(config, explicitly_authorized=args.execute_approved_phase15_pilot,
                         approval_reference=args.approval_reference, expected_binding_sha256=args.expected_binding_sha256,
                         config_path=args.config, clock=clock,
                         invocation={"kind": "cli", "interpreter": sys.executable,
                                     "argv": list(sys.argv) if argv is None else [sys.argv[0], *argv]})
        else:
            run_simulation(config, explicitly_authorized=args.execute_approved_phase15_pilot,
                           approval_reference=args.approval_reference, expected_binding_sha256=args.expected_binding_sha256)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        try:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(simulation_main())
