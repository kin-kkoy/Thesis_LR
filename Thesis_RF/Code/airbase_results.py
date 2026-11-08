"""Identity-bound scoring for the Phase 16 Airbase runs."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.env import get_gdal_config

try:
    from .validation_engine import evaluate_final_footprint_arrays
    from . import orchestrator
except ImportError:
    from validation_engine import evaluate_final_footprint_arrays
    import orchestrator


CODE_DIR = Path(__file__).resolve().parent
REFERENCE_PATH = CODE_DIR / "dataFiles" / "stack_ground_truth_airbase.tif"
BUILDINGS_PATH = CODE_DIR / "processedData" / "raster" / "raster" / "stack_buildings.tif"
REFERENCE_SHA256 = "1627ac903354f029572fa9abc9c61088909b38a5357c09228361f9021d2f52db"
BUILDINGS_SHA256 = "da53c3e63b661471b3366fc68249793c6a5a84e0d63bcc96492f60baaf4b83c7"
PARENT_SHAPE = (5489, 6896)
PARENT_TRANSFORM = Affine(3, 0, 598558.7784, 0, -3, 1144374.7699)
CRS_CODE = "EPSG:32651"
ENV_OPTIONS = {"GDAL_PAM_ENABLED": "NO", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR"}
STATE_CODES = (1, 2, 3, 4, 5)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_airbase_arrays(arrays: dict, *, termination_reason: str) -> dict:
    """Score aligned crop arrays while retaining the evaluator's censoring semantics."""
    state, reference, buildings, environment = (np.asarray(arrays[k]) for k in
                                                 ("prediction", "reference", "buildings", "environment"))
    shape = state.shape
    if len(shape) != 2 or any(np.shape(v) != shape for v in arrays.values()):
        raise ValueError("All Airbase arrays and masks must share a two-dimensional shape")
    for name in ("prediction", "reference", "buildings", "environment"):
        if arrays[name + "_valid"].dtype != np.bool_:
            raise ValueError("Raster validity masks must be boolean")
    if not np.all(np.isin(state, STATE_CODES)):
        raise ValueError("Unexpected final-state category")
    if not np.all(arrays["environment_valid"]) or not np.all(np.isin(environment, (0, 1))):
        raise ValueError("Environment must be fully valid and binary")
    if not np.all(np.isin(reference[arrays["reference_valid"]], (0, 1))):
        raise ValueError("Ground truth must contain only 0/1 on valid cells")
    if not np.all(np.isin(buildings[arrays["buildings_valid"]], (0, 10))):
        raise ValueError("Buildings must contain only 0/10 on valid cells")
    env = environment == 1
    common = (env & arrays["prediction_valid"] & arrays["reference_valid"]
              & arrays["buildings_valid"] & (buildings == 10))
    scored = evaluate_final_footprint_arrays(
        state, reference == 1, env, arrays["prediction_valid"], arrays["reference_valid"],
        arrays["buildings_valid"], buildings == 10, termination_reason=termination_reason)
    tn, fp = scored["confusion_matrix"][0]
    fn, tp = scored["confusion_matrix"][1]
    metrics = scored
    metrics["accuracy"] = (tn + tp) / (tn + fp + fn + tp) if tn + fp + fn + tp else None
    predicted_burn = np.isin(state, (3, 4, 5)) if termination_reason == "timestep_limit" else state == 5
    metrics["predicted_burned_cells"] = int(np.count_nonzero(predicted_burn & common))
    metrics["predicted_burned_full_crop"] = int(np.count_nonzero(predicted_burn))
    total = metrics["reference_coverage"]["reference_valid_burned_total"]
    excluded = metrics["reference_coverage"]["reference_burned_excluded_from_primary_domain"]
    metrics["excluded_percentage"] = 100 * excluded / total if total else None
    return {"metrics": metrics, "domain_cells": int(np.count_nonzero(common)),
            "termination_reason": termination_reason}


def _read_raster(path: Path, *, parent: bool, expected_sha256: str) -> tuple[np.ndarray, np.ndarray, dict]:
    path = Path(path)
    orchestrator._phase15_no_reparse(path)
    if not path.is_file():
        raise ValueError(f"Ordinary TIFF required: {path}")
    before = file_sha256(path)
    if before != expected_sha256:
        raise ValueError(f"Input hash mismatch for {path.name}")
    row, col = orchestrator.PHASE16_AIRBASE_WINDOW["global_origin"]
    height, width = orchestrator.PHASE16_AIRBASE_WINDOW["shape"]
    crop_transform = PARENT_TRANSFORM * Affine.translation(col, row)
    expected_shape = PARENT_SHAPE if parent else (height, width)
    expected_transform = PARENT_TRANSFORM if parent else crop_transform
    with rasterio.Env(**ENV_OPTIONS), warnings.catch_warnings():
        warnings.simplefilter("error")
        if any(get_gdal_config(k, normalize=False) != v for k, v in ENV_OPTIONS.items()):
            raise ValueError("Restricted raster controls are not effective")
        with rasterio.open(path, mode="r", driver="GTiff", sharing=False, GEOREF_SOURCES="INTERNAL") as src:
            if (src.driver != "GTiff" or src.mode != "r" or src.count != 1 or src.subdatasets
                    or src.options.get("GEOREF_SOURCES") != "INTERNAL"
                    or [Path(p).absolute() for p in src.files] != [path.absolute()]):
                raise ValueError("Expected a self-contained single-band TIFF")
            if (src.shape != expected_shape or src.crs != CRS.from_user_input(CRS_CODE)
                    or src.transform != expected_transform):
                raise ValueError("Raster geometry differs from bound Airbase grid")
            flags = {flag.name for flag in src.mask_flag_enums[0]}
            if flags not in ({"all_valid"}, {"nodata"}, {"per_dataset"}):
                raise ValueError("Unsupported native mask flags")
            window = Window(col, row, width, height) if parent else None
            if parent and src.window_transform(window) != crop_transform:
                raise ValueError("Airbase crop window alignment mismatch")
            raw, mask = src.read(1, window=window), src.read_masks(1, window=window)
            if raw.shape != (height, width) or raw.dtype.kind not in "biuf":
                raise ValueError("Unexpected native crop shape or categorical dtype")
            if flags == {"all_valid"} and not np.all(mask != 0):
                raise ValueError("Contradictory all-valid mask")
            valid = (mask != 0) & np.isfinite(raw) & (raw != -9999)
            if src.nodata is not None:
                valid &= raw != src.nodata
            identity = {"path": str(path.resolve()), "sha256": before, "dtype": str(raw.dtype),
                        "nodata": src.nodata, "mask_flags": sorted(flags)}
        if any(get_gdal_config(k, normalize=False) != v for k, v in ENV_OPTIONS.items()):
            raise ValueError("Restricted raster controls changed")
    if file_sha256(path) != before:
        raise ValueError(f"Input changed while reading: {path.name}")
    return raw, valid, identity


def score_airbase_run(run_dir: Path) -> dict:
    """Verify only identified Airbase records and TIFFs; never open a model or row dataset."""
    run_dir = Path(run_dir).resolve()
    start_path, end_path = run_dir / "start.json", run_dir / "termination.json"
    orchestrator._phase15_no_reparse(start_path)
    orchestrator._phase15_no_reparse(end_path)
    start_bytes, end_bytes = start_path.read_bytes(), end_path.read_bytes()
    start, end = json.loads(start_bytes), json.loads(end_bytes)
    binding = start["binding"]["payload"]
    calculated = orchestrator.canonical_metadata_hash(binding)
    if (start["schema_version"] != "phase15_run_start.v1"
            or binding["schema_version"] != "phase15_execution_binding.v1"
            or start["binding"]["expected_sha256"] != calculated
            or start["binding"]["calculated_sha256"] != calculated
            or end["schema_version"] != "phase15_run_termination.v1"
            or end["binding_sha256"] != calculated
            or end["start_record_sha256"] != sha256(start_bytes).hexdigest()):
        raise ValueError("Run record or execution binding identity mismatch")
    config = start["configuration"]["effective"]
    schema = config["phase15"]["schema_version"]
    if schema not in (orchestrator.PHASE16_AIRBASE_SCHEMA, orchestrator.PHASE16_AIRBASE_DURATION90_SCHEMA,
                      orchestrator.PHASE16_AIRBASE_LR_GATE_SCHEMA):
        raise ValueError("Expected an approved Airbase schema")
    lr = schema == orchestrator.PHASE16_AIRBASE_LR_GATE_SCHEMA
    disabled = json.loads(json.dumps(config))
    disabled["phase15"]["enabled"] = False
    if orchestrator.canonical_metadata_hash(orchestrator._phase15_normalize(disabled)) != orchestrator.canonical_metadata_hash(config):
        raise ValueError("Effective Airbase configuration does not satisfy its frozen contract")
    if orchestrator.canonical_metadata_hash(config) != binding["configuration"]["activated_effective_sha256"]:
        raise ValueError("Effective configuration identity mismatch")
    seed = config["simulation"]["seed"]
    horizon = "-h90" if schema == orchestrator.PHASE16_AIRBASE_DURATION90_SCHEMA else ""
    expected_run = f"p16-airbase-lr-t040-s{seed}-r1" if lr else f"p16-airbase-rf-nocutoff{horizon}-s{seed}-r1"
    if (type(seed) is not int or seed not in range(42, 47) or run_dir.name != expected_run
            or start["run_id"] != expected_run or end["run_id"] != expected_run
            or config["phase15"]["run_id"] != expected_run
            or binding["run"]["run_id"] != expected_run
            or binding["run"]["scenario_id"] != config["phase15"]["scenario_id"]
            or start["scenario_id"] != config["phase15"]["scenario_id"]
            or Path(binding["run"]["destination"]).resolve() != run_dir):
        raise ValueError("Run identity does not match its bound seed/scenario/destination")
    reason = end["termination_reason"]
    if (end["exit_code"] != 0 or not end["output_valid"] or end["violations"]
            or reason not in ("inactive", "timestep_limit")
            or end["status"] != ("complete" if reason == "inactive" else "censored")):
        raise ValueError("Only successful inactive or timestep-limit outcomes can be scored")
    model = end["observed_bindings"]["model"]
    expected_model = orchestrator.PHASE16_LR_MODEL_SHA256 if lr else orchestrator.PHASE15_CANDIDATE_MODEL_SHA256
    if (model["observed_sha256"] != expected_model or model["interface_validated"] is not True
            or binding["model"]["sha256"] != model["observed_sha256"]):
        raise ValueError("Frozen model observed identity mismatch")
    if lr:
        if "manifest" in model or "manifest" in binding["model"]:
            raise ValueError("Uncalibrated LR must not have a calibration manifest")
    elif model["manifest"]["observed_sha256"] != orchestrator.PHASE15_CANDIDATE_MANIFEST_SHA256:
        raise ValueError("Frozen RF manifest observed identity mismatch")
    for name, item in binding["inputs"].items():
        if end["observed_bindings"]["inputs"][name]["observed_sha256"] != item["sha256"]:
            raise ValueError("Environmental input identity mismatch")
    outputs = end["outputs"]
    expected_files = {item["filename"]: item for item in outputs}
    if len(outputs) != 2 or set(expected_files) != {"final_state.tif", "simulation_environment_valid.tif"}:
        raise ValueError("Unexpected or duplicate output inventory")
    arrays, identities = {}, {}
    specifications = {
        "prediction": (run_dir/"final_state.tif", False, expected_files["final_state.tif"]["sha256"]),
        "environment": (run_dir/"simulation_environment_valid.tif", False, expected_files["simulation_environment_valid.tif"]["sha256"]),
        "reference": (REFERENCE_PATH, True, REFERENCE_SHA256),
        "buildings": (BUILDINGS_PATH, True, BUILDINGS_SHA256),
    }
    for item in outputs:
        if item["verified"] is not True:
            raise ValueError("Output verification incomplete")
    for name, (path, parent, expected) in specifications.items():
        arrays[name], arrays[name+"_valid"], identities[name] = _read_raster(path, parent=parent, expected_sha256=expected)
    result = evaluate_airbase_arrays(arrays, termination_reason=reason)
    state = arrays["prediction"]
    actual_counts = {"non_burnable": int(np.count_nonzero(state == 1)), "not_yet_burning": int(np.count_nonzero(state == 2)),
        "ignited": int(np.count_nonzero(state == 3)), "blazing": int(np.count_nonzero(state == 4)), "extinguished": int(np.count_nonzero(state == 5))}
    if actual_counts != end["state_counts"] or actual_counts["ignited"] + actual_counts["blazing"] != end["active_cell_count"]:
        raise ValueError("Raster state counts disagree with termination record")
    return {"schema": schema, "run": {"run_id": expected_run,
        "scenario_id": start["scenario_id"], "seed": seed, "termination_reason": reason, "status": end["status"],
        "update_count": end["last_completed_timestep"], "elapsed_seconds": end["decision_elapsed_seconds"],
        "state_counts": actual_counts, "binding_sha256": calculated,
        "record_sha256": {"start.json": sha256(start_bytes).hexdigest(), "termination.json": sha256(end_bytes).hexdigest()}},
        "identities": identities, **result}
