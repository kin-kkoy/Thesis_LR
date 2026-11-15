"""Phase 17 matched DSWD CA-only runner and evidence readers.

Importing this module performs no input, reference, or output reads. Execution
is disabled by the versioned manifest and needs a fresh call-time authority.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

try:
    from . import orchestrator
    from .modules.automata_engine import FireAutomata
    from .modules.ca_state import STATE_CODES, validate_full_state_grid
    from .modules.transition_observer import canonical_metadata_hash, mask_sha256
    from .validation_engine import evaluate_final_footprint_arrays
except ImportError:
    import orchestrator
    from modules.automata_engine import FireAutomata
    from modules.ca_state import STATE_CODES, validate_full_state_grid
    from modules.transition_observer import canonical_metadata_hash, mask_sha256
    from validation_engine import evaluate_final_footprint_arrays


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = CODE_DIR / "config" / "phase17_dswd_ca_only_v1.yaml"
SCHEMA = "phase17_dswd_ca_only.v1"
SNAPSHOT_SCHEMA = "phase17_dswd_ca_only_snapshot_series.v1"
RUN_BINDING_SCHEMA = "phase17_dswd_ca_only_execution_binding.v1"
START_SCHEMA = "phase17_dswd_ca_only_run_start.v1"
TERMINATION_SCHEMA = "phase17_dswd_ca_only_run_termination.v1"
VALIDATION_SCHEMA = "phase17_dswd_ca_only_output_validation.v1"
SCORING_SCHEMA = "phase17_dswd_ca_only_scoring.v1"
METRICS = ("f1_score", "recall", "precision", "accuracy", "jaccard_index")
REQUIRED_SEEDS = (42, 43, 44, 45, 46)
INPUT_NAMES = ("slope", "proximity", "buildings", "materials")
EXPECTED_INPUT_SHA256 = {
    "slope": "dd61a0da016cd9c0d21d957ffedebd7ce871ef94f9084d5d4feb90e7e32b8482",
    "proximity": "1b12936cd8bc78d1e638fd31ddf1241bd753cb040cc9db556c623b8193de14b5",
    "buildings": "da53c3e63b661471b3366fc68249793c6a5a84e0d63bcc96492f60baaf4b83c7",
    "materials": "21f5c98c7936b4be6e368e7c8c0472a3c045b9c85b1abcb0a0e3fc8c1be0336a",
}
EXPECTED_INCIDENTS = {
    "airbase": {"incident_id": "airbase_sanggi_20250124", "ignition_local_row_col": [[512, 512]],
                "crop_origin": [820, 1989], "shape": [1024, 1024],
                "crop_transform": [3, 0, 604525.7784, 0, -3, 1141914.7699],
                "wind": {"speed_kmh": 10.6, "direction_deg": 60.826537}, "max_updates": 172,
                "horizons": [{"requested_update": 43, "assumed_minutes_per_update": 2,
                              "clock_role": "sensitivity"},
                             {"requested_update": 86, "assumed_minutes_per_update": 1,
                              "clock_role": "primary"},
                             {"requested_update": 172, "assumed_minutes_per_update": 0.5,
                              "clock_role": "sensitivity"}]},
    "santa_maria": {"incident_id": "santa_maria_pusok_20231212", "ignition_local_row_col": [[636, 554]],
                    "crop_origin": [256, 2176], "shape": [1024, 1024],
                    "crop_transform": [3, 0, 605086.7784, 0, -3, 1143606.7699],
                    "wind": {"speed_kmh": 17.7, "direction_deg": 48.825208}, "max_updates": 424,
                    "horizons": [{"requested_update": 106, "assumed_minutes_per_update": 2,
                                  "clock_role": "sensitivity"},
                                 {"requested_update": 212, "assumed_minutes_per_update": 1,
                                  "clock_role": "primary"},
                                 {"requested_update": 424, "assumed_minutes_per_update": 0.5,
                                  "clock_role": "sensitivity"}]},
}
EXPECTED_D397_ARTIFACT_SHA256 = "b610ac827e79ba3f1a2e0d6b6f63a6aed692e2011c5317c405990840f77b5073"
EXPECTED_BINDING_LEDGER_SHA256 = "aa97e663acdd96a6ee486bf1041f36e28404d11e7d90b0271b40628cd02f8310"
EXPECTED_ALIGNMENT = {
    "airbase": {"report_sha256": "1a6e804bec01a96a9cbc279f607bd3fdf65550abe220f6e479349f4b22e099ea",
                "domain_mask_sha256": "4a6dea5c7b60a2c3f952c59bed78060dad55f240f0e19f316a257bdf97285f7e",
                "included_positive_mask_sha256": "2d924ffbfb1302eb0cf309f0fb979a875ad7368866d59142675b5f5f9b80524c",
                "domain_cells": 215553, "reference_positive_included": 178},
    "santa_maria": {"report_sha256": "da40821ec30106407339611ab99ab9661cae5737413ea178c555f13936b65cce",
                    "domain_mask_sha256": "263385dcdc19d6602572e1eb19de282ec70bdde194756c53237dd24909379f69",
                    "included_positive_mask_sha256": "aecc271764878684ce9c85cc8aa1de6d0e8c9034b02a03e1e19627e7601dd2b8",
                    "domain_cells": 135992, "reference_positive_included": 1910},
}
_WIND_CONFIG = {
    "schema_version": "simulated_wind.v1",
    "direction_convention": "meteorological_from",
    "direction_units": "degrees_clockwise",
    "north_reference": "projected_grid_north",
    "calm_representation": "speed_zero_direction_zero",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, name: str = "digest") -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _check_binding_sha(actual: object, expected: object) -> None:
    actual = _digest(actual, "Observed execution binding")
    expected = _digest(expected, "Expected execution binding")
    if actual != expected:
        raise ValueError("Execution binding SHA-256 mismatch")


def _assert_exact_carry_forward(captured: np.ndarray, terminal: np.ndarray) -> None:
    if not np.array_equal(captured, terminal):
        raise ValueError("Inactive carry-forward state differs from terminal final_state")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        raise ValueError(f"{name} must be nonempty exact text")
    return value


def _same(actual: object, expected: object, message: str) -> None:
    if orchestrator._phase15_json_bytes(actual) != orchestrator._phase15_json_bytes(expected):
        raise ValueError(message)


def _code_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "~" in relative or "\x00" in relative:
        raise ValueError("Manifest paths must be explicit Code-relative paths")
    result = CODE_DIR / path
    # Output destinations are intentionally absent before a fresh run.
    current = result
    while not os.path.lexists(current):
        current = current.parent
    orchestrator._phase15_no_reparse(current)
    return result


def load_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict, bytes, str]:
    """Load the Phase 17 manifest through the repository's strict YAML parser."""
    path = Path(path).resolve()
    orchestrator._phase15_no_reparse(path)
    raw = path.read_bytes()
    config = orchestrator._phase15_yaml(raw)
    validate_manifest(config)
    return config, raw, _sha256_bytes(raw)


def validate_manifest(config: dict) -> None:
    if not isinstance(config, dict) or set(config) != {
        "phase17", "environment", "candidate_inputs", "d397_benchmark", "incidents", "scoring"
    }:
        raise ValueError("Phase 17 manifest sections differ")
    phase = config["phase17"]
    if phase.get("schema_version") != SCHEMA:
        raise ValueError("Phase 17 schema version differs")
    if type(phase.get("enabled")) is not bool or type(phase.get("execution_authorized")) is not bool:
        raise ValueError("Phase 17 activation flags must be boolean")
    if phase.get("trajectory_count") != 10 or phase.get("seeds") != list(REQUIRED_SEEDS):
        raise ValueError("Phase 17 requires ten trajectories over seeds 42–46")
    expected_trajectories = [{"incident": incident, "seed": seed,
                              "run_id": _trajectory_id(incident, seed)}
                             for incident in ("airbase", "santa_maria") for seed in REQUIRED_SEEDS]
    _same(phase.get("trajectories"), expected_trajectories,
          "Phase 17 manifest must enumerate the ten frozen incident/seed runs")
    if phase["snapshot_schema"] != "phase17_dswd_ca_only_snapshots.v1":
        raise ValueError("Phase 17 snapshot output version differs")
    if phase["output_root"] != "output/phase17_dswd_ca_only_v1":
        raise ValueError("Phase 17 output root differs")
    if phase["timers"] != {"T_3_to_4": 1, "T_4_to_5": 1}:
        raise ValueError("D-397 transition timers differ")
    monitor_policy = phase.get("resource_monitor")
    limit_names = {"max_wall_seconds", "max_process_rss_bytes", "min_system_available_bytes",
                   "min_free_output_bytes", "sample_interval_seconds"}
    if (not isinstance(monitor_policy, dict)
            or set(monitor_policy) != {"schema_version", "authorization_reference", "limits"}
            or monitor_policy["schema_version"] != "phase17_resource_monitor.v1"
            or not isinstance(monitor_policy["limits"], dict)
            or set(monitor_policy["limits"]) != limit_names
            or any(value is not None and (type(value) not in (int, float) or value <= 0)
                   for value in monitor_policy["limits"].values())):
        raise ValueError("Phase 17 resource monitor schema/placeholder gate differs")
    rule = phase["ignition_rule"]
    if rule != {
        "schema_version": "d014_model_free_multiplicative.v1",
        "base_ignition_prob": 0.12, "slope_weight": 0.2, "building_weight": 0.1,
        "proximity_weight": 0.2, "material_weight": 0.35, "wind_weight": 0.2,
        "law": "p_effective = clip(p_base * wind_weighted_score_t, 0, 1); seeded full-grid Bernoulli U < p_effective",
    }:
        raise ValueError("Approved D-014 multiplicative rule or coefficients differ")
    if phase["rng"] != {
        "generator": "numpy.random.default_rng", "bit_generator": "PCG64",
        "draw_policy": "existing FireAutomata full-grid draw once per transition",
    }:
        raise ValueError("Phase 17 RNG contract differs")
    if set(config["incidents"]) != set(EXPECTED_INCIDENTS):
        raise ValueError("Exactly the two D-397 incidents are required")
    for name, incident in config["incidents"].items():
        expected = EXPECTED_INCIDENTS[name]
        for key in ("incident_id", "ignition_local_row_col", "crop_origin", "shape",
                    "crop_transform", "wind", "max_updates"):
            _same(incident[key], expected[key], f"D-397 {name} {key} differs")
        _same(incident["horizons"], expected["horizons"],
              f"D-397 {name} horizon/clock mapping differs")
        _same(incident["route_a_alignment"], name, f"D-397 {name} alignment key differs")
        if (incident["max_updates"] != max(row["requested_update"] for row in incident["horizons"])
                or len(incident["horizons"]) != 3
                or len({row["requested_update"] for row in incident["horizons"]}) != 3
                or sum(row["clock_role"] == "primary" for row in incident["horizons"]) != 1):
            raise ValueError(f"Invalid Phase 17 horizon schedule for {name}")
    benchmark = config["d397_benchmark"]
    if (benchmark["decision"] != "D-397" or benchmark["design"] != "shared_rf_ca_v1"
            or benchmark["schema_version"] != "route_a_scoring_candidate.v1"
            or benchmark["artifact_path"] != "output/phase16_route_a_scoring_20261111_v1/scores.json"
            or benchmark["arms"] != ["rf", "lr"]
            or benchmark["primary_requested_update"] != {"airbase": 86, "santa_maria": 212}):
        raise ValueError("Frozen D-397 benchmark identity differs")
    _same(benchmark["artifact_sha256"], EXPECTED_D397_ARTIFACT_SHA256, "D-397 artifact pin differs")
    _same(benchmark["binding_ledger_sha256"], EXPECTED_BINDING_LEDGER_SHA256,
          "D-397 binding ledger pin differs")
    _same(benchmark["alignment"], EXPECTED_ALIGNMENT, "D-397 alignment identities/digests differ")
    if set(config["environment"]["input_sha256"]) != set(INPUT_NAMES):
        raise ValueError("Exactly four pinned environmental inputs are required")
    _same(config["environment"]["input_sha256"], EXPECTED_INPUT_SHA256,
          "D-397 environmental input identities differ")
    _same({key: config["environment"][key] for key in
           ("raster_dir", "slope_file", "proximity_file", "buildings_file", "materials_file", "nodata_value")},
          {"raster_dir": "processedData/raster/raster", "slope_file": "stack_slope_final.tif",
           "proximity_file": "stack_proximity.tif", "buildings_file": "stack_buildings.tif",
           "materials_file": "stack_materials.tif", "nodata_value": -9999},
          "D-397 environmental paths/nodata policy differ")
    _same(config["candidate_inputs"]["expected_grid"],
          {"shape": [5489, 6896], "crs": "EPSG:32651",
           "transform": [3, 0, 598558.7784, 0, -3, 1144374.7699, 0, 0, 1]},
          "D-397 parent raster grid differs")
    _same(config["candidate_inputs"]["reader_capability"], {
        "controls_verified": True,
        "environment_options": {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_PAM_ENABLED": "NO"},
        "open_options": {"GEOREF_SOURCES": "INTERNAL"}, "policy_id": "embedded_tiff_only.v1",
        "runtime": {"gdal": "3.12.1", "rasterio": "1.5.0"},
        "verification_reference": "Owner-adopted bounded mixed-dtype reader milestone; source tests/test_candidate_reader_capability.py SHA-256 14456daddc4453b34e58d79a1a6b5b491e2559526393510c5805ed16a050f597",
        "window_verification_reference": "Parent-reported native synthetic test test_installed_native_reader_window_matches_identified_synthetic_tiffs; source SHA-256 dde4cfb9fa8ba540c6e6b799eaef54edbdc72d06b135bb19e6cc4665caee6ffe; qualified validator report SHA-256 526fff81a669f48d12220304fde8923a61edf8ceb542f87ce14df7c94ba29938",
        "windowed_reads_verified": True,
    }, "Route A installed reader capability differs")
    scoring = config["scoring"]
    if (scoring["references_are_outside_simulation_path"] is not True
            or scoring["join_keys"] != ["incident", "seed", "requested_update"]
            or scoring["contrasts"] != ["rf_minus_ca", "lr_minus_ca"]
            or scoring["group_keys"] != ["incident", "arm", "requested_update"]
            or scoring["required_seeds"] != list(REQUIRED_SEEDS)
            or scoring["minimum_valid_seeds"] != 5
            or scoring["dispersion"] != "sample SD, n-1"):
        raise ValueError("Phase 17 scoring contract differs")
    _same(scoring["reference_paths"],
          {"airbase": "dataFiles/stack_ground_truth_airbase.tif",
           "santa_maria": "processedData/raster/raster/stack_ground_truth.tif"},
          "D-397 reference paths differ")


def _trajectory_id(incident: str, seed: int) -> str:
    if incident not in ("airbase", "santa_maria") or type(seed) is not int or seed not in REQUIRED_SEEDS:
        raise ValueError("Unknown Phase 17 trajectory")
    return f"p17-dswd-ca-only-v1-{incident.replace('_', '-')}-s{seed}-r1"


def _benchmark_identity(config: dict) -> tuple[dict, str]:
    benchmark = config["d397_benchmark"]
    path = _code_path(benchmark["artifact_path"])
    digest = _sha256_file(path)
    if digest != benchmark["artifact_sha256"]:
        raise ValueError("Frozen D-397 score artifact SHA-256 mismatch")
    record = json.loads(path.read_bytes())
    if (record.get("schema") != benchmark["schema_version"]
            or record.get("authorization") != "D-394"
            or record.get("design") != benchmark["design"]
            or record.get("binding_ledger_sha256") != benchmark["binding_ledger_sha256"]
            or len(record.get("scores", ())) != 60
            or len(record.get("groups", ())) != 12
            or len(record.get("seed_deltas", ())) != 30):
        raise ValueError("Frozen D-397 benchmark ledger/schema differs")
    for incident in ("airbase", "santa_maria"):
        expected = benchmark["alignment"][incident]
        observed = record["alignment"][incident]
        report = observed["report"]
        _same(observed["report_sha256"], expected["report_sha256"], "D-397 alignment report pin differs")
        _same(observed["included_positive_mask_sha256"], expected["included_positive_mask_sha256"],
              "D-397 included-positive digest differs")
        _same(report["masks"]["evaluation_domain"]["sha256"], expected["domain_mask_sha256"],
              "D-397 domain digest differs")
        _same(report["masks"]["evaluation_domain"]["count"], expected["domain_cells"],
              "D-397 domain count differs")
        _same(report["reference_positive"]["included"], expected["reference_positive_included"],
              "D-397 included-positive count differs")
    if not isinstance(record.get("source_pins"), dict) or not isinstance(record.get("runtime"), dict):
        raise ValueError("D-397 common source/runtime identities are missing")
    return record, digest


def _route_a_binding_pairs(config: dict, benchmark: dict) -> dict[tuple[str, int], dict]:
    """Select and cross-check RF/LR run bindings from the frozen D-397 ledger."""
    rows = benchmark["scores"]
    pairs = {(item["incident"], item["seed"]): item["reader_result"]
             for item in benchmark["pair_readers"]}
    expected_keys = {(incident, seed) for incident in config["incidents"] for seed in REQUIRED_SEEDS}
    if set(pairs) != expected_keys:
        raise ValueError("D-397 pairing ledger does not contain exactly ten incident/seed pairs")
    selected = {}
    for incident, seed in sorted(expected_keys):
        pair = pairs[(incident, seed)]
        if (pair.get("status") != "MATCH" or pair.get("accepted_for_shared_ca_comparison") is not True
                or pair.get("accepted_for_cross_simulator_comparison") is not False
                or len(pair.get("run_ids", ())) != 2 or len(pair.get("binding_sha256", ())) != 2):
            raise ValueError("D-397 paired run binding is not accepted for shared-CA comparison")
        identities = {}
        for arm, run_id, binding_sha in zip(("rf", "lr"), pair["run_ids"], pair["binding_sha256"]):
            _digest(binding_sha, "Route A execution binding")
            arm_rows = [row for row in rows if row["incident"] == incident and row["seed"] == seed
                        and row["arm"] == arm]
            expected_horizons = {h["requested_update"] for h in config["incidents"][incident]["horizons"]}
            if ({row["outcome"]["requested_update"] for row in arm_rows} != expected_horizons
                    or len(arm_rows) != 3
                    or any(row["outcome"].get("run_id") != run_id
                           or row["outcome"].get("execution_binding_sha256") != binding_sha
                           or row["outcome"].get("output_valid") is not True
                           or row["outcome"].get("seed") != seed for row in arm_rows)):
                raise ValueError("D-397 score rows do not bind to the selected Route A run")
            identities[arm] = {"run_id": run_id, "binding_sha256": binding_sha}
        selected[(incident, seed)] = identities
    return selected


def _input_identities(config: dict) -> dict:
    env = config["environment"]
    return {name: {"path": str((CODE_DIR / env["raster_dir"] / env[name + "_file"]).resolve()),
                   "sha256": env["input_sha256"][name]} for name in INPUT_NAMES}


def _trajectory_contract(config: dict, incident: str, seed: int) -> dict:
    incident_cfg = config["incidents"][incident]
    alignment = config["d397_benchmark"]["alignment"][incident_cfg["route_a_alignment"]]
    return {"run_id": _trajectory_id(incident, seed), "incident": incident,
            "incident_id": incident_cfg["incident_id"], "seed": seed,
            "ignition_local_row_col": deepcopy(incident_cfg["ignition_local_row_col"]),
            "ignition_global_row_col": [[row + incident_cfg["crop_origin"][0],
                                          col + incident_cfg["crop_origin"][1]]
                                         for row, col in incident_cfg["ignition_local_row_col"]],
            "crop_origin": deepcopy(incident_cfg["crop_origin"]),
            "shape": deepcopy(incident_cfg["shape"]), "transform": deepcopy(incident_cfg["crop_transform"]),
            "wind": {**_WIND_CONFIG, **incident_cfg["wind"]},
            "timers": deepcopy(config["phase17"]["timers"]),
            "ignition_rule": deepcopy(config["phase17"]["ignition_rule"]),
            "rng": {**deepcopy(config["phase17"]["rng"]), "seed": seed},
            "max_updates": incident_cfg["max_updates"], "horizons": deepcopy(incident_cfg["horizons"]),
            "domain_mask_sha256": alignment["domain_mask_sha256"],
            "included_positive_mask_sha256": alignment["included_positive_mask_sha256"]}


def _phase17_source_identity(manifest_path: Path, manifest_sha256: str) -> dict:
    source = orchestrator._phase15_source()
    repo = Path(source["repository_root"])
    source["files"][Path(__file__).resolve().relative_to(repo).as_posix()] = _sha256_file(Path(__file__))
    source["files"][manifest_path.resolve().relative_to(repo).as_posix()] = manifest_sha256
    source["phase17_schema"] = SCHEMA
    return source


def _check_route_a_common_identity(benchmark: dict, source: dict | None = None,
                                   runtime: dict | None = None,
                                   expected_gdal: str | None = None) -> dict:
    """Fail closed if the current shared source/runtime differs from accepted D-397 pins."""
    source = source or orchestrator._phase15_source()
    runtime = runtime or orchestrator._phase15_runtime()
    pins = benchmark.get("source_pins")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("D-397 source_pins are absent")
    files = source.get("files", {})
    repository = Path(source["repository_root"]).resolve()
    for path, expected in pins.items():
        _digest(expected, f"D-397 source pin {path}")
        actual = files.get(path)
        if actual is None:
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"D-397 source pin path is unsafe: {path!r}")
            pinned_path = repository / relative
            orchestrator._phase15_no_reparse(pinned_path)
            if not pinned_path.is_file():
                raise ValueError(f"D-397 source pin file is missing: {path}")
            actual = _sha256_file(pinned_path)
        if actual != expected:
            raise ValueError(f"D-397 source pin mismatch for {path}: observed={actual!r}, expected={expected!r}")
    expected_runtime = benchmark.get("runtime")
    if not isinstance(expected_runtime, dict) or not expected_runtime:
        raise ValueError("D-397 common runtime identity is absent")
    if runtime != expected_runtime:
        keys = sorted(set(runtime) | set(expected_runtime))
        field = next((key for key in keys if runtime.get(key) != expected_runtime.get(key)), "runtime")
        raise ValueError(f"D-397 runtime mismatch for {field}: observed={runtime.get(field)!r}, expected={expected_runtime.get(field)!r}")
    if expected_gdal is not None and rasterio.__gdal_version__ != expected_gdal:
        raise ValueError(f"D-397 runtime mismatch for GDAL: observed={rasterio.__gdal_version__!r}, expected={expected_gdal!r}")
    return {"source_pins": deepcopy(pins), "runtime": deepcopy(expected_runtime)}


def build_execution_binding(config: dict, manifest_path: Path, manifest_sha256: str,
                            incident: str, seed: int, authority_reference: str,
                            resource_monitor_config: dict) -> dict:
    validate_manifest(config)
    _text(authority_reference, "Fresh run authority reference")
    if config["phase17"]["enabled"] is not True or config["phase17"]["execution_authorized"] is not True:
        raise PermissionError("Phase 17 is default-disabled; explicit manifest activation is required")
    if incident not in config["incidents"] or type(seed) is not int or seed not in REQUIRED_SEEDS:
        raise ValueError("Unknown Phase 17 incident/seed")
    benchmark, benchmark_sha = _benchmark_identity(config)
    _check_route_a_common_identity(benchmark, expected_gdal=config["candidate_inputs"]["reader_capability"]["runtime"]["gdal"])
    _validate_resource_monitor_config(config, resource_monitor_config)
    pairs = _route_a_binding_pairs(config, benchmark)
    run_id = _trajectory_id(incident, seed)
    output_root = _code_path(config["phase17"]["output_root"])
    return {
        "schema_version": RUN_BINDING_SCHEMA,
        "authority_reference": authority_reference,
        "configuration": {"path": manifest_path.resolve().as_posix(), "sha256": manifest_sha256,
                          "activated_effective_sha256": canonical_metadata_hash(config)},
        "source": _phase17_source_identity(manifest_path, manifest_sha256),
        "runtime": orchestrator._phase15_runtime(),
        "phase17_adapter": {"module": Path(__file__).resolve().relative_to(CODE_DIR.parent).as_posix(),
                            "configuration": manifest_path.resolve().relative_to(CODE_DIR).as_posix(),
                            "source_sha256": _sha256_file(Path(__file__)), "manifest_sha256": manifest_sha256},
        "resource_monitor": deepcopy(resource_monitor_config),
        "route_a_common_identity": {"artifact_sha256": benchmark_sha,
                                    "binding_ledger_sha256": config["d397_benchmark"]["binding_ledger_sha256"],
                                    "source_pins": deepcopy(benchmark["source_pins"]),
                                    "runtime": deepcopy(benchmark["runtime"])},
        "route_a_bindings": deepcopy(pairs[(incident, seed)]),
        "trajectory": _trajectory_contract(config, incident, seed),
        "inputs": _input_identities(config),
        "destination": (output_root / run_id).as_posix(),
    }


def _fire_config(config: dict, incident: str, seed: int) -> dict:
    phase = config["phase17"]
    incident_config = config["incidents"][incident]
    rule = phase["ignition_rule"]
    return {
        "simulation": {"seed": seed, "max_timesteps": incident_config["max_updates"],
                       "emergency_checkpoint_enabled": False, "checkpoint_interval": 0,
                       "resume_checkpoint": None},
        "wind": {**_WIND_CONFIG, **incident_config["wind"]},
        "placeholder_transition": {"T_3_to_4": phase["timers"]["T_3_to_4"],
                                   "T_4_to_5": phase["timers"]["T_4_to_5"],
                                   "base_ignition_prob": rule["base_ignition_prob"],
                                   "slope_weight": rule["slope_weight"],
                                   "building_weight": rule["building_weight"],
                                   "proximity_weight": rule["proximity_weight"],
                                   "material_weight": rule["material_weight"],
                                   "wind_weight": rule["wind_weight"]},
        "ml_model": {"inference_mode": "stochastic_probability", "positive_label": 1,
                     "threshold": None, "inference_threshold": None,
                     "classification_threshold": None},
        "flammability_weights": {"base_weight": 0.5, "slope_weight": 0.3, "proximity_weight": 0.2},
        "output": {"output_dir": str(_code_path(phase["output_root"]))},
    }


def _validate_resource_monitor_config(config: dict, monitor_config: dict) -> None:
    required = {"schema_version", "authorization_reference", "limits"}
    if not isinstance(monitor_config, dict) or set(monitor_config) != required:
        raise ValueError("Configured resource monitor is required")
    if monitor_config["schema_version"] != "phase17_resource_monitor.v1":
        raise ValueError("Resource monitor schema differs")
    _text(monitor_config["authorization_reference"], "Resource monitor approval reference")
    expected = config["phase17"].get("resource_monitor")
    if not isinstance(expected, dict) or monitor_config != expected:
        raise ValueError("Resource monitor configuration does not match the activated manifest")
    limits = monitor_config["limits"]
    if (not isinstance(limits, dict) or set(limits) != {
            "max_wall_seconds", "max_process_rss_bytes", "min_system_available_bytes",
            "min_free_output_bytes", "sample_interval_seconds"}
            or any(type(value) not in (int, float) or value <= 0 for value in limits.values())):
        raise ValueError("Resource limits are an unconfigured execution-card gate")


def _output_allowlist(horizons: list[dict]) -> frozenset[str]:
    names = {"start.json", "termination.json", "termination.json.partial",
             "final_state.tif", "final_state.tif.partial",
             "simulation_environment_valid.tif", "simulation_environment_valid.tif.partial"}
    for row in horizons:
        for kind in ("state", "reached"):
            name = f"snapshot_h{row['requested_update']:04d}_{kind}.tif"
            names.update((name, name + ".partial"))
    return frozenset(names)


def _exclusive_destination(destination: Path, *, allowed_root: Path = CODE_DIR) -> None:
    destination = Path(destination)
    allowed_root = Path(allowed_root).resolve()
    if not destination.is_absolute():
        raise ValueError("Exclusive destination must be absolute")
    parent = destination.parent
    try:
        relative = parent.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError("Exclusive destination is outside its allowlisted root") from exc
    orchestrator._phase15_no_reparse(allowed_root)
    current = allowed_root
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current):
            orchestrator._phase15_no_reparse(current)
            if not current.is_dir():
                raise NotADirectoryError(current)
        else:
            current.mkdir()
    destination.mkdir(exist_ok=False)


def run_trajectory(manifest_path: Path = DEFAULT_MANIFEST, *, incident: str, seed: int,
                   explicitly_authorized: bool = False, authority_reference: str | None = None,
                   expected_manifest_sha256: str | None = None,
                   expected_binding_sha256: str | None = None,
                   resource_monitor=None, resource_monitor_config: dict | None = None) -> dict:
    """Execution-card entry point for a later owner-approved Phase 17 run.

    The card must pass a monitor whose configuration matches the bound
    ``resource_monitor_config``; the default manifest intentionally leaves
    resource limits unconfigured and execution disabled.
    """
    config, _, manifest_sha = load_manifest(manifest_path)
    if explicitly_authorized is not True:
        raise PermissionError("A fresh explicit Phase 17 run authority is required")
    authority_reference = _text(authority_reference, "Fresh run authority reference")
    if expected_manifest_sha256 != manifest_sha:
        raise PermissionError("Independent Phase 17 manifest digest mismatch")
    expected_binding_sha256 = _digest(expected_binding_sha256, "Expected Phase 17 execution binding")
    if resource_monitor is None or not callable(getattr(resource_monitor, "check", None)):
        raise ValueError("A configured resource monitor is required")
    monitor_identity = getattr(resource_monitor, "configuration", getattr(resource_monitor, "policy", None))
    if monitor_identity is None or resource_monitor_config is None or monitor_identity != resource_monitor_config:
        raise ValueError("Resource monitor object is unconfigured or differs from its bound configuration")
    binding = build_execution_binding(config, Path(manifest_path), manifest_sha,
                                      incident, seed, authority_reference, resource_monitor_config)
    binding_sha = canonical_metadata_hash(binding)
    if binding_sha != expected_binding_sha256:
        raise PermissionError("Independent Phase 17 execution binding mismatch")
    incident_config = config["incidents"][incident]
    destination = Path(binding["destination"])
    allowed = _output_allowlist(incident_config["horizons"])
    _exclusive_destination(destination, allowed_root=CODE_DIR)

    start_hash = None
    snapshots = None
    outputs: list[dict] = []
    captured_states: dict[int, np.ndarray] = {}
    observed: dict[str, Any] = {}
    reason = "preflight_failure"
    error: BaseException | None = None
    completed = 0
    counts = None
    started = datetime.now(timezone.utc).isoformat()
    start = {"schema_version": START_SCHEMA, "run_id": _trajectory_id(incident, seed),
             "started_at_utc": started, "binding": {"payload": binding,
             "expected_sha256": binding_sha, "calculated_sha256": canonical_metadata_hash(binding)},
             "configuration": {"sha256": manifest_sha, "effective": config},
             "authorization_reference": authority_reference,
             "source": binding["source"], "runtime": binding["runtime"],
             "route_a_common_identity": binding["route_a_common_identity"],
             "route_a_bindings": binding["route_a_bindings"],
             "output_policy": {"allowlist": sorted(allowed), "resume": False,
                               "exclusive_destination": True}}
    try:
        start_hash = orchestrator._phase15_write_json(destination / "start.json", start)
        environment_config = deepcopy(config["environment"])
        candidate_config = deepcopy(config["candidate_inputs"])
        candidate_config["window"] = {"incident_id": incident_config["incident_id"],
                                       "global_origin": deepcopy(incident_config["crop_origin"]),
                                       "shape": deepcopy(incident_config["shape"])}
        active = {"environment": environment_config, "candidate_inputs": candidate_config,
                  "simulation": {"ignition_points": deepcopy(incident_config["ignition_local_row_col"]),
                                 "seed": seed}}
        with rasterio.Env(**orchestrator.candidate_inputs.ENV_OPTIONS):
            environment, observed = orchestrator._phase15_candidate_environment(
                active, binding["inputs"], resource_monitor)
        expected_domain = binding["trajectory"]["domain_mask_sha256"]
        if observed["mask_identities"]["receiving_sha256"] != expected_domain:
            raise ValueError("Observed Route A receiving/domain digest differs from D-397")
        automata = FireAutomata(environment, _fire_config(config, incident, seed), model_free=True)
        if automata.model is not None or not automata.model_free:
            raise RuntimeError("Phase 17 requires the model-free FireAutomata arm")
        points = [tuple(point) for point in observed["converted_ignitions"]]
        if (len(points) != len(incident_config["ignition_local_row_col"])
                or any(not automata.burnable_mask[row, col] for row, col in points)):
            raise ValueError("Prescribed D-397 ignition is not in the observed receiving population")
        automata.set_ignition(points)
        observed.update(model_policy={"arm": "ca_only", "model_loaded": False,
                                     "estimator_inference": False},
                        rng_bit_generator=type(automata.rng.bit_generator).__name__,
                        timers={"T_3_to_4": automata.t_3_to_4, "T_4_to_5": automata.t_4_to_5},
                        prescribed_ignition_local_row_col=[list(point) for point in points],
                        domain_mask_sha256=observed["mask_identities"]["receiving_sha256"],
                        included_positive_mask_sha256=binding["trajectory"]["included_positive_mask_sha256"])
        basis = {"incident": incident, "seed": seed,
                 "duration_basis": None,
                 "horizons": deepcopy(incident_config["horizons"])}
        snapshots = orchestrator._Phase16Snapshots(basis)
        for row in snapshots.rows:
            row.update(schema_version=SNAPSHOT_SCHEMA, design="matched_ca_only_v1", arm="ca_only",
                       incident=incident, seed=seed, run_id=binding["trajectory"]["run_id"],
                       execution_binding_sha256=binding_sha, start_record_sha256=start_hash,
                       configuration_sha256=canonical_metadata_hash(config),
                       route_a_bindings=deepcopy(binding["route_a_bindings"]))

        def capture(row: dict, state: np.ndarray, reached: np.ndarray) -> list[dict]:
            captured_states[row["requested_update"]] = np.array(state, copy=True)
            captures = []
            for kind, array in (("state", state), ("reached", reached)):
                name = f"snapshot_h{row['requested_update']:04d}_{kind}.tif"
                record = orchestrator._phase15_raster(destination, name, array, automata,
                                                      resource_monitor, allowed_outputs=allowed)
                outputs.append(record)
                captures.append(record)
            return captures

        reason, completed, counts = snapshots.run(automata, resource_monitor.check, capture)
        terminal_state = automata.get_grid()
        if not captured_states or not np.array_equal(terminal_state, captured_states[max(captured_states)]):
            raise ValueError("Snapshot terminal state differs from final simulation state")
        for row in snapshots.rows:
            if row["absorbing_carry_forward"]:
                _assert_exact_carry_forward(captured_states[row["requested_update"]], terminal_state)
        outputs.append(orchestrator._phase15_raster(destination, "final_state.tif", terminal_state,
                                                    automata, resource_monitor, allowed_outputs=allowed))
        receiving = np.asarray(automata.burnable_mask, dtype=bool)
        environmental_valid = np.asarray(automata.simulation_valid_mask, dtype=bool)
        if (receiving.shape != terminal_state.shape or environmental_valid.shape != terminal_state.shape
                or not receiving.any() or np.any(receiving & ~environmental_valid)
                or not np.array_equal(receiving, terminal_state != STATE_CODES["non_burnable"])
                or mask_sha256(receiving) != binding["trajectory"]["domain_mask_sha256"]):
            raise ValueError("Receiving mask coverage or digest differs from the D-397 domain")
        outputs.append(orchestrator._phase15_raster(destination, "simulation_environment_valid.tif",
                                                    environmental_valid.astype(np.uint8), automata,
                                                    resource_monitor, allowed_outputs=allowed))
    except BaseException as exc:
        error = exc
        if snapshots is not None:
            snapshots.fail()
            completed, counts = snapshots.completed, snapshots.counts

    series = None
    if snapshots is not None:
        series = snapshots.record()
        series.update(schema_version=SNAPSHOT_SCHEMA, design="matched_ca_only_v1", arm="ca_only",
                      incident=incident, seed=seed, run_id=binding["trajectory"]["run_id"],
                      execution_binding_sha256=binding_sha,
                      domain_mask_sha256=binding["trajectory"]["domain_mask_sha256"],
                      included_positive_mask_sha256=binding["trajectory"]["included_positive_mask_sha256"])
    success = error is None and reason in ("inactive", "timestep_limit")
    end = {"schema_version": TERMINATION_SCHEMA, "run_id": binding["trajectory"]["run_id"],
           "start_record_sha256": start_hash, "binding_sha256": binding_sha,
           "ended_at_utc": datetime.now(timezone.utc).isoformat(),
           "status": ("complete" if reason == "inactive" else "censored") if success else "failed",
           "termination_reason": reason if success else "failure",
           "exit_code": 0 if success else 1, "last_completed_update": completed,
           "state_counts": counts, "active_cell_count": None if counts is None else counts["ignited"] + counts["blazing"],
           "output_valid": success, "observed": observed, "outputs": outputs,
           "snapshot_series": series,
           "error": None if error is None else {"type": type(error).__name__, "message": str(error)}}
    end_hash = orchestrator._phase15_write_json(destination / "termination.json.partial", end)
    orchestrator._phase15_finalize(destination / "termination.json.partial", destination / "termination.json")
    if error is not None:
        raise error
    return {**end, "termination_record_sha256": end_hash}


def _record(path: Path, max_bytes: int = 16 * 1024 * 1024) -> tuple[dict, bytes]:
    path = Path(path)
    orchestrator._phase15_no_reparse(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("Expected a bounded ordinary JSON record")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON evidence record must be an object")
    return value, raw


def _read_snapshot(path: Path, *, dtype: str, shape: tuple[int, int], crs: str,
                   transform: list[float], digest: str) -> np.ndarray:
    orchestrator._phase15_no_reparse(path)
    if _sha256_file(path) != digest:
        raise ValueError("Snapshot raster digest differs from termination evidence")
    with rasterio.open(path, mode="r", driver="GTiff", sharing=False,
                       GEOREF_SOURCES="INTERNAL") as source:
        if (source.driver != "GTiff" or source.mode != "r" or source.count != 1
                or source.subdatasets or len(source.files) != 1
                or source.shape != shape or source.dtypes != (dtype,)
                or source.crs != CRS.from_user_input(crs)
                or source.transform != Affine(*transform[:6]) or source.nodata is not None):
            raise ValueError("Snapshot TIFF metadata differs from its trajectory contract")
        array = source.read(1)
    return array


def validate_trajectory_output(run_dir: Path, *, manifest_path: Path = DEFAULT_MANIFEST,
                               explicitly_authorized: bool = False,
                               review_reference: str | None = None,
                               expected_binding_sha256: str | None = None,
                               expected_start_sha256: str | None = None,
                               expected_termination_sha256: str | None = None) -> dict:
    """Read and validate Phase 17 outputs only; it never opens reference rasters."""
    if explicitly_authorized is not True:
        raise PermissionError("Separate Phase 17 output-read authority is required")
    review_reference = _text(review_reference, "Output validation review reference")
    config, _, manifest_sha = load_manifest(manifest_path)
    run_dir = Path(run_dir)
    orchestrator._phase15_no_reparse(run_dir)
    run_dir = run_dir.resolve()
    expected_root = _code_path(config["phase17"]["output_root"]).resolve()
    if run_dir.parent != expected_root or run_dir.name not in {
        _trajectory_id(incident, seed) for incident in config["incidents"] for seed in REQUIRED_SEEDS
    }:
        raise ValueError("Output directory is outside the exact Phase 17 allowlist")
    orchestrator._phase15_no_reparse(run_dir)
    start, start_bytes = _record(run_dir / "start.json")
    end, end_bytes = _record(run_dir / "termination.json")
    partial = run_dir / "termination.json.partial"
    orchestrator._phase15_no_reparse(partial)
    if not os.path.samefile(run_dir / "termination.json", partial):
        raise ValueError("Termination is not the retained hard-link publication")
    binding = start["binding"]["payload"]
    binding_sha = canonical_metadata_hash(binding)
    _check_binding_sha(binding_sha, expected_binding_sha256)
    if (_sha256_bytes(start_bytes) != _digest(expected_start_sha256, "Expected start record")
            or _sha256_bytes(end_bytes) != _digest(expected_termination_sha256, "Expected termination record")):
        raise ValueError("Independent Phase 17 start/termination record identity differs")
    if (start.get("schema_version") != START_SCHEMA or end.get("schema_version") != TERMINATION_SCHEMA
            or end["start_record_sha256"] != _sha256_bytes(start_bytes)
            or end["output_valid"] is not True or end["status"] not in ("complete", "censored")):
        raise ValueError("Phase 17 run records are incomplete or binding-mismatched")
    if (end.get("termination_reason") not in ("inactive", "timestep_limit")
            or (end["termination_reason"] == "inactive") != (end["status"] == "complete")):
        raise ValueError("Phase 17 trajectory termination reason/status differs")
    _check_binding_sha(start["binding"]["expected_sha256"], binding_sha)
    _check_binding_sha(start["binding"]["calculated_sha256"], binding_sha)
    _check_binding_sha(end["binding_sha256"], binding_sha)
    trajectory = binding["trajectory"]
    incident, seed = trajectory["incident"], trajectory["seed"]
    if (run_dir.name != trajectory["run_id"] or trajectory["run_id"] != _trajectory_id(incident, seed)
            or end["run_id"] != run_dir.name):
        raise ValueError("Phase 17 run directory attribution differs")
    if (start["configuration"].get("sha256") != manifest_sha
            or canonical_metadata_hash(start["configuration"].get("effective")) != canonical_metadata_hash(config)
            or binding["configuration"].get("sha256") != manifest_sha
            or binding["configuration"].get("activated_effective_sha256") != canonical_metadata_hash(config)
            or binding["configuration"].get("path") != Path(manifest_path).resolve().as_posix()
            or binding.get("authority_reference") != start.get("authorization_reference")):
        raise ValueError("Phase 17 effective configuration or fresh authority link differs")
    benchmark, benchmark_sha = _benchmark_identity(config)
    selected_pairs = _route_a_binding_pairs(config, benchmark)
    expected_common = {"artifact_sha256": benchmark_sha,
                       "binding_ledger_sha256": config["d397_benchmark"]["binding_ledger_sha256"],
                       "source_pins": benchmark["source_pins"], "runtime": benchmark["runtime"]}
    _same(binding["trajectory"], _trajectory_contract(config, incident, seed),
          "D-397 trajectory controls in execution binding differ")
    _same(binding["inputs"], _input_identities(config), "D-397 environmental input binding differs")
    _same(binding["route_a_bindings"], selected_pairs[(incident, seed)],
          "Selected Route A RF/LR binding identities differ")
    _same(binding["route_a_common_identity"], expected_common,
          "Common Route A source/runtime identity differs")
    _same(binding.get("resource_monitor"), config["phase17"]["resource_monitor"],
          "Bound resource monitor configuration differs")
    observed = end.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("Observed Phase 17 execution controls are missing")
    trajectory = binding["trajectory"]
    _same(observed.get("model_policy"), {"arm": "ca_only", "model_loaded": False,
                                          "estimator_inference": False},
          "Phase 17 estimator-free execution policy differs")
    _same(observed.get("rng_bit_generator"), trajectory["rng"]["bit_generator"],
          "Phase 17 RNG identity differs")
    _same(observed.get("timers"), trajectory["timers"], "Phase 17 transition timers differ")
    _same(observed.get("prescribed_ignition_local_row_col"), trajectory["ignition_local_row_col"],
          "Phase 17 prescribed ignition differs")
    _same(observed.get("domain_mask_sha256"), trajectory["domain_mask_sha256"],
          "Observed Phase 17 domain digest differs")
    _same(observed.get("included_positive_mask_sha256"), trajectory["included_positive_mask_sha256"],
          "Observed Phase 17 included-positive digest differs")
    source_files = binding["source"].get("files", {})
    repo = Path(binding["source"]["repository_root"])
    _same(source_files.get(Path(__file__).resolve().relative_to(repo).as_posix()), _sha256_file(Path(__file__)),
          "Phase 17 source identity differs from the current entry point")
    _same(source_files.get(Path(manifest_path).resolve().relative_to(repo).as_posix()), manifest_sha,
          "Phase 17 source identity differs from the current manifest")
    expected_horizons = config["incidents"][incident]["horizons"]
    series = end["snapshot_series"]
    if (series.get("schema_version") != SNAPSHOT_SCHEMA or series.get("incident") != incident
            or series.get("seed") != seed or series.get("execution_binding_sha256") != binding_sha
            or len(series.get("outcomes", ())) != 3):
        raise ValueError("Phase 17 snapshot series is incomplete")
    completed_updates = series.get("completed_updates")
    if (type(completed_updates) is not int or completed_updates < 0
            or end.get("last_completed_update") != completed_updates):
        raise ValueError("Phase 17 completed-update identities differ")
    arrays_by_horizon = {}
    for outcome, expected in zip(sorted(series["outcomes"], key=lambda row: row["requested_update"]),
                                 sorted(expected_horizons, key=lambda row: row["requested_update"])):
        _same({key: outcome[key] for key in ("requested_update", "assumed_minutes_per_update", "clock_role")},
              expected, "Phase 17 horizon attribution differs")
        requested = outcome["requested_update"]
        termination_reason = outcome.get("termination_reason")
        actual_capture = outcome.get("actual_capture_update")
        if (outcome.get("output_valid") is not True or outcome.get("status") not in ("complete", "censored")
                or termination_reason not in ("inactive", "timestep_limit")
                or type(actual_capture) is not int or actual_capture < 0 or actual_capture > completed_updates
                or actual_capture > requested or outcome.get("horizon_outcome_update") != actual_capture
                or outcome.get("trajectory_stop_update") != completed_updates
                or outcome.get("actual_stop_update") != completed_updates
                or outcome.get("censoring") is not (outcome.get("status") == "censored")
                or len(outcome.get("outputs", ())) != 2):
            raise ValueError("A failed trajectory invalidates all snapshots")
        native_record = next((item for item in outcome["outputs"] if item["filename"].endswith("_state.tif")), None)
        reached_record = next((item for item in outcome["outputs"] if item["filename"].endswith("_reached.tif")), None)
        if native_record is None or reached_record is None:
            raise ValueError("State and reached snapshot pair is required")
        if ({native_record["filename"], reached_record["filename"]}
                != {f"snapshot_h{requested:04d}_state.tif", f"snapshot_h{requested:04d}_reached.tif"}
                or not native_record.get("verified") or not reached_record.get("verified")):
            raise ValueError("Snapshot filenames or publication verification differ")
        shape = tuple(trajectory["shape"])
        transform = trajectory["transform"]
        state = _read_snapshot(run_dir / native_record["filename"], dtype="int8", shape=shape,
                               crs="EPSG:32651", transform=transform, digest=native_record["sha256"])
        reached = _read_snapshot(run_dir / reached_record["filename"], dtype="uint8", shape=shape,
                                 crs="EPSG:32651", transform=transform, digest=reached_record["sha256"])
        state = validate_full_state_grid(state)
        expected_reached = np.isin(state, (3, 4, 5)).astype(np.uint8)
        if not np.array_equal(reached, expected_reached):
            raise ValueError("Reached export differs from the full-state raster")
        counts = {name: int(np.count_nonzero(state == code)) for code, name in enumerate(
            ("non_burnable", "not_yet_burning", "ignited", "blazing", "extinguished"), 1)}
        active = counts["ignited"] + counts["blazing"]
        reached_count = int(np.count_nonzero(expected_reached))
        if (counts != outcome["state_counts"] or active != outcome["active_cell_count"]
                or reached_count != outcome["cumulative_reached_count"]
                or (termination_reason == "inactive") != (outcome["status"] == "complete")
                or (outcome["status"] == "complete" and (active != 0 or outcome["censoring"] is not False))
                or (outcome["status"] == "censored" and (active == 0 or outcome["censoring"] is not True))
                or outcome["absorbing_carry_forward"] != (actual_capture < requested)
                or (actual_capture < requested and (termination_reason != "inactive" or active != 0))):
            raise ValueError("Snapshot counts or inactive/censoring semantics differ")
        arrays_by_horizon[outcome["requested_update"]] = {"state": state, "reached": reached,
                                                           "outcome": deepcopy(outcome)}
    output_by_name = {item.get("filename"): item for item in end.get("outputs", ())}
    if not {"final_state.tif", "simulation_environment_valid.tif"} <= set(output_by_name):
        raise ValueError("Final-state and receiving-mask publications are required")
    if any(output_by_name[name].get("verified") is not True for name in
           ("final_state.tif", "simulation_environment_valid.tif")):
        raise ValueError("Final-state and receiving-mask publications are not verified")
    final_state = _read_snapshot(run_dir / "final_state.tif", dtype="int8", shape=tuple(trajectory["shape"]),
                                 crs="EPSG:32651", transform=trajectory["transform"],
                                 digest=output_by_name["final_state.tif"]["sha256"])
    environmental_valid = _read_snapshot(run_dir / "simulation_environment_valid.tif", dtype="uint8",
                                shape=tuple(trajectory["shape"]), crs="EPSG:32651",
                                transform=trajectory["transform"],
                                digest=output_by_name["simulation_environment_valid.tif"]["sha256"])
    receiving = final_state != STATE_CODES["non_burnable"]
    environment_valid_mask = environmental_valid.astype(bool)
    expected_environment_sha = end["observed"]["mask_identities"]["environment_sha256"]
    if (environmental_valid.dtype != np.uint8 or not np.all(np.isin(environmental_valid, (0, 1)))
            or np.any(receiving & ~environment_valid_mask)
            or mask_sha256(environment_valid_mask) != expected_environment_sha
            or mask_sha256(receiving) != trajectory["domain_mask_sha256"]):
        raise ValueError("Receiving mask is outside environmental-valid coverage or differs from D-397")
    if not np.array_equal(arrays_by_horizon[max(arrays_by_horizon)]["state"], final_state):
        raise ValueError("Last captured state differs from terminal final_state raster")
    for row in series["outcomes"]:
        if row["absorbing_carry_forward"]:
            _assert_exact_carry_forward(arrays_by_horizon[row["requested_update"]]["state"], final_state)
    allowed = _output_allowlist(expected_horizons)
    if not {path.name for path in run_dir.iterdir()} <= allowed:
        raise ValueError("Unexpected files in Phase 17 trajectory output")
    return {"schema_version": VALIDATION_SCHEMA, "run_id": run_dir.name,
            "incident": incident, "seed": seed, "binding_sha256": binding_sha,
            "review_reference": review_reference, "snapshots": arrays_by_horizon,
            "route_a_bindings": deepcopy(binding["route_a_bindings"]),
            "route_a_common_identity": deepcopy(binding["route_a_common_identity"])}


def _validate_alignment(config: dict, incident: str, alignment: dict) -> dict:
    name = config["incidents"][incident]["route_a_alignment"]
    expected = config["d397_benchmark"]["alignment"][name]
    report, arrays = alignment["report"], alignment["arrays"]
    if (alignment["report_sha256"] != expected["report_sha256"]
            or canonical_metadata_hash(alignment["report"]) != alignment["report_sha256"]
            or alignment["report"].get("incident") != incident):
        raise ValueError("Native alignment report identity differs from D-397")
    prediction_valid = np.ones(tuple(config["incidents"][incident]["shape"]), dtype=bool)
    domain = (arrays["environment"] == 1) & prediction_valid & arrays["reference_valid"] \
        & arrays["buildings_valid"] & (arrays["buildings"] == 10)
    if mask_sha256(domain) != expected["domain_mask_sha256"]:
        raise ValueError("Native scoring domain digest differs from D-397")
    included_positive = domain & arrays["reference_valid"] & (arrays["reference"] == 1)
    if (mask_sha256(included_positive) != expected["included_positive_mask_sha256"]
            or int(np.count_nonzero(domain)) != expected["domain_cells"]
            or int(np.count_nonzero(included_positive)) != expected["reference_positive_included"]):
        raise ValueError("D-397 included-positive domain digest/count differs")
    _same(report["masks"]["evaluation_domain"]["sha256"], expected["domain_mask_sha256"],
          "D-397 reported domain digest differs")
    return arrays


def _signed_delta(reference_metrics: dict, ca_metrics: dict) -> dict:
    result = {}
    for name in METRICS:
        left, right = reference_metrics.get(name), ca_metrics.get(name)
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise ValueError(f"Finite {name} values are required for signed deltas")
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Finite metrics are required for signed deltas")
        result[name] = left - right
    for key, row, col in (("TN", 0, 0), ("FP", 0, 1), ("FN", 1, 0), ("TP", 1, 1)):
        result[key] = reference_metrics["confusion_matrix"][row][col] - ca_metrics["confusion_matrix"][row][col]
    return result


def _summarize_exact_five(rows: list[dict], *, grouping: tuple[str, ...], value_key: str) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in grouping), []).append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        seeds = [row["seed"] for row in group]
        if len(group) != 5 or set(seeds) != set(REQUIRED_SEEDS) or len(set(seeds)) != 5 \
                or any(row.get("valid") is not True for row in group):
            continue
        values_by_metric: dict[str, list[float]] = {}
        for row in group:
            for metric, value in row[value_key].items():
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("Exact-five group contains a nonfinite metric")
                values_by_metric.setdefault(metric, []).append(float(value))
        summaries.append({**dict(zip(grouping, key)), "n": 5, "seeds": sorted(seeds),
                          "mean_sample_sd": {metric: {"mean": statistics.mean(values),
                                                       "sample_sd": statistics.stdev(values)}
                                             for metric, values in values_by_metric.items()}})
    return summaries


def score_validated_outputs(validated_runs: list[dict], alignments: dict, *,
                            manifest_path: Path = DEFAULT_MANIFEST,
                            explicitly_authorized: bool = False,
                            review_reference: str | None = None) -> dict:
    """Join validated CA snapshots to frozen D-397 rows and report signed deltas.

    The caller supplies alignment arrays from the separately authorized Route A
    reader. This function never opens a raster or launches a simulation.
    """
    if explicitly_authorized is not True:
        raise PermissionError("Separate Phase 17 scoring authority is required")
    review_reference = _text(review_reference, "Scoring review reference")
    if not validated_runs:
        raise ValueError("At least one independently validated Phase 17 trajectory is required")
    config, _, _ = load_manifest(manifest_path)
    benchmark, benchmark_sha = _benchmark_identity(config)
    bindings = _route_a_binding_pairs(config, benchmark)
    score_rows = benchmark["scores"]
    expected_runs = {(incident, seed) for incident in config["incidents"] for seed in REQUIRED_SEEDS}
    keyed_runs = {}
    for row in validated_runs:
        key = (row["incident"], row["seed"])
        if key not in expected_runs or key in keyed_runs:
            raise ValueError("Unexpected or duplicate validated Phase 17 trajectory")
        keyed_runs[key] = row
    if len(keyed_runs) > 10:
        raise ValueError("Phase 17 has ten planned trajectories")
    result_rows, deltas = [], []
    for incident, seed in sorted(keyed_runs):
        validated = keyed_runs[(incident, seed)]
        if (validated.get("schema_version") != VALIDATION_SCHEMA
                or validated.get("route_a_bindings") != bindings[(incident, seed)]):
            raise ValueError("Validated CA output is not bound to selected D-397 identities")
        arrays = _validate_alignment(config, incident, alignments[incident])
        incident_cfg = config["incidents"][incident]
        env = arrays["environment"] == 1
        building_valid = arrays["buildings_valid"]
        mapped = building_valid & (arrays["buildings"] == 10)
        reference = arrays["reference"] == 1
        prediction_valid = np.ones(tuple(incident_cfg["shape"]), dtype=bool)
        expected_binding_pair = bindings[(incident, seed)]
        for horizon in incident_cfg["horizons"]:
            requested = horizon["requested_update"]
            snapshot = validated["snapshots"].get(requested)
            if snapshot is None:
                raise ValueError("Validated trajectory is missing a requested horizon")
            outcome = snapshot["outcome"]
            termination_reason = "inactive" if outcome["termination_reason"] == "inactive" else "timestep_limit"
            metrics = evaluate_final_footprint_arrays(
                snapshot["state"], reference, env, prediction_valid, arrays["reference_valid"],
                building_valid, mapped, termination_reason=termination_reason)
            confusion = metrics["confusion_matrix"]
            tn, fp = confusion[0]
            fn, tp = confusion[1]
            metrics["accuracy"] = (tn + tp) / (tn + fp + fn + tp)
            expected_d397 = config["d397_benchmark"]["alignment"][incident_cfg["route_a_alignment"]]
            if (metrics["evaluation_pixel_count"] != expected_d397["domain_cells"]
                    or metrics["reference_coverage"]["reference_burned_included_in_primary_domain"]
                       != expected_d397["reference_positive_included"]):
                raise ValueError("Reused footprint metric population differs from D-397")
            result_row = {"incident": incident, "seed": seed, "requested_update": requested,
                          "assumed_minutes_per_update": horizon["assumed_minutes_per_update"],
                          "clock_role": horizon["clock_role"], "valid": True,
                          "termination_reason": termination_reason, "metrics": metrics,
                          "ca_run_id": validated["run_id"], "ca_binding_sha256": validated["binding_sha256"],
                          "route_a_bindings": deepcopy(expected_binding_pair)}
            result_rows.append(result_row)
            frozen = [row for row in score_rows if row["incident"] == incident and row["seed"] == seed
                      and row["outcome"]["requested_update"] == requested]
            if len(frozen) != 2 or {row["arm"] for row in frozen} != {"rf", "lr"}:
                raise ValueError("Frozen D-397 rows are missing or duplicated")
            for row in frozen:
                arm = row["arm"]
                pinned = expected_binding_pair[arm]
                if (row["outcome"]["run_id"] != pinned["run_id"]
                        or row["outcome"]["execution_binding_sha256"] != pinned["binding_sha256"]):
                    raise ValueError("Frozen D-397 score row binding differs from selected Route A run")
                deltas.append({"incident": incident, "seed": seed, "requested_update": requested,
                               "arm": arm, "contrast": f"{arm}_minus_ca", "valid": True,
                               "route_a_binding_sha256": pinned["binding_sha256"],
                               "ca_binding_sha256": validated["binding_sha256"],
                               "deltas": _signed_delta(row["metrics"], metrics)})
    summaries = _summarize_exact_five(
        [{**row, "value": row["metrics"]} for row in result_rows],
        grouping=("incident", "requested_update"), value_key="value")
    contrast_summaries = _summarize_exact_five(
        [{**row, "value": row["deltas"]} for row in deltas],
        grouping=("incident", "requested_update", "arm"), value_key="value")
    return {"schema_version": SCORING_SCHEMA, "decision": "D-397",
            "benchmark_sha256": benchmark_sha, "review_reference": review_reference,
            "scores": result_rows, "contrasts": deltas,
            "ca_five_seed_summaries": summaries,
            "contrast_five_seed_summaries": contrast_summaries,
            "qualification": "Matched development-incident footprint agreement; not physical or held-out validation"}


