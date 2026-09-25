"""Bounded aggregate-only feasibility harness for the approved Set C pilot."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import ctypes
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import Any

import numpy as np

from .feature_pipeline import CANONICAL_FEATURE_NAMES
from .model_free_teacher import (
    COMPLETENESS_CENSORED,
    COMPLETENESS_COMPLETE,
    COMPLETENESS_FAILED,
    ModelFreeTeacherRunError,
    TEACHER_RUNNER_SCHEMA_VERSION,
    TERMINATION_FAILURE,
    TERMINATION_INACTIVE,
    TERMINATION_SAFETY_LIMIT,
    ignition_set_sha256,
    run_model_free_teacher,
)
from .set_c_collector import (
    RUNNER_OWNED_BACKPRESSURE_POLICY,
    SET_C_ROW_BATCH_SCHEMA_VERSION,
    SetCCollector,
    SetCRowBatch,
)
from .transition_observer import (
    AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
    SOURCE_ONLY_AUTHORIZATION,
    TransitionObservation,
    canonical_metadata_hash,
    mask_sha256,
)
from .wind_convention import WindContract


PILOT_ID = "phase8_set_c_feasibility_pilot_v2"
AGGREGATE_SCHEMA_VERSION = "phase8_set_c_feasibility_aggregate.v2"
APPROVAL_REFERENCE = "Phase 8 v2 owner approval: phase8_set_c_feasibility_pilot_v2"
APPROVED_RELATIVE_DESTINATION = Path(
    "output/phase8/phase8_set_c_feasibility_pilot_v2.aggregate.json"
)
MODEL_FREE_SOURCE = "model-free-fire-automata"
GRID_ID = "lapu_lapu_epsg32651_3m_5489x6896_v1"
EVENT_ID = "simulated_case_study_no_observed_event"
V1_AGGREGATE_PAYLOAD_SHA256 = (
    "23cd71ebca4837e094eabec2a02c95e3869e9b7ae5aac28b2b5eb40b3a98afc6"
)
V1_AGGREGATE_FILE_SHA256 = (
    "e4c52c15ca14286ec8926898e082166b7481ad480970db74877d10b00b7bd332"
)
EXPECTED_GRID_SHAPE = (5489, 6896)
EXPECTED_SIMULATION_VALID_COUNT = 3_959_768
EXPECTED_MAPPED_BUILDING_COUNT = 1_077_802
EXPECTED_AUTHORITATIVE_DOMAIN_COUNT = 932_864
EXPECTED_SIMULATION_VALID_SHA256 = (
    "9a3b0ec29bc846440bce35e63042507258f905dc8cbfa7b274c01dfa7a1f74eb"
)
EXPECTED_MAPPED_BUILDING_SHA256 = (
    "9fcfe93f2f066078015e862fe001e985982c5e8eaab1c339a0cd6c40457f04c1"
)
EXPECTED_AUTHORITATIVE_DOMAIN_SHA256 = (
    "f9bb9dbfe00c61a5b08d228bcbec4b4313853a47f4ca2ea82b7a4359d81814a4"
)
APPROVED_GRID_IDENTITY_SHA256 = (
    "26e9da1104da2caf15039b5f9fbcf719fea506adcee95f9c2da247e8a9e2bc46"
)
SAFETY_LIMIT_TIMESTEPS = 256
BATCH_ROWS = 16_384
MAX_BUFFER_ROWS = 932_864
MAX_BUFFER_BYTES = 176_311_296
TOTAL_ROW_CEILING = 20_557_824
CANDIDATE_BLOCK_SIZES = ((32, 32), (64, 64), (128, 128))
ACTIVE_BLOCK_SIZE = (32, 32)
BLOCK_ORIGIN = (0, 0)
LAUNCH_MEMORY_FLOOR_BYTES = 6_442_450_944
DISK_FLOOR_BYTES = 2_147_483_648
RUNTIME_MEMORY_FLOOR_BYTES = 2_147_483_648
RSS_CEILING_BYTES = 6_442_450_944
RUN_RUNTIME_LIMIT_SECONDS = 900.0
PILOT_RUNTIME_LIMIT_SECONDS = 5_400.0
MAX_RETRY_CYCLES_PER_OBSERVATION = 58
def _approved_wind_manifest(direction_deg: float) -> dict[str, object]:
    return {
        "speed_kmh": 10.0,
        "direction_deg": float(direction_deg),
        "direction_convention": "meteorological_from",
        "direction_units": "degrees_clockwise",
        "north_reference": "projected_grid_north",
        "schema_version": "simulated_wind.v1",
        "calm_representation": "speed_zero_direction_zero",
    }


APPROVED_WIND_MANIFEST = _approved_wind_manifest(315.0)
APPROVED_PLACEHOLDER_TRANSITION = {
    "base_ignition_prob": 0.12,
    "slope_weight": 0.20,
    "building_weight": 0.10,
    "proximity_weight": 0.20,
    "wind_weight": 0.20,
    "burnout_prob": 0.04,
    "material_weight": 0.35,
}
APPROVED_FLAMMABILITY_WEIGHTS = {
    "base_weight": 0.5,
    "slope_weight": 0.3,
    "proximity_weight": 0.2,
}


@dataclass(frozen=True, slots=True)
class PilotFamily:
    family_id: str
    scenario_id: str
    ignition_coordinates: tuple[tuple[int, int], ...]
    ignition_set_sha256: str
    component_size: int
    row_ceiling: int
    wind_direction_deg: float
    seeds: tuple[int, ...]

    @property
    def wind_manifest(self) -> dict[str, object]:
        return _approved_wind_manifest(self.wind_direction_deg)


FAMILIES = (
    PilotFamily(
        "p8f01_cluster_322_m1",
        "p8f01_w315_s10",
        ((894, 2730),),
        "0843f3c6db0284bb85efd670b65027ee11af833e7070f2221080a729ab6193af",
        3_976,
        1_017_856,
        315.0,
        (81001, 81002),
    ),
    PilotFamily(
        "p8f02_cluster_1052_m3",
        "p8f02_w315_s10",
        ((1198, 2523),),
        "2a938c46cd01a87b49f312257257053c2dedc33773d538776397b3394de31f65",
        5_454,
        1_396_224,
        315.0,
        (81001, 81002),
    ),
    PilotFamily(
        "p8f03_cluster_4974_m2",
        "p8f03_w315_s10",
        ((1723, 1569),),
        "4c649982c5844a54d3bb8dd7b7eb3d1fbc3d71fe9f67505c25a95fef7d0bb1b8",
        3_811,
        975_616,
        315.0,
        (81001, 81002),
    ),
    PilotFamily(
        "p8f04_cluster_160_m5",
        "p8f04_w315_s10",
        ((791, 3164),),
        "a23beb74a1d4a649446416e6ca8258b1cd315a774a8757b0b033a3496057bbaa",
        3_535,
        904_960,
        315.0,
        (81001, 81002, 82001, 82002, 82003, 82004, 82005, 82006, 82007, 82008),
    ),
    PilotFamily(
        "p8v2f05_cluster_4146_m3",
        "p8v2f05_w045_s10",
        ((1603, 2799),),
        "01fb6b26993f5651fb070825c4fe7b9380a7d8fc852d977850f9e6c86063914f",
        1_921,
        491_776,
        45.0,
        (83001, 83002, 83003, 83004, 83005, 83006, 83007, 83008),
    ),
    PilotFamily(
        "p8v2f06_cluster_6201_m5",
        "p8v2f06_w135_s10",
        ((1789, 2602),),
        "084a8e3f06d0514c0c53a04d986248080a7271b769f0a09e464fa1d0f3a0c8d1",
        388,
        99_328,
        135.0,
        (84001, 84002, 84003, 84004, 84005, 84006, 84007, 84008),
    ),
)

V1_REPLAY_EXPECTATIONS = {
    "p8f01_w315_s10__seed_81001": (10, 1, 9, 3),
    "p8f01_w315_s10__seed_81002": (33, 6, 27, 11),
    "p8f02_w315_s10__seed_81001": (5, 0, 5, 1),
    "p8f02_w315_s10__seed_81002": (10, 1, 9, 3),
    "p8f03_w315_s10__seed_81001": (4, 0, 4, 1),
    "p8f03_w315_s10__seed_81002": (10, 1, 9, 3),
    "p8f04_w315_s10__seed_81001": (8, 0, 8, 1),
    "p8f04_w315_s10__seed_81002": (8, 0, 8, 1),
}
V2_NEW_FAMILY_IDS = (
    "p8v2f05_cluster_4146_m3",
    "p8v2f06_cluster_6201_m5",
)


@dataclass(frozen=True, slots=True)
class PilotRunSpec:
    order: int
    family: PilotFamily
    seed: int
    run_id: str


RUN_MATRIX = tuple(
    PilotRunSpec(
        order=index + 1,
        family=family,
        seed=seed,
        run_id=f"{family.scenario_id}__seed_{seed}",
    )
    for index, (family, seed) in enumerate(
        (family, seed) for family in FAMILIES for seed in family.seeds
    )
)


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    available_memory_bytes: int
    process_rss_bytes: int
    free_disk_bytes: int
    captured_at_utc: str


class PilotContractError(RuntimeError):
    """Raised when an approved pilot contract or integrity gate fails."""


class PilotResourceError(PilotContractError):
    """Raised when a memory, disk, or runtime stop condition is crossed."""


def approved_destination() -> Path:
    """Return the sole approved production aggregate destination."""
    return Path(__file__).resolve().parents[1] / APPROVED_RELATIVE_DESTINATION


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"No existing parent for destination: {path}")
        candidate = candidate.parent
    return candidate


def _windows_memory() -> tuple[int, int]:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatusEx)]
    kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return int(status.ullAvailPhys), int(counters.WorkingSetSize)


def _posix_memory() -> tuple[int, int]:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    statm = Path("/proc/self/statm")
    if not statm.is_file():
        raise OSError("Process RSS telemetry is unavailable on this platform")
    fields = statm.read_text(encoding="ascii").split()
    if len(fields) < 2:
        raise OSError("Process RSS telemetry returned an invalid value")
    return available_pages * page_size, int(fields[1]) * page_size


def capture_resource_snapshot(destination: Path) -> ResourceSnapshot:
    """Capture required memory, RSS, and destination-filesystem telemetry."""
    available, rss = _windows_memory() if os.name == "nt" else _posix_memory()
    free_disk = shutil.disk_usage(_nearest_existing_parent(destination)).free
    return ResourceSnapshot(
        available_memory_bytes=available,
        process_rss_bytes=rss,
        free_disk_bytes=int(free_disk),
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def activate_approved_pilot_config(config: Mapping[str, object]) -> dict[str, Any]:
    """Return an in-memory enabled copy while leaving defaults fail-closed."""
    active = deepcopy(dict(config))
    pilot = active.get("phase8_set_c_pilot")
    if not isinstance(pilot, dict):
        raise PilotContractError("phase8_set_c_pilot configuration is required")
    if pilot.get("enabled") is not False:
        raise PilotContractError("The checked-in Phase 8 pilot must remain disabled")
    if pilot.get("pilot_id") != PILOT_ID:
        raise PilotContractError("Phase 8 pilot_id does not match the approval")
    if pilot.get("aggregate_schema_version") != AGGREGATE_SCHEMA_VERSION:
        raise PilotContractError("Phase 8 aggregate schema does not match the approval")
    if pilot.get("authorization_reference") != APPROVAL_REFERENCE:
        raise PilotContractError("Phase 8 authorization reference does not match")
    pilot["enabled"] = True
    return active


def _require_exact_contract(config: Mapping[str, object]) -> None:
    pilot = config.get("phase8_set_c_pilot")
    if not isinstance(pilot, Mapping) or pilot.get("enabled") is not True:
        raise PermissionError("Phase 8 pilot is disabled by default")
    exact = {
        "pilot_id": PILOT_ID,
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "authorization_reference": APPROVAL_REFERENCE,
        "transition_source": MODEL_FREE_SOURCE,
        "estimator_enabled": False,
        "normal_ca_ml_path_enabled": False,
        "safety_limit_timesteps": SAFETY_LIMIT_TIMESTEPS,
        "run_concurrency": 0,
        "launch_memory_floor_bytes": LAUNCH_MEMORY_FLOOR_BYTES,
        "disk_floor_bytes": DISK_FLOOR_BYTES,
        "runtime_memory_floor_bytes": RUNTIME_MEMORY_FLOOR_BYTES,
        "rss_ceiling_bytes": RSS_CEILING_BYTES,
        "run_runtime_limit_seconds": 900,
        "pilot_runtime_limit_seconds": 5400,
        "aggregate_destination": str(APPROVED_RELATIVE_DESTINATION).replace("\\", "/"),
    }
    for name, expected in exact.items():
        if pilot.get(name) != expected:
            raise PilotContractError(
                f"phase8_set_c_pilot.{name} must equal {expected!r}"
            )
    if config.get("simulation", {}).get("max_timesteps") != 300:
        raise PilotContractError("General simulation.max_timesteps must remain 300")
    configured_wind = config.get("wind")
    if not isinstance(configured_wind, Mapping):
        raise PilotContractError("Wind configuration is required")
    if dict(configured_wind) != APPROVED_WIND_MANIFEST:
        raise PilotContractError("Wind configuration does not match the approved matrix")
    wind = WindContract.from_config(dict(configured_wind)).to_manifest()
    if wind != APPROVED_WIND_MANIFEST:
        raise PilotContractError("Wind configuration does not match the approved matrix")
    placeholder_transition = config.get("placeholder_transition")
    if (
        not isinstance(placeholder_transition, Mapping)
        or dict(placeholder_transition) != APPROVED_PLACEHOLDER_TRANSITION
    ):
        raise PilotContractError(
            "placeholder_transition configuration does not match the approval"
        )
    flammability_weights = config.get("flammability_weights")
    if (
        not isinstance(flammability_weights, Mapping)
        or dict(flammability_weights) != APPROVED_FLAMMABILITY_WEIGHTS
    ):
        raise PilotContractError(
            "flammability_weights configuration does not match the approval"
        )
    collection = pilot.get("collector")
    expected_collection = {
        "sampling_policy": "all_eligible",
        "batch_rows": BATCH_ROWS,
        "max_buffer_rows": MAX_BUFFER_ROWS,
        "max_buffer_bytes": MAX_BUFFER_BYTES,
        "backpressure_policy": RUNNER_OWNED_BACKPRESSURE_POLICY,
    }
    if not isinstance(collection, Mapping) or dict(collection) != expected_collection:
        raise PilotContractError("Pilot collector bounds do not match the approval")
    if pilot.get("candidate_block_sizes") != [[32, 32], [64, 64], [128, 128]]:
        raise PilotContractError("Diagnostic candidate block sizes changed")
    if pilot.get("active_diagnostic_block_size") != [32, 32]:
        raise PilotContractError("Active diagnostic block identity must be 32x32")
    if pilot.get("block_origin") != [0, 0]:
        raise PilotContractError("Diagnostic block origin must be (0,0)")
    if pilot.get("run_count") != len(RUN_MATRIX):
        raise PilotContractError("Pilot run count changed")
    if pilot.get("total_row_ceiling") != TOTAL_ROW_CEILING:
        raise PilotContractError("Pilot total row ceiling changed")
    configured_families = pilot.get("families")
    expected_families = [
        {
            "family_id": family.family_id,
            "scenario_id": family.scenario_id,
            "ignition_coordinates": [list(point) for point in family.ignition_coordinates],
            "ignition_set_sha256": family.ignition_set_sha256,
            "component_size": family.component_size,
            "row_ceiling": family.row_ceiling,
            "wind_manifest": family.wind_manifest,
            "seeds": list(family.seeds),
        }
        for family in FAMILIES
    ]
    if configured_families != expected_families:
        raise PilotContractError("Pilot family/ignition matrix changed")
    if config.get("dataset_generation", {}).get("artifact_write_enabled") is not False:
        raise PermissionError("Dataset artifact writes must remain disabled")
    publisher = config.get("dataset_generation", {}).get("set_c_publisher", {})
    if not isinstance(publisher, Mapping) or publisher.get("enabled") is not False:
        raise PermissionError("Set C row-level publisher must remain disabled")
    for family in FAMILIES:
        if ignition_set_sha256(family.ignition_coordinates) != family.ignition_set_sha256:
            raise PilotContractError("Approved ignition identity is internally inconsistent")


def _run_config(config: Mapping[str, object], spec: PilotRunSpec) -> dict[str, Any]:
    run_config = deepcopy(dict(config))
    run_config["simulation"]["seed"] = spec.seed
    run_config["wind"] = spec.family.wind_manifest
    run_config["model_free_teacher"].update(
        {
            "enabled": True,
            "runner_schema_version": TEACHER_RUNNER_SCHEMA_VERSION,
            "safety_limit_timesteps": SAFETY_LIMIT_TIMESTEPS,
            "persistent_output_enabled": False,
        }
    )
    run_config["ml_model"]["enabled"] = False
    dataset = run_config["dataset_generation"]
    dataset["artifact_write_enabled"] = False
    dataset["sampling_policy"] = "all_eligible"
    dataset["set_c_collector"] = {
        "enabled": True,
        "batch_rows": BATCH_ROWS,
        "max_buffer_rows": MAX_BUFFER_ROWS,
        "max_buffer_bytes": MAX_BUFFER_BYTES,
        "backpressure_policy": RUNNER_OWNED_BACKPRESSURE_POLICY,
    }
    dataset["set_c_publisher"]["enabled"] = False
    dataset["spatial_block"] = {
        "origin_row": BLOCK_ORIGIN[0],
        "origin_col": BLOCK_ORIGIN[1],
        "size_rows": ACTIVE_BLOCK_SIZE[0],
        "size_cols": ACTIVE_BLOCK_SIZE[1],
    }
    return run_config


def _validate_environment(environment: Mapping[str, object], *, synthetic: bool) -> dict[str, Any]:
    shape = tuple(int(value) for value in environment.get("grid_shape", ()))
    if len(shape) != 2:
        raise PilotContractError("Environment grid_shape is invalid")
    burnable = np.asarray(environment.get("burnable_mask"))
    nodata = np.asarray(environment.get("nodata_mask"))
    buildings = np.asarray(environment.get("building_presence"))
    materials = np.asarray(environment.get("material_class"))
    if burnable.dtype != np.dtype(bool) or nodata.dtype != np.dtype(bool):
        raise PilotContractError("Environment validity masks must be boolean")
    if any(array.shape != shape for array in (burnable, nodata, buildings, materials)):
        raise PilotContractError("Environment domains do not match grid_shape")
    valid = burnable & (~nodata)
    mapped = buildings > 0
    material_mapped = materials > 0
    if not np.array_equal(mapped, material_mapped):
        raise PilotContractError(
            "building_presence and material_class mapped domains must match bidirectionally"
        )
    domain = valid & mapped
    crs = str(environment.get("crs")).strip()
    try:
        transform = tuple(float(value) for value in environment.get("transform", ()))
    except (TypeError, ValueError) as exc:
        raise PilotContractError("Environment grid transform is invalid") from exc
    if not crs or len(transform) not in (6, 9) or not np.all(np.isfinite(transform)):
        raise PilotContractError("Environment grid CRS/transform is invalid")
    grid_identity_sha256 = canonical_metadata_hash(
        {"shape": shape, "crs": crs, "transform": transform}
    )
    identity = {
        "grid_shape": list(shape),
        "grid_crs": crs,
        "grid_transform": list(transform),
        "simulation_valid_count": int(np.count_nonzero(valid)),
        "mapped_building_count": int(np.count_nonzero(mapped)),
        "authoritative_domain_count": int(np.count_nonzero(domain)),
        "simulation_valid_mask_sha256": mask_sha256(valid),
        "mapped_building_mask_sha256": mask_sha256(mapped),
        "authoritative_domain_sha256": mask_sha256(domain),
        "grid_identity_sha256": grid_identity_sha256,
    }
    if not synthetic:
        expected = {
            "grid_shape": list(EXPECTED_GRID_SHAPE),
            "grid_crs": "EPSG:32651",
            "simulation_valid_count": EXPECTED_SIMULATION_VALID_COUNT,
            "mapped_building_count": EXPECTED_MAPPED_BUILDING_COUNT,
            "authoritative_domain_count": EXPECTED_AUTHORITATIVE_DOMAIN_COUNT,
            "simulation_valid_mask_sha256": EXPECTED_SIMULATION_VALID_SHA256,
            "mapped_building_mask_sha256": EXPECTED_MAPPED_BUILDING_SHA256,
            "authoritative_domain_sha256": EXPECTED_AUTHORITATIVE_DOMAIN_SHA256,
            "grid_identity_sha256": APPROVED_GRID_IDENTITY_SHA256,
        }
        for name, expected_value in expected.items():
            if identity[name] != expected_value:
                raise PilotContractError(f"Environment {name} does not match Task 8.1")
        transform_values = identity["grid_transform"]
        if not math.isclose(transform_values[0], 3.0) or not math.isclose(transform_values[4], -3.0):
            raise PilotContractError("Environment grid resolution must remain 3m")
    return identity


def _snapshot_dict(snapshot: ResourceSnapshot) -> dict[str, object]:
    return {
        "available_memory_bytes": snapshot.available_memory_bytes,
        "process_rss_bytes": snapshot.process_rss_bytes,
        "free_disk_bytes": snapshot.free_disk_bytes,
        "captured_at_utc": snapshot.captured_at_utc,
    }


def _check_launch(snapshot: ResourceSnapshot, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite pilot report: {destination}")
    if snapshot.available_memory_bytes < LAUNCH_MEMORY_FLOOR_BYTES:
        raise PilotResourceError("Available physical memory is below the 6 GiB launch floor")
    if snapshot.free_disk_bytes < DISK_FLOOR_BYTES:
        raise PilotResourceError("Free disk is below the 2 GiB launch floor")
    if snapshot.process_rss_bytes < 0:
        raise PilotResourceError("Process RSS telemetry is invalid")


def preflight_pilot_launch(
    config: Mapping[str, object],
    destination: Path,
    *,
    resource_probe: Callable[[Path], ResourceSnapshot] = capture_resource_snapshot,
    synthetic_test: bool = False,
) -> ResourceSnapshot:
    """Validate the disabled-by-default contract and the launch resource gate."""
    if not synthetic_test and resource_probe is not capture_resource_snapshot:
        raise PermissionError(
            "Production resource telemetry must use capture_resource_snapshot"
        )
    _require_exact_contract(config)
    destination = Path(destination).resolve()
    if synthetic_test:
        if destination == approved_destination().resolve():
            raise PermissionError("Synthetic tests cannot preflight the production destination")
    elif destination != approved_destination().resolve():
        raise PermissionError("Only the approved aggregate destination is permitted")
    snapshot = resource_probe(destination)
    _check_launch(snapshot, destination)
    return snapshot


def _check_runtime(snapshot: ResourceSnapshot, elapsed: float) -> None:
    if snapshot.available_memory_bytes < RUNTIME_MEMORY_FLOOR_BYTES:
        raise PilotResourceError("Available physical memory fell below 2 GiB")
    if snapshot.process_rss_bytes > RSS_CEILING_BYTES:
        raise PilotResourceError("Pilot process RSS exceeded 6 GiB")
    if elapsed > RUN_RUNTIME_LIMIT_SECONDS:
        raise PilotResourceError("Pilot run exceeded 900 seconds")


class _UnionFind:
    def __init__(self) -> None:
        self.parent = {family.family_id: family.family_id for family in FAMILIES}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


class _RunAccumulator:
    def __init__(self, spec: PilotRunSpec) -> None:
        self.spec = spec
        self.rows = 0
        self.positive = 0
        self.negative = 0
        self.batch_count = 0
        self.observation_rows: dict[int, list[int]] = {}
        self.timestep_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        self.duplicate_groups: dict[bytes, list[int]] = {}
        self.block_groups: dict[tuple[int, int], dict[tuple[int, int], list[int]]] = {
            size: {} for size in CANDIDATE_BLOCK_SIZES
        }
        self.max_pending_rows = 0
        self.max_pending_bytes = 0
        self.backpressure_drain_count = 0
        self.max_retry_cycles = 0
        self.current_retry_cycles = 0
        self.no_progress_detected = False
        self.overflow_detected = False
        self.retry_limit_exceeded = False
        self.resource_samples: list[ResourceSnapshot] = []

    @staticmethod
    def _label_mask(labels: np.ndarray) -> int:
        mask = 0
        if np.any(labels == 0):
            mask |= 1
        if np.any(labels == 1):
            mask |= 2
        return mask

    def accept_batch(self, batch: SetCRowBatch) -> None:
        if batch.schema_version != SET_C_ROW_BATCH_SCHEMA_VERSION:
            raise PilotContractError("Set C batch schema changed during the pilot")
        if batch.feature_names != CANONICAL_FEATURE_NAMES or batch.features.dtype != np.float32:
            raise PilotContractError("Set C feature schema changed during the pilot")
        if batch.labels.dtype != np.int8 or not np.all(np.isin(batch.labels, (0, 1))):
            raise PilotContractError("Set C class contract changed during the pilot")
        if batch.row_count != len(batch.labels):
            raise PilotContractError("Set C row and class counts do not match")
        positives = int(np.count_nonzero(batch.labels == 1))
        negatives = batch.row_count - positives
        self.rows += batch.row_count
        self.positive += positives
        self.negative += negatives
        self.batch_count += 1
        counts = self.timestep_counts[batch.timestep_t]
        counts[0] += negatives
        counts[1] += positives
        observation = self.observation_rows.setdefault(batch.observation_index, [0, 0, 0])
        observation[0] += batch.row_count
        observation[1] += negatives
        observation[2] += positives
        for duplicate, label in zip(batch.duplicate_group_ids, batch.labels, strict=True):
            key = bytes(duplicate)
            value = self.duplicate_groups.setdefault(key, [0, 0])
            value[0] += 1
            value[1] |= 1 if int(label) == 0 else 2
        for size in CANDIDATE_BLOCK_SIZES:
            groups = self.block_groups[size]
            block_rows = (batch.cell_rows - BLOCK_ORIGIN[0]) // size[0]
            block_cols = (batch.cell_cols - BLOCK_ORIGIN[1]) // size[1]
            for block_row, block_col, label in zip(
                block_rows, block_cols, batch.labels, strict=True
            ):
                key = (int(block_row), int(block_col))
                value = groups.setdefault(key, [0, 0])
                value[0] += 1
                value[1] |= 1 if int(label) == 0 else 2

    def duplicate_summary(self) -> dict[str, int]:
        values = list(self.duplicate_groups.values())
        return {
            "unique_group_count": len(values),
            "repeated_group_count": sum(value[0] > 1 for value in values),
            "conflicting_label_group_count": sum(value[1] == 3 for value in values),
            "maximum_group_rows": max((value[0] for value in values), default=0),
        }

    def failure_evidence(
        self,
        *,
        termination: object | None,
        elapsed: float,
        failure: BaseException,
    ) -> dict[str, object]:
        """Return aggregate-only, explicitly non-authoritative failed-run evidence."""
        samples = self.resource_samples
        return {
            "run_id": self.spec.run_id,
            "scenario_family_id": self.spec.family.family_id,
            "seed": self.spec.seed,
            "failure": {
                "type": type(failure).__name__,
                "message": str(failure),
            },
            "termination": {
                "reason": getattr(termination, "termination_reason", TERMINATION_FAILURE),
                "completeness_status": COMPLETENESS_FAILED,
                "authoritative": False,
                "final_timestep": getattr(termination, "final_timestep", None),
                "final_ignited_count": getattr(
                    termination, "final_ignited_count", None
                ),
                "final_blazing_count": getattr(
                    termination, "final_blazing_count", None
                ),
                "expected_transition_count": getattr(
                    termination, "expected_transition_count", None
                ),
                "captured_transition_count": getattr(
                    termination,
                    "captured_transition_count",
                    len(self.observation_rows),
                ),
            },
            "partial_counts": {
                "observation_count": len(self.observation_rows),
                "rows": self.rows,
                "negative_0": self.negative,
                "positive_1": self.positive,
            },
            "memory_backpressure": {
                "maximum_pending_rows": self.max_pending_rows,
                "maximum_pending_bytes": self.max_pending_bytes,
                "backpressure_drain_count": self.backpressure_drain_count,
                "maximum_retry_cycles": self.max_retry_cycles,
                "no_progress_detected": self.no_progress_detected,
                "overflow_detected": self.overflow_detected,
                "retry_limit_exceeded": self.retry_limit_exceeded,
            },
            "resource_summary": {
                "minimum_available_memory_bytes": min(
                    (sample.available_memory_bytes for sample in samples), default=None
                ),
                "maximum_process_rss_bytes": max(
                    (sample.process_rss_bytes for sample in samples), default=None
                ),
                "minimum_free_disk_bytes": min(
                    (sample.free_disk_bytes for sample in samples), default=None
                ),
                "sample_count": len(samples),
            },
            "runtime_seconds": elapsed,
        }


class _PilotAccumulator:
    def __init__(self) -> None:
        self.runs: list[dict[str, object]] = []
        self.total_rows = 0
        self.total_positive = 0
        self.total_negative = 0
        self.timestep_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        self.duplicates: dict[bytes, list[object]] = {}
        self.blocks: dict[tuple[int, int], dict[tuple[int, int], list[object]]] = {
            size: {} for size in CANDIDATE_BLOCK_SIZES
        }
        self.authoritative_duplicates: dict[bytes, set[str]] = defaultdict(set)
        self.authoritative_blocks: dict[
            tuple[int, int], dict[tuple[int, int], set[str]]
        ] = {size: defaultdict(set) for size in CANDIDATE_BLOCK_SIZES}
        self.authoritative_family_class_masks: dict[str, int] = defaultdict(int)

    def invalidate_authoritative_evidence(self) -> None:
        """Prevent a failed pilot from presenting any partial run as authoritative."""
        self.authoritative_duplicates.clear()
        for groups in self.authoritative_blocks.values():
            groups.clear()
        self.authoritative_family_class_masks.clear()
        for run in self.runs:
            run["termination"]["authoritative"] = False

    def merge(
        self,
        run: _RunAccumulator,
        termination: object,
        elapsed: float,
        *,
        configuration_hash: str,
        run_identity_sha256: str,
        provenance_sha256: str,
    ) -> None:
        authoritative = (
            getattr(termination, "termination_reason", None) == TERMINATION_INACTIVE
            and getattr(termination, "completeness_status", None) == COMPLETENESS_COMPLETE
            and getattr(termination, "authoritative", None) is True
        )
        self.total_rows += run.rows
        self.total_positive += run.positive
        self.total_negative += run.negative
        for timestep, counts in run.timestep_counts.items():
            aggregate = self.timestep_counts[timestep]
            aggregate[0] += counts[0]
            aggregate[1] += counts[1]
        for key, (count, class_mask) in run.duplicate_groups.items():
            aggregate = self.duplicates.setdefault(key, [0, 0, set()])
            aggregate[0] = int(aggregate[0]) + count
            aggregate[1] = int(aggregate[1]) | class_mask
            aggregate[2].add(run.spec.family.family_id)
            if authoritative:
                self.authoritative_duplicates[key].add(run.spec.family.family_id)
        for size, groups in run.block_groups.items():
            for key, (count, class_mask) in groups.items():
                aggregate = self.blocks[size].setdefault(key, [0, 0, set()])
                aggregate[0] = int(aggregate[0]) + count
                aggregate[1] = int(aggregate[1]) | class_mask
                aggregate[2].add(run.spec.family.family_id)
                if authoritative:
                    self.authoritative_blocks[size][key].add(run.spec.family.family_id)
        if authoritative:
            mask = (1 if run.negative else 0) | (2 if run.positive else 0)
            self.authoritative_family_class_masks[run.spec.family.family_id] |= mask
        self.runs.append(
            {
                "order": run.spec.order,
                "run_id": run.spec.run_id,
                "scenario_id": run.spec.family.scenario_id,
                "scenario_family_id": run.spec.family.family_id,
                "seed": run.spec.seed,
                "rng_bit_generator": "PCG64",
                "configuration_sha256": configuration_hash,
                "run_identity_sha256": run_identity_sha256,
                "provenance_sha256": provenance_sha256,
                "ignition_set_sha256": run.spec.family.ignition_set_sha256,
                "component_size": run.spec.family.component_size,
                "row_ceiling": run.spec.family.row_ceiling,
                "rows": run.rows,
                "class_counts": {"negative_0": run.negative, "positive_1": run.positive},
                "positive_prevalence": run.positive / run.rows if run.rows else None,
                "observed_timestep_count": len(run.observation_rows),
                "batch_count": run.batch_count,
                "duplicate_groups": run.duplicate_summary(),
                "termination": {
                    "reason": getattr(termination, "termination_reason", "unknown"),
                    "completeness_status": getattr(termination, "completeness_status", "failed"),
                    "authoritative": authoritative,
                    "final_timestep": int(getattr(termination, "final_timestep", -1)),
                    "final_ignited_count": int(getattr(termination, "final_ignited_count", -1)),
                    "final_blazing_count": int(getattr(termination, "final_blazing_count", -1)),
                    "expected_transition_count": int(getattr(termination, "expected_transition_count", -1)),
                    "captured_transition_count": int(getattr(termination, "captured_transition_count", -1)),
                },
                "memory_backpressure": {
                    "maximum_pending_rows": run.max_pending_rows,
                    "maximum_pending_bytes": run.max_pending_bytes,
                    "backpressure_drain_count": run.backpressure_drain_count,
                    "maximum_retry_cycles": run.max_retry_cycles,
                    "no_progress_detected": run.no_progress_detected,
                    "overflow_detected": run.overflow_detected,
                    "retry_limit_exceeded": run.retry_limit_exceeded,
                    "minimum_available_memory_bytes": min(
                        (sample.available_memory_bytes for sample in run.resource_samples),
                        default=None,
                    ),
                    "maximum_process_rss_bytes": max(
                        (sample.process_rss_bytes for sample in run.resource_samples),
                        default=None,
                    ),
                },
                "runtime_seconds": elapsed,
            }
        )

    def _connectivity(self, size: tuple[int, int]) -> dict[str, object]:
        union = _UnionFind()
        for families in self.authoritative_duplicates.values():
            ordered = sorted(families)
            for family in ordered[1:]:
                union.union(ordered[0], family)
        for families in self.authoritative_blocks[size].values():
            ordered = sorted(families)
            for family in ordered[1:]:
                union.union(ordered[0], family)
        components: dict[str, set[str]] = defaultdict(set)
        for family in FAMILIES:
            family_id = family.family_id
            if family_id in self.authoritative_family_class_masks:
                components[union.find(family_id)].add(family_id)
        ordered_components = sorted(
            (tuple(sorted(families)) for families in components.values()),
            key=lambda families: families,
        )
        inventory = []
        for ordinal, family_ids in enumerate(ordered_components, start=1):
            family_set = set(family_ids)
            runs = [
                run
                for run in self.runs
                if run["scenario_family_id"] in family_set
                and run["termination"]["authoritative"] is True
            ]
            negative = sum(int(run["class_counts"]["negative_0"]) for run in runs)
            positive = sum(int(run["class_counts"]["positive_1"]) for run in runs)
            inventory.append(
                {
                    "component_ordinal": ordinal,
                    "component_sha256": canonical_metadata_hash(
                        {
                            "candidate_block_size": size,
                            "scenario_family_ids": family_ids,
                        }
                    ),
                    "scenario_family_ids": list(family_ids),
                    "family_count": len(family_ids),
                    "run_count": len(runs),
                    "rows": negative + positive,
                    "class_counts": {
                        "negative_0": negative,
                        "positive_1": positive,
                    },
                    "both_classes_present": negative > 0 and positive > 0,
                    "duplicate_group_count": sum(
                        bool(families & family_set)
                        for families in self.authoritative_duplicates.values()
                    ),
                    "block_group_count": sum(
                        bool(families & family_set)
                        for families in self.authoritative_blocks[size].values()
                    ),
                    "authoritative_only": True,
                }
            )
        both = sum(component["both_classes_present"] for component in inventory)
        return {
            "authoritative_component_count": len(inventory),
            "authoritative_components_with_both_labels": both,
            "components": inventory,
            "four_role_mathematically_feasible": len(inventory) >= 4 and both >= 4,
            "split_roles_assigned": False,
        }

    @staticmethod
    def _distribution(values: list[int]) -> dict[str, int | float | None]:
        return {
            "minimum": min(values) if values else None,
            "median": statistics.median(values) if values else None,
            "maximum": max(values) if values else None,
        }

    def aggregate_sections(self) -> dict[str, object]:
        duplicate_values = list(self.duplicates.values())
        candidate_blocks = []
        for size in CANDIDATE_BLOCK_SIZES:
            block_values = list(self.blocks[size].values())
            candidate_blocks.append(
                {
                    "size_rows": size[0],
                    "size_cols": size[1],
                    "physical_size_m": [size[0] * 3, size[1] * 3],
                    "diagnostic_only": True,
                    "active_identity": size == ACTIVE_BLOCK_SIZE,
                    "occupied_block_count": len(block_values),
                    "rows_per_occupied_block": self._distribution(
                        [int(value[0]) for value in block_values]
                    ),
                    "blocks_with_both_labels": sum(int(value[1]) == 3 for value in block_values),
                    "leakage_connectivity": self._connectivity(size),
                }
            )
        families = []
        for family in FAMILIES:
            runs = [run for run in self.runs if run["scenario_family_id"] == family.family_id]
            negative = sum(int(run["class_counts"]["negative_0"]) for run in runs)
            positive = sum(int(run["class_counts"]["positive_1"]) for run in runs)
            rows = negative + positive
            families.append(
                {
                    "scenario_family_id": family.family_id,
                    "run_count": len(runs),
                    "rows": rows,
                    "class_counts": {"negative_0": negative, "positive_1": positive},
                    "positive_prevalence": positive / rows if rows else None,
                }
            )
        return {
            "totals": {
                "rows": self.total_rows,
                "class_counts": {
                    "negative_0": self.total_negative,
                    "positive_1": self.total_positive,
                },
                "positive_prevalence": self.total_positive / self.total_rows if self.total_rows else None,
                "scenario_family_count": len(FAMILIES),
                "run_count": len(self.runs),
            },
            "scenario_families": families,
            "runs": self.runs,
            "timesteps": [
                {
                    "timestep_t": timestep,
                    "rows": counts[0] + counts[1],
                    "class_counts": {"negative_0": counts[0], "positive_1": counts[1]},
                }
                for timestep, counts in sorted(self.timestep_counts.items())
            ],
            "duplicate_groups": {
                "unique_group_count": len(duplicate_values),
                "repeated_group_count": sum(int(value[0]) > 1 for value in duplicate_values),
                "conflicting_label_group_count": sum(int(value[1]) == 3 for value in duplicate_values),
                "maximum_group_rows": max((int(value[0]) for value in duplicate_values), default=0),
            },
            "candidate_blocks": candidate_blocks,
        }


def _require_v1_replay_reconciliation(
    run: _RunAccumulator, termination: object
) -> None:
    expected = V1_REPLAY_EXPECTATIONS.get(run.spec.run_id)
    if expected is None:
        return
    observed = (
        run.rows,
        run.positive,
        run.negative,
        int(getattr(termination, "captured_transition_count", -1)),
    )
    if observed != expected:
        raise PilotContractError(
            f"V1 replay reconciliation failed for {run.spec.run_id}: "
            f"expected {expected!r}, observed {observed!r}"
        )
    if (
        getattr(termination, "termination_reason", None) != TERMINATION_INACTIVE
        or getattr(termination, "completeness_status", None) != COMPLETENESS_COMPLETE
        or getattr(termination, "authoritative", None) is not True
        or int(getattr(termination, "expected_transition_count", -1)) != expected[3]
    ):
        raise PilotContractError(
            f"V1 replay termination reconciliation failed for {run.spec.run_id}"
        )


def _v1_replay_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    by_id = {str(run["run_id"]): run for run in runs}
    entries = []
    for run_id, expected in V1_REPLAY_EXPECTATIONS.items():
        run = by_id.get(run_id)
        termination = run.get("termination", {}) if run is not None else {}
        classes = run.get("class_counts", {}) if run is not None else {}
        observed = {
            "rows": run.get("rows") if run is not None else None,
            "positive_1": classes.get("positive_1"),
            "negative_0": classes.get("negative_0"),
            "transition_count": termination.get("captured_transition_count"),
            "expected_transition_count": termination.get("expected_transition_count"),
            "termination_reason": termination.get("reason"),
            "authoritative": termination.get("authoritative"),
        }
        passed = observed == {
            "rows": expected[0],
            "positive_1": expected[1],
            "negative_0": expected[2],
            "transition_count": expected[3],
            "expected_transition_count": expected[3],
            "termination_reason": TERMINATION_INACTIVE,
            "authoritative": True,
        }
        entries.append({"run_id": run_id, "passed": passed, "observed": observed})
    return {
        "required_run_count": len(V1_REPLAY_EXPECTATIONS),
        "all_reconciled": len(entries) == len(V1_REPLAY_EXPECTATIONS)
        and all(entry["passed"] for entry in entries),
        "runs": entries,
    }


def _acceptance_summary(
    execution_status: str, sections: Mapping[str, object]
) -> dict[str, object]:
    runs = list(sections["runs"])
    families = {
        str(family["scenario_family_id"]): family
        for family in sections["scenario_families"]
    }
    all_authoritative_inactive = len(runs) == len(RUN_MATRIX) and all(
        run["termination"]["reason"] == TERMINATION_INACTIVE
        and run["termination"]["completeness_status"] == COMPLETENESS_COMPLETE
        and run["termination"]["authoritative"] is True
        for run in runs
    )
    transition_counts_exact = len(runs) == len(RUN_MATRIX) and all(
        run["termination"]["expected_transition_count"]
        == run["termination"]["captured_transition_count"]
        == run["observed_timestep_count"]
        for run in runs
    )
    replay = _v1_replay_summary(runs)
    new_family_coverage = {
        family_id: bool(
            family_id in families
            and families[family_id]["class_counts"]["negative_0"] > 0
            and families[family_id]["class_counts"]["positive_1"] > 0
        )
        for family_id in V2_NEW_FAMILY_IDS
    }
    blocks = list(sections["candidate_blocks"])
    block_coverage = {
        f"{block['size_rows']}x{block['size_cols']}": bool(
            block["leakage_connectivity"]["authoritative_component_count"] >= 4
            and block["leakage_connectivity"][
                "authoritative_components_with_both_labels"
            ]
            >= 4
            and block["leakage_connectivity"]["four_role_mathematically_feasible"]
            is True
        )
        for block in blocks
    }
    operational_contract = (
        execution_status == "complete"
        and len(runs) == len(RUN_MATRIX)
        and all(
            run["termination"]["reason"] == TERMINATION_INACTIVE
            and run["termination"]["completeness_status"] == COMPLETENESS_COMPLETE
            and run["termination"]["authoritative"] is True
            and run["memory_backpressure"]["no_progress_detected"] is False
            and run["memory_backpressure"]["overflow_detected"] is False
            and run["memory_backpressure"]["retry_limit_exceeded"] is False
            for run in runs
        )
    )
    gates = {
        "all_32_runs_inactive_and_authoritative": all_authoritative_inactive,
        "every_expected_transition_captured_once": transition_counts_exact,
        "v1_replay_exactly_reconciled": bool(replay["all_reconciled"]),
        "both_new_families_contain_both_labels": all(new_family_coverage.values()),
        "four_qualifying_components_at_every_candidate_block_size": bool(
            block_coverage and all(block_coverage.values())
        ),
        "no_backpressure_provenance_resource_censoring_or_partial_failure": operational_contract,
        "no_split_roles_assigned": all(
            block["leakage_connectivity"]["split_roles_assigned"] is False
            for block in blocks
        ),
        "no_protected_output_operation": True,
        "stack_ground_truth_never_accessed": True,
        "v2_aggregate_is_sole_persistent_destination": True,
    }
    passed = execution_status == "complete" and all(gates.values())
    return {
        "status": "pass" if passed else "fail",
        "all_gates_passed": passed,
        "gates": gates,
        "v1_replay_reconciliation": replay,
        "new_family_both_label_coverage": new_family_coverage,
        "candidate_block_coverage": block_coverage,
        "split_roles_assigned": False,
    }


def _drain_collector(collector: SetCCollector, accumulator: _RunAccumulator) -> None:
    accumulator.max_pending_rows = max(accumulator.max_pending_rows, collector.pending_rows)
    accumulator.max_pending_bytes = max(accumulator.max_pending_bytes, collector.pending_bytes)
    if collector.pending_rows > MAX_BUFFER_ROWS or collector.pending_bytes > MAX_BUFFER_BYTES:
        accumulator.overflow_detected = True
        raise PilotResourceError("Collector retained payload crossed its approved bound")
    while collector.pending_batches():
        batch = collector.pop_batch()
        accumulator.accept_batch(batch)
        del batch


def _provenance(
    config: Mapping[str, object],
    spec: PilotRunSpec,
    environment_identity: Mapping[str, object],
    source_context: Mapping[str, object],
) -> dict[str, object]:
    wind = WindContract.from_config(dict(config["wind"])).to_manifest()
    return {
        "set_id": "SetC",
        "experiment_id": PILOT_ID,
        "event_id": EVENT_ID,
        "scenario_id": spec.family.scenario_id,
        "scenario_family_id": spec.family.family_id,
        "run_id": spec.run_id,
        "grid_id": GRID_ID,
        "simulation_valid_mask_id": "phase8_simulation_valid_mask_v1",
        "mapped_building_mask_id": "phase8_mapped_building_mask_v1",
        "simulator_id": MODEL_FREE_SOURCE,
        "simulator_version": TEACHER_RUNNER_SCHEMA_VERSION,
        "wind": wind,
        "source_revision": source_context["source_revision"],
        "source_dirty": source_context["source_dirty"],
        "configuration_hash": canonical_metadata_hash(config),
        "input_hashes": source_context["input_hashes"],
        "authorization": SOURCE_ONLY_AUTHORIZATION,
        "authorization_reference": APPROVAL_REFERENCE,
        "transition_source_type": AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
        "seed": spec.seed,
        "rng_bit_generator": "PCG64",
        "wind_manifest": wind,
        "wind_weight": float(config["placeholder_transition"]["wind_weight"]),
        "simulation_valid_mask_sha256": environment_identity["simulation_valid_mask_sha256"],
        "mapped_building_mask_sha256": environment_identity["mapped_building_mask_sha256"],
        "environment_id": GRID_ID,
        "execution_id": f"{PILOT_ID}:{spec.run_id}",
        "capture_source_id": "model_free_teacher.step",
        "ignition_coordinate_system": "grid_row_col",
        "ignition_coordinates": spec.family.ignition_coordinates,
        "ignition_set_sha256": spec.family.ignition_set_sha256,
    }


def _aggregate_safe_teacher_contract(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Return the validated aggregate-only model-free teacher contract."""
    pilot = config["phase8_set_c_pilot"]
    configured_families = pilot["families"]
    return {
        "wind_manifest": dict(
            WindContract.from_config(dict(config["wind"])).to_manifest()
        ),
        "wind_weight": float(config["placeholder_transition"]["wind_weight"]),
        "placeholder_transition": dict(config["placeholder_transition"]),
        "flammability_weights": dict(config["flammability_weights"]),
        "scenario_families": [
            {
                "family_id": family["family_id"],
                "scenario_id": family["scenario_id"],
                "ignition_set_sha256": family["ignition_set_sha256"],
                "component_size": family["component_size"],
                "row_ceiling": family["row_ceiling"],
                "wind_manifest": dict(family["wind_manifest"]),
                "seeds": list(family["seeds"]),
            }
            for family in configured_families
        ],
    }


def _contract_manifest(config: Mapping[str, object]) -> dict[str, object]:
    return {
        "pilot_id": PILOT_ID,
        "transition_source": MODEL_FREE_SOURCE,
        "simulator_version": TEACHER_RUNNER_SCHEMA_VERSION,
        "safety_limit_timesteps": SAFETY_LIMIT_TIMESTEPS,
        "inactive_early_stop": True,
        "seeds_by_family": {
            family.family_id: list(family.seeds) for family in FAMILIES
        },
        "run_order": [spec.run_id for spec in RUN_MATRIX],
        "run_concurrency": 0,
        "sampling_policy": "all_eligible",
        "collector": {
            "batch_rows": BATCH_ROWS,
            "max_buffer_rows": MAX_BUFFER_ROWS,
            "max_buffer_bytes": MAX_BUFFER_BYTES,
            "backpressure_policy": RUNNER_OWNED_BACKPRESSURE_POLICY,
            "maximum_retry_cycles_per_observation": MAX_RETRY_CYCLES_PER_OBSERVATION,
        },
        "row_ceilings": {
            family.family_id: family.row_ceiling for family in FAMILIES
        },
        "total_row_ceiling": TOTAL_ROW_CEILING,
        "v1_replay_reconciliation": {
            run_id: {
                "rows": expected[0],
                "positive_1": expected[1],
                "negative_0": expected[2],
                "transition_count": expected[3],
                "termination_reason": TERMINATION_INACTIVE,
                "authoritative": True,
            }
            for run_id, expected in V1_REPLAY_EXPECTATIONS.items()
        },
        "v1_parent_evidence": {
            "pilot_id": "phase8_set_c_feasibility_pilot_v1",
            "payload_sha256": V1_AGGREGATE_PAYLOAD_SHA256,
            "file_sha256": V1_AGGREGATE_FILE_SHA256,
        },
        "candidate_block_sizes": [list(size) for size in CANDIDATE_BLOCK_SIZES],
        "candidate_block_origin": list(BLOCK_ORIGIN),
        "candidate_blocks_diagnostic_only": True,
        "final_block_dimensions_selected": False,
        "split_roles_assigned": False,
        "publisher_enabled": False,
        "approved_model_free_teacher_contract": _aggregate_safe_teacher_contract(
            config
        ),
        "forbidden_inputs": ["stack_ground_truth.tif"],
        "forbidden_outputs": [
            "row_level_dataset",
            "model",
            "checkpoint",
            "ca_raster",
            "split_manifest",
            "calibration_record",
            "manuscript",
        ],
    }


def _validate_no_row_payload(value: object) -> None:
    forbidden = {
        "features",
        "labels",
        "cell_rows",
        "cell_cols",
        "duplicate_group_ids",
        "spatial_block_ids",
        "ignition_coordinates",
        "duplicate_ids",
        "block_ids",
        "cell_ids",
        "feature_values",
        "row_level_data",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in forbidden:
                raise PilotContractError(f"Aggregate report contains forbidden key: {key}")
            _validate_no_row_payload(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_no_row_payload(item)


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def publish_aggregate_report(report: Mapping[str, object], destination: Path) -> tuple[Path, str]:
    """Atomically publish one canonical, self-hashed, non-overwriting JSON report."""
    if "payload_sha256" in report:
        raise PilotContractError("Caller must not pre-populate payload_sha256")
    _validate_no_row_payload(report)
    payload_hash = sha256(_canonical_json(report)).hexdigest()
    final_report = dict(report)
    final_report["payload_sha256"] = payload_hash
    payload = _canonical_json(final_report)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite pilot report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        loaded = json.loads(staging.read_text(encoding="utf-8"))
        loaded_hash = loaded.pop("payload_sha256", None)
        if loaded_hash != sha256(_canonical_json(loaded)).hexdigest():
            raise PilotContractError("Staged aggregate payload hash validation failed")
        os.link(staging, destination)
        return destination, payload_hash
    finally:
        staging.unlink(missing_ok=True)


def run_set_c_pilot(
    environment: Mapping[str, object],
    config: Mapping[str, object],
    *,
    destination: Path,
    source_context: Mapping[str, object],
    resource_probe: Callable[[Path], ResourceSnapshot] = capture_resource_snapshot,
    teacher_runner: Callable[..., object] = run_model_free_teacher,
    synthetic_test: bool = False,
) -> dict[str, object]:
    """Execute the approved matrix and publish only its aggregate report."""
    if not synthetic_test and teacher_runner is not run_model_free_teacher:
        raise PermissionError(
            "Production pilot must use the exact approved run_model_free_teacher"
        )
    if not synthetic_test and resource_probe is not capture_resource_snapshot:
        raise PermissionError(
            "Production pilot must use the exact approved resource telemetry"
        )
    _require_exact_contract(config)
    contract_manifest = _contract_manifest(config)
    production_destination = approved_destination().resolve()
    destination = Path(destination).resolve()
    if synthetic_test:
        if destination == production_destination:
            raise PermissionError("Synthetic tests cannot use the approved production destination")
    elif destination != production_destination:
        raise PermissionError("Only the approved aggregate destination is permitted")
    required_context = {
        "source_revision",
        "source_dirty",
        "source_hashes",
        "input_hashes",
        "generation_command",
        "environment_record",
    }
    if set(source_context) != required_context:
        raise PilotContractError("Source/input provenance context is incomplete or contains extras")
    if not isinstance(source_context["source_dirty"], bool):
        raise PilotContractError("source_dirty must be boolean")
    for field in ("source_hashes", "input_hashes"):
        hashes = source_context[field]
        if not isinstance(hashes, Mapping) or not hashes:
            raise PilotContractError(f"{field} must be a non-empty mapping")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        ):
            raise PilotContractError(f"{field} must contain SHA-256 digests")
        if any("ground_truth" in str(name).lower() for name in hashes):
            raise PermissionError("stack_ground_truth.tif must not be read or recorded")
    launch = preflight_pilot_launch(
        config,
        destination,
        resource_probe=resource_probe,
        synthetic_test=synthetic_test,
    )
    environment_identity = _validate_environment(environment, synthetic=synthetic_test)
    pilot_start = time.monotonic()
    aggregate = _PilotAccumulator()
    resource_samples = [launch]
    status = "complete"
    failure: dict[str, object] | None = None
    current_run: _RunAccumulator | None = None
    current_termination: object | None = None
    current_run_start: float | None = None

    try:
        for spec in RUN_MATRIX:
            run = _RunAccumulator(spec)
            current_run = run
            current_termination = None
            run_start = time.monotonic()
            current_run_start = run_start
            if time.monotonic() - pilot_start > PILOT_RUNTIME_LIMIT_SECONDS:
                raise PilotResourceError("Pilot exceeded 5400 seconds before the next run")
            before = resource_probe(destination)
            resource_samples.append(before)
            run.resource_samples.append(before)
            _check_launch(before, destination)
            run_config = _run_config(config, spec)
            collector = SetCCollector(environment, run_config)

            def drain_for_backpressure(active_collector: SetCCollector) -> None:
                run.backpressure_drain_count += 1
                run.current_retry_cycles += 1
                run.max_retry_cycles = max(
                    run.max_retry_cycles, run.current_retry_cycles
                )
                if run.current_retry_cycles > MAX_RETRY_CYCLES_PER_OBSERVATION:
                    run.retry_limit_exceeded = True
                    raise PilotContractError(
                        "Backpressure retries exceeded the approved bound"
                    )
                before_rows = active_collector.pending_rows
                _drain_collector(active_collector, run)
                if before_rows <= 0:
                    run.no_progress_detected = True
                    raise PilotContractError("Backpressure drain had no accepted rows")

            def observe(observation: TransitionObservation) -> None:
                if observation.timestep_t in run.observation_rows:
                    raise PilotContractError("Duplicate pilot observation timestep")
                _drain_collector(collector, run)
                expected_rows = int(np.count_nonzero(observation.eligible_mask_t))
                observed = run.observation_rows.setdefault(
                    observation.timestep_t, [0, 0, 0]
                )
                if observed[0] != expected_rows:
                    raise PilotContractError("Eligible and collected row counts do not match")
                if run.rows > spec.family.row_ceiling:
                    raise PilotContractError("Per-run row ceiling exceeded")
                if aggregate.total_rows + run.rows > TOTAL_ROW_CEILING:
                    raise PilotContractError("Pilot total row ceiling exceeded")
                snapshot = resource_probe(destination)
                resource_samples.append(snapshot)
                run.resource_samples.append(snapshot)
                _check_runtime(snapshot, time.monotonic() - run_start)
                if collector.live_full_grid_observation_count > 1:
                    raise PilotResourceError("More than one full-grid observation is live")
                run.current_retry_cycles = 0

            provenance = _provenance(
                run_config, spec, environment_identity, source_context
            )
            try:
                termination = teacher_runner(
                    dict(environment),
                    run_config,
                    ignition_points=spec.family.ignition_coordinates,
                    transition_provenance=provenance,
                    transition_observer=observe,
                    set_c_collector=collector,
                    backpressure_drain=drain_for_backpressure,
                )
            except ModelFreeTeacherRunError as exc:
                current_termination = exc.termination_record
                raise
            current_termination = termination
            sealed = collector.finalize(termination)
            if collector.pending_rows != 0 or collector.pending_bytes != 0:
                raise PilotContractError("Row payload remained after streaming aggregation")
            if sealed.released_row_count != run.rows or sealed.row_count != run.rows:
                raise PilotContractError("Collector released-row ledger does not match")
            if sealed.observation_count != len(run.observation_rows):
                raise PilotContractError("Collector observation ledger does not match")
            termination_expected = int(getattr(termination, "expected_transition_count"))
            termination_captured = int(getattr(termination, "captured_transition_count"))
            if termination_expected != termination_captured or termination_captured != sealed.observation_count:
                raise PilotContractError("Termination transition counts do not match")
            termination_reason = getattr(termination, "termination_reason", None)
            completeness = getattr(termination, "completeness_status", None)
            authoritative = getattr(termination, "authoritative", None)
            final_ignited = int(getattr(termination, "final_ignited_count"))
            final_blazing = int(getattr(termination, "final_blazing_count"))
            if termination_reason == TERMINATION_INACTIVE:
                if (
                    completeness != COMPLETENESS_COMPLETE
                    or authoritative is not True
                    or final_ignited != 0
                    or final_blazing != 0
                ):
                    raise PilotContractError("Inactive termination contract mismatch")
            elif termination_reason == TERMINATION_SAFETY_LIMIT:
                if (
                    completeness != COMPLETENESS_CENSORED
                    or authoritative is not False
                    or final_ignited + final_blazing <= 0
                ):
                    raise PilotContractError("Safety-limit censoring contract mismatch")
            else:
                raise PilotContractError("Unexpected teacher termination classification")
            if (
                int(getattr(termination, "final_timestep")) != sealed.final_timestep_t1
                or sealed.final_timestep_t1 > SAFETY_LIMIT_TIMESTEPS
                or tuple(getattr(termination, "ignition_coordinates"))
                != spec.family.ignition_coordinates
                or getattr(termination, "ignition_set_sha256")
                != spec.family.ignition_set_sha256
            ):
                raise PilotContractError("Termination identity or timestep mismatch")
            elapsed = time.monotonic() - run_start
            after = resource_probe(destination)
            resource_samples.append(after)
            run.resource_samples.append(after)
            _check_runtime(after, elapsed)
            _require_v1_replay_reconciliation(run, termination)
            aggregate.merge(
                run,
                termination,
                elapsed,
                configuration_hash=canonical_metadata_hash(run_config),
                run_identity_sha256=sealed.run_identity_sha256,
                provenance_sha256=sealed.provenance_sha256,
            )
            merged = resource_probe(destination)
            resource_samples.append(merged)
            run.resource_samples.append(merged)
            _check_runtime(merged, time.monotonic() - run_start)
            if time.monotonic() - pilot_start > PILOT_RUNTIME_LIMIT_SECONDS:
                raise PilotResourceError(
                    "Pilot exceeded 5400 seconds after aggregate merge"
                )
            current_run = None
            current_termination = None
            current_run_start = None
    except Exception as exc:
        status = "failed"
        aggregate.invalidate_authoritative_evidence()
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if current_run is not None:
            failure["run"] = current_run.failure_evidence(
                termination=current_termination,
                elapsed=(
                    time.monotonic() - current_run_start
                    if current_run_start is not None
                    else 0.0
                ),
                failure=exc,
            )

    sections = aggregate.aggregate_sections()
    available = [sample.available_memory_bytes for sample in resource_samples]
    rss = [sample.process_rss_bytes for sample in resource_samples]
    free_disk = [sample.free_disk_bytes for sample in resource_samples]
    acceptance = _acceptance_summary(status, sections)
    report: dict[str, object] = {
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "pilot_id": PILOT_ID,
        "status": status,
        "authorization_reference": APPROVAL_REFERENCE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract_manifest,
        "provenance": {
            "source_revision": source_context["source_revision"],
            "source_dirty": source_context["source_dirty"],
            "source_hashes": source_context["source_hashes"],
            "input_hashes": source_context["input_hashes"],
            "generation_command": source_context["generation_command"],
            "environment_record": source_context["environment_record"],
            "grid_and_domain": environment_identity,
        },
        "aggregate": sections,
        "acceptance": acceptance,
        "resource_summary": {
            "launch_memory_floor_bytes": LAUNCH_MEMORY_FLOOR_BYTES,
            "runtime_memory_floor_bytes": RUNTIME_MEMORY_FLOOR_BYTES,
            "rss_ceiling_bytes": RSS_CEILING_BYTES,
            "disk_floor_bytes": DISK_FLOOR_BYTES,
            "minimum_available_memory_bytes": min(available),
            "maximum_process_rss_bytes": max(rss),
            "minimum_free_disk_bytes": min(free_disk),
            "sample_count": len(resource_samples),
            "elapsed_seconds": time.monotonic() - pilot_start,
        },
        "integrity": {
            "row_level_payload_retained": False,
            "row_level_payload_published": False,
            "publisher_enabled": False,
            "model_accessed": False,
            "stack_ground_truth_accessed": False,
            "split_roles_assigned": False,
            "training_performed": False,
            "calibration_performed": False,
            "raster_or_checkpoint_written": False,
            "count_contract_passed": status == "complete",
            "provenance_contract_passed": status == "complete",
        },
        "failure": failure,
    }
    if status == "complete" and len(aggregate.runs) != len(RUN_MATRIX):
        raise PilotContractError("Complete pilot report is missing runs")
    path, payload_hash = publish_aggregate_report(report, destination)
    return {
        "path": path,
        "payload_sha256": payload_hash,
        "status": status,
        "run_count": len(aggregate.runs),
        "acceptance_status": acceptance["status"],
    }


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "APPROVAL_REFERENCE",
    "APPROVED_RELATIVE_DESTINATION",
    "FAMILIES",
    "PILOT_ID",
    "RUN_MATRIX",
    "PilotContractError",
    "PilotResourceError",
    "ResourceSnapshot",
    "activate_approved_pilot_config",
    "approved_destination",
    "capture_resource_snapshot",
    "preflight_pilot_launch",
    "publish_aggregate_report",
    "run_set_c_pilot",
]
