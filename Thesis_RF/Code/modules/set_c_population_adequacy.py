"""Disabled aggregate-only Phase 9 population-adequacy harness."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
import yaml

from .model_free_teacher import (
    COMPLETENESS_COMPLETE,
    TERMINATION_INACTIVE,
    TEACHER_RUNNER_SCHEMA_VERSION,
    run_model_free_teacher,
)
from .set_c_collector import (
    RUNNER_OWNED_BACKPRESSURE_POLICY,
    SetCCollector,
    _sha256_rows,
)
from .set_c_pilot import (
    EXPECTED_MAPPED_BUILDING_SHA256,
    EXPECTED_SIMULATION_VALID_SHA256,
    ResourceSnapshot,
    capture_resource_snapshot,
    publish_aggregate_report,
    _validate_environment as validate_phase8_environment,
)
from .set_c_site_inventory import (
    EXPECTED_ENVIRONMENT_FILES,
    EXPECTED_RASTER_SHA256,
)
from .transition_observer import (
    AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
    SOURCE_ONLY_AUTHORIZATION,
    TransitionObservation,
    canonical_metadata_hash,
)
from .wind_convention import WindContract


EXPANSION_ID = "phase9_set_c_population_adequacy_edge_diagnostic_v2"
AGGREGATE_SCHEMA_VERSION = "phase9_set_c_population_adequacy_aggregate.v2"
APPROVAL_REFERENCE = "Phase 9 Gate 6 edge diagnostic source and synthetic tests only; production execution requires separate one-run authorization"
APPROVED_RELATIVE_DESTINATION = Path(
    "output/phase9/phase9_set_c_population_adequacy_edge_diagnostic_v2.aggregate.json"
)
APPROVED_INVENTORY_RELATIVE_PATH = Path(
    "output/phase9/phase9_set_c_site_inventory_v2.aggregate.json"
)
APPROVED_INVENTORY_PAYLOAD_SHA256 = (
    "25a0a27ac81db42a38eaf339b101fbc6c41c44cd60c1525cc5686112332158cc"
)
APPROVED_INVENTORY_FILE_SHA256 = (
    "26fb06f324b7197a6b6cb5242d7b09012400451e5467d3bdb6fdb30e5dc1de75"
)
APPROVED_CANDIDATE_SHA256 = (
    "7336d5f124ce8164275855f10f54eeae4f13c3af81f20ce81f2014a3c56012de"
)
APPROVED_PRIMARY_SHA256 = (
    "d97e3cb7598dbc9c4b4cc53398c0d0daad0a5236f2d59e357db3cd8cf86a7d33"
)
APPROVED_RESERVE_SHA256 = (
    "7459ba860d583c6790d033bd3eaf48c8fce7b54ad200a0d6e1c8560581fa79e0"
)

MODEL_FREE_SOURCE = "model-free-fire-automata"
EVENT_ID = "simulated_case_study_no_observed_event"
GRID_ID = "lapu_lapu_epsg32651_3m_5489x6896_v1"
BLOCK_SIZE = (128, 128)
BLOCK_ORIGIN = (0, 0)
WIND_DIRECTIONS = tuple(range(0, 360, 45))
WIND_CONDITIONS = ((0, 0),) + tuple(
    (speed, direction) for speed in (10, 20) for direction in WIND_DIRECTIONS
)
REPLICATES = (1, 2)
SAFETY_LIMIT_TIMESTEPS = 300
PER_RUN_ROW_CEILING = 1_800_000
TOTAL_ROW_CEILING = 1_418_667_000
BATCH_ROWS = 16_384
MAX_BUFFER_ROWS = 932_864
MAX_BUFFER_BYTES = 176_311_296
RETRY_LIMIT = 58
LAUNCH_MEMORY_FLOOR_BYTES = 6_442_450_944
RUNTIME_MEMORY_FLOOR_BYTES = 2_147_483_648
RSS_CEILING_BYTES = 6_442_450_944
FREE_DISK_FLOOR_BYTES = 2_147_483_648
RUN_RUNTIME_LIMIT_SECONDS = 900.0
EXPANSION_RUNTIME_LIMIT_SECONDS = 21_600.0
EXPECTED_FAMILY_COUNT = 26
EXPECTED_RUN_COUNT = 884
SQLITE_CACHE_KIB = 8_192
PRODUCTION_SOURCE_RELATIVE_PATHS = (
    "config/default_experiment.yaml",
    "modules/automata_engine.py",
    "modules/ca_state.py",
    "modules/data_loader.py",
    "modules/feature_pipeline.py",
    "modules/model_free_teacher.py",
    "modules/set_c_collector.py",
    "modules/set_c_pilot.py",
    "modules/set_c_site_inventory.py",
    "modules/set_c_population_adequacy.py",
    "modules/transition_observer.py",
    "modules/wind_convention.py",
    "orchestrator.py",
    "run_set_c_population_adequacy.py",
)
PRODUCTION_GENERATION_COMMAND = (
    ".venv\\Scripts\\python.exe -B "
    "Thesis_RF\\Code\\run_set_c_population_adequacy.py "
    "--config Thesis_RF\\Code\\config\\default_experiment.yaml "
    "--execute-approved-edge-diagnostic"
)

ROLE_REQUIREMENTS = (
    ("train", 8, 200, 800),
    ("hyperparameter_validation", 5, 100, 400),
    ("calibration", 5, 100, 400),
    ("final_test", 5, 100, 400),
)

FORBIDDEN_REPORT_KEYS = {
    "features",
    "predictors",
    "labels",
    "cell_rows",
    "cell_cols",
    "cell_inventory",
    "duplicate_group_ids",
    "spatial_block_ids",
    "ignition_coordinates",
    "split_role",
    "split_assignments",
    "model",
    "estimator",
    "ground_truth",
    "raster",
    "checkpoint",
}


class PopulationAdequacyError(RuntimeError):
    """Raised when a Gate 5 contract or integrity condition fails."""


class PopulationResourceError(PopulationAdequacyError):
    """Raised when a Gate 5 resource bound is crossed."""


@dataclass(frozen=True, slots=True)
class PopulationFamily:
    order: int
    family_id: str
    material_class: int
    component_ordinal: int
    component_size: int
    ignition_row: int
    ignition_col: int
    ignition_set_sha256: str
    block_footprint_sha256: str
    source_tier: str

    @property
    def ignition_coordinates(self) -> tuple[tuple[int, int], ...]:
        return ((self.ignition_row, self.ignition_col),)


@dataclass(frozen=True, slots=True)
class PopulationRunSpec:
    order: int
    family: PopulationFamily
    wind_condition_ordinal: int
    speed_kmh: int
    direction_deg: int
    replicate_ordinal: int
    scenario_id: str
    seed: int
    run_id: str


@dataclass(frozen=True, slots=True)
class PopulationRunResult:
    run_id: str
    scenario_id: str
    family_id: str
    seed: int
    rows: int
    batch_count: int
    observation_count: int
    released_rows: int
    pending_rows: int
    pending_bytes: int
    termination_reason: str
    completeness_status: str
    authoritative: bool
    final_timestep: int
    final_ignited_count: int
    final_blazing_count: int
    expected_transition_count: int
    captured_transition_count: int
    configuration_sha256: str
    run_identity_sha256: str
    provenance_sha256: str
    ignition_set_sha256: str
    estimator_accessed: bool = False
    unexpected_outputs: tuple[str, ...] = ()
    max_live_full_grid_observations: int = 1
    retry_limit_exceeded: bool = False


RunExecutor = Callable[
    [PopulationRunSpec, Callable[[object], None]], PopulationRunResult
]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_temp_descendant(path: Path, field: str) -> Path:
    resolved = path.resolve()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    try:
        resolved.relative_to(temporary_root)
    except ValueError as exc:
        raise PermissionError(f"Synthetic {field} must be inside the system temp directory") from exc
    if resolved == temporary_root:
        raise PermissionError(f"Synthetic {field} must be below the system temp directory")
    return resolved


def validate_approved_environment_config(config: Mapping[str, object]) -> None:
    environment = config.get("environment")
    if not isinstance(environment, Mapping):
        raise PopulationAdequacyError("Environment configuration is required")
    code_root = Path(__file__).resolve().parents[1]
    raster_dir = Path(str(environment.get("raster_dir", "")))
    if not raster_dir.is_absolute():
        raster_dir = code_root / raster_dir
    expected_raster_dir = (code_root / "processedData" / "raster" / "raster").resolve()
    expected = {**EXPECTED_ENVIRONMENT_FILES, "nodata_value": -9999}
    if (
        raster_dir.resolve() != expected_raster_dir
        or {key: environment.get(key) for key in expected} != expected
        or set(environment) != {"raster_dir", *expected}
    ):
        raise PopulationAdequacyError("Environmental raster filename mapping drifted")
    if any(
        "ground_truth" in str(key).lower() or "ground_truth" in str(value).lower()
        for key, value in environment.items()
    ):
        raise PermissionError("Ground-truth inputs are forbidden")


def approved_destination() -> Path:
    return Path(__file__).resolve().parents[1] / APPROVED_RELATIVE_DESTINATION


def approved_inventory_path() -> Path:
    return Path(__file__).resolve().parents[1] / APPROVED_INVENTORY_RELATIVE_PATH


def preflight_population_launch(
    config: Mapping[str, object],
    destination: Path,
    *,
    resource_probe: Callable[[Path], ResourceSnapshot] = capture_resource_snapshot,
) -> ResourceSnapshot:
    """Fail before raster loading when contract, manifest, output, or resources drift."""
    _require_exact_contract(config)
    validate_approved_environment_config(config)
    destination = destination.resolve()
    if destination != approved_destination().resolve():
        raise PermissionError("Only the approved aggregate destination is permitted")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite adequacy report: {destination}")
    load_approved_families()
    snapshot = resource_probe(destination)
    _check_resources(snapshot, launch=True)
    return snapshot


def activate_approved_population_config(
    config: Mapping[str, object],
) -> dict[str, Any]:
    """Enable only an in-memory copy; the checked-in configuration stays off."""
    active = deepcopy(dict(config))
    section = active.get("phase9_set_c_population_adequacy")
    if not isinstance(section, dict):
        raise PopulationAdequacyError(
            "phase9_set_c_population_adequacy configuration is required"
        )
    if section.get("enabled") is not False:
        raise PopulationAdequacyError(
            "The checked-in population adequacy harness must remain disabled"
        )
    section["enabled"] = True
    _require_exact_contract(active)
    return active


def _require_exact_contract(config: Mapping[str, object]) -> None:
    section = config.get("phase9_set_c_population_adequacy")
    if not isinstance(section, Mapping) or section.get("enabled") is not True:
        raise PermissionError("The population adequacy harness is disabled by default")
    expected = {
        "expansion_id": EXPANSION_ID,
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "authorization_reference": APPROVAL_REFERENCE,
        "aggregate_destination": str(APPROVED_RELATIVE_DESTINATION).replace("\\", "/"),
        "inventory_path": str(APPROVED_INVENTORY_RELATIVE_PATH).replace("\\", "/"),
        "inventory_payload_sha256": APPROVED_INVENTORY_PAYLOAD_SHA256,
        "inventory_file_sha256": APPROVED_INVENTORY_FILE_SHA256,
        "ordered_candidate_sha256": APPROVED_CANDIDATE_SHA256,
        "ordered_primary_sha256": APPROVED_PRIMARY_SHA256,
        "ordered_reserve_sha256": APPROVED_RESERVE_SHA256,
        "existing_family_count": 6,
        "primary_family_count": 20,
        "wind_condition_count": 17,
        "replicates_per_condition": 2,
        "run_count": EXPECTED_RUN_COUNT,
        "safety_limit_timesteps": SAFETY_LIMIT_TIMESTEPS,
        "block_size": list(BLOCK_SIZE),
        "block_origin": list(BLOCK_ORIGIN),
        "per_run_row_ceiling": PER_RUN_ROW_CEILING,
        "total_row_ceiling": TOTAL_ROW_CEILING,
        "batch_rows": BATCH_ROWS,
        "max_buffer_rows": MAX_BUFFER_ROWS,
        "max_buffer_bytes": MAX_BUFFER_BYTES,
        "retry_limit": RETRY_LIMIT,
        "launch_memory_floor_bytes": LAUNCH_MEMORY_FLOOR_BYTES,
        "runtime_memory_floor_bytes": RUNTIME_MEMORY_FLOOR_BYTES,
        "rss_ceiling_bytes": RSS_CEILING_BYTES,
        "free_disk_floor_bytes": FREE_DISK_FLOOR_BYTES,
        "run_runtime_limit_seconds": 900,
        "expansion_runtime_limit_seconds": 21600,
    }
    for key, value in expected.items():
        if section.get(key) != value:
            raise PopulationAdequacyError(
                f"phase9_set_c_population_adequacy.{key} must equal {value!r}"
            )
    dataset = config.get("dataset_generation", {})
    if not isinstance(dataset, Mapping) or dataset.get("artifact_write_enabled") is not False:
        raise PermissionError("Dataset artifact writes must remain disabled")
    publisher = dataset.get("set_c_publisher", {})
    if not isinstance(publisher, Mapping) or publisher.get("enabled") is not False:
        raise PermissionError("The row-level Set C publisher must remain disabled")


def _family_from_record(
    record: Mapping[str, object], *, order: int, source_tier: str
) -> PopulationFamily:
    family_key = "existing_family_id" if source_tier == "existing" else "proposed_family_id"
    material_key = "material_class" if source_tier == "existing" else "assigned_class"
    return PopulationFamily(
        order=order,
        family_id=str(record[family_key]),
        material_class=int(record[material_key]),
        component_ordinal=int(record["component_ordinal"]),
        component_size=int(record["component_size"]),
        ignition_row=int(record["ignition_row"]),
        ignition_col=int(record["ignition_col"]),
        ignition_set_sha256=str(record["ignition_set_sha256"]),
        block_footprint_sha256=str(record["block_footprint_sha256"]),
        source_tier=source_tier,
    )


def load_approved_families(path: Path | None = None) -> tuple[PopulationFamily, ...]:
    """Load and hash-bind the exact approved six existing and twenty primaries."""
    inventory = (path or approved_inventory_path()).resolve()
    if _sha256_file(inventory) != APPROVED_INVENTORY_FILE_SHA256:
        raise PopulationAdequacyError("Approved inventory whole-file hash mismatch")
    report = json.loads(inventory.read_text(encoding="utf-8"))
    payload_hash = report.pop("payload_sha256", None)
    if payload_hash != APPROVED_INVENTORY_PAYLOAD_SHA256:
        raise PopulationAdequacyError("Approved inventory payload identity mismatch")
    if sha256(_canonical_json(report)).hexdigest() != payload_hash:
        raise PopulationAdequacyError("Approved inventory self-hash mismatch")
    if report.get("status") != "complete" or report.get("contract_status") != "pass":
        raise PopulationAdequacyError("Approved inventory is not complete and passing")
    candidates = list(report.get("candidates", ()))
    primaries = [item for item in candidates if item.get("candidate_tier") == "primary"]
    reserves = [item for item in candidates if item.get("candidate_tier") == "reserve"]
    if sha256(_canonical_json(candidates)).hexdigest() != APPROVED_CANDIDATE_SHA256:
        raise PopulationAdequacyError("Approved ordered candidate manifest drifted")
    if sha256(_canonical_json(primaries)).hexdigest() != APPROVED_PRIMARY_SHA256:
        raise PopulationAdequacyError("Approved ordered primary manifest drifted")
    if sha256(_canonical_json(reserves)).hexdigest() != APPROVED_RESERVE_SHA256:
        raise PopulationAdequacyError("Approved ordered reserve manifest drifted")
    existing = list(report.get("existing_sites", ()))
    if len(existing) != 6 or len(primaries) != 20 or len(reserves) != 20:
        raise PopulationAdequacyError("Approved family counts drifted")
    expected_primary_order = sorted(
        primaries,
        key=lambda item: (
            int(item["assigned_class"]),
            int(item["rank"]),
            int(item["component_ordinal"]),
        ),
    )
    if primaries != expected_primary_order:
        raise PopulationAdequacyError("Approved primary family order drifted")
    records = [(item, "existing") for item in existing] + [
        (item, "primary") for item in primaries
    ]
    families = tuple(
        _family_from_record(item, order=index, source_tier=tier)
        for index, (item, tier) in enumerate(records, start=1)
    )
    if len({family.family_id for family in families}) != EXPECTED_FAMILY_COUNT:
        raise PopulationAdequacyError("Family identities are not unique")
    return families


def build_run_matrix(
    families: Sequence[PopulationFamily],
) -> tuple[PopulationRunSpec, ...]:
    """Return the D-015 family, wind, replicate ordering and identities."""
    if len(families) != EXPECTED_FAMILY_COUNT:
        raise PopulationAdequacyError("The controlled population requires 26 families")
    if tuple(family.order for family in families) != tuple(range(1, 27)):
        raise PopulationAdequacyError("Family ordinals must be contiguous from one")
    specs: list[PopulationRunSpec] = []
    for family in families:
        for wind_ordinal, (speed, direction) in enumerate(WIND_CONDITIONS, start=1):
            scenario_id = f"{family.family_id}__w{direction:03d}_s{speed:02d}"
            for replicate in REPLICATES:
                seed = (
                    9_000_000
                    + 10_000 * family.order
                    + 100 * wind_ordinal
                    + replicate
                )
                specs.append(
                    PopulationRunSpec(
                        order=len(specs) + 1,
                        family=family,
                        wind_condition_ordinal=wind_ordinal,
                        speed_kmh=speed,
                        direction_deg=direction,
                        replicate_ordinal=replicate,
                        scenario_id=scenario_id,
                        seed=seed,
                        run_id=f"{scenario_id}__rep{replicate}__seed_{seed}",
                    )
                )
    if len(specs) != EXPECTED_RUN_COUNT or len({spec.run_id for spec in specs}) != EXPECTED_RUN_COUNT:
        raise PopulationAdequacyError("The controlled population must contain 884 unique runs")
    return tuple(specs)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


class _IdentityStore:
    """Exact file-backed identity aggregates with a fixed SQLite page cache."""

    def __init__(self, directory: Path | None = None, *, checkpoint: Path | None = None,
                 resume: bool = False) -> None:
        if checkpoint is None:
            descriptor, name = tempfile.mkstemp(
                prefix="set-c-adequacy-", suffix=".sqlite3", dir=directory
            )
            os.close(descriptor)
            self.path = Path(name)
        else:
            self.path = checkpoint
            if resume:
                if not self.path.is_file():
                    raise PopulationAdequacyError("Checkpoint is missing")
            else:
                with self.path.open("xb"):
                    pass
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(f"PRAGMA cache_size=-{SQLITE_CACHE_KIB}")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA journal_mode=DELETE")
        self.connection.execute("PRAGMA synchronous=FULL")
        if not resume:
            self.connection.execute(
            """
            CREATE TABLE identities (
                kind INTEGER NOT NULL,
                identity BLOB NOT NULL,
                row_count INTEGER NOT NULL,
                class_mask INTEGER NOT NULL,
                family_mask INTEGER NOT NULL,
                PRIMARY KEY (kind, identity)
            ) WITHOUT ROWID
            """
            )
            if checkpoint is not None:
                self.connection.execute("CREATE TABLE checkpoint_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
                self.connection.execute("CREATE TABLE completed_runs (run_order INTEGER PRIMARY KEY, payload TEXT NOT NULL, digest TEXT NOT NULL)")
        self.connection.commit()
        self.closed = False
        self.checkpoint = checkpoint is not None

    def merge(
        self,
        kind: int,
        identities: np.ndarray,
        labels: np.ndarray,
        family_order: int,
    ) -> None:
        unique, inverse, counts = np.unique(
            identities.astype("S64", copy=False),
            return_inverse=True,
            return_counts=True,
        )
        masks = np.zeros(unique.size, dtype=np.uint8)
        np.bitwise_or.at(masks, inverse, np.where(labels == 0, 1, 2).astype(np.uint8))
        family_mask = 1 << (family_order - 1)
        self.connection.executemany(
            """
            INSERT INTO identities(kind, identity, row_count, class_mask, family_mask)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(kind, identity) DO UPDATE SET
                row_count = row_count + excluded.row_count,
                class_mask = class_mask | excluded.class_mask,
                family_mask = family_mask | excluded.family_mask
            """,
            (
                (kind, bytes(identity), int(count), int(mask), family_mask)
                for identity, count, mask in zip(unique, counts, masks, strict=True)
            ),
        )

    def commit(self) -> None:
        self.connection.commit()

    def family_masks(self, kind: int):
        for (mask,) in self.connection.execute(
            "SELECT family_mask FROM identities WHERE kind = ?", (kind,)
        ):
            yield int(mask)

    def summary(self, kind: int) -> dict[str, int]:
        unique, repeated, conflicting = self.connection.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(row_count > 1), 0),
                   COALESCE(SUM(class_mask = 3), 0)
            FROM identities WHERE kind = ?
            """,
            (kind,),
        ).fetchone()
        return {
            "unique_group_count": int(unique),
            "repeated_group_count": int(repeated),
            "conflicting_label_group_count": int(conflicting),
        }

    def close(self, *, keep_checkpoint: bool = False) -> None:
        if self.closed:
            return
        try:
            self.connection.rollback()
            self.connection.close()
        finally:
            if not (keep_checkpoint and self.checkpoint):
                for suffix in ("", "-journal", "-wal", "-shm"):
                    Path(f"{self.path}{suffix}").unlink(missing_ok=True)
            self.closed = True


class _AggregateAccumulator:
    """Consume complete bounded batches and retain aggregate identities only."""

    def __init__(
        self,
        families: Sequence[PopulationFamily],
        *,
        store_directory: Path | None = None,
        checkpoint: Path | None = None,
        resume: bool = False,
    ) -> None:
        self.families = {family.family_id: family for family in families}
        self.family_counts: dict[str, list[int]] = {
            family.family_id: [0, 0] for family in families
        }
        self.identities = _IdentityStore(store_directory, checkpoint=checkpoint, resume=resume)
        self.runs: list[dict[str, object]] = []
        self.total_rows = 0
        self.total_positive = 0
        self.total_negative = 0
        self.current_spec: PopulationRunSpec | None = None
        self.current_rows = 0
        self.current_positive = 0
        self.current_negative = 0
        self.current_batches = 0

    def begin(self, spec: PopulationRunSpec) -> None:
        if self.current_spec is not None:
            raise PopulationAdequacyError("More than one run is live")
        self.current_spec = spec
        self.current_rows = self.current_positive = self.current_negative = 0
        self.current_batches = 0

    def consume(self, batch: object) -> None:
        spec = self.current_spec
        if spec is None:
            raise PopulationAdequacyError("A batch arrived without an active run")
        labels = np.asarray(getattr(batch, "labels", None))
        duplicates = np.asarray(getattr(batch, "duplicate_group_ids", None))
        blocks = np.asarray(getattr(batch, "spatial_block_ids", None))
        row_count = int(getattr(batch, "row_count", labels.size))
        if labels.dtype != np.int8 or labels.ndim != 1:
            raise PopulationAdequacyError("Batch labels must be one-dimensional np.int8")
        if row_count <= 0 or row_count > BATCH_ROWS:
            raise PopulationAdequacyError("Batch row count crossed the approved bound")
        if labels.size != row_count or duplicates.shape != (row_count,) or blocks.shape != (row_count,):
            raise PopulationAdequacyError("Batch aggregate arrays are not count-matched")
        if not np.all((labels == 0) | (labels == 1)):
            raise PopulationAdequacyError("Batch labels must use the binary Set C target")
        positive = int(np.count_nonzero(labels))
        negative = row_count - positive
        self.current_rows += row_count
        self.current_positive += positive
        self.current_negative += negative
        self.current_batches += 1
        if self.current_rows > PER_RUN_ROW_CEILING:
            raise PopulationAdequacyError("Per-run row ceiling exceeded")
        if self.total_rows + self.current_rows > TOTAL_ROW_CEILING:
            raise PopulationAdequacyError("Total row ceiling exceeded")
        # V2 aggregate kind 0 means exact predictor collisions, never record identity.
        self.identities.merge(0, _sha256_rows(batch.features), labels, spec.family.order)
        self.identities.merge(1, blocks, labels, spec.family.order)

    def finish(self, spec: PopulationRunSpec, result: PopulationRunResult, elapsed: float) -> None:
        if self.current_spec != spec:
            raise PopulationAdequacyError("Run completion order drifted")
        checks = (
            result.run_id == spec.run_id,
            result.scenario_id == spec.scenario_id,
            result.family_id == spec.family.family_id,
            result.seed == spec.seed,
            result.rows == self.current_rows == result.released_rows,
            result.batch_count == self.current_batches,
            result.pending_rows == 0,
            result.pending_bytes == 0,
            result.termination_reason == TERMINATION_INACTIVE,
            result.completeness_status == COMPLETENESS_COMPLETE,
            result.authoritative is True,
            result.final_ignited_count == 0,
            result.final_blazing_count == 0,
            result.expected_transition_count
            == result.captured_transition_count
            == result.observation_count,
            0 < result.final_timestep <= SAFETY_LIMIT_TIMESTEPS,
            result.ignition_set_sha256 == spec.family.ignition_set_sha256,
            result.estimator_accessed is False,
            not result.unexpected_outputs,
            result.max_live_full_grid_observations <= 1,
            result.retry_limit_exceeded is False,
            elapsed <= RUN_RUNTIME_LIMIT_SECONDS,
        )
        hashes = (
            result.configuration_sha256,
            result.run_identity_sha256,
            result.provenance_sha256,
        )
        if not all(checks) or any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise PopulationAdequacyError(
                f"Run {spec.run_id} is failed, partial, censored, count-mismatched, or identity-invalid"
            )
        self.total_rows += self.current_rows
        self.total_positive += self.current_positive
        self.total_negative += self.current_negative
        family_counts = self.family_counts[spec.family.family_id]
        family_counts[0] += self.current_negative
        family_counts[1] += self.current_positive
        self.runs.append(
            {
                "order": spec.order,
                "run_id": spec.run_id,
                "scenario_family_id": spec.family.family_id,
                "scenario_id": spec.scenario_id,
                "seed": spec.seed,
                "wind_condition_ordinal": spec.wind_condition_ordinal,
                "replicate_ordinal": spec.replicate_ordinal,
                "rows": self.current_rows,
                "class_counts": {
                    "negative_0": self.current_negative,
                    "positive_1": self.current_positive,
                },
                "batch_count": self.current_batches,
                "observation_count": result.observation_count,
                "termination": {
                    "reason": result.termination_reason,
                    "completeness_status": result.completeness_status,
                    "authoritative": result.authoritative,
                    "final_timestep": result.final_timestep,
                    "expected_transition_count": result.expected_transition_count,
                    "captured_transition_count": result.captured_transition_count,
                },
                "configuration_sha256": result.configuration_sha256,
                "run_identity_sha256": result.run_identity_sha256,
                "provenance_sha256": result.provenance_sha256,
                "runtime_seconds": elapsed,
            }
        )
        self.current_spec = None

    def leakage_components(self) -> list[dict[str, object]]:
        union = _UnionFind()
        for family_id in self.families:
            union.find(family_id)
        ordered_families = tuple(self.families)
        for kind in (0, 1):
            for mask in self.identities.family_masks(kind):
                ordered = [
                    family_id
                    for index, family_id in enumerate(ordered_families)
                    if mask & (1 << index)
                ]
                for family_id in ordered[1:]:
                    union.union(ordered[0], family_id)
        groups: dict[str, set[str]] = defaultdict(set)
        for family_id in self.families:
            groups[union.find(family_id)].add(family_id)
        components = []
        for family_ids in sorted(tuple(sorted(group)) for group in groups.values()):
            negative = sum(self.family_counts[item][0] for item in family_ids)
            positive = sum(self.family_counts[item][1] for item in family_ids)
            materials = sorted({self.families[item].material_class for item in family_ids})
            components.append(
                {
                    "component_sha256": canonical_metadata_hash(
                        {"block_size": BLOCK_SIZE, "scenario_family_ids": family_ids}
                    ),
                    "family_count": len(family_ids),
                    "run_count": 34 * len(family_ids),
                    "rows": negative + positive,
                    "class_counts": {"negative_0": negative, "positive_1": positive},
                    "material_classes": materials,
                    "qualifies": negative >= 5 and positive >= 5,
                }
            )
        return components

    def edge_attribution(self) -> dict[str, object]:
        """Count shared identities per family pair without exposing identity values."""
        pair_counts: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
        unions = [_UnionFind(), _UnionFind(), _UnionFind()]
        family_orders = tuple(range(1, EXPECTED_FAMILY_COUNT + 1))
        for union in unions:
            for order in family_orders:
                union.find(str(order))
        for kind in (0, 1):
            for mask in self.identities.family_masks(kind):
                if mask.bit_count() < 2:
                    continue
                present = [order for order in family_orders if mask & (1 << (order - 1))]
                for left, right in combinations(present, 2):
                    pair_counts[left, right][kind] += 1
                    unions[kind].union(str(left), str(right))
                    unions[2].union(str(left), str(right))

        def sizes(union: _UnionFind) -> list[int]:
            counts: dict[str, int] = defaultdict(int)
            for order in family_orders:
                counts[union.find(str(order))] += 1
            return sorted(counts.values(), reverse=True)

        duplicate_sizes, block_sizes, joint_sizes = (sizes(union) for union in unions)
        return {
            "family_pair_links": [
                {
                    "family_ordinals": [left, right],
                    "duplicate_identity_count": counts[0],
                    "spatial_block_count": counts[1],
                }
                for (left, right), counts in sorted(pair_counts.items())
            ],
            "duplicate_only": {
                "component_count": len(duplicate_sizes),
                "family_counts": duplicate_sizes,
            },
            "block_only": {
                "component_count": len(block_sizes),
                "family_counts": block_sizes,
            },
            "joint": {"component_count": len(joint_sizes), "family_counts": joint_sizes},
        }

    def close(self, *, keep_checkpoint: bool = False) -> None:
        self.identities.close(keep_checkpoint=keep_checkpoint)


def _checkpoint_binding(config, source_context, families, specs):
    return sha256(_canonical_json({
        "schema": AGGREGATE_SCHEMA_VERSION,
        "expansion": EXPANSION_ID,
        "config": config,
        "source_context": source_context,
        "families": [asdict(family) for family in families],
        "run_order": [spec.run_id for spec in specs],
    })).hexdigest()


def _checkpoint_load(aggregate, binding, specs):
    connection = aggregate.identities.connection
    try:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise ValueError("SQLite integrity check failed")
        names = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if names != {"identities", "checkpoint_meta", "completed_runs"}:
            raise ValueError("Checkpoint table inventory drifted")
        if tuple(row[1] for row in connection.execute("PRAGMA table_info(identities)")) != (
                "kind", "identity", "row_count", "class_mask", "family_mask"):
            raise ValueError("Checkpoint identity schema drifted")
        meta_rows = connection.execute("SELECT key, value FROM checkpoint_meta").fetchall()
        if len(meta_rows) != 2 or {key for key, _ in meta_rows} != {"binding", "state"}:
            raise ValueError("Checkpoint metadata is incomplete")
        meta = dict(meta_rows)
        if meta["binding"] != binding:
            raise ValueError("Checkpoint provenance, manifest, schema, or run order drifted")
        state = json.loads(meta["state"])
        if set(state) != {"elapsed_seconds", "sample_count", "launch_count", "attempt_started_at_utc", "minimum_available_memory_bytes",
                          "maximum_process_rss_bytes", "minimum_free_disk_bytes"}:
            raise ValueError("Checkpoint resource state is invalid")
        if (not isinstance(state["elapsed_seconds"], (int, float))
                or not 0 <= state["elapsed_seconds"] <= EXPANSION_RUNTIME_LIMIT_SECONDS
                or not isinstance(state["sample_count"], int)
                or state["sample_count"] < 1
                or type(state["launch_count"]) is not int or state["launch_count"] < 1
                or not all(type(state[key]) is int and state[key] >= 0 for key in (
                    "minimum_available_memory_bytes", "maximum_process_rss_bytes",
                    "minimum_free_disk_bytes"))
                or state["minimum_available_memory_bytes"] < RUNTIME_MEMORY_FLOOR_BYTES
                or state["maximum_process_rss_bytes"] > RSS_CEILING_BYTES
                or state["minimum_free_disk_bytes"] < FREE_DISK_FLOOR_BYTES):
            raise ValueError("Checkpoint resource bounds are invalid")
        attempt = state["attempt_started_at_utc"]
        if attempt is not None:
            if not isinstance(attempt, str):
                raise ValueError("Checkpoint attempt timestamp is invalid")
            started = datetime.fromisoformat(attempt)
            if started.tzinfo is None:
                raise ValueError("Checkpoint attempt timestamp has no timezone")
            downtime = (datetime.now(timezone.utc) - started).total_seconds()
            if downtime < 0:
                raise ValueError("Checkpoint attempt timestamp is in the future")
            state["elapsed_seconds"] += downtime
            if state["elapsed_seconds"] > EXPANSION_RUNTIME_LIMIT_SECONDS:
                raise ValueError("Checkpoint attempt exceeded aggregate runtime limit")
            state["attempt_started_at_utc"] = None
        completed = connection.execute(
            "SELECT run_order, payload, digest FROM completed_runs ORDER BY run_order"
        ).fetchall()
        if len(completed) > EXPECTED_RUN_COUNT:
            raise ValueError("Checkpoint has excess runs")
        for expected_order, (order, payload, digest) in enumerate(completed, start=1):
            if order != expected_order or sha256(payload.encode("utf-8")).hexdigest() != digest:
                raise ValueError("Checkpoint run gap or corrupted journal")
            record = json.loads(payload)
            _validate_aggregate_only(record)
            spec = specs[expected_order - 1]
            if (set(record) != {"order", "run_id", "scenario_family_id", "scenario_id",
                                "seed", "wind_condition_ordinal", "replicate_ordinal",
                                "rows", "class_counts", "batch_count", "observation_count",
                                "termination", "configuration_sha256", "run_identity_sha256",
                                "provenance_sha256", "runtime_seconds"}
                    or set(record.get("class_counts", {})) != {"negative_0", "positive_1"}
                    or set(record.get("termination", {})) != {"reason", "completeness_status",
                                                                  "authoritative", "final_timestep",
                                                                  "expected_transition_count",
                                                                  "captured_transition_count"}
                    or record.get("order") != order or record.get("run_id") != spec.run_id
                    or record.get("scenario_family_id") != spec.family.family_id
                    or record.get("scenario_id") != spec.scenario_id
                    or record.get("seed") != spec.seed
                    or record.get("wind_condition_ordinal") != spec.wind_condition_ordinal
                    or record.get("replicate_ordinal") != spec.replicate_ordinal
                    or record.get("termination", {}).get("reason") != TERMINATION_INACTIVE
                    or record.get("termination", {}).get("completeness_status") != COMPLETENESS_COMPLETE
                    or record.get("termination", {}).get("authoritative") is not True
                    or record.get("rows") != sum(record.get("class_counts", {}).values())
                    or record.get("rows", 0) < 1 or record["rows"] > PER_RUN_ROW_CEILING):
                raise ValueError("Checkpoint run record is invalid")
            if (not all(isinstance(record.get(key), str) and len(record[key]) == 64
                        and all(character in "0123456789abcdef" for character in record[key])
                        for key in ("configuration_sha256", "run_identity_sha256", "provenance_sha256"))
                    or not 0 <= record["runtime_seconds"] <= RUN_RUNTIME_LIMIT_SECONDS
                    or not 0 < record["termination"]["final_timestep"] <= SAFETY_LIMIT_TIMESTEPS
                    or not (record["termination"]["expected_transition_count"] ==
                            record["termination"]["captured_transition_count"] ==
                            record["observation_count"])):
                raise ValueError("Checkpoint run validation drifted")
            aggregate.runs.append(record)
            negative = record["class_counts"]["negative_0"]
            positive = record["class_counts"]["positive_1"]
            if not all(type(value) is int and value >= 0 for value in (negative, positive)):
                raise ValueError("Checkpoint class counts are invalid")
            aggregate.total_rows += record["rows"]
            aggregate.total_negative += negative
            aggregate.total_positive += positive
            counts = aggregate.family_counts[spec.family.family_id]
            counts[0] += negative
            counts[1] += positive
        if aggregate.total_rows > TOTAL_ROW_CEILING:
            raise ValueError("Checkpoint total row bound exceeded")
        for kind in (0, 1):
            summary = connection.execute(
                "SELECT COALESCE(SUM(row_count), 0), COUNT(*) FROM identities WHERE kind=?",
                (kind,),
            ).fetchone()
            if summary[0] != aggregate.total_rows or (completed and summary[1] == 0):
                raise ValueError("Checkpoint identity counts disagree with runs")
        invalid = connection.execute(
            "SELECT COUNT(*) FROM identities WHERE kind NOT IN (0,1) OR length(identity)=0 "
            "OR row_count<1 OR class_mask NOT IN (1,2,3) OR family_mask<1 "
            "OR family_mask>=?", (1 << EXPECTED_FAMILY_COUNT,)
        ).fetchone()[0]
        if invalid:
            raise ValueError("Checkpoint identity domain is invalid")
        if state["sample_count"] < state["launch_count"] + 2 * len(completed):
            raise ValueError("Checkpoint resource sample count drifted")
        return state
    except (sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        raise PopulationAdequacyError(f"Invalid synthetic checkpoint: {exc}") from exc


def _checkpoint_commit(aggregate, state):
    record = aggregate.runs[-1]
    payload = _canonical_json(record).decode("utf-8")
    connection = aggregate.identities.connection
    connection.execute(
        "INSERT INTO completed_runs(run_order,payload,digest) VALUES (?,?,?)",
        (record["order"], payload, sha256(payload.encode("utf-8")).hexdigest()),
    )
    _checkpoint_save_state(connection, state)
    aggregate.identities.commit()


def _checkpoint_save_state(connection, state):
    connection.execute(
        "UPDATE checkpoint_meta SET value=? WHERE key='state'",
        (_canonical_json(state).decode("utf-8"),),
    )


def _unexposed_allocation_feasible(components: Sequence[Mapping[str, object]]) -> bool:
    """Return true only when a deterministic construction proves D-015 feasibility."""
    remaining = [component for component in components if component["qualifies"] is True]
    if len(remaining) < 23:
        return False
    # Allocate the three five-component roles first so the eight-component train
    # minimum cannot starve them. The chosen identities are deliberately discarded.
    for _, component_minimum, positive_minimum, negative_minimum in (
        ROLE_REQUIREMENTS[1:] + ROLE_REQUIREMENTS[:1]
    ):
        selected: list[Mapping[str, object]] = []
        uncovered = set(range(1, 6))
        while uncovered:
            candidates = [
                component
                for component in remaining
                if uncovered.intersection(component["material_classes"])
            ]
            if not candidates:
                return False
            chosen = max(
                candidates,
                key=lambda component: (
                    len(uncovered.intersection(component["material_classes"])),
                    min(
                        int(component["class_counts"]["positive_1"]),
                        int(component["class_counts"]["negative_0"]),
                    ),
                    component["component_sha256"],
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
            uncovered.difference_update(chosen["material_classes"])
        while True:
            positive = sum(int(item["class_counts"]["positive_1"]) for item in selected)
            negative = sum(int(item["class_counts"]["negative_0"]) for item in selected)
            maximum_positive = max(
                (int(item["class_counts"]["positive_1"]) for item in selected),
                default=0,
            )
            if (
                len(selected) >= component_minimum
                and positive >= positive_minimum
                and negative >= negative_minimum
                and maximum_positive * 2 <= positive
            ):
                break
            if not remaining:
                return False
            chosen = max(
                remaining,
                key=lambda component: (
                    min(
                        int(component["class_counts"]["positive_1"]),
                        int(component["class_counts"]["negative_0"]),
                    ),
                    component["component_sha256"],
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
    return True


def _validate_source_context(
    source_context: Mapping[str, object], *, synthetic: bool
) -> None:
    required = {
        "source_revision",
        "source_dirty",
        "source_hashes",
        "input_hashes",
        "generation_command",
        "environment_record",
    }
    if set(source_context) != required or not isinstance(source_context["source_dirty"], bool):
        raise PopulationAdequacyError("Source/input provenance context is incomplete")
    if source_context["source_dirty"] is not False:
        raise PopulationAdequacyError("Production adequacy execution requires a clean source")
    revision = source_context["source_revision"]
    if not synthetic and (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise PopulationAdequacyError("Production source revision is not a Git commit")
    for field in ("source_hashes", "input_hashes"):
        values = source_context[field]
        if not isinstance(values, Mapping) or not values:
            raise PopulationAdequacyError(f"{field} must be a non-empty mapping")
        for key, value in values.items():
            if "ground_truth" in str(key).lower():
                raise PermissionError("stack_ground_truth.tif is forbidden")
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise PopulationAdequacyError(f"{field} contains an invalid digest")
    if not synthetic:
        if set(source_context["source_hashes"]) != set(PRODUCTION_SOURCE_RELATIVE_PATHS):
            raise PopulationAdequacyError("Production source hash inventory drifted")
        inputs = source_context["input_hashes"]
        expected_inputs = set(EXPECTED_RASTER_SHA256) | {
            "simulation_valid_mask",
            "mapped_building_mask",
        }
        if set(inputs) != expected_inputs:
            raise PopulationAdequacyError("Production input hash inventory drifted")
        if {name: inputs[name] for name in EXPECTED_RASTER_SHA256} != EXPECTED_RASTER_SHA256:
            raise PopulationAdequacyError("Approved environmental raster identity drifted")
        if inputs.get("simulation_valid_mask") != EXPECTED_SIMULATION_VALID_SHA256:
            raise PopulationAdequacyError("Simulation-valid mask identity drifted")
        if inputs.get("mapped_building_mask") != EXPECTED_MAPPED_BUILDING_SHA256:
            raise PopulationAdequacyError("Mapped-building mask identity drifted")


def _current_source_hashes() -> dict[str, str]:
    code_root = Path(__file__).resolve().parents[1]
    return {
        relative: _sha256_file(code_root / relative)
        for relative in PRODUCTION_SOURCE_RELATIVE_PATHS
    }


def _current_git_state() -> tuple[str, bool]:
    code_root = Path(__file__).resolve().parents[1]
    repository_root = code_root.parents[1]
    git = [
        "git",
        "-c",
        f"safe.directory={repository_root.as_posix()}",
        "-C",
        str(repository_root),
    ]
    revision = subprocess.run(
        [*git, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            *git,
            "status",
            "--porcelain",
            "--",
            ".",
            ":(exclude)Thesis-try.qgz",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return revision, bool(dirty)


def _verify_production_evidence(
    config: Mapping[str, object],
    source_context: Mapping[str, object],
    environment: Mapping[str, object],
) -> None:
    """Recompute all production evidence instead of trusting caller claims."""
    validate_approved_environment_config(config)
    code_root = Path(__file__).resolve().parents[1]
    default_config_path = code_root / "config" / "default_experiment.yaml"
    expected_config = yaml.safe_load(default_config_path.read_text(encoding="utf-8"))
    expected_config["phase9_set_c_population_adequacy"]["enabled"] = True
    expected_config["environment"]["raster_dir"] = str(
        (code_root / expected_config["environment"]["raster_dir"]).resolve()
    )
    if canonical_metadata_hash(config) != canonical_metadata_hash(expected_config):
        raise PopulationAdequacyError("Production configuration differs from checked-in default")
    if source_context["source_hashes"] != _current_source_hashes():
        raise PopulationAdequacyError("Production source digest claims do not match current files")
    revision, dirty = _current_git_state()
    if dirty or source_context["source_revision"] != revision or source_context["source_dirty"] is not False:
        raise PopulationAdequacyError("Production Git provenance is dirty or mismatched")
    if source_context["generation_command"] != PRODUCTION_GENERATION_COMMAND:
        raise PopulationAdequacyError("Production generation command drifted")
    identity = validate_phase8_environment(environment, synthetic=False)
    if identity["simulation_valid_mask_sha256"] != source_context["input_hashes"]["simulation_valid_mask"]:
        raise PopulationAdequacyError("Supplied simulation-valid mask provenance is false")
    if identity["mapped_building_mask_sha256"] != source_context["input_hashes"]["mapped_building_mask"]:
        raise PopulationAdequacyError("Supplied mapped-building mask provenance is false")
    raster_dir = code_root / "processedData" / "raster" / "raster"
    actual_rasters = {
        filename: _sha256_file(raster_dir / filename)
        for filename in EXPECTED_RASTER_SHA256
    }
    if actual_rasters != EXPECTED_RASTER_SHA256:
        raise PopulationAdequacyError("Current environmental raster digests drifted")
    if {
        filename: source_context["input_hashes"][filename]
        for filename in EXPECTED_RASTER_SHA256
    } != actual_rasters:
        raise PopulationAdequacyError("Supplied environmental raster provenance is false")


def _check_resources(snapshot: ResourceSnapshot, *, launch: bool) -> None:
    memory_floor = LAUNCH_MEMORY_FLOOR_BYTES if launch else RUNTIME_MEMORY_FLOOR_BYTES
    if snapshot.available_memory_bytes < memory_floor:
        raise PopulationResourceError("Available memory crossed the approved floor")
    if snapshot.process_rss_bytes > RSS_CEILING_BYTES:
        raise PopulationResourceError("Process RSS crossed the approved ceiling")
    if snapshot.free_disk_bytes < FREE_DISK_FLOOR_BYTES:
        raise PopulationResourceError("Free disk crossed the approved floor")


def _validate_aggregate_only(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_REPORT_KEYS or "ground_truth" in normalized:
                raise PopulationAdequacyError(
                    f"Aggregate report contains forbidden key: {key}"
                )
            _validate_aggregate_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_aggregate_only(item)


def _wind_manifest(spec: PopulationRunSpec) -> dict[str, object]:
    return {
        "speed_kmh": float(spec.speed_kmh),
        "direction_deg": float(spec.direction_deg),
        "direction_convention": "meteorological_from",
        "direction_units": "degrees_clockwise",
        "north_reference": "projected_grid_north",
        "schema_version": "simulated_wind.v1",
        "calm_representation": "speed_zero_direction_zero",
    }


def _run_config(config: Mapping[str, object], spec: PopulationRunSpec) -> dict[str, Any]:
    active = deepcopy(dict(config))
    active["simulation"]["seed"] = spec.seed
    active["wind"] = _wind_manifest(spec)
    active["model_free_teacher"].update(
        {
            "enabled": True,
            "runner_schema_version": TEACHER_RUNNER_SCHEMA_VERSION,
            "safety_limit_timesteps": SAFETY_LIMIT_TIMESTEPS,
            "persistent_output_enabled": False,
        }
    )
    active["ml_model"]["enabled"] = False
    dataset = active["dataset_generation"]
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
        "size_rows": BLOCK_SIZE[0],
        "size_cols": BLOCK_SIZE[1],
    }
    return active


def build_model_free_executor(
    environment: Mapping[str, object],
    config: Mapping[str, object],
    source_context: Mapping[str, object],
    *,
    destination: Path,
    resource_probe: Callable[[Path], ResourceSnapshot],
) -> RunExecutor:
    """Wire the existing model-free teacher and bounded Set C collector."""

    expansion_start = time.monotonic()
    output_root = Path(__file__).resolve().parents[1] / str(
        config.get("output", {}).get("output_dir", "output")
    )

    def output_snapshot() -> dict[str, tuple[int, int]]:
        if not output_root.exists():
            return {}
        return {
            str(path.relative_to(output_root)).replace("\\", "/"): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in output_root.rglob("*")
            if path.is_file()
        }

    def execute(
        spec: PopulationRunSpec, consume_batch: Callable[[object], None]
    ) -> PopulationRunResult:
        run_config = _run_config(config, spec)
        collector = SetCCollector(environment, run_config)
        rows = batches = retry_cycles = maximum_live = 0
        run_start = time.monotonic()
        outputs_before = output_snapshot()

        def check_runtime() -> None:
            _check_resources(resource_probe(destination), launch=False)
            if time.monotonic() - run_start > RUN_RUNTIME_LIMIT_SECONDS:
                raise PopulationResourceError("Per-run runtime limit exceeded")
            if time.monotonic() - expansion_start > EXPANSION_RUNTIME_LIMIT_SECONDS:
                raise PopulationResourceError("Aggregate expansion runtime limit exceeded")

        def drain(active_collector: SetCCollector) -> None:
            nonlocal rows, batches
            if active_collector.pending_rows > MAX_BUFFER_ROWS or active_collector.pending_bytes > MAX_BUFFER_BYTES:
                raise PopulationResourceError("Collector crossed its retained payload bound")
            while active_collector.pending_batches():
                batch = active_collector.pop_batch()
                consume_batch(batch)
                rows += batch.row_count
                batches += 1
                del batch
            check_runtime()

        def backpressure(active_collector: SetCCollector) -> None:
            nonlocal retry_cycles
            retry_cycles += 1
            if retry_cycles > RETRY_LIMIT:
                raise PopulationAdequacyError("Backpressure retry limit exceeded")
            before = active_collector.pending_rows
            drain(active_collector)
            if before <= 0:
                raise PopulationAdequacyError("Backpressure drain made no progress")
            check_runtime()

        def observe(observation: TransitionObservation) -> None:
            nonlocal retry_cycles, maximum_live
            drain(collector)
            maximum_live = max(maximum_live, collector.live_full_grid_observation_count)
            retry_cycles = 0
            if rows > PER_RUN_ROW_CEILING:
                raise PopulationAdequacyError("Per-run row ceiling exceeded")
            check_runtime()

        wind = WindContract.from_config(run_config["wind"]).to_manifest()
        provenance = {
            "set_id": "SetC",
            "experiment_id": EXPANSION_ID,
            "event_id": EVENT_ID,
            "scenario_id": spec.scenario_id,
            "scenario_family_id": spec.family.family_id,
            "run_id": spec.run_id,
            "grid_id": GRID_ID,
            "simulation_valid_mask_id": "phase9_simulation_valid_mask_v1",
            "mapped_building_mask_id": "phase9_mapped_building_mask_v1",
            "simulator_id": MODEL_FREE_SOURCE,
            "simulator_version": TEACHER_RUNNER_SCHEMA_VERSION,
            "wind": wind,
            "source_revision": source_context["source_revision"],
            "source_dirty": source_context["source_dirty"],
            "configuration_hash": canonical_metadata_hash(run_config),
            "input_hashes": source_context["input_hashes"],
            "authorization": SOURCE_ONLY_AUTHORIZATION,
            "authorization_reference": APPROVAL_REFERENCE,
            "transition_source_type": AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
            "seed": spec.seed,
            "rng_bit_generator": "PCG64",
            "wind_manifest": wind,
            "wind_weight": float(run_config["placeholder_transition"]["wind_weight"]),
            "simulation_valid_mask_sha256": source_context["input_hashes"]["simulation_valid_mask"],
            "mapped_building_mask_sha256": source_context["input_hashes"]["mapped_building_mask"],
            "environment_id": GRID_ID,
            "execution_id": f"{EXPANSION_ID}:{spec.run_id}",
            "capture_source_id": "model_free_teacher.step",
            "ignition_coordinate_system": "grid_row_col",
            "ignition_coordinates": spec.family.ignition_coordinates,
            "ignition_set_sha256": spec.family.ignition_set_sha256,
        }
        termination = run_model_free_teacher(
            dict(environment),
            run_config,
            ignition_points=spec.family.ignition_coordinates,
            transition_provenance=provenance,
            transition_observer=observe,
            set_c_collector=collector,
            backpressure_drain=backpressure,
        )
        drain(collector)
        sealed = collector.finalize(termination)
        outputs_after = output_snapshot()
        unexpected_outputs = tuple(
            sorted(
                name
                for name in set(outputs_before) | set(outputs_after)
                if outputs_before.get(name) != outputs_after.get(name)
            )
        )
        return PopulationRunResult(
            run_id=spec.run_id,
            scenario_id=spec.scenario_id,
            family_id=spec.family.family_id,
            seed=spec.seed,
            rows=rows,
            batch_count=batches,
            observation_count=sealed.observation_count,
            released_rows=sealed.released_row_count,
            pending_rows=collector.pending_rows,
            pending_bytes=collector.pending_bytes,
            termination_reason=str(termination.termination_reason),
            completeness_status=str(termination.completeness_status),
            authoritative=bool(termination.authoritative),
            final_timestep=int(termination.final_timestep),
            final_ignited_count=int(termination.final_ignited_count),
            final_blazing_count=int(termination.final_blazing_count),
            expected_transition_count=int(termination.expected_transition_count),
            captured_transition_count=int(termination.captured_transition_count),
            configuration_sha256=canonical_metadata_hash(run_config),
            run_identity_sha256=sealed.run_identity_sha256,
            provenance_sha256=sealed.provenance_sha256,
            ignition_set_sha256=sealed.ignition_set_sha256,
            unexpected_outputs=unexpected_outputs,
            max_live_full_grid_observations=maximum_live,
            retry_limit_exceeded=retry_cycles > RETRY_LIMIT,
        )

    return execute


def _run_population_adequacy_impl(
    config: Mapping[str, object],
    *,
    destination: Path,
    source_context: Mapping[str, object],
    environment: Mapping[str, object] | None = None,
    families: Sequence[PopulationFamily] | None = None,
    run_executor: RunExecutor | None = None,
    resource_probe: Callable[[Path], ResourceSnapshot] = capture_resource_snapshot,
    synthetic_test: bool = False,
    aggregate_store_directory: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    _cleanup: list[_AggregateAccumulator],
) -> dict[str, object]:
    """Run sequentially and publish one immutable aggregate report on success."""
    _require_exact_contract(config)
    _validate_source_context(source_context, synthetic=synthetic_test)
    destination = destination.resolve()
    production_destination = approved_destination().resolve()
    if synthetic_test:
        destination = _require_temp_descendant(destination, "destination")
        if aggregate_store_directory is not None:
            aggregate_store_directory = _require_temp_descendant(
                aggregate_store_directory, "aggregate_store_directory"
            )
        if destination == production_destination:
            raise PermissionError("Synthetic tests cannot use the production destination")
        if families is None or run_executor is None:
            raise PopulationAdequacyError("Synthetic tests require injected families and executor")
        if checkpoint_path is not None:
            checkpoint_path = _require_temp_descendant(checkpoint_path, "checkpoint_path")
            if checkpoint_path == destination:
                raise PermissionError("Checkpoint cannot be the aggregate destination")
        elif resume:
            raise PermissionError("Resume requires a synthetic checkpoint path")
    else:
        if checkpoint_path is not None or resume:
            raise PermissionError("Production checkpoints and resume require separate approval")
        if resource_probe is not capture_resource_snapshot:
            raise PermissionError("Production execution requires exact resource telemetry")
        if aggregate_store_directory is not None:
            raise PermissionError("Production aggregate storage location cannot be injected")
        if destination != production_destination:
            raise PermissionError("Only the approved aggregate destination is permitted")
        if families is not None or run_executor is not None:
            raise PermissionError("Production execution cannot inject families or executor")
        if environment is None:
            raise PopulationAdequacyError("Production execution requires the loaded environment")
        _verify_production_evidence(config, source_context, environment)
        families = load_approved_families()
        run_executor = build_model_free_executor(
            environment,
            config,
            source_context,
            destination=destination,
            resource_probe=resource_probe,
        )
    assert families is not None and run_executor is not None
    specs = build_run_matrix(families)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite adequacy report: {destination}")
    if checkpoint_path is not None and checkpoint_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite synthetic checkpoint: {checkpoint_path}")
    if resume and checkpoint_path is not None and not checkpoint_path.is_file():
        raise PopulationAdequacyError("Synthetic checkpoint is missing")
    launch = resource_probe(destination)
    _check_resources(launch, launch=True)
    aggregate = _AggregateAccumulator(
        families, store_directory=aggregate_store_directory,
        checkpoint=checkpoint_path, resume=resume,
    )
    _cleanup.append(aggregate)
    binding = _checkpoint_binding(config, source_context, families, specs) if checkpoint_path else None
    if resume:
        state = _checkpoint_load(aggregate, binding, specs)
        state["sample_count"] += 1
        state["launch_count"] += 1
        state["minimum_available_memory_bytes"] = min(
            state["minimum_available_memory_bytes"], launch.available_memory_bytes)
        state["maximum_process_rss_bytes"] = max(
            state["maximum_process_rss_bytes"], launch.process_rss_bytes)
        state["minimum_free_disk_bytes"] = min(
            state["minimum_free_disk_bytes"], launch.free_disk_bytes)
    else:
        state = {
            "elapsed_seconds": 0.0,
            "sample_count": 1,
            "launch_count": 1,
            "attempt_started_at_utc": None,
            "minimum_available_memory_bytes": launch.available_memory_bytes,
            "maximum_process_rss_bytes": launch.process_rss_bytes,
            "minimum_free_disk_bytes": launch.free_disk_bytes,
        }
        if checkpoint_path is not None:
            aggregate.identities.connection.executemany(
                "INSERT INTO checkpoint_meta(key,value) VALUES (?,?)",
                (("binding", binding), ("state", _canonical_json(state).decode("utf-8"))),
            )
            aggregate.identities.commit()
    start = time.monotonic() - state["elapsed_seconds"]
    for spec in specs[len(aggregate.runs):]:
        if time.monotonic() - start > EXPANSION_RUNTIME_LIMIT_SECONDS:
            raise PopulationResourceError("Aggregate expansion runtime limit exceeded")
        if checkpoint_path is not None:
            state["attempt_started_at_utc"] = datetime.now(timezone.utc).isoformat()
            _checkpoint_save_state(aggregate.identities.connection, state)
            aggregate.identities.commit()
        try:
            before = resource_probe(destination)
            for sample in (before,):
                state["minimum_available_memory_bytes"] = min(
                    state["minimum_available_memory_bytes"], sample.available_memory_bytes)
                state["maximum_process_rss_bytes"] = max(
                    state["maximum_process_rss_bytes"], sample.process_rss_bytes)
                state["minimum_free_disk_bytes"] = min(
                    state["minimum_free_disk_bytes"], sample.free_disk_bytes)
                state["sample_count"] += 1
            _check_resources(before, launch=False)
            aggregate.begin(spec)
            run_start = time.monotonic()
            result = run_executor(spec, aggregate.consume)
            elapsed = time.monotonic() - run_start
            aggregate.finish(spec, result, elapsed)
            after = resource_probe(destination)
            state["minimum_available_memory_bytes"] = min(
                state["minimum_available_memory_bytes"], after.available_memory_bytes)
            state["maximum_process_rss_bytes"] = max(
                state["maximum_process_rss_bytes"], after.process_rss_bytes)
            state["minimum_free_disk_bytes"] = min(
                state["minimum_free_disk_bytes"], after.free_disk_bytes)
            state["sample_count"] += 1
            _check_resources(after, launch=False)
            state["elapsed_seconds"] = time.monotonic() - start
            if state["elapsed_seconds"] > EXPANSION_RUNTIME_LIMIT_SECONDS:
                raise PopulationResourceError("Aggregate expansion runtime limit exceeded")
            state["attempt_started_at_utc"] = None
            if checkpoint_path is None:
                aggregate.identities.commit()
            else:
                _checkpoint_commit(aggregate, state)
        except BaseException:
            if checkpoint_path is not None:
                aggregate.identities.connection.rollback()
                state["attempt_started_at_utc"] = None
                state["elapsed_seconds"] = time.monotonic() - start
                _checkpoint_save_state(aggregate.identities.connection, state)
                aggregate.identities.commit()
            raise
        if spec.order % 100 == 0 or spec.order == EXPECTED_RUN_COUNT:
            print(f"Gate 6 progress: {spec.order}/{EXPECTED_RUN_COUNT} runs complete", flush=True)
    if len(aggregate.runs) != EXPECTED_RUN_COUNT or aggregate.total_rows > TOTAL_ROW_CEILING:
        raise PopulationAdequacyError("Complete expansion count contract failed")
    components = aggregate.leakage_components()
    edge_attribution = aggregate.edge_attribution()
    if edge_attribution["joint"]["family_counts"] != sorted(
        (component["family_count"] for component in components), reverse=True
    ):
        raise PopulationAdequacyError("Edge attribution disagrees with the joint leakage graph")
    qualifying = [component for component in components if component["qualifies"] is True]
    allocation_feasible = _unexposed_allocation_feasible(components)
    gates = {
        "all_884_runs_complete_inactive_authoritative_and_count_matched": True,
        "at_least_23_qualifying_components": len(qualifying) >= 23,
        "every_qualifying_component_has_five_of_each_class": all(
            int(component["class_counts"]["positive_1"]) >= 5
            and int(component["class_counts"]["negative_0"]) >= 5
            for component in qualifying
        ),
        "unexposed_role_minima_material_coverage_and_concentration_feasible": allocation_feasible,
        "one_live_run_and_bounded_complete_batch_draining": True,
        "no_estimator_protected_input_row_split_or_model_output": True,
    }
    report: dict[str, object] = {
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "expansion_id": EXPANSION_ID,
        "status": "complete",
        "contract_status": "pass" if all(gates.values()) else "fail",
        "authorization_reference": APPROVAL_REFERENCE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved_inventory": {
            "payload_sha256": APPROVED_INVENTORY_PAYLOAD_SHA256,
            "file_sha256": APPROVED_INVENTORY_FILE_SHA256,
            "ordered_candidate_sha256": APPROVED_CANDIDATE_SHA256,
            "ordered_primary_sha256": APPROVED_PRIMARY_SHA256,
            "ordered_reserve_sha256": APPROVED_RESERVE_SHA256,
        },
        "population_contract": {
            "family_count": EXPECTED_FAMILY_COUNT,
            "existing_family_count": 6,
            "approved_primary_family_count": 20,
            "wind_condition_count": len(WIND_CONDITIONS),
            "replicates_per_condition": len(REPLICATES),
            "run_count": len(specs),
            "run_order_sha256": canonical_metadata_hash([spec.run_id for spec in specs]),
            "seed_formula": "9000000 + 10000*family_ordinal + 100*wind_condition_ordinal + replicate_ordinal",
            "run_concurrency": 1,
            "block_size": list(BLOCK_SIZE),
            "block_origin": list(BLOCK_ORIGIN),
            "split_roles_assigned": False,
            "prospective_role_minima": {
                name: {
                    "components": components,
                    "positive_1": positives,
                    "negative_0": negatives,
                    "material_classes": [1, 2, 3, 4, 5],
                    "maximum_single_component_positive_share": 0.5,
                }
                for name, components, positives, negatives in ROLE_REQUIREMENTS
            },
        },
        "provenance": dict(source_context),
        "aggregate": {
            "totals": {
                "rows": aggregate.total_rows,
                "class_counts": {
                    "negative_0": aggregate.total_negative,
                    "positive_1": aggregate.total_positive,
                },
                "family_count": len(families),
                "run_count": len(aggregate.runs),
            },
            "scenario_families": [
                {
                    "scenario_family_id": family.family_id,
                    "material_class": family.material_class,
                    "component_ordinal": family.component_ordinal,
                    "source_tier": family.source_tier,
                    "run_count": 34,
                    "class_counts": {
                        "negative_0": aggregate.family_counts[family.family_id][0],
                        "positive_1": aggregate.family_counts[family.family_id][1],
                    },
                }
                for family in families
            ],
            "runs": aggregate.runs,
            "duplicate_groups": aggregate.identities.summary(0),
            "occupied_spatial_block_count": aggregate.identities.summary(1)[
                "unique_group_count"
            ],
            "leakage": {
                "component_count": len(components),
                "qualifying_component_count": len(qualifying),
                "components": components,
                "edge_attribution": edge_attribution,
                "unexposed_allocation_feasible": allocation_feasible,
                "role_identities_exposed": False,
            },
        },
        "adequacy": {"status": "pass" if all(gates.values()) else "fail", "gates": gates},
        "resources": {
            "minimum_available_memory_bytes": state["minimum_available_memory_bytes"],
            "maximum_process_rss_bytes": state["maximum_process_rss_bytes"],
            "minimum_free_disk_bytes": state["minimum_free_disk_bytes"],
            "sample_count": state["sample_count"],
            "elapsed_seconds": time.monotonic() - start,
        },
        "integrity": {
            "row_level_payload_retained": False,
            "row_level_payload_published": False,
            "split_roles_assigned": False,
            "estimator_accessed": False,
            "protected_spatial_reference_accessed": False,
            "training_performed": False,
            "calibration_performed": False,
            "raster_checkpoint_or_model_written": False,
            "sole_persistent_output_is_aggregate_json": True,
            "aggregate_identity_store": {
                "backend": "temporary_sqlite",
                "cache_kib": SQLITE_CACHE_KIB,
                "per_identity_values": [
                    "row_count",
                    "class_mask",
                    "family_bitmask_26",
                ],
                "committed_per_run": True,
                "cleanup_required_on_success_and_failure": True,
            },
        },
    }
    _validate_aggregate_only(report)
    path, payload_hash = publish_aggregate_report(report, destination)
    return {
        "path": path,
        "payload_sha256": payload_hash,
        "status": report["status"],
        "adequacy_status": report["adequacy"]["status"],
        "run_count": len(aggregate.runs),
    }


def run_population_adequacy(
    config: Mapping[str, object],
    *,
    destination: Path,
    source_context: Mapping[str, object],
    environment: Mapping[str, object] | None = None,
    families: Sequence[PopulationFamily] | None = None,
    run_executor: RunExecutor | None = None,
    resource_probe: Callable[[Path], ResourceSnapshot] = capture_resource_snapshot,
    synthetic_test: bool = False,
    aggregate_store_directory: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Run the expansion and always remove its temporary exact aggregate store."""
    cleanup: list[_AggregateAccumulator] = []
    try:
        return _run_population_adequacy_impl(
            config,
            destination=destination,
            source_context=source_context,
            environment=environment,
            families=families,
            run_executor=run_executor,
            resource_probe=resource_probe,
            synthetic_test=synthetic_test,
            aggregate_store_directory=aggregate_store_directory,
            checkpoint_path=checkpoint_path,
            resume=resume,
            _cleanup=cleanup,
        )
    finally:
        for aggregate in cleanup:
            aggregate.close(keep_checkpoint=checkpoint_path is not None and
                            not Path(destination).exists())


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "APPROVED_RELATIVE_DESTINATION",
    "EXPANSION_ID",
    "PopulationAdequacyError",
    "PopulationFamily",
    "PopulationResourceError",
    "PopulationRunResult",
    "PopulationRunSpec",
    "ResourceSnapshot",
    "activate_approved_population_config",
    "approved_destination",
    "build_run_matrix",
    "load_approved_families",
    "preflight_population_launch",
    "run_population_adequacy",
]
