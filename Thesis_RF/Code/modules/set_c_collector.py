"""Bounded in-memory collection and test-only publication for Set C rows."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
import zipfile

import numpy as np

from .ca_state import STATE_ENCODING
from .feature_pipeline import (
    CANONICAL_FEATURE_NAMES,
    POSITIVE_LABEL,
    SET_C_SCHEMA_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
    FeatureAssembler,
    validate_feature_values,
)
from .transition_observer import (
    AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
    TRANSITION_OBSERVATION_SCHEMA_VERSION,
    TransitionObservation,
    canonical_metadata_hash,
    validate_grid_provenance,
    validate_transition_observation,
    validate_transition_provenance,
)


SET_C_ROW_BATCH_SCHEMA_VERSION = "set_c_row_batch.v1"
SET_C_RUN_SCHEMA_VERSION = "set_c_collected_run.v1"
SET_C_PUBLICATION_MANIFEST_VERSION = "set_c_test_package_manifest.v1"
DUPLICATE_IDENTITY_VERSION = "sha256_canonical_float32_feature_row.v1"
SPATIAL_BLOCK_IDENTITY_VERSION = "sha256_grid_block_coordinates.v1"
SYNTHETIC_TEST_PUBLISHER_MODE = "synthetic_test_only"
RUNNER_OWNED_BACKPRESSURE_POLICY = "runner_owned_exact_retry.v1"
MEMORY_BOUND_SCOPE = "retained_numpy_row_payload_arrays_only_not_peak_process_memory"
PAYLOAD_MUTABILITY_CONTRACT = "all_numpy_payload_arrays_read_only"
_LOWER_SHA256_RE = re.compile(rb"[0-9a-f]{64}")

_RUN_IDENTITY_TEXT_FIELDS = (
    "set_id",
    "experiment_id",
    "event_id",
    "scenario_family_id",
    "scenario_id",
    "run_id",
    "grid_id",
    "simulation_valid_mask_id",
    "mapped_building_mask_id",
    "simulator_id",
    "simulator_version",
    "source_revision",
    "environment_id",
    "execution_id",
    "capture_source_id",
    "authorization",
    "authorization_reference",
    "transition_source_type",
    "ignition_coordinate_system",
    "ignition_set_sha256",
    "rng_bit_generator",
)

_ARRAY_BYTES_PER_ROW = (
    len(CANONICAL_FEATURE_NAMES) * np.dtype(np.float32).itemsize
    + np.dtype(np.int8).itemsize
    + 2 * np.dtype(np.int64).itemsize
    + 2 * np.dtype("S64").itemsize
)


class CollectorBackpressureError(BufferError):
    """Signal that the caller must drain accepted batches before retrying."""

    def __init__(self, message: str, *, permanent: bool) -> None:
        super().__init__(message)
        self.permanent = permanent


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an explicit integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _readonly_array(values: object, dtype: object) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON metadata")


def _sha256_rows(values: np.ndarray) -> np.ndarray:
    canonical = np.ascontiguousarray(values, dtype="<f4")
    identifiers = np.asarray(
        [sha256(row.tobytes()).hexdigest().encode("ascii") for row in canonical],
        dtype="S64",
    )
    identifiers.setflags(write=False)
    return identifiers


def _require_lower_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PermissionError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _validate_sha256_identity_array(values: np.ndarray, name: str) -> None:
    if values.dtype != np.dtype("S64"):
        raise PermissionError(f"{name} dtype must be fixed-width S64")
    if not all(
        isinstance(value, bytes) and _LOWER_SHA256_RE.fullmatch(value) is not None
        for value in values.tolist()
    ):
        raise PermissionError(
            f"{name} values must be lowercase 64-character SHA-256 encodings"
        )


def _spatial_block_ids(
    *,
    grid_id: str,
    rows: np.ndarray,
    cols: np.ndarray,
    origin_row: int,
    origin_col: int,
    size_rows: int,
    size_cols: int,
) -> np.ndarray:
    block_rows = np.floor_divide(rows - origin_row, size_rows)
    block_cols = np.floor_divide(cols - origin_col, size_cols)
    identifiers = np.asarray(
        [
            sha256(
                (
                    f"{SPATIAL_BLOCK_IDENTITY_VERSION}|{grid_id}|"
                    f"{origin_row}|{origin_col}|{size_rows}|{size_cols}|"
                    f"{int(block_row)}|{int(block_col)}"
                ).encode("utf-8")
            ).hexdigest().encode("ascii")
            for block_row, block_col in zip(block_rows, block_cols, strict=True)
        ],
        dtype="S64",
    )
    identifiers.setflags(write=False)
    return identifiers


def _run_identity(observation: TransitionObservation) -> Mapping[str, object]:
    provenance = observation.provenance
    missing = [
        name
        for name in _RUN_IDENTITY_TEXT_FIELDS
        if not isinstance(provenance.get(name), str)
        or not str(provenance[name]).strip()
    ]
    for name in (
        "source_dirty",
        "input_hashes",
        "seed",
        "configuration_hash",
        "wind_manifest",
        "wind_weight",
        "simulation_valid_mask_sha256",
        "mapped_building_mask_sha256",
        "ignition_coordinates",
    ):
        if name not in provenance:
            missing.append(name)
    if missing:
        raise ValueError(
            "Incomplete D-014 run identity provenance: "
            + ", ".join(sorted(set(missing)))
        )
    if not isinstance(provenance["source_dirty"], (bool, np.bool_)):
        raise TypeError("D-014 run identity source_dirty must be boolean")
    if int(provenance["seed"]) != observation.seed:
        raise ValueError("D-014 run identity seed does not match the observation")
    if dict(provenance["wind_manifest"]) != dict(observation.wind_manifest):
        raise ValueError("D-014 run identity wind does not match the observation")
    if float(provenance["wind_weight"]) != observation.wind_weight:
        raise ValueError("D-014 run identity wind weight does not match the observation")
    if provenance["simulation_valid_mask_sha256"] != observation.domain_mask_hashes[
        "simulation_valid_mask_sha256"
    ]:
        raise ValueError("D-014 valid-domain identity does not match the observation")
    if provenance["mapped_building_mask_sha256"] != observation.domain_mask_hashes[
        "mapped_building_mask_sha256"
    ]:
        raise ValueError(
            "D-014 mapped-building identity does not match the observation"
        )
    return MappingProxyType(
        {
            **{name: provenance[name] for name in _RUN_IDENTITY_TEXT_FIELDS},
            "source_dirty": bool(provenance["source_dirty"]),
            "input_hashes": provenance["input_hashes"],
            "seed": int(provenance["seed"]),
            "wind_manifest": observation.wind_manifest,
            "wind_weight": observation.wind_weight,
            "configuration_hash": provenance["configuration_hash"],
            "simulation_valid_mask_sha256": provenance[
                "simulation_valid_mask_sha256"
            ],
            "mapped_building_mask_sha256": provenance[
                "mapped_building_mask_sha256"
            ],
            "ignition_coordinates": provenance["ignition_coordinates"],
            "observation_schema_version": observation.schema_version,
            "state_encoding": observation.state_encoding,
            "grid_shape": observation.grid_shape,
            "grid_crs": observation.grid_crs,
            "grid_transform": observation.grid_transform,
        }
    )


@dataclass(frozen=True, slots=True)
class SetCRowBatch:
    """One bounded immutable batch with metadata outside the predictor matrix."""

    schema_version: str
    feature_schema_version: str
    feature_names: tuple[str, ...]
    target_name: str
    target_version: str
    positive_label: int
    source_observation_schema_version: str
    transition_source_type: str
    state_encoding: str
    run_identity_sha256: str
    provenance_sha256: str
    observation_index: int
    batch_index: int
    timestep_t: int
    timestep_t1: int
    seed: int
    grid_shape: tuple[int, int]
    grid_crs: str
    grid_transform: tuple[float, ...]
    wind_manifest: Mapping[str, object]
    wind_weight: float
    domain_mask_hashes: Mapping[str, str]
    features: np.ndarray
    labels: np.ndarray
    cell_rows: np.ndarray
    cell_cols: np.ndarray
    duplicate_group_ids: np.ndarray
    spatial_block_ids: np.ndarray
    provenance: Mapping[str, object]

    @property
    def row_count(self) -> int:
        return int(self.features.shape[0])

    @property
    def payload_nbytes(self) -> int:
        return int(
            self.features.nbytes
            + self.labels.nbytes
            + self.cell_rows.nbytes
            + self.cell_cols.nbytes
            + self.duplicate_group_ids.nbytes
            + self.spatial_block_ids.nbytes
        )


@dataclass(frozen=True, slots=True)
class CollectedSetCRun:
    """Sealed in-memory run; publication eligibility is checked separately."""

    schema_version: str
    run_identity_sha256: str
    provenance_sha256: str
    run_identity: Mapping[str, object]
    provenance: Mapping[str, object]
    batches: tuple[SetCRowBatch, ...]
    observation_count: int
    row_count: int
    payload_nbytes: int
    released_batch_count: int
    released_row_count: int
    first_timestep_t: int
    final_timestep_t1: int
    final_ignited_count: int
    final_blazing_count: int
    block_origin_row: int
    block_origin_col: int
    block_size_rows: int
    block_size_cols: int
    ignition_coordinates: tuple[tuple[int, int], ...]
    ignition_set_sha256: str
    observation_pairs: tuple[tuple[int, int, int], ...]
    termination_record: object

    @property
    def complete_in_memory(self) -> bool:
        return self.released_batch_count == 0 and self.released_row_count == 0


@dataclass(frozen=True, slots=True)
class PublishedSetCTestArtifact:
    """Identity returned for one atomically published synthetic test package."""

    path: Path
    sha256: str
    manifest: Mapping[str, object]


class SetCCollector:
    """Convert observations and bound retained NumPy row payload arrays.

    ``max_buffer_bytes`` does not claim to bound Python object overhead,
    transient validation arrays, compression buffers, or total process memory.
    """

    def __init__(self, environment: Mapping[str, object], config: Mapping[str, object]):
        dataset_config = config.get("dataset_generation")
        if not isinstance(dataset_config, Mapping):
            raise ValueError("dataset_generation configuration is required")
        collector_config = dataset_config.get("set_c_collector")
        if not isinstance(collector_config, Mapping):
            raise ValueError("dataset_generation.set_c_collector is required")
        if collector_config.get("enabled") is not True:
            raise PermissionError("Set C collection is disabled by default")
        if dataset_config.get("sampling_policy") != "all_eligible":
            raise ValueError(
                "dataset_generation.sampling_policy must be 'all_eligible'; "
                "sampling and pressure dropping are prohibited"
            )

        self.max_rows = _positive_int(
            collector_config.get("max_buffer_rows"),
            "set_c_collector.max_buffer_rows",
        )
        self.max_bytes = _positive_int(
            collector_config.get("max_buffer_bytes"),
            "set_c_collector.max_buffer_bytes",
        )
        self.batch_rows = _positive_int(
            collector_config.get("batch_rows"),
            "set_c_collector.batch_rows",
        )
        if self.batch_rows > self.max_rows:
            raise ValueError("set_c_collector.batch_rows cannot exceed max_buffer_rows")

        block_config = dataset_config.get("spatial_block")
        if not isinstance(block_config, Mapping):
            raise ValueError("dataset_generation.spatial_block is required")
        self.block_rows = _positive_int(
            block_config.get("size_rows"), "spatial_block.size_rows"
        )
        self.block_cols = _positive_int(
            block_config.get("size_cols"), "spatial_block.size_cols"
        )
        self.block_origin_row = int(block_config.get("origin_row", 0))
        self.block_origin_col = int(block_config.get("origin_col", 0))

        wind_config = config.get("wind")
        if not isinstance(wind_config, Mapping):
            raise ValueError("wind configuration is required")
        self._assembler = FeatureAssembler(
            dict(environment),
            dict(wind_config),
            dict(config.get("flammability_weights", {})),
        )
        self._configuration_hash = canonical_metadata_hash(config)
        self._expected_wind_manifest = self._assembler.wind.to_manifest()

        grid_shape = tuple(environment.get("grid_shape", ()))
        if grid_shape != tuple(self._assembler.grid_shape):
            raise ValueError("Feature environment grid shape is inconsistent")
        raw_burnable = np.asarray(environment.get("burnable_mask"))
        raw_nodata = np.asarray(environment.get("nodata_mask"))
        raw_buildings = np.asarray(environment.get("building_presence"))
        if any(
            array.shape != grid_shape
            for array in (raw_burnable, raw_nodata, raw_buildings)
        ):
            raise ValueError("Collector environment domain arrays must match grid_shape")
        if raw_burnable.dtype != np.dtype(bool) or raw_nodata.dtype != np.dtype(bool):
            raise TypeError("Collector burnable_mask and nodata_mask must be boolean")
        raw_mapped = raw_buildings > 0
        assembled_mapped = self._assembler.building_presence > 0
        if not np.array_equal(raw_mapped, assembled_mapped):
            raise ValueError(
                "Building/material disagreement changes the observer eligible domain"
            )
        self._simulation_valid_mask = raw_burnable & (~raw_nodata)
        self._mapped_building_mask = raw_mapped

        self._batches: deque[SetCRowBatch] = deque()
        self._pending_rows = 0
        self._pending_bytes = 0
        self._accepted_observations = 0
        self._accepted_rows = 0
        self._released_batches = 0
        self._released_rows = 0
        self._processing = False
        self._sealed_run: CollectedSetCRun | None = None
        self._run_identity: Mapping[str, object] | None = None
        self._run_identity_sha256: str | None = None
        self._provenance_sha256: str | None = None
        self._provenance: Mapping[str, object] | None = None
        self._first_timestep_t: int | None = None
        self._last_timestep_t1: int | None = None
        self._observation_pairs: list[tuple[int, int, int]] = []

    @property
    def pending_rows(self) -> int:
        return self._pending_rows

    @property
    def pending_bytes(self) -> int:
        return self._pending_bytes

    @property
    def observation_count(self) -> int:
        return self._accepted_observations

    @property
    def row_count(self) -> int:
        return self._accepted_rows

    @property
    def live_full_grid_observation_count(self) -> int:
        return int(self._processing)

    def _validate_binding(self, observation: TransitionObservation) -> None:
        if observation.grid_shape != tuple(self._assembler.grid_shape):
            raise ValueError("Observation grid shape is not bound to collector environment")
        if dict(observation.wind_manifest) != self._expected_wind_manifest:
            raise ValueError("Observation wind is not bound to collector configuration")
        if not np.array_equal(
            observation.simulation_valid_mask, self._simulation_valid_mask
        ):
            raise ValueError("Observation valid domain is not bound to collector environment")
        if not np.array_equal(
            observation.mapped_building_mask, self._mapped_building_mask
        ):
            raise ValueError(
                "Observation mapped-building domain is not bound to collector environment"
            )
        if observation.provenance["configuration_hash"] != self._configuration_hash:
            raise ValueError("Observation provenance is not bound to collector config")

    def _validated_run_identity(
        self, observation: TransitionObservation
    ) -> tuple[Mapping[str, object], str]:
        identity = _run_identity(observation)
        digest = canonical_metadata_hash(identity)
        provenance_digest = canonical_metadata_hash(observation.provenance)
        if self._run_identity_sha256 is not None and digest != self._run_identity_sha256:
            raise ValueError(
                "Mixed-run contamination: D-014 run identity changed after collection began"
            )
        if (
            self._provenance_sha256 is not None
            and provenance_digest != self._provenance_sha256
        ):
            raise ValueError(
                "Mixed-run contamination: complete D-014 provenance changed"
            )
        return identity, digest

    def _validate_timestep_sequence(self, observation: TransitionObservation) -> None:
        if self._last_timestep_t1 is None:
            if observation.timestep_t != 0:
                raise ValueError(
                    "The first Set C transition must be the post-seeding 0->1 pair"
                )
            return
        if observation.timestep_t != self._last_timestep_t1:
            if observation.timestep_t < self._last_timestep_t1:
                raise ValueError(
                    "Duplicate or reordered Set C transition timestep pair"
                )
            raise ValueError("Missing Set C transition timestep pair")

    def __call__(self, observation: TransitionObservation) -> None:
        if self._sealed_run is not None:
            raise RuntimeError("Set C collector is sealed")
        if self._processing:
            raise RuntimeError(
                "Set C collector cannot retain more than one live full-grid observation"
            )
        self._processing = True
        try:
            validated = validate_transition_observation(observation)
            self._validate_binding(validated)
            identity, identity_digest = self._validated_run_identity(validated)
            self._validate_timestep_sequence(validated)
            if self._run_identity is None:
                self._run_identity = identity
                self._run_identity_sha256 = identity_digest
                self._provenance_sha256 = canonical_metadata_hash(
                    validated.provenance
                )
                self._provenance = validated.provenance
                self._first_timestep_t = validated.timestep_t
            selected = np.flatnonzero(validated.eligible_mask_t.ravel())
            candidate_rows = int(selected.size)
            candidate_bytes = candidate_rows * _ARRAY_BYTES_PER_ROW
            if candidate_rows > self.max_rows or candidate_bytes > self.max_bytes:
                raise CollectorBackpressureError(
                    "One transition exceeds the configured Set C row/byte buffer; "
                    "increase explicitly approved bounds before retrying",
                    permanent=True,
                )
            if (
                self._pending_rows + candidate_rows > self.max_rows
                or self._pending_bytes + candidate_bytes > self.max_bytes
            ):
                raise CollectorBackpressureError(
                    "Set C buffer is full; drain batches and retry the same observation",
                    permanent=False,
                )

            flat = selected
            features = np.column_stack(
                (
                    self._assembler.slope_risk.ravel()[flat],
                    self._assembler.proximity_risk.ravel()[flat],
                    self._assembler.building_presence.ravel()[flat],
                    self._assembler.material_risk.ravel()[flat],
                    self._assembler.material_class.ravel()[flat],
                    np.full(candidate_rows, self._assembler.wind_speed, dtype=np.float32),
                    np.full(candidate_rows, self._assembler.wind_sin, dtype=np.float32),
                    np.full(candidate_rows, self._assembler.wind_cos, dtype=np.float32),
                    validated.blazing_neighbor_count_t.ravel()[flat],
                    self._assembler.composite_flammability.ravel()[flat],
                    validated.wind_weighted_score_t.ravel()[flat],
                )
            ).astype(np.float32, copy=False)
            features = validate_feature_values(features)
            labels = validated.newly_ignited_mask_t1.ravel()[flat].astype(
                np.int8, copy=False
            )
            if not np.all(np.isin(labels, (0, POSITIVE_LABEL))):
                raise ValueError("Set C labels must be integer 0 or 1")
            cell_rows, cell_cols = np.unravel_index(flat, validated.grid_shape)
            cell_rows = np.asarray(cell_rows, dtype=np.int64)
            cell_cols = np.asarray(cell_cols, dtype=np.int64)
            duplicate_ids = _sha256_rows(features)
            block_ids = _spatial_block_ids(
                grid_id=str(validated.provenance["grid_id"]),
                rows=cell_rows,
                cols=cell_cols,
                origin_row=self.block_origin_row,
                origin_col=self.block_origin_col,
                size_rows=self.block_rows,
                size_cols=self.block_cols,
            )

            observation_index = self._accepted_observations
            pending: list[SetCRowBatch] = []
            for batch_index, start in enumerate(
                range(0, candidate_rows, self.batch_rows)
            ):
                stop = min(start + self.batch_rows, candidate_rows)
                batch = SetCRowBatch(
                    schema_version=SET_C_ROW_BATCH_SCHEMA_VERSION,
                    feature_schema_version=SET_C_SCHEMA_VERSION,
                    feature_names=CANONICAL_FEATURE_NAMES,
                    target_name=TARGET_NAME,
                    target_version=TARGET_VERSION,
                    positive_label=POSITIVE_LABEL,
                    source_observation_schema_version=validated.schema_version,
                    transition_source_type=AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
                    state_encoding=validated.state_encoding,
                    run_identity_sha256=identity_digest,
                    provenance_sha256=canonical_metadata_hash(
                        validated.provenance
                    ),
                    observation_index=observation_index,
                    batch_index=batch_index,
                    timestep_t=validated.timestep_t,
                    timestep_t1=validated.timestep_t1,
                    seed=validated.seed,
                    grid_shape=validated.grid_shape,
                    grid_crs=validated.grid_crs,
                    grid_transform=validated.grid_transform,
                    wind_manifest=validated.wind_manifest,
                    wind_weight=validated.wind_weight,
                    domain_mask_hashes=validated.domain_mask_hashes,
                    features=_readonly_array(features[start:stop], np.float32),
                    labels=_readonly_array(labels[start:stop], np.int8),
                    cell_rows=_readonly_array(cell_rows[start:stop], np.int64),
                    cell_cols=_readonly_array(cell_cols[start:stop], np.int64),
                    duplicate_group_ids=_readonly_array(
                        duplicate_ids[start:stop], "S64"
                    ),
                    spatial_block_ids=_readonly_array(block_ids[start:stop], "S64"),
                    provenance=validated.provenance,
                )
                pending.append(batch)

            actual_bytes = sum(batch.payload_nbytes for batch in pending)
            if actual_bytes != candidate_bytes:
                raise RuntimeError("Set C payload byte accounting is inconsistent")
            self._batches.extend(pending)
            self._pending_rows += candidate_rows
            self._pending_bytes += actual_bytes
            self._accepted_observations += 1
            self._accepted_rows += candidate_rows
            self._last_timestep_t1 = validated.timestep_t1
            self._observation_pairs.append(
                (validated.timestep_t, validated.timestep_t1, candidate_rows)
            )
        finally:
            self._processing = False

    def pop_batch(self) -> SetCRowBatch:
        """Release one accepted batch, providing explicit backpressure handling."""
        if self._sealed_run is not None:
            raise RuntimeError("Set C collector is sealed")
        if not self._batches:
            raise IndexError("No Set C batches are pending")
        batch = self._batches.popleft()
        self._pending_rows -= batch.row_count
        self._pending_bytes -= batch.payload_nbytes
        self._released_batches += 1
        self._released_rows += batch.row_count
        return batch

    def pending_batches(self) -> tuple[SetCRowBatch, ...]:
        """Return the bounded immutable batch objects without removing them."""
        return tuple(self._batches)

    def finalize(self, termination_record: object) -> CollectedSetCRun:
        if self._processing:
            raise RuntimeError("Cannot finalize while an observation is live")
        if self._sealed_run is not None:
            if self._sealed_run.termination_record != termination_record:
                raise RuntimeError("Set C collector was finalized with another outcome")
            return self._sealed_run
        if (
            self._run_identity is None
            or self._run_identity_sha256 is None
            or self._provenance_sha256 is None
            or self._provenance is None
        ):
            raise ValueError("Cannot seal a Set C run without a locked run identity")
        termination_ignitions = tuple(
            tuple(int(value) for value in point)
            for point in getattr(termination_record, "ignition_coordinates")
        )
        identity_ignitions = tuple(
            tuple(int(value) for value in point)
            for point in self._run_identity["ignition_coordinates"]
        )
        if termination_ignitions != identity_ignitions or str(
            getattr(termination_record, "ignition_set_sha256")
        ) != str(self._run_identity["ignition_set_sha256"]):
            raise ValueError(
                "Termination ignition identity does not match the collected run"
            )
        captured = int(getattr(termination_record, "captured_transition_count"))
        if captured != self._accepted_observations:
            raise ValueError(
                "Termination observation count does not match collected observations"
            )
        if self._last_timestep_t1 is None:
            raise ValueError("Cannot seal a run without an accepted transition")
        final_timestep = int(getattr(termination_record, "final_timestep"))
        if final_timestep != self._last_timestep_t1:
            raise ValueError(
                "Termination final timestep does not match the contiguous collected run"
            )
        final_counts: dict[str, int] = {}
        for name in ("final_ignited_count", "final_blazing_count"):
            final_counts[name] = int(getattr(termination_record, name))
            if final_counts[name] < 0:
                raise ValueError(f"Termination {name} must be non-negative")
        if int(getattr(termination_record, "expected_transition_count")) != captured and bool(
            getattr(termination_record, "authoritative")
        ):
            raise ValueError("An authoritative run cannot have transition count mismatch")
        self._sealed_run = CollectedSetCRun(
            schema_version=SET_C_RUN_SCHEMA_VERSION,
            run_identity_sha256=self._run_identity_sha256,
            provenance_sha256=self._provenance_sha256,
            run_identity=self._run_identity,
            provenance=self._provenance,
            batches=tuple(self._batches),
            observation_count=self._accepted_observations,
            row_count=self._accepted_rows,
            payload_nbytes=self._pending_bytes,
            released_batch_count=self._released_batches,
            released_row_count=self._released_rows,
            first_timestep_t=int(self._first_timestep_t),
            final_timestep_t1=self._last_timestep_t1,
            final_ignited_count=final_counts["final_ignited_count"],
            final_blazing_count=final_counts["final_blazing_count"],
            block_origin_row=self.block_origin_row,
            block_origin_col=self.block_origin_col,
            block_size_rows=self.block_rows,
            block_size_cols=self.block_cols,
            ignition_coordinates=identity_ignitions,
            ignition_set_sha256=str(self._run_identity["ignition_set_sha256"]),
            observation_pairs=tuple(self._observation_pairs),
            termination_record=termination_record,
        )
        return self._sealed_run


class SyntheticSetCPublisher:
    """Publish only complete synthetic test runs as one atomic package."""

    def __init__(self, config: Mapping[str, object]):
        dataset_config = config.get("dataset_generation")
        publisher_config = (
            dataset_config.get("set_c_publisher")
            if isinstance(dataset_config, Mapping)
            else None
        )
        if not isinstance(publisher_config, Mapping):
            raise ValueError("dataset_generation.set_c_publisher is required")
        if publisher_config.get("enabled") is not True:
            raise PermissionError("Set C publication is disabled by default")
        if publisher_config.get("mode") != SYNTHETIC_TEST_PUBLISHER_MODE:
            raise PermissionError(
                "Only synthetic_test_only publication is implemented; production "
                "publication requires separate future authorization"
            )
        authorization = publisher_config.get("authorization_reference")
        if not isinstance(authorization, str) or not authorization.strip():
            raise PermissionError(
                "Synthetic test publication requires an authorization reference"
            )
        self.authorization_reference = authorization.strip()

    @staticmethod
    def _termination_manifest(record: object) -> dict[str, object]:
        if is_dataclass(record):
            return asdict(record)
        names = (
            "schema_version",
            "termination_reason",
            "final_timestep",
            "final_ignited_count",
            "final_blazing_count",
            "expected_transition_count",
            "captured_transition_count",
            "completeness_status",
            "authoritative",
            "ignition_coordinates",
            "ignition_set_sha256",
        )
        return {name: getattr(record, name) for name in names}

    @staticmethod
    def _validate_locked_context(run: CollectedSetCRun) -> None:
        _require_lower_sha256(run.run_identity_sha256, "Run identity digest")
        _require_lower_sha256(run.provenance_sha256, "Run provenance digest")
        if not isinstance(run.run_identity, Mapping) or not isinstance(
            run.provenance, Mapping
        ):
            raise PermissionError("Sealed run identity and provenance must be mappings")
        try:
            identity_digest = canonical_metadata_hash(run.run_identity)
            provenance_digest = canonical_metadata_hash(run.provenance)
        except (TypeError, ValueError) as exc:
            raise PermissionError("Sealed run metadata is not canonical") from exc
        if identity_digest != run.run_identity_sha256:
            raise PermissionError("Sealed run identity digest is inconsistent")
        if provenance_digest != run.provenance_sha256:
            raise PermissionError("Sealed run provenance digest is inconsistent")

        identity = run.run_identity
        provenance = run.provenance
        for name in _RUN_IDENTITY_TEXT_FIELDS:
            if not isinstance(identity.get(name), str) or not str(
                identity[name]
            ).strip():
                raise PermissionError(f"Sealed run identity {name} is invalid")
        if not isinstance(identity.get("source_dirty"), (bool, np.bool_)):
            raise PermissionError("Sealed source_dirty identity must be boolean")
        shared_names = (*_RUN_IDENTITY_TEXT_FIELDS, "source_dirty", "input_hashes")
        for name in shared_names:
            if name not in identity or name not in provenance or (
                identity[name] != provenance[name]
            ):
                raise PermissionError(
                    f"Sealed run identity and provenance disagree on {name}"
                )
        for name in (
            "seed",
            "configuration_hash",
            "wind_manifest",
            "wind_weight",
            "simulation_valid_mask_sha256",
            "mapped_building_mask_sha256",
            "ignition_coordinates",
        ):
            if name not in identity or name not in provenance or (
                identity[name] != provenance[name]
            ):
                raise PermissionError(
                    f"Sealed run identity and provenance disagree on {name}"
                )
        if identity["transition_source_type"] != AUTHORIZED_SIMULATED_TRANSITION_SOURCE:
            raise PermissionError("Sealed transition source is not authorized simulated")
        if identity.get("observation_schema_version") != (
            TRANSITION_OBSERVATION_SCHEMA_VERSION
        ):
            raise PermissionError("Sealed observation schema is inconsistent")
        if identity.get("state_encoding") != STATE_ENCODING:
            raise PermissionError("Sealed state encoding is inconsistent")
        expected_rng = np.random.default_rng(
            int(identity["seed"])
        ).bit_generator.__class__.__name__
        if identity["rng_bit_generator"] != expected_rng:
            raise PermissionError("Sealed RNG identity does not match the seed contract")
        try:
            validate_transition_provenance(
                provenance,
                expected_configuration_hash=str(identity["configuration_hash"]),
                expected_seed=int(identity["seed"]),
                expected_wind_manifest=identity["wind_manifest"],
                expected_wind_weight=float(identity["wind_weight"]),
                expected_simulation_valid_mask_sha256=str(
                    identity["simulation_valid_mask_sha256"]
                ),
                expected_mapped_building_mask_sha256=str(
                    identity["mapped_building_mask_sha256"]
                ),
            )
            _, canonical_transform = validate_grid_provenance(
                identity.get("grid_crs"), identity.get("grid_transform")
            )
        except (TypeError, ValueError) as exc:
            raise PermissionError("Sealed run context is invalid") from exc
        if tuple(identity["grid_transform"]) != canonical_transform:
            raise PermissionError("Sealed grid transform is not canonical")
        if identity["ignition_coordinate_system"] != "grid_row_col":
            raise PermissionError("Sealed ignition coordinate system is inconsistent")
        _require_lower_sha256(
            identity["ignition_set_sha256"], "Sealed ignition-set digest"
        )
        try:
            raw_ignitions = tuple(identity["ignition_coordinates"])
            if any(
                isinstance(point, (str, bytes))
                or len(point) != 2
                or any(
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    for value in point
                )
                for point in raw_ignitions
            ):
                raise TypeError
            ignition_coordinates = tuple(
                (int(point[0]), int(point[1])) for point in raw_ignitions
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise PermissionError("Sealed ignition coordinates are invalid") from exc
        if (
            not ignition_coordinates
            or ignition_coordinates != tuple(sorted(set(ignition_coordinates)))
            or canonical_metadata_hash(ignition_coordinates)
            != identity["ignition_set_sha256"]
        ):
            raise PermissionError("Sealed ignition coordinates/hash are inconsistent")
        if tuple(identity["ignition_coordinates"]) != run.ignition_coordinates or (
            identity["ignition_set_sha256"] != run.ignition_set_sha256
        ):
            raise PermissionError("Sealed ignition identity is inconsistent")
        grid_shape = tuple(identity.get("grid_shape", ()))
        if len(grid_shape) != 2 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in grid_shape
        ):
            raise PermissionError("Sealed grid shape is invalid")
        grid_rows, grid_cols = (int(value) for value in grid_shape)
        if any(
            row < 0 or col < 0 or row >= grid_rows or col >= grid_cols
            for row, col in ignition_coordinates
        ):
            raise PermissionError("Sealed ignition coordinates are outside the grid")
        for name, value, positive in (
            ("block_origin_row", run.block_origin_row, False),
            ("block_origin_col", run.block_origin_col, False),
            ("block_size_rows", run.block_size_rows, True),
            ("block_size_cols", run.block_size_cols, True),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise PermissionError(f"Sealed {name} must be an integer")
            if positive and int(value) <= 0:
                raise PermissionError(f"Sealed {name} must be positive")

    @staticmethod
    def _validate_run(run: CollectedSetCRun) -> None:
        record = run.termination_record
        if run.schema_version != SET_C_RUN_SCHEMA_VERSION:
            raise ValueError("Unsupported collected Set C run schema")
        SyntheticSetCPublisher._validate_locked_context(run)
        if not run.complete_in_memory:
            raise PermissionError("A partial or previously drained run cannot be published")
        if getattr(record, "termination_reason", None) != "inactive":
            raise PermissionError("Only inactive termination can be published")
        if getattr(record, "completeness_status", None) != "complete":
            raise PermissionError("Only complete termination can be published")
        if getattr(record, "authoritative", None) is not True:
            raise PermissionError("Only authoritative termination can be published")
        if int(getattr(record, "final_ignited_count", -1)) != 0 or int(
            getattr(record, "final_blazing_count", -1)
        ) != 0:
            raise PermissionError("An active or incomplete run cannot be published")
        expected = int(getattr(record, "expected_transition_count", -1))
        captured = int(getattr(record, "captured_transition_count", -2))
        if expected != captured or captured != run.observation_count:
            raise PermissionError("Transition counts must match before publication")
        if tuple(getattr(record, "ignition_coordinates", ())) != run.ignition_coordinates:
            raise PermissionError("Termination ignition coordinates changed after sealing")
        if str(getattr(record, "ignition_set_sha256", "")) != run.ignition_set_sha256:
            raise PermissionError("Termination ignition hash changed after sealing")
        if int(getattr(record, "final_timestep", -1)) != run.final_timestep_t1:
            raise PermissionError("Termination timestep changed after sealing")
        if int(getattr(record, "final_ignited_count", -1)) != run.final_ignited_count or int(
            getattr(record, "final_blazing_count", -1)
        ) != run.final_blazing_count:
            raise PermissionError("Termination final counts changed after sealing")
        if len(run.observation_pairs) != run.observation_count:
            raise PermissionError("Observation ledger count is inconsistent")
        expected_timestep = run.first_timestep_t
        ledger_rows = 0
        for timestep_t, timestep_t1, row_count in run.observation_pairs:
            if timestep_t != expected_timestep or timestep_t1 != timestep_t + 1:
                raise PermissionError("Observation ledger is not ordered and contiguous")
            if row_count < 0:
                raise PermissionError("Observation ledger row count is invalid")
            expected_timestep = timestep_t1
            ledger_rows += row_count
        if expected_timestep != run.final_timestep_t1 or ledger_rows != run.row_count:
            raise PermissionError("Observation ledger does not bind the sealed run")
        if sum(batch.row_count for batch in run.batches) != run.row_count:
            raise PermissionError("Batch row counts do not match the collected run")
        if run.row_count <= 0:
            raise PermissionError("An empty Set C run cannot be published")

        observation_batch_indices: dict[int, list[int]] = {}
        observation_domain_hashes: dict[int, dict[str, str]] = {}
        for batch in run.batches:
            if batch.schema_version != SET_C_ROW_BATCH_SCHEMA_VERSION:
                raise PermissionError("Batch schema version is inconsistent")
            if batch.feature_schema_version != SET_C_SCHEMA_VERSION:
                raise PermissionError("Batch feature schema version is inconsistent")
            if batch.feature_names != CANONICAL_FEATURE_NAMES:
                raise PermissionError("Batch feature order is inconsistent")
            if batch.target_name != TARGET_NAME or batch.target_version != TARGET_VERSION:
                raise PermissionError("Batch target schema is inconsistent")
            if batch.positive_label != POSITIVE_LABEL:
                raise PermissionError("Batch positive label is inconsistent")
            if batch.positive_label != 1:
                raise PermissionError("Set C positive label must be integer 1")
            _require_lower_sha256(batch.run_identity_sha256, "Batch run identity digest")
            _require_lower_sha256(batch.provenance_sha256, "Batch provenance digest")
            if batch.run_identity_sha256 != run.run_identity_sha256:
                raise PermissionError("Batch run identity is inconsistent")
            try:
                batch_provenance_digest = canonical_metadata_hash(batch.provenance)
            except (TypeError, ValueError) as exc:
                raise PermissionError("Batch provenance is not canonical") from exc
            if batch.provenance_sha256 != run.provenance_sha256 or (
                batch_provenance_digest != run.provenance_sha256
            ):
                raise PermissionError("Batch provenance is inconsistent")
            identity = run.run_identity
            expected_domain_hashes = {
                "simulation_valid_mask_sha256": identity[
                    "simulation_valid_mask_sha256"
                ],
                "mapped_building_mask_sha256": identity[
                    "mapped_building_mask_sha256"
                ],
            }
            batch_domain_hashes = dict(batch.domain_mask_hashes)
            if set(batch_domain_hashes) != {
                "simulation_valid_mask_sha256",
                "mapped_building_mask_sha256",
                "eligible_mask_t_sha256",
            }:
                raise PermissionError("Batch domain-mask hash schema is inconsistent")
            for name, digest in batch_domain_hashes.items():
                _require_lower_sha256(digest, f"Batch domain-mask hash {name}")
            context_matches = (
                batch.transition_source_type == identity["transition_source_type"]
                and batch.source_observation_schema_version
                == identity["observation_schema_version"]
                and batch.state_encoding == identity["state_encoding"]
                and batch.seed == identity["seed"]
                and tuple(batch.grid_shape) == tuple(identity["grid_shape"])
                and batch.grid_crs == identity["grid_crs"]
                and tuple(batch.grid_transform) == tuple(identity["grid_transform"])
                and dict(batch.wind_manifest) == dict(identity["wind_manifest"])
                and batch.wind_weight == identity["wind_weight"]
                and all(
                    batch_domain_hashes[name] == digest
                    for name, digest in expected_domain_hashes.items()
                )
            )
            if not context_matches:
                raise PermissionError("Batch duplicated run context is inconsistent")
            if batch.features.dtype != np.float32 or batch.features.ndim != 2 or (
                batch.features.shape[1] != len(CANONICAL_FEATURE_NAMES)
            ):
                raise PermissionError("Batch predictor matrix is inconsistent")
            try:
                validate_feature_values(batch.features)
            except (TypeError, ValueError) as exc:
                raise PermissionError("Batch predictor values are invalid") from exc
            if batch.labels.dtype != np.int8 or batch.labels.shape != (
                batch.row_count,
            ):
                raise PermissionError("Batch labels are inconsistent")
            if not np.all(np.isin(batch.labels, (0, 1))):
                raise PermissionError("Batch labels must contain only integer 0 or 1")
            if batch.cell_rows.dtype != np.int64 or batch.cell_cols.dtype != np.int64:
                raise PermissionError("Batch cell coordinates must use int64")
            for values in (batch.cell_rows, batch.cell_cols):
                if values.shape != (batch.row_count,):
                    raise PermissionError("Batch row metadata is inconsistent")
            for values in (
                batch.features,
                batch.labels,
                batch.cell_rows,
                batch.cell_cols,
                batch.duplicate_group_ids,
                batch.spatial_block_ids,
            ):
                if values.flags.writeable:
                    raise PermissionError(
                        "Batch payload arrays must remain read-only after sealing"
                    )
            rows, cols = tuple(int(value) for value in batch.grid_shape)
            if np.any(batch.cell_rows < 0) or np.any(batch.cell_rows >= rows) or (
                np.any(batch.cell_cols < 0) or np.any(batch.cell_cols >= cols)
            ):
                raise PermissionError("Batch cell coordinates are outside grid bounds")
            _validate_sha256_identity_array(
                batch.duplicate_group_ids, "Duplicate identity"
            )
            _validate_sha256_identity_array(
                batch.spatial_block_ids, "Spatial identity"
            )
            if batch.duplicate_group_ids.shape != (batch.row_count,) or (
                batch.spatial_block_ids.shape != (batch.row_count,)
            ):
                raise PermissionError("Batch identity array shape is inconsistent")
            if not np.array_equal(
                batch.duplicate_group_ids, _sha256_rows(batch.features)
            ):
                raise PermissionError("Batch duplicate identities are inconsistent")
            expected_spatial_ids = _spatial_block_ids(
                grid_id=str(identity["grid_id"]),
                rows=batch.cell_rows,
                cols=batch.cell_cols,
                origin_row=run.block_origin_row,
                origin_col=run.block_origin_col,
                size_rows=run.block_size_rows,
                size_cols=run.block_size_cols,
            )
            if not np.array_equal(batch.spatial_block_ids, expected_spatial_ids):
                raise PermissionError("Batch spatial identities are inconsistent")
            if batch.timestep_t1 != batch.timestep_t + 1:
                raise PermissionError("Batch timestep pair is inconsistent")
            if batch.observation_index < 0 or batch.observation_index >= len(
                run.observation_pairs
            ):
                raise PermissionError("Batch observation index is inconsistent")
            pair = run.observation_pairs[batch.observation_index]
            if (batch.timestep_t, batch.timestep_t1) != pair[:2]:
                raise PermissionError("Batch timestep does not match observation ledger")
            observation_batch_indices.setdefault(batch.observation_index, []).append(
                batch.batch_index
            )
            prior_domain_hashes = observation_domain_hashes.setdefault(
                batch.observation_index, batch_domain_hashes
            )
            if prior_domain_hashes != batch_domain_hashes:
                raise PermissionError(
                    "Batch domain-mask hashes differ within one observation"
                )
        if [batch.observation_index for batch in run.batches] != sorted(
            batch.observation_index for batch in run.batches
        ):
            raise PermissionError("Batches are reordered across observations")
        for observation_index, indices in observation_batch_indices.items():
            if indices != list(range(len(indices))):
                raise PermissionError("Batch order is not contiguous within observation")
            if sum(
                batch.row_count
                for batch in run.batches
                if batch.observation_index == observation_index
            ) != run.observation_pairs[observation_index][2]:
                raise PermissionError("Batch rows do not match observation ledger")
        if sum(batch.payload_nbytes for batch in run.batches) != run.payload_nbytes:
            raise PermissionError("Batch byte counts do not match the collected run")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _batch_payload(batch: SetCRowBatch) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            features=batch.features,
            labels=batch.labels,
            cell_rows=batch.cell_rows,
            cell_cols=batch.cell_cols,
            duplicate_group_ids=batch.duplicate_group_ids,
            spatial_block_ids=batch.spatial_block_ids,
        )
        return buffer.getvalue()

    def publish(
        self,
        run: CollectedSetCRun,
        *,
        destination_root: Path,
        artifact_id: str,
        artifact_version: str,
    ) -> PublishedSetCTestArtifact:
        self._validate_run(run)
        for name, value in (
            ("artifact_id", artifact_id),
            ("artifact_version", artifact_version),
        ):
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
                raise ValueError(f"{name} must be a safe, path-free identifier")
        root = Path(destination_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                "Synthetic test destination_root must already exist as a directory"
            )
        final_path = root / f"{artifact_id}-{artifact_version}.setc.zip"
        if final_path.exists():
            raise FileExistsError(f"Refusing to overwrite Set C package: {final_path}")

        descriptor, staging_name = tempfile.mkstemp(
            prefix=f".{artifact_id}-{artifact_version}-",
            suffix=".tmp",
            dir=root,
        )
        os.close(descriptor)
        staging_path = Path(staging_name)
        try:
            batch_entries: list[dict[str, object]] = []
            with zipfile.ZipFile(
                staging_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for index, batch in enumerate(run.batches):
                    payload = self._batch_payload(batch)
                    entry_name = f"batches/{index:06d}.npz"
                    archive.writestr(entry_name, payload)
                    batch_entries.append(
                        {
                            "entry": entry_name,
                            "sha256": sha256(payload).hexdigest(),
                            "row_count": batch.row_count,
                            "payload_nbytes": batch.payload_nbytes,
                            "observation_index": batch.observation_index,
                            "batch_index": batch.batch_index,
                            "timestep_t": batch.timestep_t,
                            "timestep_t1": batch.timestep_t1,
                            "run_identity_sha256": batch.run_identity_sha256,
                            "state_encoding": batch.state_encoding,
                            "seed": batch.seed,
                            "grid_shape": list(batch.grid_shape),
                            "grid_crs": batch.grid_crs,
                            "grid_transform": list(batch.grid_transform),
                            "wind_manifest": _json_safe(batch.wind_manifest),
                            "wind_weight": batch.wind_weight,
                            "domain_mask_hashes": _json_safe(
                                batch.domain_mask_hashes
                            ),
                            "provenance": _json_safe(batch.provenance),
                            "provenance_sha256": batch.provenance_sha256,
                        }
                    )
                manifest = {
                    "manifest_schema_version": SET_C_PUBLICATION_MANIFEST_VERSION,
                    "artifact_kind": "synthetic_set_c_test_package",
                    "artifact_id": artifact_id,
                    "artifact_version": artifact_version,
                    "authorization_reference": self.authorization_reference,
                    "feature_schema_version": SET_C_SCHEMA_VERSION,
                    "feature_names": list(CANONICAL_FEATURE_NAMES),
                    "feature_dtype": "float32",
                    "target_name": TARGET_NAME,
                    "target_version": TARGET_VERSION,
                    "positive_label": POSITIVE_LABEL,
                    "sampling_policy": "all_eligible",
                    "duplicate_identity_version": DUPLICATE_IDENTITY_VERSION,
                    "spatial_block_identity_version": SPATIAL_BLOCK_IDENTITY_VERSION,
                    "observation_count": run.observation_count,
                    "row_count": run.row_count,
                    "final_ignited_count": run.final_ignited_count,
                    "final_blazing_count": run.final_blazing_count,
                    "retained_numpy_payload_nbytes": run.payload_nbytes,
                    "memory_bound_scope": MEMORY_BOUND_SCOPE,
                    "payload_mutability_contract": PAYLOAD_MUTABILITY_CONTRACT,
                    "run_identity_sha256": run.run_identity_sha256,
                    "provenance_sha256": run.provenance_sha256,
                    "run_identity": _json_safe(run.run_identity),
                    "provenance": _json_safe(run.provenance),
                    "spatial_block": {
                        "origin_row": run.block_origin_row,
                        "origin_col": run.block_origin_col,
                        "size_rows": run.block_size_rows,
                        "size_cols": run.block_size_cols,
                    },
                    "observation_pairs": [
                        list(pair) for pair in run.observation_pairs
                    ],
                    "termination": self._termination_manifest(
                        run.termination_record
                    ),
                    "batches": batch_entries,
                }
                manifest_text = json.dumps(
                    manifest, sort_keys=True, separators=(",", ":")
                )
                manifest = json.loads(manifest_text)
                archive.writestr("manifest.json", manifest_text)

            with staging_path.open("rb+") as file_obj:
                os.fsync(file_obj.fileno())
            with zipfile.ZipFile(staging_path, "r") as archive:
                loaded_manifest = json.loads(
                    archive.read("manifest.json").decode("utf-8")
                )
                if loaded_manifest != manifest:
                    raise RuntimeError("Staged Set C manifest validation failed")
                for entry in batch_entries:
                    payload = archive.read(str(entry["entry"]))
                    if sha256(payload).hexdigest() != entry["sha256"]:
                        raise RuntimeError("Staged Set C batch hash validation failed")

            artifact_hash = self._file_sha256(staging_path)
            os.link(staging_path, final_path)
            return PublishedSetCTestArtifact(
                path=final_path,
                sha256=artifact_hash,
                manifest=MappingProxyType(manifest),
            )
        finally:
            staging_path.unlink(missing_ok=True)
