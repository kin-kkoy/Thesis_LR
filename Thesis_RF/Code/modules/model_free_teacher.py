"""Dedicated in-memory runner for the approved model-free Set C teacher."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import operator

import numpy as np

from .automata_engine import FireAutomata
from .ca_state import STATE_BLAZING
from .set_c_collector import (
    RUNNER_OWNED_BACKPRESSURE_POLICY,
    CollectorBackpressureError,
    SetCCollector,
)
from .transition_observer import (
    TransitionObservation,
    TransitionObserver,
    canonical_metadata_hash,
)


TEACHER_RUNNER_SCHEMA_VERSION = "model_free_teacher_run.v1"
TERMINATION_INACTIVE = "inactive"
TERMINATION_SAFETY_LIMIT = "safety_limit"
TERMINATION_FAILURE = "failure"
COMPLETENESS_COMPLETE = "complete"
COMPLETENESS_CENSORED = "censored"
COMPLETENESS_FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TeacherTerminationRecord:
    """Immutable outcome for one in-memory model-free teacher run."""

    schema_version: str
    termination_reason: str
    final_timestep: int
    final_ignited_count: int
    final_blazing_count: int
    expected_transition_count: int
    captured_transition_count: int
    completeness_status: str
    authoritative: bool
    ignition_coordinates: tuple[tuple[int, int], ...]
    ignition_set_sha256: str


class ModelFreeTeacherRunError(RuntimeError):
    """Expose a non-authoritative termination record when a run fails."""

    def __init__(
        self,
        message: str,
        termination_record: TeacherTerminationRecord,
    ) -> None:
        super().__init__(message)
        self.termination_record = termination_record


def canonicalize_ignition_coordinates(
    points: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Return unique grid coordinates in deterministic row-major order."""
    canonical: list[tuple[int, int]] = []
    for point in points:
        if isinstance(point, (str, bytes)) or len(point) != 2:
            raise ValueError("Each ignition coordinate must contain [row, col]")
        try:
            row = operator.index(point[0])
            col = operator.index(point[1])
        except TypeError as exc:
            raise TypeError("Ignition row and column values must be integers") from exc
        if isinstance(point[0], (bool, np.bool_)) or isinstance(
            point[1], (bool, np.bool_)
        ):
            raise TypeError("Ignition row and column values must not be boolean")
        canonical.append((int(row), int(col)))

    if not canonical:
        raise ValueError("At least one explicit ignition coordinate is required")
    if len(set(canonical)) != len(canonical):
        raise ValueError("Ignition coordinates must not contain duplicates")
    return tuple(sorted(canonical))


def ignition_set_sha256(
    canonical_coordinates: tuple[tuple[int, int], ...],
) -> str:
    """Hash canonical grid coordinates using the repository metadata contract."""
    return canonical_metadata_hash(canonical_coordinates)


def _teacher_safety_limit(config: Mapping[str, object]) -> int:
    teacher_config = config.get("model_free_teacher")
    if not isinstance(teacher_config, Mapping):
        raise ValueError("model_free_teacher configuration is required")
    if teacher_config.get("enabled") is not True:
        raise ValueError("model_free_teacher.enabled must be true for an explicit run")
    if teacher_config.get("runner_schema_version") != TEACHER_RUNNER_SCHEMA_VERSION:
        raise ValueError(
            "model_free_teacher.runner_schema_version must be "
            f"{TEACHER_RUNNER_SCHEMA_VERSION!r}"
        )
    if teacher_config.get("persistent_output_enabled", False) is not False:
        raise ValueError(
            "model_free_teacher.persistent_output_enabled must remain false"
        )
    safety_limit = teacher_config.get("safety_limit_timesteps")
    if isinstance(safety_limit, (bool, np.bool_)) or not isinstance(
        safety_limit, (int, np.integer)
    ):
        raise TypeError(
            "model_free_teacher.safety_limit_timesteps must be an explicit integer"
        )
    if int(safety_limit) <= 0:
        raise ValueError("model_free_teacher.safety_limit_timesteps must be positive")
    return int(safety_limit)


def _termination_record(
    automata: FireAutomata,
    *,
    reason: str,
    expected_transition_count: int,
    captured_transition_count: int,
    ignition_coordinates: tuple[tuple[int, int], ...],
    ignition_hash: str,
) -> TeacherTerminationRecord:
    counts = automata.get_state_counts()
    if reason == TERMINATION_INACTIVE:
        completeness = COMPLETENESS_COMPLETE
        authoritative = expected_transition_count == captured_transition_count
    elif reason == TERMINATION_SAFETY_LIMIT:
        completeness = COMPLETENESS_CENSORED
        authoritative = False
    else:
        completeness = COMPLETENESS_FAILED
        authoritative = False
    return TeacherTerminationRecord(
        schema_version=TEACHER_RUNNER_SCHEMA_VERSION,
        termination_reason=reason,
        final_timestep=int(automata.timestep),
        final_ignited_count=counts["ignited"],
        final_blazing_count=counts["blazing"],
        expected_transition_count=expected_transition_count,
        captured_transition_count=captured_transition_count,
        completeness_status=completeness,
        authoritative=authoritative,
        ignition_coordinates=ignition_coordinates,
        ignition_set_sha256=ignition_hash,
    )


def run_model_free_teacher(
    environment: dict,
    config: dict,
    *,
    ignition_points: Sequence[Sequence[int]],
    transition_provenance: Mapping[str, object],
    transition_observer: TransitionObserver | None = None,
    set_c_collector: SetCCollector | None = None,
    backpressure_drain: Callable[[SetCCollector], None] | None = None,
) -> TeacherTerminationRecord:
    """Run the stochastic teacher entirely in memory and return its outcome."""
    if transition_observer is not None and not callable(transition_observer):
        raise TypeError("transition_observer must be callable when provided")
    if set_c_collector is not None and not isinstance(
        set_c_collector, SetCCollector
    ):
        raise TypeError("set_c_collector must be a SetCCollector")
    if transition_observer is None and set_c_collector is None:
        raise ValueError("A transition observer or Set C collector is required")
    if backpressure_drain is not None and not callable(backpressure_drain):
        raise TypeError("backpressure_drain must be callable when provided")
    if backpressure_drain is not None and set_c_collector is None:
        raise ValueError("backpressure_drain requires a Set C collector")
    if backpressure_drain is not None:
        collector_config = config.get("dataset_generation", {}).get(
            "set_c_collector", {}
        )
        if (
            collector_config.get("backpressure_policy")
            != RUNNER_OWNED_BACKPRESSURE_POLICY
        ):
            raise PermissionError(
                "Runner-owned exact-retry backpressure requires explicit approval"
            )
    if not isinstance(transition_provenance, Mapping):
        raise TypeError("transition_provenance must be a mapping")

    safety_limit = _teacher_safety_limit(config)
    canonical_ignitions = canonicalize_ignition_coordinates(ignition_points)
    ignition_hash = ignition_set_sha256(canonical_ignitions)

    grid_shape = tuple(environment.get("grid_shape", ()))
    if len(grid_shape) != 2:
        raise ValueError("environment.grid_shape must contain two dimensions")
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    burnable = np.asarray(environment.get("burnable_mask"), dtype=bool)
    nodata = np.asarray(environment.get("nodata_mask"), dtype=bool)
    buildings = np.asarray(environment.get("building_presence"))
    if any(array.shape != (rows, cols) for array in (burnable, nodata, buildings)):
        raise ValueError("Teacher domain arrays must match environment.grid_shape")
    teacher_domain = burnable & (~nodata) & (buildings > 0)
    for row, col in canonical_ignitions:
        if row < 0 or col < 0 or row >= rows or col >= cols:
            raise ValueError(f"Ignition coordinate ({row}, {col}) is out of bounds")
        if not teacher_domain[row, col]:
            raise ValueError(
                f"Ignition coordinate ({row}, {col}) is outside the burnable domain"
            )

    provenance = dict(transition_provenance)
    for name in ("scenario_family_id", "rng_bit_generator"):
        if not isinstance(provenance.get(name), str) or not provenance[name].strip():
            raise ValueError(f"transition_provenance.{name} is required")
    expected_bit_generator = np.random.default_rng(
        int(config["simulation"]["seed"])
    ).bit_generator.__class__.__name__
    if provenance["rng_bit_generator"] != expected_bit_generator:
        raise ValueError(
            "transition_provenance.rng_bit_generator does not match default_rng"
        )
    expected_ignition_fields = {
        "ignition_coordinate_system": "grid_row_col",
        "ignition_coordinates": canonical_ignitions,
        "ignition_set_sha256": ignition_hash,
    }
    for name, expected_value in expected_ignition_fields.items():
        if name in provenance and provenance[name] != expected_value:
            raise ValueError(
                f"transition_provenance.{name} does not bind the active ignition set"
            )
        provenance[name] = expected_value

    captured_transition_count = 0

    def counted_observer(observation: TransitionObservation) -> None:
        nonlocal captured_transition_count
        if set_c_collector is not None:
            while True:
                try:
                    set_c_collector(observation)
                    break
                except CollectorBackpressureError as exc:
                    if exc.permanent or backpressure_drain is None:
                        raise
                    before = (
                        set_c_collector.pending_rows,
                        set_c_collector.pending_bytes,
                    )
                    backpressure_drain(set_c_collector)
                    after = (
                        set_c_collector.pending_rows,
                        set_c_collector.pending_bytes,
                    )
                    if after[0] >= before[0] and after[1] >= before[1]:
                        raise RuntimeError(
                            "Approved backpressure drain made no progress"
                        ) from exc
        if transition_observer is not None and transition_observer is not set_c_collector:
            transition_observer(observation)
        captured_transition_count += 1

    def failed_record(record: TeacherTerminationRecord) -> TeacherTerminationRecord:
        return TeacherTerminationRecord(
            schema_version=record.schema_version,
            termination_reason=TERMINATION_FAILURE,
            final_timestep=record.final_timestep,
            final_ignited_count=record.final_ignited_count,
            final_blazing_count=record.final_blazing_count,
            expected_transition_count=record.expected_transition_count,
            captured_transition_count=record.captured_transition_count,
            completeness_status=COMPLETENESS_FAILED,
            authoritative=False,
            ignition_coordinates=record.ignition_coordinates,
            ignition_set_sha256=record.ignition_set_sha256,
        )

    def finalize_collector(record: TeacherTerminationRecord) -> None:
        if set_c_collector is None:
            return
        try:
            set_c_collector.finalize(record)
        except Exception as exc:
            failure = failed_record(record)
            raise ModelFreeTeacherRunError(
                "Set C collector finalization failed and the run is non-authoritative",
                failure,
            ) from exc

    automata = FireAutomata(
        environment,
        config,
        transition_observer=counted_observer,
        transition_provenance=provenance,
        model_free=True,
    )
    automata.set_ignition(list(canonical_ignitions))
    if any(automata.grid[row, col] != STATE_BLAZING for row, col in canonical_ignitions):
        raise RuntimeError("The complete canonical ignition set was not initialized")

    expected_transition_count = 0
    for _ in range(safety_limit):
        expected_transition_count += 1
        try:
            automata.step()
        except Exception as exc:
            record = _termination_record(
                automata,
                reason=TERMINATION_FAILURE,
                expected_transition_count=expected_transition_count,
                captured_transition_count=captured_transition_count,
                ignition_coordinates=canonical_ignitions,
                ignition_hash=ignition_hash,
            )
            finalize_collector(record)
            raise ModelFreeTeacherRunError(
                "Model-free teacher run failed and is non-authoritative",
                record,
            ) from exc

        if not automata.is_active():
            record = _termination_record(
                automata,
                reason=TERMINATION_INACTIVE,
                expected_transition_count=expected_transition_count,
                captured_transition_count=captured_transition_count,
                ignition_coordinates=canonical_ignitions,
                ignition_hash=ignition_hash,
            )
            finalize_collector(record)
            return record

    record = _termination_record(
        automata,
        reason=TERMINATION_SAFETY_LIMIT,
        expected_transition_count=expected_transition_count,
        captured_transition_count=captured_transition_count,
        ignition_coordinates=canonical_ignitions,
        ignition_hash=ignition_hash,
    )
    finalize_collector(record)
    return record
