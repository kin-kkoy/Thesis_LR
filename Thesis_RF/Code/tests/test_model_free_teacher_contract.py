from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from modules.model_free_teacher import (
    COMPLETENESS_CENSORED,
    COMPLETENESS_COMPLETE,
    COMPLETENESS_FAILED,
    ModelFreeTeacherRunError,
    TERMINATION_FAILURE,
    TERMINATION_INACTIVE,
    TERMINATION_SAFETY_LIMIT,
    canonicalize_ignition_coordinates,
    ignition_set_sha256,
    run_model_free_teacher,
)
from modules.set_c_collector import SetCCollector
from modules.transition_observer import (
    AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
    SOURCE_ONLY_AUTHORIZATION,
    canonical_metadata_hash,
    mask_sha256,
)
from modules.wind_convention import (
    CALM_REPRESENTATION,
    DIRECTION_CONVENTION,
    DIRECTION_UNITS,
    NORTH_REFERENCE,
    WIND_SCHEMA_VERSION,
    WindContract,
)


def _environment() -> dict:
    shape = (1, 2)
    return {
        "slope_risk": np.zeros(shape, dtype=np.float32),
        "proximity_risk": np.zeros(shape, dtype=np.float32),
        "building_presence": np.ones(shape, dtype=np.float32),
        "material_class": np.ones(shape, dtype=np.int8),
        "burnable_mask": np.ones(shape, dtype=bool),
        "nodata_mask": np.zeros(shape, dtype=bool),
        "grid_shape": shape,
        "transform": (1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
        "crs": "EPSG:32651",
    }


def _config(*, safety_limit: int = 3, blazing_duration: int = 1) -> dict:
    return {
        "simulation": {"seed": 17},
        "model_free_teacher": {
            "enabled": True,
            "runner_schema_version": "model_free_teacher_run.v1",
            "safety_limit_timesteps": safety_limit,
            "persistent_output_enabled": False,
        },
        "wind": {
            "speed_kmh": 10.0,
            "direction_deg": 0.0,
            "direction_convention": DIRECTION_CONVENTION,
            "direction_units": DIRECTION_UNITS,
            "north_reference": NORTH_REFERENCE,
            "schema_version": WIND_SCHEMA_VERSION,
            "calm_representation": CALM_REPRESENTATION,
        },
        "placeholder_transition": {
            "base_ignition_prob": 0.0,
            "wind_weight": 0.2,
            "T_3_to_4": 1,
            "T_4_to_5": blazing_duration,
        },
        "ml_model": {
            "enabled": True,
            "inference_mode": "stochastic_probability",
            "threshold": None,
        },
        "dataset_generation": {
            "sampling_policy": "all_eligible",
            "set_c_collector": {
                "enabled": True,
                "max_buffer_rows": 8,
                "max_buffer_bytes": 4096,
                "batch_rows": 2,
                "backpressure_policy": "runner_owned_exact_retry.v1",
            },
            "spatial_block": {
                "origin_row": 0,
                "origin_col": 0,
                "size_rows": 1,
                "size_cols": 1,
            },
        },
    }


def _provenance(config: dict) -> dict:
    domain = np.ones((1, 2), dtype=bool)
    return {
        "set_id": "synthetic-set-c",
        "experiment_id": "synthetic-experiment",
        "event_id": "synthetic-event",
        "scenario_id": "synthetic-scenario",
        "scenario_family_id": "synthetic-scenario-family",
        "run_id": "synthetic-run",
        "grid_id": "synthetic-grid",
        "simulation_valid_mask_id": "synthetic-valid",
        "mapped_building_mask_id": "synthetic-buildings",
        "simulator_id": "model-free-fire-automata",
        "simulator_version": "model_free_teacher_run.v1",
        "source_revision": "synthetic-revision",
        "source_dirty": False,
        "configuration_hash": canonical_metadata_hash(config),
        "input_hashes": {"synthetic-input": "a" * 64},
        "authorization": SOURCE_ONLY_AUTHORIZATION,
        "authorization_reference": "D-014 implementation-only approval",
        "transition_source_type": AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
        "seed": config["simulation"]["seed"],
        "rng_bit_generator": "PCG64",
        "wind_manifest": WindContract.from_config(config["wind"]).to_manifest(),
        "wind_weight": config["placeholder_transition"]["wind_weight"],
        "simulation_valid_mask_sha256": mask_sha256(domain),
        "mapped_building_mask_sha256": mask_sha256(domain),
        "environment_id": "synthetic-environment",
        "execution_id": "synthetic-execution",
        "capture_source_id": "synthetic-model-free-step",
    }


def test_inactive_run_is_authoritative_and_records_canonical_ignition():
    config = _config()
    observations = []

    record = run_model_free_teacher(
        _environment(),
        config,
        ignition_points=[(0, 0)],
        transition_observer=observations.append,
        transition_provenance=_provenance(config),
    )

    assert record.termination_reason == TERMINATION_INACTIVE
    assert record.completeness_status == COMPLETENESS_COMPLETE
    assert record.authoritative is True
    assert record.final_timestep == 1
    assert record.final_ignited_count == 0
    assert record.final_blazing_count == 0
    assert record.expected_transition_count == 1
    assert record.captured_transition_count == 1
    assert record.ignition_coordinates == ((0, 0),)
    assert record.ignition_set_sha256 == ignition_set_sha256(((0, 0),))
    assert observations[0].state_t.dtype == np.int8
    assert observations[0].state_t1.dtype == np.int8
    assert observations[0].provenance["ignition_coordinates"] == ((0, 0),)
    assert (
        observations[0].provenance["ignition_set_sha256"]
        == record.ignition_set_sha256
    )
    with pytest.raises(FrozenInstanceError):
        record.final_timestep = 99


def test_safety_limit_is_censored_and_non_authoritative():
    config = _config(safety_limit=1, blazing_duration=5)

    record = run_model_free_teacher(
        _environment(),
        config,
        ignition_points=[(0, 0)],
        transition_observer=lambda observation: None,
        transition_provenance=_provenance(config),
    )

    assert record.termination_reason == TERMINATION_SAFETY_LIMIT
    assert record.completeness_status == COMPLETENESS_CENSORED
    assert record.authoritative is False
    assert record.final_timestep == 1
    assert record.final_blazing_count == 1
    assert record.expected_transition_count == 1
    assert record.captured_transition_count == 1


def test_observer_failure_returns_failed_record_on_exception():
    config = _config()

    def fail_observer(_observation):
        raise RuntimeError("synthetic observer failure")

    with pytest.raises(ModelFreeTeacherRunError) as exc_info:
        run_model_free_teacher(
            _environment(),
            config,
            ignition_points=[(0, 0)],
            transition_observer=fail_observer,
            transition_provenance=_provenance(config),
        )

    record = exc_info.value.termination_record
    assert record.termination_reason == TERMINATION_FAILURE
    assert record.completeness_status == COMPLETENESS_FAILED
    assert record.authoritative is False
    assert record.expected_transition_count == 1
    assert record.captured_transition_count == 0


def test_teacher_requires_explicit_enablement_and_non_persistent_mode():
    config = _config()
    config["model_free_teacher"]["enabled"] = False
    with pytest.raises(ValueError, match="enabled must be true"):
        run_model_free_teacher(
            _environment(),
            config,
            ignition_points=[(0, 0)],
            transition_observer=lambda observation: None,
            transition_provenance=_provenance(config),
        )

    config = _config()
    config["model_free_teacher"]["persistent_output_enabled"] = True
    with pytest.raises(ValueError, match="must remain false"):
        run_model_free_teacher(
            _environment(),
            config,
            ignition_points=[(0, 0)],
            transition_observer=lambda observation: None,
            transition_provenance=_provenance(config),
        )


def test_ignition_coordinates_are_canonical_and_fail_closed():
    assert canonicalize_ignition_coordinates([(1, 2), (0, 3)]) == (
        (0, 3),
        (1, 2),
    )
    with pytest.raises(ValueError, match="duplicates"):
        canonicalize_ignition_coordinates([(0, 0), (0, 0)])
    with pytest.raises(TypeError, match="integers"):
        canonicalize_ignition_coordinates([(0.5, 1)])


def test_teacher_rejects_ignition_outside_burnable_domain():
    config = _config()
    environment = _environment()
    environment["building_presence"][0, 1] = 0.0

    with pytest.raises(ValueError, match="outside the burnable domain"):
        run_model_free_teacher(
            environment,
            config,
            ignition_points=[(0, 1)],
            transition_observer=lambda observation: None,
            transition_provenance=_provenance(config),
        )


def test_teacher_immediately_routes_each_observation_to_set_c_collector():
    config = _config()
    environment = _environment()
    collector = SetCCollector(environment, config)
    raw_observations = []

    record = run_model_free_teacher(
        environment,
        config,
        ignition_points=[(0, 0)],
        transition_provenance=_provenance(config),
        transition_observer=raw_observations.append,
        set_c_collector=collector,
    )

    sealed = collector.finalize(record)
    assert record.authoritative is True
    assert collector.observation_count == 1
    assert sealed.observation_count == record.captured_transition_count == 1
    assert sealed.row_count == 1
    assert sealed.batches[0].features.shape == (1, 11)
    assert len(raw_observations) == 1
    assert collector.live_full_grid_observation_count == 0


def test_runner_drains_then_retries_the_exact_rejected_observation_once():
    config = _config(safety_limit=3, blazing_duration=2)
    config["dataset_generation"]["set_c_collector"]["max_buffer_rows"] = 1
    config["dataset_generation"]["set_c_collector"]["batch_rows"] = 1
    environment = _environment()

    class RecordingCollector(SetCCollector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.attempts = []

        def __call__(self, observation):
            self.attempts.append((id(observation), observation.timestep_t))
            return super().__call__(observation)

    collector = RecordingCollector(environment, config)
    drained = []

    def drain_one(owned_collector):
        batch = owned_collector.pop_batch()
        drained.append((batch.timestep_t, batch.timestep_t1, batch.row_count))

    record = run_model_free_teacher(
        environment,
        config,
        ignition_points=[(0, 0)],
        transition_provenance=_provenance(config),
        set_c_collector=collector,
        backpressure_drain=drain_one,
    )

    assert record.authoritative is True
    assert record.captured_transition_count == 2
    assert collector.observation_count == 2
    assert [attempt[1] for attempt in collector.attempts] == [0, 1, 1]
    assert collector.attempts[1][0] == collector.attempts[2][0]
    assert drained == [(0, 1, 1)]
    assert [(batch.timestep_t, batch.timestep_t1) for batch in collector.pending_batches()] == [
        (1, 2)
    ]


def test_runner_fails_nonauthoritatively_when_one_observation_cannot_fit():
    config = _config()
    config["dataset_generation"]["set_c_collector"]["max_buffer_bytes"] = 188
    collector = SetCCollector(_environment(), config)
    drain_calls = []

    with pytest.raises(ModelFreeTeacherRunError) as exc_info:
        run_model_free_teacher(
            _environment(),
            config,
            ignition_points=[(0, 0)],
            transition_provenance=_provenance(config),
            set_c_collector=collector,
            backpressure_drain=lambda _collector: drain_calls.append(True),
        )

    assert drain_calls == []
    assert collector.observation_count == 0
    assert exc_info.value.termination_record.termination_reason == TERMINATION_FAILURE
    assert exc_info.value.termination_record.completeness_status == COMPLETENESS_FAILED
    assert exc_info.value.termination_record.authoritative is False


def test_collector_finalization_failure_replaces_authoritative_record(monkeypatch):
    config = _config()
    collector = SetCCollector(_environment(), config)

    def fail_finalize(_record):
        raise ValueError("synthetic finalization failure")

    monkeypatch.setattr(collector, "finalize", fail_finalize)
    with pytest.raises(ModelFreeTeacherRunError) as exc_info:
        run_model_free_teacher(
            _environment(),
            config,
            ignition_points=[(0, 0)],
            transition_provenance=_provenance(config),
            set_c_collector=collector,
        )

    failed = exc_info.value.termination_record
    assert failed.termination_reason == TERMINATION_FAILURE
    assert failed.completeness_status == COMPLETENESS_FAILED
    assert failed.authoritative is False
    assert failed.final_timestep == 1
    assert failed.captured_transition_count == 1
