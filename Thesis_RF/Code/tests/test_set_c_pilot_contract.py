from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import modules.set_c_pilot as pilot
import orchestrator
import run_set_c_pilot as pilot_cli
from modules.model_free_teacher import (
    COMPLETENESS_CENSORED,
    COMPLETENESS_COMPLETE,
    COMPLETENESS_FAILED,
    ModelFreeTeacherRunError,
    TERMINATION_FAILURE,
    TERMINATION_INACTIVE,
    TERMINATION_SAFETY_LIMIT,
    TeacherTerminationRecord,
)
from modules.transition_observer import (
    AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
    build_transition_observation,
)
from modules.wind_convention import WindContract


def _raw_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "config" / "default_experiment.yaml"
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def _active_config() -> dict:
    return pilot.activate_approved_pilot_config(_raw_config())


def _environment() -> dict:
    shape = (3, 3)
    return {
        "slope_risk": np.full(shape, 0.25, dtype=np.float32),
        "proximity_risk": np.full(shape, 0.5, dtype=np.float32),
        "building_presence": np.ones(shape, dtype=np.int8),
        "material_class": np.ones(shape, dtype=np.int8),
        "burnable_mask": np.ones(shape, dtype=bool),
        "nodata_mask": np.zeros(shape, dtype=bool),
        "grid_shape": shape,
        "transform": (3.0, 0.0, 100.0, 0.0, -3.0, 200.0, 0.0, 0.0, 1.0),
        "crs": "EPSG:32651",
    }


def _source_context() -> dict:
    return {
        "source_revision": "synthetic-revision",
        "source_dirty": False,
        "source_hashes": {"synthetic-source.py": "a" * 64},
        "input_hashes": {"synthetic-environment.tif": "b" * 64},
        "generation_command": "synthetic-test-only",
        "environment_record": {"python": "synthetic"},
    }


def _resource_probe(
    _destination: Path,
    *,
    available: int = pilot.LAUNCH_MEMORY_FLOOR_BYTES + 1,
) -> pilot.ResourceSnapshot:
    return pilot.ResourceSnapshot(
        available_memory_bytes=available,
        process_rss_bytes=10_000_000,
        free_disk_bytes=pilot.DISK_FLOOR_BYTES + 1,
        captured_at_utc="2026-09-24T00:00:00+00:00",
    )


def _observation(
    environment: dict,
    config: dict,
    provenance: dict,
    *,
    timestep: int = 0,
    row_count: int = 8,
    positive_count: int = 1,
):
    if not 0 <= positive_count <= row_count <= 8:
        raise ValueError("synthetic observation counts are invalid")
    state_t = np.full((3, 3), 5, dtype=np.int8)
    state_t[1, 1] = 4
    eligible = [
        (row, col)
        for row in range(3)
        for col in range(3)
        if (row, col) != (1, 1)
    ][:row_count]
    for row, col in eligible:
        state_t[row, col] = 2
    state_t1 = state_t.copy()
    for row, col in eligible[:positive_count]:
        state_t1[row, col] = 3
    domain = np.ones((3, 3), dtype=bool)
    return build_transition_observation(
        state_t=state_t,
        state_t1=state_t1,
        transition_source_type=AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
        timestep_t=timestep,
        timestep_t1=timestep + 1,
        simulation_valid_mask=domain,
        mapped_building_mask=domain,
        wind_manifest=WindContract.from_config(config["wind"]).to_manifest(),
        wind_weight=config["placeholder_transition"]["wind_weight"],
        seed=config["simulation"]["seed"],
        provenance=provenance,
        grid_crs=environment["crs"],
        grid_transform=environment["transform"],
    )


def _teacher(*args, censored_seed: int | None = None, **kwargs):
    environment, config = args
    collector = kwargs["set_c_collector"]
    observer = kwargs["transition_observer"]
    run_id = kwargs["transition_provenance"]["run_id"]
    expected = pilot.V1_REPLAY_EXPECTATIONS.get(run_id, (8, 1, 7, 1))
    rows, positives, _negatives, transitions = expected
    rows_remaining = rows
    positives_remaining = positives
    for timestep in range(transitions):
        transitions_remaining = transitions - timestep
        row_count = min(8, rows_remaining - (transitions_remaining - 1))
        positive_count = min(positives_remaining, row_count)
        observation = _observation(
            environment,
            config,
            kwargs["transition_provenance"],
            timestep=timestep,
            row_count=row_count,
            positive_count=positive_count,
        )
        collector(observation)
        observer(observation)
        rows_remaining -= row_count
        positives_remaining -= positive_count
    assert rows_remaining == 0
    assert positives_remaining == 0
    censored = config["simulation"]["seed"] == censored_seed
    return TeacherTerminationRecord(
        schema_version="model_free_teacher_run.v1",
        termination_reason=(
            TERMINATION_SAFETY_LIMIT if censored else TERMINATION_INACTIVE
        ),
        final_timestep=transitions,
        final_ignited_count=1 if censored else 0,
        final_blazing_count=1 if censored else 0,
        expected_transition_count=transitions,
        captured_transition_count=transitions,
        completeness_status=(
            COMPLETENESS_CENSORED if censored else COMPLETENESS_COMPLETE
        ),
        authoritative=not censored,
        ignition_coordinates=tuple(kwargs["ignition_points"]),
        ignition_set_sha256=kwargs["transition_provenance"]["ignition_set_sha256"],
    )


def _canonical(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


def _expected_aggregate_teacher_contract(config: dict) -> dict:
    return {
        "wind_manifest": {
            "speed_kmh": 10.0,
            "direction_deg": 315.0,
            "direction_convention": "meteorological_from",
            "direction_units": "degrees_clockwise",
            "north_reference": "projected_grid_north",
            "schema_version": "simulated_wind.v1",
            "calm_representation": "speed_zero_direction_zero",
        },
        "wind_weight": 0.20,
        "placeholder_transition": {
            "base_ignition_prob": 0.12,
            "slope_weight": 0.20,
            "building_weight": 0.10,
            "proximity_weight": 0.20,
            "wind_weight": 0.20,
            "burnout_prob": 0.04,
            "material_weight": 0.35,
        },
        "flammability_weights": {
            "base_weight": 0.5,
            "slope_weight": 0.3,
            "proximity_weight": 0.2,
        },
        "scenario_families": [
            {
                "family_id": family["family_id"],
                "scenario_id": family["scenario_id"],
                "ignition_set_sha256": family["ignition_set_sha256"],
                "component_size": family["component_size"],
                "row_ceiling": family["row_ceiling"],
                "wind_manifest": family["wind_manifest"],
                "seeds": family["seeds"],
            }
            for family in config["phase8_set_c_pilot"]["families"]
        ],
    }


def test_exact_approved_matrix_and_checked_in_gate():
    config = _raw_config()
    assert config["simulation"]["max_timesteps"] == 300
    assert config["phase8_set_c_pilot"]["enabled"] is False
    assert config["phase8_set_c_pilot"]["safety_limit_timesteps"] == 256
    assert [spec.run_id for spec in pilot.RUN_MATRIX[:8]] == [
        "p8f01_w315_s10__seed_81001",
        "p8f01_w315_s10__seed_81002",
        "p8f02_w315_s10__seed_81001",
        "p8f02_w315_s10__seed_81002",
        "p8f03_w315_s10__seed_81001",
        "p8f03_w315_s10__seed_81002",
        "p8f04_w315_s10__seed_81001",
        "p8f04_w315_s10__seed_81002",
    ]
    assert len(pilot.RUN_MATRIX) == 32
    assert [spec.seed for spec in pilot.RUN_MATRIX[8:16]] == list(range(82001, 82009))
    assert [spec.seed for spec in pilot.RUN_MATRIX[16:24]] == list(range(83001, 83009))
    assert [spec.seed for spec in pilot.RUN_MATRIX[24:32]] == list(range(84001, 84009))
    assert [family.component_size for family in pilot.FAMILIES] == [
        3976,
        5454,
        3811,
        3535,
        1921,
        388,
    ]
    assert [family.row_ceiling for family in pilot.FAMILIES] == [
        1_017_856,
        1_396_224,
        975_616,
        904_960,
        491_776,
        99_328,
    ]
    assert [family.wind_direction_deg for family in pilot.FAMILIES] == [
        315.0,
        315.0,
        315.0,
        315.0,
        45.0,
        135.0,
    ]
    assert pilot.TOTAL_ROW_CEILING == 20_557_824
    assert pilot.V1_AGGREGATE_PAYLOAD_SHA256 == (
        "23cd71ebca4837e094eabec2a02c95e3869e9b7ae5aac28b2b5eb40b3a98afc6"
    )
    assert pilot.V1_AGGREGATE_FILE_SHA256 == (
        "e4c52c15ca14286ec8926898e082166b7481ad480970db74877d10b00b7bd332"
    )


def test_each_family_uses_its_exact_approved_wind_and_seed_matrix(tmp_path):
    observed_winds = {}

    def recording_teacher(*args, **kwargs):
        run_id = kwargs["transition_provenance"]["run_id"]
        observed_winds[run_id] = dict(args[1]["wind"])
        return _teacher(*args, **kwargs)

    result = pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=tmp_path / "pilot.aggregate.json",
        source_context=_source_context(),
        resource_probe=_resource_probe,
        teacher_runner=recording_teacher,
        synthetic_test=True,
    )

    assert result["status"] == "complete"
    assert observed_winds == {
        spec.run_id: spec.family.wind_manifest for spec in pilot.RUN_MATRIX
    }


def test_v1_replay_mismatch_fails_closed_with_aggregate_only_report(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"

    def mismatched_teacher(*args, **kwargs):
        environment, config = args
        observation = _observation(
            environment, config, kwargs["transition_provenance"]
        )
        kwargs["set_c_collector"](observation)
        kwargs["transition_observer"](observation)
        return TeacherTerminationRecord(
            schema_version="model_free_teacher_run.v1",
            termination_reason=TERMINATION_INACTIVE,
            final_timestep=1,
            final_ignited_count=0,
            final_blazing_count=0,
            expected_transition_count=1,
            captured_transition_count=1,
            completeness_status=COMPLETENESS_COMPLETE,
            authoritative=True,
            ignition_coordinates=tuple(kwargs["ignition_points"]),
            ignition_set_sha256=kwargs["transition_provenance"][
                "ignition_set_sha256"
            ],
        )

    result = pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        teacher_runner=mismatched_teacher,
        synthetic_test=True,
    )
    report = json.loads(destination.read_text(encoding="utf-8"))

    assert result["status"] == "failed"
    assert result["acceptance_status"] == "fail"
    assert report["failure"]["type"] == "PilotContractError"
    assert "V1 replay reconciliation failed" in report["failure"]["message"]
    assert report["aggregate"]["runs"] == []
    assert report["acceptance"]["gates"]["v1_replay_exactly_reconciled"] is False
    assert not {
        "features",
        "labels",
        "duplicate_group_ids",
        "spatial_block_ids",
        "ignition_coordinates",
    }.intersection(_keys(report))


def test_pilot_provenance_hashes_the_memory_optimized_loader():
    assert "modules/data_loader.py" in orchestrator.PILOT_SOURCE_RELATIVE_PATHS


def test_pilot_is_disabled_before_environment_or_model_access(monkeypatch):
    environment_accessed = False

    class ForbiddenEnvironment:
        def __init__(self, _config):
            nonlocal environment_accessed
            environment_accessed = True

    monkeypatch.setattr(orchestrator, "EnvironmentManager", ForbiddenEnvironment)
    monkeypatch.setattr(
        orchestrator,
        "load_model",
        lambda *_args, **_kwargs: pytest.fail("pilot attempted estimator loading"),
    )
    with pytest.raises(PermissionError, match="disabled"):
        orchestrator.run_set_c_feasibility_pilot(_raw_config())
    assert not environment_accessed


def test_cli_requires_exact_execution_flag(monkeypatch):
    monkeypatch.setattr(
        pilot_cli,
        "load_config",
        lambda *_args: pytest.fail("disabled CLI accessed configuration"),
    )
    with pytest.raises(SystemExit):
        pilot_cli.main([])


def test_streaming_pilot_writes_one_self_hashed_aggregate_only_report(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"
    result = pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        teacher_runner=_teacher,
        synthetic_test=True,
    )

    assert result["status"] == "complete"
    assert result["run_count"] == 32
    assert list(tmp_path.iterdir()) == [destination]
    report = json.loads(destination.read_text(encoding="utf-8"))
    payload_hash = report.pop("payload_sha256")
    assert payload_hash == sha256(_canonical(report)).hexdigest()
    assert report["aggregate"]["totals"] == {
        "rows": 280,
        "class_counts": {"negative_0": 247, "positive_1": 33},
        "positive_prevalence": 33 / 280,
        "scenario_family_count": 6,
        "run_count": 32,
    }
    assert result["acceptance_status"] == "fail"
    assert report["acceptance"]["gates"]["v1_replay_exactly_reconciled"] is True
    assert report["acceptance"]["gates"]["both_new_families_contain_both_labels"] is True
    assert report["integrity"]["row_level_payload_retained"] is False
    assert report["integrity"]["row_level_payload_published"] is False
    assert report["integrity"]["model_accessed"] is False
    assert report["integrity"]["stack_ground_truth_accessed"] is False
    teacher_contract = report["contract"]["approved_model_free_teacher_contract"]
    assert teacher_contract == _expected_aggregate_teacher_contract(_active_config())
    assert not {
        "features",
        "labels",
        "duplicate_group_ids",
        "spatial_block_ids",
        "block_ids",
        "cell_ids",
        "cell_rows",
        "cell_cols",
        "ignition_coordinates",
        "row_level_data",
    }.intersection(_keys(report))
    assert report["provenance"]["grid_and_domain"]["grid_identity_sha256"] == (
        pilot.canonical_metadata_hash(
            {
                "shape": _environment()["grid_shape"],
                "crs": _environment()["crs"],
                "transform": _environment()["transform"],
            }
        )
    )
    assert not {
        "features",
        "labels",
        "cell_rows",
        "cell_cols",
        "duplicate_group_ids",
        "spatial_block_ids",
        "ignition_coordinates",
    }.intersection(_keys(report))
    assert all(
        candidate["diagnostic_only"]
        for candidate in report["aggregate"]["candidate_blocks"]
    )
    assert all(
        candidate["leakage_connectivity"]["split_roles_assigned"] is False
        for candidate in report["aggregate"]["candidate_blocks"]
    )


def test_pilot_refuses_overwrite_without_invoking_teacher(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"
    destination.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        pilot.run_set_c_pilot(
            _environment(),
            _active_config(),
            destination=destination,
            source_context=_source_context(),
            resource_probe=_resource_probe,
            teacher_runner=lambda *_args, **_kwargs: pytest.fail("teacher invoked"),
            synthetic_test=True,
        )
    assert destination.read_text(encoding="utf-8") == "existing"


def test_memory_preflight_fails_without_teacher_or_output(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"

    def low_memory(path):
        return _resource_probe(path, available=pilot.LAUNCH_MEMORY_FLOOR_BYTES - 1)

    with pytest.raises(pilot.PilotResourceError, match="6 GiB"):
        pilot.run_set_c_pilot(
            _environment(),
            _active_config(),
            destination=destination,
            source_context=_source_context(),
            resource_probe=low_memory,
            teacher_runner=lambda *_args, **_kwargs: pytest.fail("teacher invoked"),
            synthetic_test=True,
        )
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_censored_run_is_diagnostic_and_does_not_assign_roles(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"

    def teacher(*args, **kwargs):
        return _teacher(*args, censored_seed=83001, **kwargs)

    pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        teacher_runner=teacher,
        synthetic_test=True,
    )
    report = json.loads(destination.read_text(encoding="utf-8"))
    censored = [
        run
        for run in report["aggregate"]["runs"]
        if run["termination"]["reason"] == TERMINATION_SAFETY_LIMIT
    ]
    assert len(censored) == 1
    assert all(run["termination"]["authoritative"] is False for run in censored)
    assert report["contract"]["split_roles_assigned"] is False
    assert report["acceptance"]["status"] == "fail"


def test_midpilot_failure_creates_only_non_authoritative_aggregate(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"

    def failing_teacher(*_args, **_kwargs):
        raise RuntimeError("synthetic fail-closed stop")

    result = pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        teacher_runner=failing_teacher,
        synthetic_test=True,
    )
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert report["status"] == "failed"
    assert report["failure"]["type"] == "RuntimeError"
    assert report["integrity"]["count_contract_passed"] is False
    assert report["contract"]["approved_model_free_teacher_contract"] == (
        _expected_aggregate_teacher_contract(_active_config())
    )
    assert not {
        "features",
        "labels",
        "duplicate_group_ids",
        "spatial_block_ids",
        "block_ids",
        "cell_ids",
        "cell_rows",
        "cell_cols",
        "ignition_coordinates",
        "row_level_data",
    }.intersection(_keys(report))
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(
    ("section", "field", "drifted_value"),
    [
        ("wind", "speed_kmh", 11.0),
        ("wind", "direction_deg", 314.0),
        ("wind", "direction_convention", "mathematical_to"),
        ("wind", "direction_units", "radians"),
        ("wind", "north_reference", "true_north"),
        ("wind", "schema_version", "simulated_wind.v2"),
        ("wind", "calm_representation", "null_direction"),
        ("placeholder_transition", "base_ignition_prob", 0.13),
        ("placeholder_transition", "slope_weight", 0.21),
        ("placeholder_transition", "building_weight", 0.11),
        ("placeholder_transition", "proximity_weight", 0.21),
        ("placeholder_transition", "wind_weight", 0.21),
        ("placeholder_transition", "burnout_prob", 0.05),
        ("placeholder_transition", "material_weight", 0.36),
        ("flammability_weights", "base_weight", 0.6),
        ("flammability_weights", "slope_weight", 0.4),
        ("flammability_weights", "proximity_weight", 0.3),
    ],
)
def test_explicit_teacher_contract_drift_fails_before_teacher_or_output(
    tmp_path, section, field, drifted_value
):
    config = _active_config()
    config[section][field] = drifted_value
    invoked = False

    def forbidden_teacher(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        pytest.fail("teacher invoked after aggregate contract drift")

    with pytest.raises(pilot.PilotContractError, match="does not match"):
        pilot.run_set_c_pilot(
            _environment(),
            config,
            destination=tmp_path / "pilot.aggregate.json",
            source_context=_source_context(),
            resource_probe=_resource_probe,
            teacher_runner=forbidden_teacher,
            synthetic_test=True,
        )
    assert invoked is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [
        ("family_id", "changed_family"),
        ("scenario_id", "changed_scenario"),
        ("ignition_set_sha256", "f" * 64),
        ("component_size", 1),
        ("row_ceiling", 1),
        ("wind_manifest", {"speed_kmh": 99.0}),
        ("seeds", [1]),
    ],
)
def test_aggregate_safe_family_contract_drift_fails_before_teacher_or_output(
    tmp_path, field, drifted_value
):
    config = _active_config()
    config["phase8_set_c_pilot"]["families"][0][field] = drifted_value
    with pytest.raises(pilot.PilotContractError, match="family/ignition matrix"):
        pilot.run_set_c_pilot(
            _environment(),
            config,
            destination=tmp_path / "pilot.aggregate.json",
            source_context=_source_context(),
            resource_probe=_resource_probe,
            teacher_runner=lambda *_args, **_kwargs: pytest.fail("teacher invoked"),
            synthetic_test=True,
        )
    assert list(tmp_path.iterdir()) == []


def test_aggregate_publisher_rejects_row_payload_and_atomic_overwrite(tmp_path):
    destination = tmp_path / "aggregate.json"
    with pytest.raises(pilot.PilotContractError, match="forbidden key"):
        pilot.publish_aggregate_report({"labels": [0, 1]}, destination)
    assert list(tmp_path.iterdir()) == []

    pilot.publish_aggregate_report({"status": "synthetic"}, destination)
    original = destination.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        pilot.publish_aggregate_report({"status": "replacement"}, destination)
    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transition_source", "rf-assisted"),
        ("estimator_enabled", True),
        ("normal_ca_ml_path_enabled", True),
        ("safety_limit_timesteps", 300),
    ],
)
def test_pilot_rejects_model_or_matrix_contract_drift(tmp_path, field, value):
    config = _active_config()
    config["phase8_set_c_pilot"][field] = value
    with pytest.raises(pilot.PilotContractError):
        pilot.run_set_c_pilot(
            _environment(),
            config,
            destination=tmp_path / "pilot.aggregate.json",
            source_context=_source_context(),
            resource_probe=_resource_probe,
            teacher_runner=lambda *_args, **_kwargs: pytest.fail("teacher invoked"),
            synthetic_test=True,
        )
    assert list(tmp_path.iterdir()) == []


def test_ground_truth_provenance_is_rejected_without_teacher_or_write(tmp_path):
    context = _source_context()
    context["input_hashes"] = {"stack_ground_truth.tif": "a" * 64}
    with pytest.raises(PermissionError, match="ground_truth"):
        pilot.run_set_c_pilot(
            _environment(),
            _active_config(),
            destination=tmp_path / "pilot.aggregate.json",
            source_context=context,
            resource_probe=_resource_probe,
            teacher_runner=lambda *_args, **_kwargs: pytest.fail("teacher invoked"),
            synthetic_test=True,
        )
    assert list(tmp_path.iterdir()) == []


def test_production_rejects_teacher_substitution_before_environment_or_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        pilot,
        "_validate_environment",
        lambda *_args, **_kwargs: pytest.fail("environment validation was reached"),
    )
    destination = tmp_path / "pilot.aggregate.json"
    with pytest.raises(PermissionError, match="exact approved run_model_free_teacher"):
        pilot.run_set_c_pilot(
            _environment(),
            _active_config(),
            destination=destination,
            source_context=_source_context(),
            teacher_runner=_teacher,
            synthetic_test=False,
        )
    assert list(tmp_path.iterdir()) == []


def test_production_rejects_resource_probe_substitution_before_environment_or_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        pilot,
        "_validate_environment",
        lambda *_args, **_kwargs: pytest.fail("environment validation was reached"),
    )
    destination = tmp_path / "pilot.aggregate.json"
    with pytest.raises(PermissionError, match="exact approved resource telemetry"):
        pilot.run_set_c_pilot(
            _environment(),
            _active_config(),
            destination=destination,
            source_context=_source_context(),
            resource_probe=_resource_probe,
            synthetic_test=False,
        )
    assert list(tmp_path.iterdir()) == []


def test_observation_resource_extrema_and_post_merge_samples_are_reported(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"
    calls = 0
    observation_minimum = pilot.RUNTIME_MEMORY_FLOOR_BYTES + 123
    observation_maximum_rss = pilot.RSS_CEILING_BYTES - 123

    def sequenced_probe(path):
        nonlocal calls
        index = calls
        calls += 1
        snapshot = _resource_probe(path)
        if index == 2:
            return pilot.ResourceSnapshot(
                available_memory_bytes=observation_minimum,
                process_rss_bytes=observation_maximum_rss,
                free_disk_bytes=snapshot.free_disk_bytes,
                captured_at_utc=snapshot.captured_at_utc,
            )
        return snapshot

    pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        resource_probe=sequenced_probe,
        teacher_runner=_teacher,
        synthetic_test=True,
    )
    report = json.loads(destination.read_text(encoding="utf-8"))
    summary = report["resource_summary"]
    assert summary["minimum_available_memory_bytes"] == observation_minimum
    assert summary["maximum_process_rss_bytes"] == observation_maximum_rss
    # launch + pre/post/post-merge per run + every transition observation
    expected_observations = sum(
        pilot.V1_REPLAY_EXPECTATIONS.get(spec.run_id, (0, 0, 0, 1))[3]
        for spec in pilot.RUN_MATRIX
    )
    assert summary["sample_count"] == (
        1 + 3 * len(pilot.RUN_MATRIX) + expected_observations
    )


def test_teacher_failure_report_preserves_only_aggregate_partial_evidence(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"

    def partially_failing_teacher(*args, **kwargs):
        environment, config = args
        observation = _observation(
            environment, config, kwargs["transition_provenance"]
        )
        kwargs["set_c_collector"](observation)
        kwargs["transition_observer"](observation)
        raise RuntimeError("synthetic teacher failure after one observation")

    result = pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        teacher_runner=partially_failing_teacher,
        synthetic_test=True,
    )
    report = json.loads(destination.read_text(encoding="utf-8"))
    payload_hash = report.pop("payload_sha256")
    assert payload_hash == sha256(_canonical(report)).hexdigest()
    failed = report["failure"]["run"]
    assert result["status"] == "failed"
    assert failed["run_id"] == pilot.RUN_MATRIX[0].run_id
    assert failed["scenario_family_id"] == pilot.RUN_MATRIX[0].family.family_id
    assert failed["seed"] == pilot.RUN_MATRIX[0].seed
    assert failed["termination"]["authoritative"] is False
    assert failed["termination"]["expected_transition_count"] is None
    assert failed["termination"]["captured_transition_count"] == 1
    assert failed["partial_counts"] == {
        "observation_count": 1,
        "rows": 8,
        "negative_0": 7,
        "positive_1": 1,
    }
    assert failed["memory_backpressure"]["no_progress_detected"] is False
    assert failed["memory_backpressure"]["overflow_detected"] is False
    assert not {
        "features",
        "labels",
        "cell_rows",
        "cell_cols",
        "duplicate_group_ids",
        "spatial_block_ids",
        "ignition_coordinates",
    }.intersection(_keys(failed))
    with pytest.raises(FileExistsError, match="overwrite"):
        pilot.run_set_c_pilot(
            _environment(),
            _active_config(),
            destination=destination,
            source_context=_source_context(),
            resource_probe=_resource_probe,
            teacher_runner=partially_failing_teacher,
            synthetic_test=True,
        )


def test_teacher_failure_report_preserves_available_termination_counts(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"

    def recorded_failure(*args, **kwargs):
        environment, config = args
        observation = _observation(
            environment, config, kwargs["transition_provenance"]
        )
        kwargs["set_c_collector"](observation)
        kwargs["transition_observer"](observation)
        raise ModelFreeTeacherRunError(
            "synthetic recorded teacher failure",
            TeacherTerminationRecord(
                schema_version="model_free_teacher_run.v1",
                termination_reason=TERMINATION_FAILURE,
                final_timestep=1,
                final_ignited_count=3,
                final_blazing_count=4,
                expected_transition_count=2,
                captured_transition_count=1,
                completeness_status=COMPLETENESS_FAILED,
                authoritative=False,
                ignition_coordinates=tuple(kwargs["ignition_points"]),
                ignition_set_sha256=kwargs["transition_provenance"][
                    "ignition_set_sha256"
                ],
            ),
        )

    pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        teacher_runner=recorded_failure,
        synthetic_test=True,
    )
    report = json.loads(destination.read_text(encoding="utf-8"))
    termination = report["failure"]["run"]["termination"]
    assert termination == {
        "reason": TERMINATION_FAILURE,
        "completeness_status": COMPLETENESS_FAILED,
        "authoritative": False,
        "final_timestep": 1,
        "final_ignited_count": 3,
        "final_blazing_count": 4,
        "expected_transition_count": 2,
        "captured_transition_count": 1,
    }


def test_resource_failure_report_includes_failing_observation_extrema(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"
    calls = 0
    failed_available = pilot.RUNTIME_MEMORY_FLOOR_BYTES - 1

    def failing_probe(path):
        nonlocal calls
        index = calls
        calls += 1
        snapshot = _resource_probe(path)
        if index == 2:
            return pilot.ResourceSnapshot(
                available_memory_bytes=failed_available,
                process_rss_bytes=snapshot.process_rss_bytes,
                free_disk_bytes=snapshot.free_disk_bytes,
                captured_at_utc=snapshot.captured_at_utc,
            )
        return snapshot

    result = pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        resource_probe=failing_probe,
        teacher_runner=_teacher,
        synthetic_test=True,
    )
    report = json.loads(destination.read_text(encoding="utf-8"))
    failed = report["failure"]["run"]
    assert result["status"] == "failed"
    assert report["failure"]["type"] == "PilotResourceError"
    assert failed["termination"]["authoritative"] is False
    assert failed["partial_counts"]["observation_count"] == 1
    assert failed["resource_summary"]["minimum_available_memory_bytes"] == failed_available
    assert report["resource_summary"]["minimum_available_memory_bytes"] == failed_available


def test_leakage_component_inventory_is_deterministic_and_aggregate_only(tmp_path):
    destination = tmp_path / "pilot.aggregate.json"
    pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        teacher_runner=_teacher,
        synthetic_test=True,
    )
    report = json.loads(destination.read_text(encoding="utf-8"))
    expected_families = sorted(family.family_id for family in pilot.FAMILIES)
    for candidate in report["aggregate"]["candidate_blocks"]:
        connectivity = candidate["leakage_connectivity"]
        assert connectivity["authoritative_component_count"] == 1
        assert connectivity["authoritative_components_with_both_labels"] == 1
        assert connectivity["four_role_mathematically_feasible"] is False
        assert connectivity["split_roles_assigned"] is False
        component = connectivity["components"][0]
        size = (candidate["size_rows"], candidate["size_cols"])
        assert component["component_ordinal"] == 1
        assert component["scenario_family_ids"] == expected_families
        assert component["family_count"] == 6
        assert component["run_count"] == 32
        assert component["rows"] == 280
        assert component["class_counts"] == {
            "negative_0": 247,
            "positive_1": 33,
        }
        assert component["both_classes_present"] is True
        assert component["authoritative_only"] is True
        assert component["component_sha256"] == pilot.canonical_metadata_hash(
            {
                "candidate_block_size": size,
                "scenario_family_ids": tuple(expected_families),
            }
        )
        assert component["duplicate_group_count"] > 0
        assert component["block_group_count"] > 0
    assert not {
        "duplicate_group_ids",
        "spatial_block_ids",
        "cell_rows",
        "cell_cols",
        "ignition_coordinates",
    }.intersection(_keys(report))


def test_acceptance_requires_both_new_labels_and_four_components_at_every_size(
    tmp_path,
):
    destination = tmp_path / "pilot.aggregate.json"
    pilot.run_set_c_pilot(
        _environment(),
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        teacher_runner=_teacher,
        synthetic_test=True,
    )
    sections = json.loads(destination.read_text(encoding="utf-8"))["aggregate"]
    for candidate in sections["candidate_blocks"]:
        connectivity = candidate["leakage_connectivity"]
        connectivity["authoritative_component_count"] = 4
        connectivity["authoritative_components_with_both_labels"] = 4
        connectivity["four_role_mathematically_feasible"] = True

    accepted = pilot._acceptance_summary("complete", sections)
    assert accepted["status"] == "pass"
    assert accepted["all_gates_passed"] is True
    assert accepted["candidate_block_coverage"] == {
        "32x32": True,
        "64x64": True,
        "128x128": True,
    }

    new_family = next(
        family
        for family in sections["scenario_families"]
        if family["scenario_family_id"] == "p8v2f05_cluster_4146_m3"
    )
    new_family["class_counts"]["positive_1"] = 0
    rejected = pilot._acceptance_summary("complete", sections)
    assert rejected["status"] == "fail"
    assert (
        rejected["gates"]["both_new_families_contain_both_labels"] is False
    )


def _patch_small_production_grid_contract(monkeypatch, environment):
    shape = environment["grid_shape"]
    valid = environment["burnable_mask"] & (~environment["nodata_mask"])
    mapped = environment["building_presence"] > 0
    domain = valid & mapped
    monkeypatch.setattr(pilot, "EXPECTED_GRID_SHAPE", shape)
    monkeypatch.setattr(
        pilot, "EXPECTED_SIMULATION_VALID_COUNT", int(np.count_nonzero(valid))
    )
    monkeypatch.setattr(
        pilot, "EXPECTED_MAPPED_BUILDING_COUNT", int(np.count_nonzero(mapped))
    )
    monkeypatch.setattr(
        pilot, "EXPECTED_AUTHORITATIVE_DOMAIN_COUNT", int(np.count_nonzero(domain))
    )
    monkeypatch.setattr(
        pilot, "EXPECTED_SIMULATION_VALID_SHA256", pilot.mask_sha256(valid)
    )
    monkeypatch.setattr(
        pilot, "EXPECTED_MAPPED_BUILDING_SHA256", pilot.mask_sha256(mapped)
    )
    monkeypatch.setattr(
        pilot, "EXPECTED_AUTHORITATIVE_DOMAIN_SHA256", pilot.mask_sha256(domain)
    )
    monkeypatch.setattr(
        pilot,
        "APPROVED_GRID_IDENTITY_SHA256",
        pilot.canonical_metadata_hash(
            {
                "shape": shape,
                "crs": environment["crs"],
                "transform": environment["transform"],
            }
        ),
    )


def test_grid_identity_is_recomputed_from_shape_crs_and_full_transform(monkeypatch):
    environment = _environment()
    _patch_small_production_grid_contract(monkeypatch, environment)
    identity = pilot._validate_environment(environment, synthetic=False)
    assert identity["grid_identity_sha256"] == pilot.APPROVED_GRID_IDENTITY_SHA256
    assert "approved_grid_identity_sha256" not in identity


@pytest.mark.parametrize(
    "mutation",
    [
        "shifted_origin",
        "altered_crs",
        "altered_dimensions",
        "altered_pixel_transform",
    ],
)
def test_grid_identity_rejects_any_spatial_contract_change(monkeypatch, mutation):
    baseline = _environment()
    _patch_small_production_grid_contract(monkeypatch, baseline)
    environment = dict(baseline)
    if mutation == "shifted_origin":
        transform = list(environment["transform"])
        transform[2] += 3.0
        environment["transform"] = tuple(transform)
    elif mutation == "altered_crs":
        environment["crs"] = "EPSG:4326"
    elif mutation == "altered_dimensions":
        shape = (3, 4)
        environment.update(
            {
                "grid_shape": shape,
                "burnable_mask": np.ones(shape, dtype=bool),
                "nodata_mask": np.zeros(shape, dtype=bool),
                "building_presence": np.ones(shape, dtype=np.int8),
                "material_class": np.ones(shape, dtype=np.int8),
            }
        )
    else:
        transform = list(environment["transform"])
        transform[0] = 6.0
        environment["transform"] = tuple(transform)
    with pytest.raises(pilot.PilotContractError, match="does not match Task 8.1"):
        pilot._validate_environment(environment, synthetic=False)


@pytest.mark.parametrize("mismatch_direction", ["building_without_material", "material_without_building"])
def test_building_and_material_domains_must_match_bidirectionally(mismatch_direction):
    environment = _environment()
    if mismatch_direction == "building_without_material":
        environment["material_class"] = environment["material_class"].copy()
        environment["material_class"][0, 0] = 0
    else:
        environment["building_presence"] = environment["building_presence"].copy()
        environment["building_presence"][0, 0] = 0
    with pytest.raises(pilot.PilotContractError, match="bidirectionally"):
        pilot._validate_environment(environment, synthetic=True)
