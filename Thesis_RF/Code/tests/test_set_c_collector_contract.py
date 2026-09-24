from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from modules.feature_pipeline import CANONICAL_FEATURE_NAMES
from modules.model_free_teacher import (
    COMPLETENESS_CENSORED,
    COMPLETENESS_COMPLETE,
    COMPLETENESS_FAILED,
    TERMINATION_FAILURE,
    TERMINATION_INACTIVE,
    TERMINATION_SAFETY_LIMIT,
    TeacherTerminationRecord,
    ignition_set_sha256,
)
from modules.set_c_collector import (
    CollectorBackpressureError,
    SET_C_PUBLICATION_MANIFEST_VERSION,
    SET_C_ROW_BATCH_SCHEMA_VERSION,
    SetCCollector,
    SyntheticSetCPublisher,
)
from modules.transition_observer import (
    AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
    SOURCE_ONLY_AUTHORIZATION,
    build_transition_observation,
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


def _wind() -> dict:
    return {
        "speed_kmh": 10.0,
        "direction_deg": 0.0,
        "direction_convention": DIRECTION_CONVENTION,
        "direction_units": DIRECTION_UNITS,
        "north_reference": NORTH_REFERENCE,
        "schema_version": WIND_SCHEMA_VERSION,
        "calm_representation": CALM_REPRESENTATION,
    }


def _environment() -> dict:
    shape = (3, 3)
    return {
        "slope_risk": np.full(shape, 0.25, dtype=np.float32),
        "proximity_risk": np.full(shape, 0.5, dtype=np.float32),
        "building_presence": np.ones(shape, dtype=np.float32),
        "material_class": np.ones(shape, dtype=np.int8),
        "burnable_mask": np.ones(shape, dtype=bool),
        "nodata_mask": np.zeros(shape, dtype=bool),
        "grid_shape": shape,
        "transform": (1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
        "crs": "EPSG:32651",
    }


def _config(
    *,
    max_rows: int = 32,
    max_bytes: int = 16_384,
    batch_rows: int = 3,
    publisher_enabled: bool = False,
) -> dict:
    return {
        "simulation": {"seed": 23},
        "wind": _wind(),
        "placeholder_transition": {"wind_weight": 0.2},
        "flammability_weights": {},
        "dataset_generation": {
            "sampling_policy": "all_eligible",
            "set_c_collector": {
                "enabled": True,
                "max_buffer_rows": max_rows,
                "max_buffer_bytes": max_bytes,
                "batch_rows": batch_rows,
                "backpressure_policy": "runner_owned_exact_retry.v1",
            },
            "set_c_publisher": {
                "enabled": publisher_enabled,
                "mode": "synthetic_test_only",
                "authorization_reference": "Task 7.3 synthetic tests",
            },
            "spatial_block": {
                "origin_row": 0,
                "origin_col": 0,
                "size_rows": 2,
                "size_cols": 2,
            },
        },
    }


def _provenance(config: dict) -> dict:
    domain = np.ones((3, 3), dtype=bool)
    return {
        "set_id": "SetC",
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
        "ignition_coordinate_system": "grid_row_col",
        "ignition_coordinates": ((1, 1),),
        "ignition_set_sha256": ignition_set_sha256(((1, 1),)),
    }


def _observation(config: dict, *, timestep_t: int = 0, provenance: dict | None = None):
    state_t = np.full((3, 3), 2, dtype=np.int8)
    state_t[1, 1] = 4
    state_t1 = state_t.copy()
    state_t1[0, 1] = 3
    domain = np.ones((3, 3), dtype=bool)
    return build_transition_observation(
        state_t=state_t,
        state_t1=state_t1,
        transition_source_type=AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
        timestep_t=timestep_t,
        timestep_t1=timestep_t + 1,
        simulation_valid_mask=domain,
        mapped_building_mask=domain,
        wind_manifest=WindContract.from_config(config["wind"]).to_manifest(),
        wind_weight=config["placeholder_transition"]["wind_weight"],
        seed=config["simulation"]["seed"],
        provenance=_provenance(config) if provenance is None else provenance,
        grid_crs="EPSG:32651",
        grid_transform=(1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
    )


def _complete_record(observation_count: int = 1) -> TeacherTerminationRecord:
    return TeacherTerminationRecord(
        schema_version="model_free_teacher_run.v1",
        termination_reason=TERMINATION_INACTIVE,
        final_timestep=observation_count,
        final_ignited_count=0,
        final_blazing_count=0,
        expected_transition_count=observation_count,
        captured_transition_count=observation_count,
        completeness_status=COMPLETENESS_COMPLETE,
        authoritative=True,
        ignition_coordinates=((1, 1),),
        ignition_set_sha256=ignition_set_sha256(((1, 1),)),
    )


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def test_collector_emits_every_eligible_row_in_canonical_order():
    config = _config()
    observation = _observation(config)
    collector = SetCCollector(_environment(), config)

    collector(observation)
    batches = collector.pending_batches()

    assert [batch.row_count for batch in batches] == [3, 3, 2]
    features = np.concatenate([batch.features for batch in batches])
    labels = np.concatenate([batch.labels for batch in batches])
    rows = np.concatenate([batch.cell_rows for batch in batches])
    cols = np.concatenate([batch.cell_cols for batch in batches])
    duplicate_ids = np.concatenate(
        [batch.duplicate_group_ids for batch in batches]
    )
    selected = np.flatnonzero(observation.eligible_mask_t.ravel())

    assert features.shape == (8, 11)
    assert features.dtype == np.float32
    assert batches[0].schema_version == SET_C_ROW_BATCH_SCHEMA_VERSION
    assert batches[0].feature_names == CANONICAL_FEATURE_NAMES
    assert labels.dtype == np.int8
    assert labels.tolist() == observation.newly_ignited_mask_t1.ravel()[
        selected
    ].astype(np.int8).tolist()
    assert features[:, CANONICAL_FEATURE_NAMES.index("neighbor_burning_count")].tolist() == (
        observation.blazing_neighbor_count_t.ravel()[selected]
        .astype(np.float32)
        .tolist()
    )
    assert features[:, CANONICAL_FEATURE_NAMES.index("wind_weighted_score")].tolist() == pytest.approx(
        observation.wind_weighted_score_t.ravel()[selected]
    )
    expected_rows, expected_cols = np.unravel_index(selected, (3, 3))
    assert rows.tolist() == expected_rows.tolist()
    assert cols.tolist() == expected_cols.tolist()
    assert len(set(duplicate_ids.tolist())) <= 8
    assert all(len(value) == 64 for value in duplicate_ids.tolist())
    assert all(batch.provenance["run_id"] == "synthetic-run" for batch in batches)
    assert batches[0].grid_crs == "EPSG:32651"
    assert batches[0].grid_transform == (
        1.0,
        0.0,
        100.0,
        0.0,
        -1.0,
        200.0,
    )
    assert batches[0].domain_mask_hashes["simulation_valid_mask_sha256"] == (
        observation.domain_mask_hashes["simulation_valid_mask_sha256"]
    )
    assert all(not batch.features.flags.writeable for batch in batches)
    assert collector.observation_count == 1
    assert collector.row_count == 8
    assert collector.live_full_grid_observation_count == 0


def test_duplicate_and_spatial_identities_are_deterministic():
    config = _config()
    observation = _observation(config)
    first = SetCCollector(_environment(), config)
    second = SetCCollector(_environment(), config)
    first(observation)
    second(observation)

    first_duplicate = np.concatenate(
        [batch.duplicate_group_ids for batch in first.pending_batches()]
    )
    second_duplicate = np.concatenate(
        [batch.duplicate_group_ids for batch in second.pending_batches()]
    )
    first_blocks = np.concatenate(
        [batch.spatial_block_ids for batch in first.pending_batches()]
    )
    second_blocks = np.concatenate(
        [batch.spatial_block_ids for batch in second.pending_batches()]
    )
    assert np.array_equal(first_duplicate, second_duplicate)
    assert np.array_equal(first_blocks, second_blocks)


def test_backpressure_never_drops_or_partially_accepts_an_observation():
    config = _config(max_rows=8)
    observation = _observation(config)
    next_observation = _observation(config, timestep_t=1)
    collector = SetCCollector(_environment(), config)
    collector(observation)

    with pytest.raises(CollectorBackpressureError, match="drain batches"):
        collector(next_observation)
    assert collector.observation_count == 1
    assert collector.row_count == 8

    released_rows = 0
    while collector.pending_rows:
        released_rows += collector.pop_batch().row_count
    assert released_rows == 8
    collector(next_observation)
    assert collector.observation_count == 2
    assert collector.row_count == 16
    assert collector.pending_rows == 8


def test_collector_rejects_mixed_run_identity_without_accepting_rows():
    config = _config()
    collector = SetCCollector(_environment(), config)
    collector(_observation(config))
    mixed = _provenance(config)
    mixed["run_id"] = "different-run"

    with pytest.raises(ValueError, match="[Mm]ixed-run"):
        collector(_observation(config, timestep_t=1, provenance=mixed))
    assert collector.observation_count == 1
    assert collector.row_count == 8


@pytest.mark.parametrize("missing", ["scenario_family_id", "rng_bit_generator"])
def test_collector_requires_scenario_family_and_rng_identity(missing):
    config = _config()
    provenance = _provenance(config)
    del provenance[missing]
    collector = SetCCollector(_environment(), config)

    with pytest.raises(ValueError, match=missing):
        collector(_observation(config, provenance=provenance))
    assert collector.observation_count == 0


def test_collector_rejects_duplicate_missing_and_reordered_timestep_pairs():
    config = _config()

    duplicate = SetCCollector(_environment(), config)
    duplicate(_observation(config, timestep_t=0))
    with pytest.raises(ValueError, match="[Dd]uplicate or reordered"):
        duplicate(_observation(config, timestep_t=0))

    missing = SetCCollector(_environment(), config)
    missing(_observation(config, timestep_t=0))
    with pytest.raises(ValueError, match="[Mm]issing .* transition"):
        missing(_observation(config, timestep_t=2))

    reordered = SetCCollector(_environment(), config)
    reordered(_observation(config, timestep_t=0))
    reordered(_observation(config, timestep_t=1))
    with pytest.raises(ValueError, match="[Dd]uplicate or reordered"):
        reordered(_observation(config, timestep_t=0))


def test_finalization_binds_ignition_identity_and_final_timestep():
    config = _config()
    collector = SetCCollector(_environment(), config)
    collector(_observation(config))

    different_ignition = ((0, 0),)
    with pytest.raises(ValueError, match="ignition identity"):
        collector.finalize(
            replace(
                _complete_record(),
                ignition_coordinates=different_ignition,
                ignition_set_sha256=ignition_set_sha256(different_ignition),
            )
        )
    with pytest.raises(ValueError, match="final timestep"):
        collector.finalize(replace(_complete_record(), final_timestep=2))


def test_single_observation_and_byte_bounds_fail_before_acceptance():
    row_limited_config = _config(max_rows=7)
    row_observation = _observation(row_limited_config)
    row_collector = SetCCollector(_environment(), row_limited_config)
    with pytest.raises(CollectorBackpressureError, match="exceeds"):
        row_collector(row_observation)
    assert row_collector.observation_count == 0

    byte_limited_config = _config(max_bytes=8 * 189 - 1)
    byte_observation = _observation(byte_limited_config)
    collector = SetCCollector(_environment(), byte_limited_config)
    with pytest.raises(CollectorBackpressureError, match="exceeds"):
        collector(byte_observation)
    assert collector.observation_count == 0
    assert collector.row_count == 0


def test_collector_fails_closed_for_sampling_blocks_and_domain_disagreement():
    config = _config()
    config["dataset_generation"]["sampling_policy"] = "case_control"
    with pytest.raises(ValueError, match="all_eligible"):
        SetCCollector(_environment(), config)

    config = _config()
    config["dataset_generation"]["spatial_block"]["size_rows"] = None
    with pytest.raises(TypeError, match="spatial_block.size_rows"):
        SetCCollector(_environment(), config)

    config = _config()
    environment = _environment()
    environment["material_class"][0, 0] = 0
    with pytest.raises(ValueError, match="Building/material disagreement"):
        SetCCollector(environment, config)


def test_synthetic_publisher_is_atomic_manifest_bound_and_no_overwrite(tmp_path):
    config = _config(publisher_enabled=True)
    collector = SetCCollector(_environment(), config)
    collector(_observation(config))
    run = collector.finalize(_complete_record())
    publisher = SyntheticSetCPublisher(config)

    artifact = publisher.publish(
        run,
        destination_root=tmp_path,
        artifact_id="set-c-synthetic",
        artifact_version="v1",
    )

    assert artifact.path.exists()
    assert artifact.path.parent == Path(tmp_path).resolve()
    assert artifact.manifest["manifest_schema_version"] == (
        SET_C_PUBLICATION_MANIFEST_VERSION
    )
    assert artifact.manifest["row_count"] == 8
    assert not list(tmp_path.glob("*.tmp"))
    with zipfile.ZipFile(artifact.path, "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["row_count"] == 8
        assert len(manifest["batches"]) == 3
        assert manifest["batches"][0]["provenance"]["run_id"] == "synthetic-run"
        assert manifest["batches"][0]["grid_crs"] == "EPSG:32651"
        assert manifest["memory_bound_scope"] == (
            "retained_numpy_row_payload_arrays_only_not_peak_process_memory"
        )
        assert manifest["payload_mutability_contract"] == (
            "all_numpy_payload_arrays_read_only"
        )
    with pytest.raises(FileExistsError, match="overwrite"):
        publisher.publish(
            run,
            destination_root=tmp_path,
            artifact_id="set-c-synthetic",
            artifact_version="v1",
        )


@pytest.mark.parametrize(
    "record",
    [
        replace(
            _complete_record(),
            termination_reason=TERMINATION_SAFETY_LIMIT,
            completeness_status=COMPLETENESS_CENSORED,
            authoritative=False,
            final_blazing_count=1,
        ),
        replace(
            _complete_record(),
            termination_reason=TERMINATION_FAILURE,
            completeness_status=COMPLETENESS_FAILED,
            authoritative=False,
        ),
        replace(_complete_record(), authoritative=False),
    ],
)
def test_publisher_rejects_censored_incomplete_and_count_mismatched_runs(
    tmp_path, record
):
    config = _config(publisher_enabled=True)
    collector = SetCCollector(_environment(), config)
    collector(_observation(config))
    run = collector.finalize(record)
    publisher = SyntheticSetCPublisher(config)

    with pytest.raises((PermissionError, ValueError)):
        publisher.publish(
            run,
            destination_root=tmp_path,
            artifact_id="rejected",
            artifact_version="v1",
        )
    assert list(tmp_path.iterdir()) == []


def test_collector_rejects_termination_observation_count_mismatch_before_seal():
    config = _config(publisher_enabled=True)
    collector = SetCCollector(_environment(), config)
    collector(_observation(config))

    with pytest.raises(ValueError, match="observation count"):
        collector.finalize(
            replace(
                _complete_record(),
                captured_transition_count=0,
                authoritative=False,
            )
        )


def test_publisher_rejects_partial_drained_run_and_is_disabled_by_default(tmp_path):
    config = _config()
    with pytest.raises(PermissionError, match="disabled by default"):
        SyntheticSetCPublisher(config)

    enabled = _config(publisher_enabled=True)
    collector = SetCCollector(_environment(), enabled)
    collector(_observation(enabled))
    collector.pop_batch()
    run = collector.finalize(_complete_record())
    with pytest.raises(PermissionError, match="partial"):
        SyntheticSetCPublisher(enabled).publish(
            run,
            destination_root=tmp_path,
            artifact_id="partial",
            artifact_version="v1",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("corruption", ["schema", "provenance", "termination-count"])
def test_publisher_revalidates_batch_schema_and_provenance_uniformity(
    tmp_path, corruption
):
    config = _config(publisher_enabled=True)
    collector = SetCCollector(_environment(), config)
    collector(_observation(config))
    run = collector.finalize(_complete_record())
    first = run.batches[0]
    if corruption == "schema":
        bad_first = replace(first, feature_names=tuple(reversed(first.feature_names)))
        corrupted = replace(run, batches=(bad_first, *run.batches[1:]))
        expected_error = "feature order"
    elif corruption == "provenance":
        bad_first = replace(first, provenance_sha256="b" * 64)
        corrupted = replace(run, batches=(bad_first, *run.batches[1:]))
        expected_error = "provenance"
    else:
        corrupted = replace(run, final_blazing_count=1)
        expected_error = "final counts"

    with pytest.raises(PermissionError, match=expected_error):
        SyntheticSetCPublisher(config).publish(
            corrupted,
            destination_root=tmp_path,
            artifact_id=f"bad-{corruption}",
            artifact_version="v1",
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    [
        ("feature_nan", ".*predictor values"),
        ("feature_domain", ".*predictor values"),
        ("label_value", ".*labels must contain"),
        ("cell_dtype", ".*coordinates must use int64"),
        ("cell_bounds", ".*outside grid bounds"),
        ("duplicate_identity", ".*duplicate identities"),
        ("spatial_identity", ".*spatial identities"),
        ("writable_payload", ".*read-only"),
        ("seed", ".*duplicated run context"),
        ("state_encoding", ".*duplicated run context"),
        ("transition_source", ".*duplicated run context"),
        ("observation_schema", ".*duplicated run context"),
        ("grid", ".*duplicated run context"),
        ("wind", ".*duplicated run context"),
        ("domain", ".*duplicated run context"),
        ("provenance", ".*provenance"),
    ],
)
def test_publisher_rejects_semantic_payload_and_context_corruption_before_staging(
    tmp_path, corruption, expected_error
):
    config = _config(publisher_enabled=True)
    collector = SetCCollector(_environment(), config)
    collector(_observation(config))
    run = collector.finalize(_complete_record())
    first = run.batches[0]

    if corruption in {"feature_nan", "feature_domain", "writable_payload"}:
        features = first.features.copy()
        if corruption == "feature_nan":
            features[0, 0] = np.nan
            features = _readonly(features)
        elif corruption == "feature_domain":
            features[0, 0] = 2.0
            features = _readonly(features)
        bad_first = replace(first, features=features)
    elif corruption == "label_value":
        labels = first.labels.copy()
        labels[0] = 2
        bad_first = replace(first, labels=_readonly(labels))
    elif corruption == "cell_dtype":
        bad_first = replace(
            first, cell_rows=_readonly(first.cell_rows.astype(np.int32))
        )
    elif corruption == "cell_bounds":
        cell_rows = first.cell_rows.copy()
        cell_rows[0] = first.grid_shape[0]
        bad_first = replace(first, cell_rows=_readonly(cell_rows))
    elif corruption == "duplicate_identity":
        identities = first.duplicate_group_ids.copy()
        identities[0] = b"0" * 64
        bad_first = replace(first, duplicate_group_ids=_readonly(identities))
    elif corruption == "spatial_identity":
        identities = first.spatial_block_ids.copy()
        identities[0] = b"0" * 64
        bad_first = replace(first, spatial_block_ids=_readonly(identities))
    elif corruption == "seed":
        bad_first = replace(first, seed=first.seed + 1)
    elif corruption == "state_encoding":
        bad_first = replace(first, state_encoding="corrupt-state-encoding")
    elif corruption == "transition_source":
        bad_first = replace(first, transition_source_type="corrupt-source")
    elif corruption == "observation_schema":
        bad_first = replace(
            first, source_observation_schema_version="corrupt-observation-schema"
        )
    elif corruption == "grid":
        bad_first = replace(first, grid_shape=(first.grid_shape[0] + 1, 3))
    elif corruption == "wind":
        bad_first = replace(first, wind_weight=first.wind_weight + 0.1)
    elif corruption == "domain":
        hashes = dict(first.domain_mask_hashes)
        hashes["simulation_valid_mask_sha256"] = "b" * 64
        bad_first = replace(first, domain_mask_hashes=hashes)
    else:
        provenance = dict(first.provenance)
        provenance["simulator_id"] = "corrupt-simulator"
        bad_first = replace(first, provenance=provenance)

    corrupted = replace(run, batches=(bad_first, *run.batches[1:]))
    with pytest.raises(PermissionError, match=expected_error):
        SyntheticSetCPublisher(config).publish(
            corrupted,
            destination_root=tmp_path,
            artifact_id=f"corrupt-{corruption}",
            artifact_version="v1",
        )
    assert list(tmp_path.iterdir()) == []


def test_publisher_rejects_corrupted_locked_identity_before_staging(tmp_path):
    config = _config(publisher_enabled=True)
    collector = SetCCollector(_environment(), config)
    collector(_observation(config))
    run = collector.finalize(_complete_record())
    identity = dict(run.run_identity)
    identity["scenario_family_id"] = "corrupt-family"

    with pytest.raises(PermissionError, match="identity digest"):
        SyntheticSetCPublisher(config).publish(
            replace(run, run_identity=identity),
            destination_root=tmp_path,
            artifact_id="corrupt-locked-identity",
            artifact_version="v1",
        )
    assert list(tmp_path.iterdir()) == []
