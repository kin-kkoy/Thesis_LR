import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evaluate_rf_final_test import evaluate_authorized_final_test
from modules.feature_pipeline import (
    AUTHORITATIVE_CSV_COLUMNS,
    CANONICAL_FEATURE_NAMES,
    FINAL_TEST_ROLE,
    SET_C_SCHEMA_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
)
from modules.ca_state import STATE_ENCODING
from modules.set_c_collector import SET_C_ROW_BATCH_SCHEMA_VERSION
from modules.set_c_leakage import (
    build_joint_leakage_components,
    deterministic_allocation_policy_contract,
)
from modules.set_c_publication import SetCPublicationSession
from modules.transition_observer import canonical_metadata_hash
from modules.transition_observer import (
    AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
    TRANSITION_OBSERVATION_SCHEMA_VERSION,
)
from train_rf_optuna import load_dataset_manifest


def _publisher_config() -> dict:
    return {
        "dataset_generation": {
            "set_c_publisher": {
                "enabled": True,
                "mode": "authoritative_versioned",
                "authorization_reference": "synthetic temporary-path test",
                "package_id": "synthetic_set_c",
                "package_version": "1",
                "destination_pattern": "output/datasets/{package_id}/v{package_version}",
            }
        }
    }


def _package_identity() -> dict:
    return {
        "experiment_id": "synthetic-experiment",
        "source_revision": "abc123",
        "source_dirty": False,
        "configuration_sha256": "a" * 64,
        "environment_id": "synthetic-env",
        "environment_sha256": "b" * 64,
        "rng_algorithm": "PCG64",
        "input_hashes": {"synthetic": "c" * 64},
        "simulation_valid_mask_sha256": "d" * 64,
        "mapped_building_mask_sha256": "e" * 64,
        "teacher_id": "model-free-fire-automata",
        "teacher_version": "v1",
        "feature_schema_sha256": "f" * 64,
        "target_schema_sha256": "0" * 64,
        "authorization_reference": "synthetic temporary-path test",
    }


def _run_identity() -> dict:
    package = _package_identity()
    return {
        "set_id": "SetC",
        "experiment_id": "synthetic-experiment",
        "event_id": "event-1",
        "scenario_family_id": "family-1",
        "scenario_id": "scenario-1",
        "run_id": "run-1",
        "grid_id": "grid-1",
        "ignition_set_sha256": "1" * 64,
        "ignition_coordinates": [[1, 3]],
        "source_revision": package["source_revision"],
        "source_dirty": package["source_dirty"],
        "configuration_hash": package["configuration_sha256"],
        "environment_id": package["environment_id"],
        "rng_bit_generator": package["rng_algorithm"],
        "input_hashes": package["input_hashes"],
        "simulation_valid_mask_sha256": package["simulation_valid_mask_sha256"],
        "mapped_building_mask_sha256": package["mapped_building_mask_sha256"],
        "simulator_id": package["teacher_id"],
        "simulator_version": package["teacher_version"],
        "authorization_reference": package["authorization_reference"],
        "wind_manifest": {"schema_version": "simulated_wind.v1", "speed_kmh": 0.0},
    }


def _batch() -> SimpleNamespace:
    features = np.zeros((2, len(CANONICAL_FEATURE_NAMES)), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int8)
    identity = _run_identity()
    identity_hash = canonical_metadata_hash(identity)
    return SimpleNamespace(
        schema_version=SET_C_ROW_BATCH_SCHEMA_VERSION,
        feature_names=CANONICAL_FEATURE_NAMES,
        feature_schema_version=SET_C_SCHEMA_VERSION,
        target_name=TARGET_NAME,
        target_version=TARGET_VERSION,
        positive_label=1,
        state_encoding=STATE_ENCODING,
        source_observation_schema_version=TRANSITION_OBSERVATION_SCHEMA_VERSION,
        transition_source_type=AUTHORIZED_SIMULATED_TRANSITION_SOURCE,
        features=features,
        labels=labels,
        cell_rows=np.array([1, 2], dtype=np.int64),
        cell_cols=np.array([3, 4], dtype=np.int64),
        duplicate_group_ids=np.array([b"2" * 64, b"3" * 64], dtype="S64"),
        spatial_block_ids=np.array([b"4" * 64, b"5" * 64], dtype="S64"),
        run_identity_sha256=identity_hash,
        provenance_sha256=identity_hash,
        observation_index=0,
        batch_index=0,
        timestep_t=0,
        timestep_t1=1,
        seed=42,
        wind_manifest=identity["wind_manifest"],
        domain_mask_hashes={
            "simulation_valid_mask_sha256": "d" * 64,
            "mapped_building_mask_sha256": "e" * 64,
        },
        provenance=identity,
    )


def _run(*, authoritative: bool = True) -> SimpleNamespace:
    identity = _run_identity()
    identity_hash = canonical_metadata_hash(identity)
    termination = SimpleNamespace(
        authoritative=authoritative,
        termination_reason="inactive" if authoritative else "failed",
        completeness_status="complete" if authoritative else "failed",
        final_ignited_count=0,
        final_blazing_count=0,
        expected_transition_count=1,
        captured_transition_count=1,
    )
    return SimpleNamespace(
        run_identity_sha256=identity_hash,
        provenance_sha256=identity_hash,
        run_identity=identity,
        provenance=identity,
        row_count=2,
        released_row_count=2,
        batches=(),
        termination_record=termination,
    )


def test_publication_is_bounded_atomic_hash_bound_and_has_no_split_role(tmp_path):
    with SetCPublicationSession(
        _publisher_config(), _package_identity(), destination_root=tmp_path
    ) as session:
        session.stage_batch(_batch())
        session.seal_run(_run())
        package = session.publish(expected_run_count=1)

    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    with (package / "set_c.v1.csv").open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == AUTHORITATIVE_CSV_COLUMNS
    assert "split_role" not in header
    assert manifest["predictor_dtype"] == "float32"
    assert manifest["label_dtype"] == manifest["state_dtype"] == "int8"
    assert manifest["split_status"] == "unassigned"
    assert manifest["row_count"] == 2
    loaded = load_dataset_manifest(package / "manifest.json", package / "set_c.v1.csv")
    assert loaded["experiment_id"] == "synthetic-experiment"

    with pytest.raises(FileExistsError, match="overwrite"):
        SetCPublicationSession(
            _publisher_config(), _package_identity(), destination_root=tmp_path
        )


def test_publication_cleans_failed_staging_and_rejects_tampering(tmp_path):
    with pytest.raises(PermissionError, match="cannot be promoted"):
        with SetCPublicationSession(
            _publisher_config(), _package_identity(), destination_root=tmp_path
        ) as session:
            session.stage_batch(_batch())
            staging = session.staging_path
            session.seal_run(_run(authoritative=False))
    assert not staging.exists()

    with pytest.raises(PermissionError, match="hash mismatch"):
        with SetCPublicationSession(
            _publisher_config(), _package_identity(), destination_root=tmp_path
        ) as session:
            batch_path = session.stage_batch(_batch())
            session.seal_run(_run())
            batch_path.write_text("tampered", encoding="utf-8")
            session.publish(expected_run_count=1)


@pytest.mark.parametrize("case", ["censored", "count_mismatch", "partial"])
def test_publication_rejects_every_noncomplete_run_shape(tmp_path, case):
    run = _run()
    if case == "censored":
        run.termination_record.completeness_status = "censored"
    elif case == "count_mismatch":
        run.termination_record.expected_transition_count = 2
    else:
        run.released_row_count = 1
    with pytest.raises(PermissionError):
        with SetCPublicationSession(
            _publisher_config(), _package_identity(), destination_root=tmp_path / case
        ) as session:
            session.stage_batch(_batch())
            session.seal_run(run)


def test_publication_rejects_invalid_features_and_mixed_batch_provenance(tmp_path):
    invalid = _batch()
    invalid.features[0, 0] = np.nan
    with SetCPublicationSession(
        _publisher_config(), _package_identity(), destination_root=tmp_path / "invalid"
    ) as session:
        with pytest.raises(ValueError, match="finite"):
            session.stage_batch(invalid)

    first = _batch()
    second = _batch()
    second.batch_index = 1
    second.provenance = {**second.provenance, "unexpected": "different"}
    second.provenance_sha256 = canonical_metadata_hash(second.provenance)
    with SetCPublicationSession(
        _publisher_config(), _package_identity(), destination_root=tmp_path / "mixed"
    ) as session:
        session.stage_batch(first)
        with pytest.raises(PermissionError, match="Mixed batch provenance"):
            session.stage_batch(second)


def test_joint_leakage_graph_contains_family_block_cell_and_conflicting_duplicate_edges():
    rows = pd.DataFrame(
        {
            "event_id": ["event-a", "event-b", "event-c", "event-d"],
            "scenario_family_id": ["family-1", "family-1", "family-3", "family-4"],
            "spatial_block_id": ["block-1", "block-2", "block-2", "block-4"],
            "stable_cell_id": ["cell-1", "cell-2", "cell-3", "cell-4"],
            "duplicate_group_id": ["dup-1", "dup-2", "dup-3", "dup-3"],
            TARGET_NAME: [0, 1, 0, 1],
        }
    )
    components = build_joint_leakage_components(rows)
    assert components.nunique() == 1
    assert rows.groupby(components)[TARGET_NAME].nunique().iloc[0] == 2

    event_components = build_joint_leakage_components(
        rows.assign(
            scenario_family_id=["f1", "f2", "f3", "f4"],
            spatial_block_id=["b1", "b2", "b3", "b4"],
            stable_cell_id=["c1", "c2", "c3", "c4"],
            duplicate_group_id=["d1", "d2", "d3", "d4"],
            event_id=["complete", "complete", "other-1", "other-2"],
        ),
        complete_event_ids={"complete"},
    )
    assert event_components.iloc[0] == event_components.iloc[1]
    policy = deterministic_allocation_policy_contract("a" * 64)
    assert policy["assignments"] is None
    assert policy["numerical_role_targets"] is None
    assert policy["dataset_sha256"] == "a" * 64

    with pytest.raises(ValueError, match="cannot be null"):
        build_joint_leakage_components(rows.assign(stable_cell_id=["c1", None, "c3", "c4"]))


class _Model:
    feature_names_in_ = np.asarray(CANONICAL_FEATURE_NAMES)
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        probability = np.linspace(0.1, 0.9, len(features))
        return np.column_stack((1.0 - probability, probability))


def test_final_test_evaluator_is_separate_and_fail_closed():
    rows = pd.DataFrame({name: [0.0, 1.0] for name in CANONICAL_FEATURE_NAMES})
    rows[TARGET_NAME] = pd.Series([0, 1], dtype="int8")
    rows["split_role"] = FINAL_TEST_ROLE
    with pytest.raises(PermissionError, match="disabled"):
        evaluate_authorized_final_test(_Model(), rows, {"enabled": False})
    with pytest.raises(PermissionError, match="protected_storage_mechanism"):
        evaluate_authorized_final_test(
            _Model(), rows, {"enabled": True, "authorization_reference": "approval"}
        )
    result = evaluate_authorized_final_test(
        _Model(),
        rows,
        {
            "enabled": True,
            "authorization_reference": "synthetic test",
            "protected_storage_mechanism": "synthetic in-memory fixture",
        },
    )
    assert result["evaluation_scope"] == FINAL_TEST_ROLE
    assert result["threshold_metrics"] is None
