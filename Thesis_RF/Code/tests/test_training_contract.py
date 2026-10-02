from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pandas as pd
import pytest

from modules.feature_pipeline import (
    AUTHORITATIVE_OBSERVATION_METADATA_NAMES,
    AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES,
    CANONICAL_FEATURE_NAMES,
    SET_C_SCHEMA_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
)
import modules.model_trainer as model_trainer_module
from modules.model_trainer import ModelTrainer, load_verified_development_release
from modules import set_c_split_release as release_module
from modules.set_c_split_release import publish_synthetic_development_release
from modules.set_c_leakage import build_joint_leakage_components
from train_rf_optuna import (
    _split_membership_hash,
    _validate_observation_wind_manifest,
    class_weight_for_estimator,
    load_calibration_selection_record,
    load_grouped_data,
    validate_class_weight_candidates,
)
from modules.feature_pipeline import CALIBRATION_SELECTION_VERSION
from modules.wind_convention import (
    CALM_REPRESENTATION,
    DIRECTION_CONVENTION,
    DIRECTION_UNITS,
    NORTH_REFERENCE,
    WIND_SCHEMA_VERSION,
    WindContract,
)


def _contract_frame() -> pd.DataFrame:
    rows = []
    for index, role in enumerate(("train", "validation", "calibration")):
        for target in (0, 1):
            row_number = index * 2 + target
            row = {
                "set_id": "SetC",
                "experiment_id": "experiment-1",
                "event_id": "single-event",
                "scenario_family_id": f"family-{role}",
                "scenario_id": f"scenario-{role}",
                "run_id": f"run-{role}",
                "seed": 42 + index,
                "timestep_t": index,
                "timestep_t1": index + 1,
                "cell_row": row_number,
                "cell_col": row_number,
                "grid_id": "grid-1",
                "stable_cell_id": f"cell-{row_number}",
                "duplicate_group_id": f"duplicate-{role}-{target}",
                "spatial_block_id": f"block-{role}",
                "ignition_set_sha256": "a" * 64,
                "run_identity_sha256": f"{row_number + 1:064x}",
                "provenance_sha256": "b" * 64,
                "split_role": role,
            }
            row.update({name: 0.0 for name in CANONICAL_FEATURE_NAMES})
            row["building_presence"] = 1.0
            row["material_class"] = 1.0
            row["material_risk"] = 0.95
            row["neighbor_burning_count"] = 1.0
            row["composite_flammability"] = 0.95
            row[TARGET_NAME] = target
            row["feature_schema_version"] = SET_C_SCHEMA_VERSION
            row["target_version"] = TARGET_VERSION
            rows.append(row)
    columns = (
        list(AUTHORITATIVE_OBSERVATION_METADATA_NAMES) + ["split_role"]
        + list(CANONICAL_FEATURE_NAMES)
        + [TARGET_NAME, "feature_schema_version", "target_version"]
    )
    return pd.DataFrame(rows).loc[:, columns]


def _wind_manifest() -> dict[str, object]:
    return WindContract.from_config(
        {
            "speed_kmh": 10.0,
            "direction_deg": 315.0,
            "direction_convention": DIRECTION_CONVENTION,
            "direction_units": DIRECTION_UNITS,
            "north_reference": NORTH_REFERENCE,
            "schema_version": WIND_SCHEMA_VERSION,
            "calm_representation": CALM_REPRESENTATION,
        }
    ).to_manifest()


def _synthetic_role_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for role in ("train", "validation", "calibration"):
        rows = []
        for label in (0, 1):
            row = {name: 0.0 for name in CANONICAL_FEATURE_NAMES}
            row.update(building_presence=1.0, material_class=1.0, material_risk=0.95,
                       composite_flammability=0.95, neighbor_burning_count=1.0)
            row[TARGET_NAME] = label
            rows.append(row)
        frames[role] = pd.DataFrame(
            rows, columns=[*CANONICAL_FEATURE_NAMES, TARGET_NAME]
        )
    return frames


def test_grouped_dataset_contract_accepts_separate_roles():
    ModelTrainer._validate_dataset_contract(_contract_frame())


def test_both_development_entry_points_accept_pinned_synthetic_release(tmp_path):
    root, digest = publish_synthetic_development_release(
        _synthetic_role_frames(), tmp_path / "development_release"
    )

    trainer = ModelTrainer(
        str(root), str(tmp_path / "models"),
        expected_attestation_sha256=digest, allow_synthetic=True,
    )
    data, partitions = load_grouped_data(
        root, expected_attestation_sha256=digest, allow_synthetic=True
    )

    assert len(trainer.data) == 6
    assert len(data) == 6
    assert set(partitions) == {"train", "validation", "calibration"}


def test_development_entry_points_accept_hash_bound_production_shape(tmp_path, monkeypatch):
    root, _ = publish_synthetic_development_release(
        _synthetic_role_frames(), tmp_path / "development_release"
    )
    source_digest, manifest_digest = "a" * 64, "b" * 64
    monkeypatch.setattr(release_module, "APPROVED_CSV_SHA256", source_digest)
    monkeypatch.setattr(release_module, "APPROVED_MANIFEST_SHA256", manifest_digest)
    monkeypatch.setattr(model_trainer_module, "APPROVED_CSV_SHA256", source_digest)
    manifest = {
        "manifest_schema_version": "set_c_authoritative_package.v1",
        "artifact_kind": "authoritative_set_c_observation_package",
        "split_status": "unassigned",
        "dataset_sha256": source_digest,
        "row_count": 6,
        "canonical_csv_columns": list(release_module.AUTHORITATIVE_CSV_COLUMNS),
        "manifest_payload_sha256": "c" * 64,
    }
    provenance = release_module._production_provenance(
        manifest, manifest_digest=manifest_digest, source_digest=source_digest,
        row_count=6, source_revision="d" * 40,
        authorization_reference="synthetic-production-shape-test",
    )
    attestation_path = root / "attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation.update(
        source_csv_sha256=source_digest,
        policy_version=release_module.APPROVED_POLICY_VERSION,
        synthetic=False,
        source_provenance=provenance,
    )
    attestation["publication_receipt"].update(
        protected_attestation_sha256="e" * 64,
        source_csv_sha256=source_digest,
        policy_version=release_module.APPROVED_POLICY_VERSION,
        source_revision="d" * 40,
    )
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    digest = sha256(attestation_path.read_bytes()).hexdigest()

    trainer = ModelTrainer(
        str(root), str(tmp_path / "models"),
        expected_attestation_sha256=digest, allow_synthetic=True,
    )
    data, partitions = load_grouped_data(
        root, expected_attestation_sha256=digest, allow_synthetic=True
    )
    assert len(trainer.data) == len(data) == 6
    assert set(partitions) == {"train", "validation", "calibration"}


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "SetC_Protected/set_c_v1",
        "output/datasets/authoritative_set_c/v1/set_c.v1.csv",
        "SetC_Protected/set_c_v1/complete_mapping.csv",
        "SetC_Protected/set_c_v1/final_test_rows.csv",
    ],
)
def test_training_entry_points_reject_protected_inputs_before_open(
    tmp_path, monkeypatch, forbidden_path
):
    def unexpected_open(*args, **kwargs):
        raise AssertionError("protected input reached the release reader")

    monkeypatch.setattr(model_trainer_module, "load_development_release", unexpected_open)
    supplied = tmp_path / forbidden_path
    digest = "a" * 64

    with pytest.raises(PermissionError, match="cannot read canonical, mapping, or final-test"):
        load_verified_development_release(
            supplied, expected_attestation_sha256=digest, allow_synthetic=True
        )
    with pytest.raises(PermissionError, match="cannot read canonical, mapping, or final-test"):
        ModelTrainer(
            str(supplied), str(tmp_path / "models"),
            expected_attestation_sha256=digest, allow_synthetic=True,
        )
    with pytest.raises(PermissionError, match="cannot read canonical, mapping, or final-test"):
        load_grouped_data(
            supplied, expected_attestation_sha256=digest, allow_synthetic=True
        )


def test_legacy_full_dataset_and_assignment_arguments_are_rejected(tmp_path, monkeypatch):
    def unexpected_open(*args, **kwargs):
        raise AssertionError("legacy inputs reached the release reader")

    monkeypatch.setattr(model_trainer_module, "load_development_release", unexpected_open)
    with pytest.raises(TypeError):
        ModelTrainer(
            str(tmp_path / "release"), str(tmp_path / "models"),
            expected_attestation_sha256="a" * 64,
            csv_path=tmp_path / "canonical.csv",
            assignment_path=tmp_path / "complete_mapping.csv",
        )
    with pytest.raises(TypeError):
        load_grouped_data(
            tmp_path / "release",
            expected_attestation_sha256="a" * 64,
            csv_path=tmp_path / "canonical.csv",
            assignment_path=tmp_path / "complete_mapping.csv",
        )


def test_development_release_rejects_temporary_reparse_alias_before_open(
    tmp_path, monkeypatch
):
    target = tmp_path / "release_target"
    target.mkdir()
    alias = tmp_path / "release_alias"
    try:
        os.symlink(target, alias, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Host cannot create a temporary symlink: {error}")

    def unexpected_open(*args, **kwargs):
        raise AssertionError("reparse path reached the release reader")

    monkeypatch.setattr(model_trainer_module, "load_development_release", unexpected_open)
    with pytest.raises(PermissionError, match="contains a reparse point"):
        load_verified_development_release(
            alias, expected_attestation_sha256="a" * 64, allow_synthetic=True
        )


def test_development_release_rejects_reparse_attribute_before_open(tmp_path, monkeypatch):
    release = tmp_path / "synthetic_release"
    release.mkdir()
    original_lstat = Path.lstat

    def simulated_reparse_lstat(path):
        if path == release:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original_lstat(path)

    def unexpected_open(*args, **kwargs):
        raise AssertionError("reparse path reached the release reader")

    monkeypatch.setattr(Path, "lstat", simulated_reparse_lstat)
    monkeypatch.setattr(model_trainer_module, "load_development_release", unexpected_open)
    with pytest.raises(PermissionError, match="contains a reparse point"):
        load_verified_development_release(
            release, expected_attestation_sha256="a" * 64, allow_synthetic=True
        )


def test_observation_manifest_accepts_complete_canonical_wind_contract():
    wind = _wind_manifest()

    assert _validate_observation_wind_manifest(wind) == wind


@pytest.mark.parametrize(
    "mutation",
    [
        lambda wind: wind.pop("north_reference"),
        lambda wind: wind.update(direction_convention="to_bearing"),
        lambda wind: wind.update(unapproved_field="value"),
    ],
)
def test_observation_manifest_wind_contract_fails_closed(mutation):
    wind = _wind_manifest()
    mutation(wind)

    with pytest.raises(ValueError, match="wind provenance"):
        _validate_observation_wind_manifest(wind)


def test_split_roles_merge_from_a_separate_manifest():
    merged = _contract_frame()
    observations = merged.drop(columns=["split_role"])
    assignments = merged.loc[:, list(AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES) + ["split_role"]]

    reconstructed = ModelTrainer.merge_split_assignments(observations, assignments)

    assert reconstructed["split_role"].tolist() == merged["split_role"].tolist()


def test_duplicate_group_cannot_cross_roles():
    frame = _contract_frame()
    frame.loc[2, "duplicate_group_id"] = frame.loc[0, "duplicate_group_id"]
    with pytest.raises(ValueError, match="duplicate_group_id crosses split roles"):
        ModelTrainer._validate_dataset_contract(frame)


def test_equal_predictors_do_not_create_cross_family_leakage_without_record_identity():
    observations = pd.DataFrame([
        {"event_id": "event-a", "scenario_family_id": "family-a",
         "spatial_block_id": "block-a", "stable_cell_id": "cell-a",
         "duplicate_group_id": "record-a"},
        {"event_id": "event-a", "scenario_family_id": "family-b",
         "spatial_block_id": "block-b", "stable_cell_id": "cell-b",
         "duplicate_group_id": "record-b"},
    ])
    components = build_joint_leakage_components(observations)
    assert components.nunique() == 2


def test_actual_record_identity_still_forms_a_leakage_component():
    observations = pd.DataFrame([
        {"event_id": "event-a", "scenario_family_id": "family-a",
         "spatial_block_id": "block-a", "stable_cell_id": "cell-a",
         "duplicate_group_id": "record-a"},
        {"event_id": "event-a", "scenario_family_id": "family-b",
         "spatial_block_id": "block-b", "stable_cell_id": "cell-b",
         "duplicate_group_id": "record-a"},
    ])
    components = build_joint_leakage_components(observations)
    assert components.nunique() == 1


def test_conflicting_label_duplicate_group_is_retained_within_one_role():
    frame = _contract_frame()
    frame.loc[1, "duplicate_group_id"] = frame.loc[0, "duplicate_group_id"]
    ModelTrainer._validate_dataset_contract(frame)


def test_feature_order_and_unassigned_roles_fail_closed():
    frame = _contract_frame()
    swapped = frame.columns.tolist()
    left = swapped.index(CANONICAL_FEATURE_NAMES[0])
    right = swapped.index(CANONICAL_FEATURE_NAMES[1])
    swapped[left], swapped[right] = swapped[right], swapped[left]
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        ModelTrainer._validate_dataset_contract(frame.loc[:, swapped])

    frame.loc[0, "split_role"] = ""
    with pytest.raises(PermissionError, match="cannot access final-test membership"):
        ModelTrainer._validate_dataset_contract(frame)


def test_split_manifest_cannot_contain_foreign_assignment_rows():
    merged = _contract_frame()
    observations = merged.drop(columns=["split_role"])
    assignments = merged.loc[:, list(AUTHORITATIVE_SPLIT_ASSIGNMENT_KEY_NAMES) + ["split_role"]].copy()
    foreign = assignments.iloc[[0]].copy()
    foreign["cell_row"] = 9999
    foreign["cell_col"] = 9999
    assignments = pd.concat([assignments, foreign], ignore_index=True)

    with pytest.raises(ValueError, match="must match exactly"):
        ModelTrainer.merge_split_assignments(observations, assignments)


def test_each_split_role_requires_both_classes():
    frame = _contract_frame()
    frame = frame.drop(frame[(frame["split_role"] == "calibration") & (frame[TARGET_NAME] == 1)].index)
    with pytest.raises(ValueError, match="must contain both target classes"):
        ModelTrainer._validate_dataset_contract(frame)


def test_reporting_threshold_requires_calibration_provenance():
    with pytest.raises(ValueError, match="selected on calibration"):
        ModelTrainer._validate_threshold_record(
            {"value": 0.5, "selected_on": "test", "objective": "example"}
        )
    assert ModelTrainer._validate_threshold_record(
        {"value": 0.4, "selected_on": "calibration", "objective": "approved-cost-policy"}
    ) == pytest.approx(0.4)


def test_development_loader_rejects_final_test_membership():
    frame = _contract_frame()
    frame.loc[0, "split_role"] = "final_test"
    with pytest.raises(PermissionError, match="cannot access final-test membership"):
        ModelTrainer._validate_dataset_contract(frame)


def test_fixed_numeric_class_weight_cost_ratios_are_rejected():
    assert validate_class_weight_candidates(
        ["none", "balanced", "balanced_subsample"]
    ) == ("none", "balanced", "balanced_subsample")
    assert class_weight_for_estimator("none") is None
    assert class_weight_for_estimator("balanced") == "balanced"
    assert class_weight_for_estimator("balanced_subsample") == "balanced_subsample"
    with pytest.raises(ValueError, match="fixed numeric cost ratios are not approved"):
        validate_class_weight_candidates(["cw_1_5"])


def test_reusable_trainer_requires_explicit_approved_class_weight_token():
    trainer = object.__new__(ModelTrainer)
    with pytest.raises(ValueError, match="must be explicitly"):
        trainer.train_model()


def test_calibration_method_requires_grouped_non_test_selection_record(tmp_path):
    dataset_path = tmp_path / "observations.csv"
    split_path = tmp_path / "splits.csv"
    dataset_path.write_text("evidence", encoding="utf-8")
    split_path.write_text("assignments", encoding="utf-8")
    membership_hash = _split_membership_hash(_contract_frame())
    record = {
        "record_schema_version": CALIBRATION_SELECTION_VERSION,
        "artifact_kind": "set_c_calibration_method_selection",
        "created_at_utc": "2026-09-15T00:00:00+00:00",
        "set_id": "SetC",
        "experiment_id": "experiment-1",
        "source_dataset_sha256": sha256(b"evidence").hexdigest(),
        "split_assignment_sha256": sha256(b"assignments").hexdigest(),
        "split_membership_sha256": membership_hash,
        "candidate_methods": ["sigmoid", "isotonic"],
        "selection_metric": "grouped validation log loss",
        "grouped_selection_design": "complete groups; no final-test observations",
        "selected_method": "sigmoid",
        "selected_on_roles": ["train", "validation"],
        "final_test_used": False,
        "source_revision": "abc123",
        "selection_command": "recorded-command",
        "environment_record": "recorded-environment",
    }
    record_path = tmp_path / "calibration-selection.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    loaded = load_calibration_selection_record(
        record_path,
        configured_method="sigmoid",
        csv_path=dataset_path,
        split_manifest_path=split_path,
        split_membership_sha256=membership_hash,
        experiment_id="experiment-1",
    )
    assert loaded["selected_method"] == "sigmoid"

    record["final_test_used"] = True
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="final test was untouched"):
        load_calibration_selection_record(
            record_path,
            configured_method="sigmoid",
            csv_path=dataset_path,
            split_manifest_path=split_path,
            split_membership_sha256=membership_hash,
            experiment_id="experiment-1",
        )
