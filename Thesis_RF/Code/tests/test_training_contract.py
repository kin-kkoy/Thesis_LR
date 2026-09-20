from hashlib import sha256
import json

import pandas as pd
import pytest

from modules.feature_pipeline import (
    CANONICAL_FEATURE_NAMES,
    PROVENANCE_METADATA_NAMES,
    SPLIT_ASSIGNMENT_KEY_NAMES,
    SET_C_SCHEMA_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
)
from modules.model_trainer import ModelTrainer
from train_rf_optuna import (
    _split_membership_hash,
    _validate_observation_wind_manifest,
    class_weight_for_estimator,
    load_calibration_selection_record,
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
    for index, role in enumerate(("train", "validation", "calibration", "test")):
        for target in (0, 1):
            row_number = index * 2 + target
            row = {
                "set_id": "SetC",
                "experiment_id": "experiment-1",
                "event_id": "single-event",
                "scenario_id": f"scenario-{role}",
                "run_id": f"run-{role}",
                "seed": 42 + index,
                "timestep_t": index,
                "cell_row": row_number,
                "cell_col": row_number,
                "grid_id": "grid-1",
                "duplicate_group_id": f"duplicate-{role}-{target}",
                "spatial_block_id": f"block-{role}",
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
        list(PROVENANCE_METADATA_NAMES)
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


def test_grouped_dataset_contract_accepts_separate_roles():
    ModelTrainer._validate_dataset_contract(_contract_frame())


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
    assignments = merged.loc[:, list(SPLIT_ASSIGNMENT_KEY_NAMES) + ["split_role"]]

    reconstructed = ModelTrainer.merge_split_assignments(observations, assignments)

    assert reconstructed["split_role"].tolist() == merged["split_role"].tolist()


def test_duplicate_group_cannot_cross_roles():
    frame = _contract_frame()
    frame.loc[2, "duplicate_group_id"] = frame.loc[0, "duplicate_group_id"]
    with pytest.raises(ValueError, match="duplicate_group_id crosses split roles"):
        ModelTrainer._validate_dataset_contract(frame)


def test_conflicting_label_duplicate_group_requires_investigation():
    frame = _contract_frame()
    frame.loc[1, "duplicate_group_id"] = frame.loc[0, "duplicate_group_id"]
    with pytest.raises(ValueError, match="require investigation"):
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
    with pytest.raises(ValueError, match="Invalid or unassigned split_role"):
        ModelTrainer._validate_dataset_contract(frame)


def test_split_manifest_cannot_contain_foreign_assignment_rows():
    merged = _contract_frame()
    observations = merged.drop(columns=["split_role"])
    assignments = merged.loc[:, list(SPLIT_ASSIGNMENT_KEY_NAMES) + ["split_role"]].copy()
    foreign = assignments.iloc[[0]].copy()
    foreign["cell_row"] = 9999
    foreign["cell_col"] = 9999
    assignments = pd.concat([assignments, foreign], ignore_index=True)

    with pytest.raises(ValueError, match="must match exactly"):
        ModelTrainer.merge_split_assignments(observations, assignments)


def test_each_split_role_requires_both_classes():
    frame = _contract_frame()
    frame = frame.drop(frame[(frame["split_role"] == "test") & (frame[TARGET_NAME] == 1)].index)
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
