"""Synthetic, temporary-path checks for the development projection boundary."""

from hashlib import sha256
import json

import pandas as pd
import pytest

from modules.feature_pipeline import CANONICAL_FEATURE_NAMES, DEVELOPMENT_SPLIT_ROLES, TARGET_NAME
from modules.model_trainer import ModelTrainer
from modules.set_c_split_release import (
    APPROVED_CSV_SHA256,
    ATTESTATION_SCHEMA_VERSION,
    RELEASE_SCHEMA_VERSION,
    PAIR_PUBLICATION_SCHEMA_VERSION,
    APPROVED_POLICY_VERSION,
    _synthetic_test_provenance,
    allocate_synthetic_components,
    load_development_release,
    publish_synthetic_development_release,
)
from train_rf_optuna import load_grouped_data, main


def _synthetic_release(tmp_path):
    root = tmp_path / "development_release"
    root.mkdir()
    files = {}
    for role in DEVELOPMENT_SPLIT_ROLES:
        rows = []
        for label in (0, 1):
            row = {name: 0.0 for name in CANONICAL_FEATURE_NAMES}
            row.update(building_presence=1.0, material_class=1.0, material_risk=0.95,
                       composite_flammability=0.95, neighbor_burning_count=1.0)
            row[TARGET_NAME] = label
            rows.append(row)
        path = root / f"{role}.csv"
        pd.DataFrame(rows, columns=[*CANONICAL_FEATURE_NAMES, TARGET_NAME]).to_csv(path, index=False)
        files[role] = {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "rows": 2,
            "positives": 1,
            "negatives": 1,
        }
    attestation = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "source_csv_sha256": "0" * 64,
        "policy_version": APPROVED_POLICY_VERSION,
        "feature_schema_version": "set_c.v1",
        "target_version": "ignition_t_plus_1.v1",
        "feature_names": list(CANONICAL_FEATURE_NAMES),
        "roles": list(DEVELOPMENT_SPLIT_ROLES),
        "role_files": files,
        "synthetic": True,
        "final_test_used": False,
        "source_provenance": _synthetic_test_provenance("0" * 64, 6),
        "publication_receipt": {
            "schema_version": PAIR_PUBLICATION_SCHEMA_VERSION,
            "state": "complete",
            "protected_attestation_sha256": "0" * 64,
            "source_csv_sha256": "0" * 64,
            "policy_version": APPROVED_POLICY_VERSION,
            "source_revision": "synthetic:" + "d" * 40,
        },
    }
    (root / "attestation.json").write_text(json.dumps(attestation), encoding="utf-8")
    return root, sha256((root / "attestation.json").read_bytes()).hexdigest()


def test_three_role_release_loads_without_canonical_data(tmp_path):
    root, digest = _synthetic_release(tmp_path)
    data, _ = load_development_release(root, expected_attestation_sha256=digest,
                                       allow_synthetic=True)
    assert len(data) == 6
    assert set(data["split_role"]) == set(DEVELOPMENT_SPLIT_ROLES)
    assert "scenario_family_id" not in data
    trainer = ModelTrainer(str(root), str(tmp_path / "models"),
                           expected_attestation_sha256=digest, allow_synthetic=True)
    assert len(trainer.X_train) == 2
    _, partitions = load_grouped_data(root, expected_attestation_sha256=digest,
                                      allow_synthetic=True)
    assert all(len(partitions[role][0]) == 2 for role in DEVELOPMENT_SPLIT_ROLES)


def test_release_rejects_tampering_and_extra_membership(tmp_path):
    root, digest = _synthetic_release(tmp_path)
    with pytest.raises(PermissionError, match="Synthetic release"):
        load_development_release(root, expected_attestation_sha256=digest)
    (root / "final_test.csv").write_text("membership", encoding="utf-8")
    with pytest.raises(PermissionError, match="unexpected files"):
        load_development_release(root, expected_attestation_sha256=digest,
                                 allow_synthetic=True)
    (root / "final_test.csv").unlink()
    (root / "train.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_development_release(root, expected_attestation_sha256=digest,
                                 allow_synthetic=True)


def test_release_rejects_canonical_path_and_unpinned_attestation(tmp_path):
    root, digest = _synthetic_release(tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        load_development_release(root, expected_attestation_sha256="", allow_synthetic=True)
    canonical = tmp_path / "authoritative_set_c"
    root.rename(canonical)
    with pytest.raises(PermissionError, match="canonical package"):
        load_development_release(canonical, expected_attestation_sha256=digest,
                                 allow_synthetic=True)


def test_training_entry_point_remains_disabled():
    with pytest.raises(PermissionError, match="not authorized|remains disabled"):
        main()


def test_synthetic_publication_is_temp_only_and_no_overwrite(tmp_path):
    frames = {}
    for role in DEVELOPMENT_SPLIT_ROLES:
        rows = []
        for label in (0, 1):
            row = {name: 0.0 for name in CANONICAL_FEATURE_NAMES}
            row.update(building_presence=1.0, material_class=1.0, material_risk=0.95,
                       composite_flammability=0.95, neighbor_burning_count=1.0)
            row[TARGET_NAME] = label
            rows.append(row)
        frames[role] = pd.DataFrame(rows, columns=[*CANONICAL_FEATURE_NAMES, TARGET_NAME])
    destination = tmp_path / "synthetic_release"
    published, digest = publish_synthetic_development_release(frames, destination)
    loaded, _ = load_development_release(published, expected_attestation_sha256=digest,
                                         allow_synthetic=True)
    assert len(loaded) == 6
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_synthetic_development_release(frames, destination)
    assert sha256((destination / "attestation.json").read_bytes()).hexdigest() == digest


def test_synthetic_allocator_proves_hard_gates_without_real_set_c():
    inventory = pd.DataFrame([
        {"component_id": f"synthetic-{index}", "positives": 21, "negatives": 81,
         **{f"material_{number}": 1 for number in range(1, 6)}}
        for index in range(26)
    ])
    roles = allocate_synthetic_components(inventory, dataset_sha256="0" * 64)
    assert len(roles) == 26
    assert [list(roles.values()).count(role) for role in (*DEVELOPMENT_SPLIT_ROLES, "final_test")] == [11, 5, 5, 5]
    ordered = sorted(roles, key=lambda name: sha256(f"{'0' * 64}:{name}".encode()).hexdigest())
    assert [roles[name] for name in ordered] == (["train"] * 11 + ["validation"] * 5
                                                + ["calibration"] * 5 + ["final_test"] * 5)
    with pytest.raises(PermissionError, match="fixed synthetic"):
        allocate_synthetic_components(inventory, dataset_sha256=APPROVED_CSV_SHA256)
