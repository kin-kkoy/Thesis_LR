from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import modules.set_c_split_release as release_module
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES, TARGET_NAME
from modules.model_trainer import load_verified_train_validation_release
from modules.set_c_split_release import (
    PHASE13_SET_C_RF_PROTOCOL_ID,
    PHASE13_SET_C_RF_PROTOCOL_SHA256,
    load_development_release,
    publish_synthetic_development_release,
)


def _role_frames():
    frames = {}
    for role in ("train", "validation", "calibration"):
        rows = []
        for label in (0, 1):
            row = {name: 0.0 for name in CANONICAL_FEATURE_NAMES}
            row.update(
                building_presence=1.0,
                material_class=1.0,
                material_risk=0.95,
                neighbor_burning_count=1.0,
                composite_flammability=0.95,
            )
            row[TARGET_NAME] = label
            rows.append(row)
        frames[role] = pd.DataFrame(rows, columns=[*CANONICAL_FEATURE_NAMES, TARGET_NAME])
    return frames


def _release(tmp_path):
    return publish_synthetic_development_release(_role_frames(), tmp_path / "release")


def test_selective_loader_reads_only_train_and_validation_and_preserves_contract(tmp_path, monkeypatch):
    root, digest = _release(tmp_path)
    calibration = root / "calibration.csv"
    original_open = Path.open
    original_hash = release_module._sha256

    def guarded_open(path, *args, **kwargs):
        if Path(path) == calibration:
            raise AssertionError("calibration content was opened")
        return original_open(path, *args, **kwargs)

    def guarded_hash(path):
        if Path(path) == calibration:
            raise AssertionError("calibration content was hashed")
        return original_hash(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(release_module, "_sha256", guarded_hash)
    x_train, y_train, x_validation, y_validation, attestation, provenance = (
        load_verified_train_validation_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )
    )

    assert tuple(x_train.columns) == tuple(x_validation.columns) == CANONICAL_FEATURE_NAMES
    assert all(dtype == np.dtype("float32") for dtype in x_train.dtypes)
    assert all(dtype == np.dtype("float32") for dtype in x_validation.dtypes)
    assert y_train.dtype == y_validation.dtype == np.dtype("int8")
    assert set(y_train) == set(y_validation) == {0, 1}
    assert provenance == {
        "protocol_id": PHASE13_SET_C_RF_PROTOCOL_ID,
        "protocol_sha256": PHASE13_SET_C_RF_PROTOCOL_SHA256,
        "source_provenance": attestation["source_provenance"],
    }


def test_selective_loader_rejects_tampered_selected_role_hash(tmp_path):
    root, digest = _release(tmp_path)
    with (root / "train.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="train hash mismatch"):
        load_verified_train_validation_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


def test_selective_loader_rejects_selected_role_schema_even_with_rehashed_attestation(tmp_path):
    root, _ = _release(tmp_path)
    train_path = root / "train.csv"
    train = pd.read_csv(train_path)
    train = train.loc[:, [*reversed(CANONICAL_FEATURE_NAMES), TARGET_NAME]]
    train.to_csv(train_path, index=False)
    attestation_path = root / "attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["role_files"]["train"]["sha256"] = sha256(train_path.read_bytes()).hexdigest()
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    digest = sha256(attestation_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="forbidden or reordered columns"):
        load_verified_train_validation_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


def test_selective_loader_rejects_selected_role_invalid_feature_domain(tmp_path):
    root, _ = _release(tmp_path)
    train_path = root / "train.csv"
    train = pd.read_csv(train_path)
    train.loc[0, "wind_speed"] = -1
    train.to_csv(train_path, index=False)
    attestation_path = root / "attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["role_files"]["train"]["sha256"] = sha256(train_path.read_bytes()).hexdigest()
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    digest = sha256(attestation_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="wind_speed must be non-negative"):
        load_verified_train_validation_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


def test_selective_loader_rejects_selected_role_missing_class(tmp_path):
    root, _ = _release(tmp_path)
    train_path = root / "train.csv"
    train = pd.read_csv(train_path)
    train[TARGET_NAME] = 0
    train.to_csv(train_path, index=False)
    attestation_path = root / "attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    record = attestation["role_files"]["train"]
    record.update(
        sha256=sha256(train_path.read_bytes()).hexdigest(),
    )
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    digest = sha256(attestation_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="needs both labels"):
        load_verified_train_validation_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


def test_selective_loader_rejects_unsafe_release_path(tmp_path):
    with pytest.raises(
        PermissionError,
        match="Training loaders accept only the approved D-038 development release",
    ):
        load_verified_train_validation_release(
            tmp_path, expected_attestation_sha256="a" * 64, allow_synthetic=False
        )


def test_legacy_loader_still_loads_all_three_roles(tmp_path):
    root, digest = _release(tmp_path)
    data, attestation = load_development_release(
        root, expected_attestation_sha256=digest, allow_synthetic=True
    )
    assert set(data["split_role"]) == {"train", "validation", "calibration"}
    assert len(data) == 6
    assert attestation["synthetic"] is True
