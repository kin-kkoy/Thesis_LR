import builtins
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import modules.set_c_split_release as release_module
import modules.phase14_calibration as calibration_module
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES, TARGET_NAME
from modules.set_c_split_release import (
    APPROVED_CALIBRATION_ATTESTATION_SHA256,
    APPROVED_CALIBRATION_COUNTS,
    APPROVED_CALIBRATION_CSV_SHA256,
    load_development_calibration_release,
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


def _repin(root, *, role_frame=None, attestation_update=None, update_counts=True):
    if role_frame is not None:
        role_frame.to_csv(root / "calibration.csv", index=False)
    attestation_path = root / "attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if role_frame is not None:
        role_record = attestation["role_files"]["calibration"]
        role_record["sha256"] = sha256((root / "calibration.csv").read_bytes()).hexdigest()
        if update_counts:
            role_record["rows"] = len(role_frame)
            role_record["positives"] = int(role_frame[TARGET_NAME].sum())
            role_record["negatives"] = len(role_frame) - role_record["positives"]
    if attestation_update is not None:
        attestation_update(attestation)
    attestation_path.write_text(
        json.dumps(attestation, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return sha256(attestation_path.read_bytes()).hexdigest()


def test_calibration_loader_reads_only_selected_contents_and_returns_contract(tmp_path, monkeypatch):
    root, digest = _release(tmp_path)
    original_path_open = Path.open
    original_builtin_open = builtins.open
    original_hash = release_module._sha256
    opened = []
    hashed = []
    forbidden = {root / "train.csv", root / "validation.csv"}

    def check_path(path):
        if isinstance(path, (str, os.PathLike)):
            resolved_path = Path(path)
            if resolved_path in forbidden:
                raise AssertionError(f"unselected role contents accessed: {resolved_path.name}")
            if resolved_path.parent == root:
                opened.append(resolved_path.name)

    def guarded_path_open(path, *args, **kwargs):
        check_path(path)
        return original_path_open(path, *args, **kwargs)

    def guarded_builtin_open(path, *args, **kwargs):
        check_path(path)
        return original_builtin_open(path, *args, **kwargs)

    def guarded_hash(path):
        resolved_path = Path(path)
        if resolved_path in forbidden:
            raise AssertionError(f"unselected role contents hashed: {resolved_path.name}")
        if resolved_path.parent == root:
            hashed.append(resolved_path.name)
        return original_hash(path)

    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(release_module, "_sha256", guarded_hash)
    features, labels, provenance = load_development_calibration_release(
        root, expected_attestation_sha256=digest, allow_synthetic=True
    )

    assert tuple(features.columns) == CANONICAL_FEATURE_NAMES
    assert all(dtype == np.dtype("float32") for dtype in features.dtypes)
    assert labels.dtype == np.dtype("int8")
    assert set(labels) == {0, 1}
    assert len(features) == len(labels) == 2
    assert not {"split_role", "stable_cell_id", "cell_row", "cell_col"}.intersection(features.columns)
    assert provenance["attestation_sha256"] == digest
    assert provenance["rows"] == 2
    assert set(provenance) == {
        "attestation_sha256", "calibration_csv_sha256", "rows", "positives",
        "negatives", "source_csv_sha256", "source_revision", "synthetic",
    }
    assert "train.csv" not in opened and "validation.csv" not in opened
    assert "train.csv" not in hashed and "validation.csv" not in hashed
    assert set(opened).issubset({"attestation.json", "calibration.csv"})
    assert set(hashed).issubset({"attestation.json", "calibration.csv"})
    assert {item.name for item in root.iterdir()} == {
        "attestation.json", "train.csv", "validation.csv", "calibration.csv"
    }


def test_production_calibration_identity_is_pinned_without_opening_release():
    assert APPROVED_CALIBRATION_ATTESTATION_SHA256 == (
        "4fcf286156892c9da0289e56712210e90cccdcd608641c7ee573a8db1dcea23f"
    )
    assert APPROVED_CALIBRATION_CSV_SHA256 == (
        "e424865be9143823f5a15fce27a3eb6df74abe802be84f142505f1209e3a77dc"
    )
    assert APPROVED_CALIBRATION_COUNTS == (2060, 153, 1907)


def test_calibration_loader_rejects_tampered_attestation_hash(tmp_path):
    root, _ = _release(tmp_path)
    with pytest.raises(ValueError, match="attestation hash mismatch"):
        load_development_calibration_release(
            root, expected_attestation_sha256="0" * 64, allow_synthetic=True
        )


def test_calibration_loader_rejects_tampered_selected_role_hash(tmp_path):
    root, digest = _release(tmp_path)
    with (root / "calibration.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="calibration hash mismatch"):
        load_development_calibration_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


def test_calibration_loader_rejects_malformed_attestation_schema(tmp_path):
    root, _ = _release(tmp_path)
    digest = _repin(root, attestation_update=lambda record: record.update(unexpected=True))
    with pytest.raises(ValueError, match="attestation fields"):
        load_development_calibration_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


def test_calibration_loader_rejects_reordered_selected_role_columns(tmp_path):
    root, _ = _release(tmp_path)
    frame = pd.read_csv(root / "calibration.csv")
    frame = frame.loc[:, [*reversed(CANONICAL_FEATURE_NAMES), TARGET_NAME]]
    digest = _repin(root, role_frame=frame, update_counts=False)
    with pytest.raises(ValueError, match="forbidden or reordered columns"):
        load_development_calibration_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


def test_calibration_loader_rejects_invalid_feature_values(tmp_path):
    root, _ = _release(tmp_path)
    frame = pd.read_csv(root / "calibration.csv")
    frame.loc[0, "wind_speed"] = -1
    digest = _repin(root, role_frame=frame, update_counts=False)
    with pytest.raises(ValueError, match="wind_speed must be non-negative"):
        load_development_calibration_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


@pytest.mark.parametrize("invalid_label", [2, None], ids=["nonbinary", "missing"])
def test_calibration_loader_rejects_invalid_or_missing_labels(tmp_path, invalid_label):
    root, _ = _release(tmp_path)
    frame = pd.read_csv(root / "calibration.csv")
    frame.loc[0, TARGET_NAME] = invalid_label
    digest = _repin(root, role_frame=frame, update_counts=False)
    with pytest.raises(ValueError, match="invalid labels"):
        load_development_calibration_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


def test_calibration_loader_rejects_inconsistent_selected_role_counts(tmp_path):
    root, _ = _release(tmp_path)
    digest = _repin(
        root,
        attestation_update=lambda record: record["role_files"]["calibration"].update(rows=3),
    )
    with pytest.raises(ValueError, match="inventory is invalid"):
        load_development_calibration_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("SetC_Protected") / "set_c_v1",
        Path("canonical") / "set_c_v1",
        Path(".set-c-development-unpublished"),
    ],
    ids=["protected", "canonical", "unpublished-staging"],
)
def test_calibration_loader_rejects_protected_canonical_and_staging_paths(tmp_path, relative_path):
    root = tmp_path / relative_path
    root.mkdir(parents=True)
    with pytest.raises(PermissionError):
        load_development_calibration_release(
            root, expected_attestation_sha256="0" * 64, allow_synthetic=True
        )


def test_calibration_loader_rejects_parent_traversal_and_nonproduction_path(tmp_path):
    root, digest = _release(tmp_path)
    traversed = root.parent / "unused" / ".." / root.name
    with pytest.raises(PermissionError, match="parent traversal"):
        load_development_calibration_release(
            traversed, expected_attestation_sha256=digest, allow_synthetic=True
        )
    with pytest.raises(PermissionError, match="approved D-038"):
        load_development_calibration_release(root, expected_attestation_sha256=digest)


def test_calibration_loader_rejects_reparse_point_paths(tmp_path, monkeypatch):
    root, digest = _release(tmp_path)
    original_lstat = os.lstat

    class ReparseStat:
        def __init__(self, result):
            self.result = result
            self.st_file_attributes = 0x400

        def __getattr__(self, name):
            return getattr(self.result, name)

    def report_reparse(path):
        result = original_lstat(path)
        if Path(path) == root:
            return ReparseStat(result)
        return result

    monkeypatch.setattr(release_module.os, "lstat", report_reparse)
    with pytest.raises(PermissionError, match="reparse point"):
        load_development_calibration_release(
            root, expected_attestation_sha256=digest, allow_synthetic=True
        )


def test_existing_three_role_loader_remains_compatible(tmp_path):
    root, digest = _release(tmp_path)
    data, attestation = load_development_release(
        root, expected_attestation_sha256=digest, allow_synthetic=True
    )
    assert set(data["split_role"]) == {"train", "validation", "calibration"}
    assert len(data) == 6
    assert attestation["synthetic"] is True


def _valid_rows(indices):
    risk_by_class = {1: 0.95, 2: 0.8, 3: 0.48, 4: 0.30, 5: 0.15}
    rows = []
    for index in indices:
        material_class = 1 if index % 2 else 5
        material_risk = risk_by_class[material_class]
        building = 1.0
        rows.append({
            "slope_risk": np.float32((index % 5) / 10),
            "proximity_risk": np.float32((index % 7) / 10),
            "building_presence": np.float32(building),
            "material_risk": np.float32(material_risk),
            "material_class": np.float32(material_class),
            "wind_speed": np.float32(index % 3),
            "wind_sin": np.float32(0.0),
            "wind_cos": np.float32(1.0),
            "neighbor_burning_count": np.float32(index % 4),
            "composite_flammability": np.float32(building * material_risk),
            "wind_weighted_score": np.float32(material_risk),
            TARGET_NAME: np.int8(material_class == 1),
        })
    return pd.DataFrame(rows, columns=[*CANONICAL_FEATURE_NAMES, TARGET_NAME])


def _synthetic_forest():
    train = _valid_rows(range(100, 160))
    estimator = RandomForestClassifier(n_estimators=7, max_depth=4, random_state=42)
    estimator.fit(train.loc[:, list(CANONICAL_FEATURE_NAMES)], train[TARGET_NAME])
    return estimator


def test_frozen_sigmoid_calibration_preserves_base_forest_and_generic_interface(tmp_path):
    base = _synthetic_forest()
    calibration = _valid_rows(range(20))
    x = calibration.loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    y = calibration[TARGET_NAME].astype(np.int8)
    before_bytes = calibration_module.forest_fingerprint(base)
    original_model = tmp_path / "synthetic_base.joblib"
    calibration_module.joblib.dump(base, original_model)
    original_file_hash = calibration_module._sha256_file(original_model)
    original_fit = RandomForestClassifier.fit

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("RF fit must not be called during calibration")

    RandomForestClassifier.fit = forbidden_fit
    try:
        calibrated, fingerprints = calibration_module.calibrate_frozen_estimator(base, x, y)
    finally:
        RandomForestClassifier.fit = original_fit

    tree = base.estimators_[0].tree_
    original_threshold = tree.threshold[0]
    tree.threshold[0] = original_threshold + 1.0
    assert calibration_module.forest_fingerprint(base) != before_bytes
    tree.threshold[0] = original_threshold
    assert fingerprints["before_sha256"] == before_bytes
    assert fingerprints["after_sha256"] == before_bytes
    assert fingerprints["embedded_sha256"] == before_bytes
    assert calibration_module._sha256_file(original_model) == original_file_hash
    assert callable(calibrated.predict_proba)
    assert calibration_module._positive_index(calibrated.classes_) == 1
    assert calibration_module._positive_index([1, 0]) == 0
    assert np.isfinite(calibration_module.probability_report(calibrated, x, y)["prevalence"])
    artifact_path = tmp_path / "synthetic_calibrated.joblib"
    calibration_module.joblib.dump(calibrated, artifact_path)
    round_trip = calibration_module.joblib.load(artifact_path)
    np.testing.assert_allclose(round_trip.predict_proba(x), calibrated.predict_proba(x), rtol=0, atol=0)
    assert calibration_module.forest_fingerprint(calibration_module._find_forest(round_trip)) == before_bytes


class _FixedProbabilityModel:
    classes_ = np.asarray([0, 1], dtype=np.int8)
    feature_names_in_ = np.asarray(CANONICAL_FEATURE_NAMES, dtype=object)

    def __init__(self, positive):
        self.positive = np.asarray(positive, dtype=np.float64)

    def predict_proba(self, features):
        return np.column_stack([1 - self.positive, self.positive])


def test_probability_report_matches_small_independent_case_and_exact_bin_edges():
    x = _valid_rows(range(11)).loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1], dtype=np.int8)
    p = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    report = calibration_module.probability_report(_FixedProbabilityModel(p), x, y)
    assert report["metrics"]["brier_score"]["value"] == pytest.approx(np.mean((p - y) ** 2))
    assert report["metrics"]["log_loss"]["value"] == pytest.approx(log_loss(y, p, labels=[0, 1]))
    assert report["metrics"]["average_precision"]["value"] == pytest.approx(557 / 720)
    # Exact p=1 is included in the last bin, so the last bin has two values.
    assert [item["count"] for item in report["reliability_bins"]] == [1] * 9 + [2]
    assert report["reliability_bins"][0]["mean_predicted_probability"] == 0.0
    assert report["reliability_bins"][9]["mean_predicted_probability"] == pytest.approx(0.95)
    assert report["classification_threshold"] is None
    assert report["diagnostic_scope"] == "in-sample calibration-fit diagnostics"


def test_every_literal_bin_edge_and_adjacent_values_uses_protocol_intervals():
    scores = [0.0, 1.0]
    for edge in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        scores.extend([np.nextafter(edge, 0.0), edge, np.nextafter(edge, 1.0)])
    x = _valid_rows(range(len(scores))).loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    y = np.asarray([index % 2 for index in range(len(scores))], dtype=np.int8)
    report = calibration_module.probability_report(_FixedProbabilityModel(scores), x, y)
    assert [item["count"] for item in report["reliability_bins"]] == [2] + [3] * 9


def test_probability_report_accepts_endpoint_only_scores_and_marks_empty_bins():
    x = _valid_rows(range(4)).loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    report = calibration_module.probability_report(_FixedProbabilityModel([0, 1, 0, 1]), x, y)
    assert report["metrics"]["roc_auc"]["value"] == 1.0
    assert report["reliability_bins"][1]["count"] == 0
    assert report["reliability_bins"][1]["mean_predicted_probability"] is None
    assert report["reliability_bins"][1]["observed_positive_fraction"] is None
    assert report["reliability_bins"][1]["status"] == "empty"


@pytest.mark.parametrize("bad", [np.nan, -0.01, 1.01])
def test_probability_report_rejects_invalid_probabilities(bad):
    x = _valid_rows(range(2)).loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    with pytest.raises(ValueError):
        calibration_module.probability_report(_FixedProbabilityModel([bad, 0.5]), x, np.asarray([0, 1], dtype=np.int8))


def test_calibration_rejects_constant_base_scores_and_invalid_labels():
    x = _valid_rows(range(4)).loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)

    class ConstantForest(_FixedProbabilityModel):
        def __init__(self):
            self.estimators_ = [_SyntheticTree()]
            self.positive = np.full(4, 0.5)

        def get_params(self, deep=False):
            return {"random_state": 42}

    class _SyntheticTree:
        tree_ = type("Tree", (), {
            "children_left": np.asarray([-1]), "children_right": np.asarray([-1]),
            "feature": np.asarray([-2]), "threshold": np.asarray([-2.0]),
            "value": np.asarray([[[1.0, 1.0]]]), "impurity": np.asarray([0.5]),
            "n_node_samples": np.asarray([2]), "weighted_n_node_samples": np.asarray([2.0]),
        })()

    with pytest.raises(ValueError, match="nonconstant"):
        calibration_module.calibrate_frozen_estimator(ConstantForest(), x, y)
    with pytest.raises(ValueError, match="both binary classes"):
        calibration_module.calibrate_frozen_estimator(_synthetic_forest(), x, np.zeros(4, dtype=np.int8))


def test_synthetic_package_uses_calibration_loader_and_retains_computation_only_receipt(tmp_path, monkeypatch):
    source = _valid_rows(range(10))
    frames = _role_frames()
    frames["calibration"] = source
    release, digest = publish_synthetic_development_release(frames, tmp_path / "release")
    destination = tmp_path / "phase14" / "synthetic-run"
    called = []
    original = calibration_module.load_development_calibration_release

    def checked_loader(*args, **kwargs):
        called.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(calibration_module, "load_development_calibration_release", checked_loader)
    result = calibration_module.run_synthetic_calibration_from_release(
        _synthetic_forest(), release, destination, expected_attestation_sha256=digest
    )
    assert called and called[0]["allow_synthetic"] is True
    assert set(result["manifest"]["input"]) == {
        "attestation_sha256", "calibration_csv_sha256", "rows", "positives", "negatives",
        "source_csv_sha256", "source_revision", "synthetic",
    }
    receipt = json.loads((destination / "execution_receipt.json").read_text(encoding="utf-8"))
    assert receipt["computation_status"] == "complete"
    assert receipt["source_check"] is None and receipt["supervisor_handoff"] is None
    assert receipt["source_revision"] == "synthetic-fixture-not-applicable"
    assert receipt["configuration_sha256"] == result["manifest"]["configuration_sha256"]
    assert receipt["working_directory"] == str(Path.cwd())
    assert receipt["computation_started_at_local"].endswith("+08:00")
    assert receipt["computation_ended_at_local"].endswith("+08:00")
    assert result["manifest"]["phase13_expected_policy"]["base_model_sha256"] == calibration_module.BASE_MODEL_SHA256
    assert result["manifest"]["execution_scope"] == "synthetic_test_only"
    assert "supervisor_completion" not in receipt
    assert "review_record" not in receipt
    assert "execution_receipt.json" not in receipt["finalized_outputs"]
    with pytest.raises(FileExistsError):
        calibration_module.run_synthetic_calibration_from_release(
            _synthetic_forest(), release, destination, expected_attestation_sha256=digest
        )


def test_synthetic_package_failure_retains_original_exception_and_receipt(tmp_path, monkeypatch):
    features = _valid_rows(range(4)).loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    destination = tmp_path / "failed-run"
    provenance = {
        "attestation_sha256": "0" * 64, "calibration_csv_sha256": "0" * 64,
        "rows": 4, "positives": 2, "negatives": 2, "source_csv_sha256": "0" * 64,
        "source_revision": "synthetic", "synthetic": True,
    }

    def fail_calibration(*args, **kwargs):
        raise RuntimeError("original calibration error")

    monkeypatch.setattr(calibration_module, "calibrate_frozen_estimator", fail_calibration)
    with pytest.raises(RuntimeError, match="RuntimeError at frozen calibration") as error:
        calibration_module.run_synthetic_calibration_package(
            _synthetic_forest(), features, labels, destination, provenance=provenance
        )
    receipt = json.loads((destination / "execution_receipt.json").read_text(encoding="utf-8"))
    assert receipt["computation_status"] == "failed"
    assert receipt["failure_reason"] == "RuntimeError at frozen calibration"
    assert "original calibration error" not in json.dumps(receipt)
    assert error.value.__cause__ is None and error.value.__suppress_context__ is True


def test_failure_inventory_error_does_not_replace_primary_exception(tmp_path, monkeypatch, capsys):
    features = _valid_rows(range(4)).loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    destination = tmp_path / "failed-inventory-run"
    provenance = {
        "attestation_sha256": "0" * 64, "calibration_csv_sha256": "0" * 64,
        "rows": 4, "positives": 2, "negatives": 2, "source_csv_sha256": "0" * 64,
        "source_revision": "synthetic", "synthetic": True,
    }

    def fail_calibration(*args, **kwargs):
        raise RuntimeError("primary calibration failure")

    original_iterdir = Path.iterdir
    monkeypatch.setattr(calibration_module, "calibrate_frozen_estimator", fail_calibration)

    def fail_inventory(path):
        if path == destination:
            raise OSError("inventory failure")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_inventory)
    with pytest.raises(RuntimeError, match="RuntimeError at frozen calibration"):
        calibration_module.run_synthetic_calibration_package(
            _synthetic_forest(), features, labels, destination, provenance=provenance
        )
    stderr = capsys.readouterr().err
    assert "partial inventory unavailable (OSError)" in stderr
    assert "inventory failure" not in stderr


def test_production_manifest_shape_and_hash_gate_precede_deserialization(tmp_path, monkeypatch):
    root = tmp_path / "output" / "phase13" / "set_c_rf_v1"
    root.mkdir(parents=True)
    code = tmp_path / "Code"
    (code / "config").mkdir(parents=True)
    config_path = code / "config" / "phase13_set_c_rf_v1.yaml"
    config_path.write_bytes(b"synthetic pinned config")
    config_digest = calibration_module._sha256_file(config_path)
    model_path = root / "base_rf.joblib"
    model_path.write_bytes(b"synthetic pinned model bytes")
    manifest = {
        "schema_version": "set_c_rf_phase13_run.v1",
        "artifact_kind": "set_c_rf_uncalibrated_candidate_run",
        "protocol_sha256": calibration_module.PHASE13_PROTOCOL_SHA256,
        "run_id": calibration_module.PHASE13_RUN_ID,
        "completion_status": "complete",
        "training_source": {"revision": calibration_module.PHASE13_REVISION,
                            "configuration_sha256": config_digest},
        "inputs": {"attestation_sha256": calibration_module.DEVELOPMENT_ATTESTATION_SHA256,
                   "source_csv_sha256": calibration_module.SOURCE_CSV_SHA256,
                   "source_manifest_sha256": calibration_module.SOURCE_MANIFEST_SHA256},
        "schema_population": {"feature_schema_version": "set_c.v1", "target_version": "ignition_t_plus_1.v1",
                              "feature_names": list(CANONICAL_FEATURE_NAMES), "feature_dtype": "float32",
                              "target_name": TARGET_NAME, "target_dtype": "int8", "positive_label": 1},
        "models": {"base_model": {"filename": "base_rf.joblib",
                                    "training_roles_ordered": ["train", "validation"],
                                    "feature_names_in": list(CANONICAL_FEATURE_NAMES),
                                    "classes": [0, 1], "calibrated": False}},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(calibration_module, "_PHASE13_ROOT", root)
    monkeypatch.setattr(calibration_module, "_CODE_ROOT", code)
    monkeypatch.setattr(calibration_module, "PHASE13_CONFIG_SHA256", config_digest)
    monkeypatch.setattr(calibration_module, "PHASE13_MANIFEST_SHA256", calibration_module._sha256_file(manifest_path))
    monkeypatch.setattr(calibration_module, "BASE_MODEL_SHA256", calibration_module._sha256_file(model_path))
    estimator = _synthetic_forest()
    loaded = []
    monkeypatch.setattr(calibration_module.joblib, "load", lambda path: loaded.append(path) or estimator)
    model, evidence = calibration_module.load_verified_base_estimator()
    assert model is estimator and loaded == [model_path]
    assert evidence["manifest_sha256"] == calibration_module.PHASE13_MANIFEST_SHA256

    manifest["training_source"]["revision"] = "wrong-revision"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(calibration_module, "PHASE13_MANIFEST_SHA256", calibration_module._sha256_file(manifest_path))
    loaded.clear()
    with pytest.raises(ValueError, match="manifest contract"):
        calibration_module.load_verified_base_estimator()
    assert loaded == []


def test_production_reparse_root_is_rejected_before_deserialization(tmp_path, monkeypatch):
    root = tmp_path / "approved-looking-root"
    root.mkdir()
    monkeypatch.setattr(calibration_module, "_PHASE13_ROOT", root)
    called = []
    monkeypatch.setattr(calibration_module.joblib, "load", lambda path: called.append(path))
    original_lstat = os.lstat

    class ReparseStat:
        def __init__(self, result):
            self.result = result
            self.st_file_attributes = 0x400

        def __getattr__(self, name):
            return getattr(self.result, name)

    def report_reparse(path):
        result = original_lstat(path)
        return ReparseStat(result) if Path(path) == root else result

    monkeypatch.setattr(calibration_module.os, "lstat", report_reparse)
    with pytest.raises(PermissionError, match="Reparse path"):
        calibration_module.load_verified_base_estimator()
    assert called == []


def test_config_is_activated_with_complete_environment_lock():
    import yaml

    config_path = calibration_module._CODE_ROOT / "config" / "phase14_setc_calibration_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["execution_enabled"] is True
    assert config["calibration"]["method"] == "sigmoid"
    assert config["calibration"]["n_jobs"] == 1
    assert config["inputs"]["base_model_sha256"] == calibration_module.BASE_MODEL_SHA256
    assert config["calibration"]["rows"] == 2060
    assert config["resources"]["min_available_physical_memory_bytes"] == 4 * 1024**3
    assert config["resources"]["min_free_output_volume_bytes"] == 2 * 1024**3
    calibration_module._validate_environment_envelope(config["environment"])
    assert config["paths"]["phase13_root"] == "output/phase13/set_c_rf_v1"
    assert config["paths"]["calibration_release"] == "../../SetC_Development/set_c_v1"
    assert config["execution"]["status"] == "unresolved_separate_run_approval_required"


def test_config_hash_and_production_environment_lock_fail_closed_before_artifacts(tmp_path, monkeypatch):
    import yaml

    source = calibration_module._PHASE14_CONFIG.read_bytes()
    enabled_flag = b"execution_enabled: true"
    assert source.count(enabled_flag) == 1
    disabled_source = source.replace(enabled_flag, b"execution_enabled: false", 1)
    config_path = tmp_path / "phase14.yaml"
    config_path.write_bytes(disabled_source)
    digest = calibration_module._sha256_file(config_path)
    with pytest.raises(PermissionError, match="execution is disabled"):
        calibration_module._load_phase14_config(config_path, digest, synthetic=False)
    with pytest.raises(ValueError, match="hash mismatch"):
        calibration_module._load_phase14_config(config_path, "0" * 64, synthetic=True)
    config = yaml.safe_load(disabled_source)
    assert config["execution_enabled"] is False
    calibration_module._validate_environment_envelope(config["environment"])

    accessed = []
    monkeypatch.setattr(calibration_module, "load_verified_base_estimator", lambda: accessed.append("model"))
    monkeypatch.setattr(calibration_module, "load_development_calibration_release", lambda *a, **k: accessed.append("release"))
    with pytest.raises(PermissionError, match="execution is disabled"):
        calibration_module.run_calibration_job(
            config_path=config_path,
            expected_config_sha256=digest,
            execution_context={},
        )
    assert accessed == []


def test_production_context_contract_requires_clean_revision_approval_and_fixed_empty_paths(tmp_path):
    code = tmp_path / "Code"
    config = {
        "paths": {
            "phase13_root": "output/phase13/set_c_rf_v1",
            "calibration_release": "../../SetC_Development/set_c_v1",
            "output_root": "output/phase14/phase14-setc-rf-61e8f740-r1/calibration",
        },
    }
    context = {
        "source_revision": "a" * 40, "source_clean": True,
        "approval_reference": "owner-ref", "argv": ["fixed-adapter"],
        "working_directory": str(code), "run_id": "approved-run-01",
    }
    destination = calibration_module._validate_production_context(config, context, code)
    assert destination == code / config["paths"]["output_root"] / "approved-run-01"
    with pytest.raises(PermissionError, match="clean"):
        calibration_module._validate_production_context(config, {**context, "source_clean": False}, code)
    assert calibration_module._validate_production_context(
        config, {**context, "approval_reference": "another-procedural-reference"}, code
    ) == destination
    with pytest.raises(PermissionError, match="approval"):
        calibration_module._validate_production_context(config, {**context, "approval_reference": "   "}, code)
    with pytest.raises(PermissionError, match="working directory"):
        calibration_module._validate_production_context(config, {**context, "working_directory": str(tmp_path)}, code)
    with pytest.raises(PermissionError, match="run ID"):
        calibration_module._validate_production_context(config, {**context, "run_id": "../escape"}, code)


def test_environment_lock_digest_baseline_and_observed_runtime_are_bound():
    observed = calibration_module._software_versions()
    lock = dict(observed)
    lock_sha256 = calibration_module.sha256(calibration_module._json_bytes(lock)).hexdigest()
    env = {"status": "complete_locked", "versions": lock, "lock_sha256": lock_sha256,
           "interpreter": {"path": str(Path(__import__("sys").executable))},
           "powershell": {"version": "5.1.26100.9711", "edition": "Desktop"},
           "auxiliary_versions": {"rasterio": "synthetic-reviewed-auxiliary"}}
    assert calibration_module._verify_environment_lock(env, observed=observed) == lock_sha256
    changed = {**lock, "platform": "different-platform"}
    changed_env = {**env, "status": "complete_locked", "versions": changed,
                   "lock_sha256": calibration_module.sha256(calibration_module._json_bytes(changed)).hexdigest()}
    with pytest.raises(PermissionError, match="runtime does not match"):
        calibration_module._verify_environment_lock(changed_env, observed=observed)
    with pytest.raises(PermissionError, match="digest is invalid"):
        calibration_module._verify_environment_lock({**env, "lock_sha256": "0" * 64}, observed=observed)


@pytest.mark.parametrize(("version", "edition", "accepted"), [
    ("5.1.26100.9711", "Desktop", True),
    ("5.1.26100.9444", "Desktop", False),
    ("5.1.26100.9712", "Desktop", False),
    ("5.1.26100.9711", "Core", False),
], ids=["current-host", "old-build", "other-build", "wrong-edition"])
def test_environment_envelope_accepts_only_rebased_powershell(version, edition, accepted):
    versions = dict.fromkeys(("python", "scikit_learn", "optuna", "joblib", "numpy", "pandas", "scipy", "pyyaml", "platform"), "synthetic")
    environment = {
        "status": "complete_locked",
        "versions": versions,
        "lock_sha256": calibration_module.sha256(calibration_module._json_bytes(versions)).hexdigest(),
        "interpreter": {"path": "synthetic-python"},
        "powershell": {"version": version, "edition": edition},
        "auxiliary_versions": {"rasterio": "synthetic"},
    }
    if accepted:
        calibration_module._validate_environment_envelope(environment)
    else:
        with pytest.raises(PermissionError, match="mismatched"):
            calibration_module._validate_environment_envelope(environment)


def test_real_cli_denies_before_artifact_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled CLI attempted artifact access")

    monkeypatch.setattr(calibration_module.joblib, "load", forbidden)
    monkeypatch.setattr(calibration_module, "load_development_calibration_release", forbidden)
    with pytest.raises(SystemExit, match="execution is disabled"):
        calibration_module.main(["--config", "must-not-be-opened"])


def _reviewed_authorization(job="calibration", run_id="synthetic-context"):
    import sys
    versions = calibration_module._software_versions()
    import rasterio
    environment = {"status": "complete_locked", "versions": versions,
        "lock_sha256": calibration_module.sha256(calibration_module._json_bytes(versions)).hexdigest(),
        "interpreter": {"path": sys.executable}, "powershell": {"version": "5.1.26100.9711", "edition": "Desktop"},
        "auxiliary_versions": {"rasterio": rasterio.__version__}}
    return {"schema_version": "phase14a_job_authorization.v1", "job": job, "run_id": run_id,
        "approval_reference": "synthetic-owner-approval", "expected_revision": "a" * 40,
        "created_at_utc": "2026-10-10T00:00:00+00:00", "configuration_sha256": "b" * 64,
        "protocol_sha256": calibration_module.PROTOCOL_SHA256,
        "source_files": {**{name: "c" * 64 for name in calibration_module.SOURCE_FILES},
            "docs/Phase14A_Calibration_Final_Test_Protocol_Proposal.md": calibration_module.PROTOCOL_SHA256,
            "Thesis_RF/Code/config/" + ("phase14_setc_calibration_v1.yaml" if job == "calibration" else "phase14_setc_final_test_v1.yaml"): "b" * 64},
        "environment": environment, "environment_sha256": calibration_module.sha256(calibration_module._json_bytes(environment)).hexdigest(),
        "pretest_lock": None if job == "calibration" else {"lock_id": "synthetic-lock", "sha256": "d" * 64, "owner_approval_reference": "synthetic-lock-approval"},
        "primary_evidence": ({"run_id": "synthetic-primary", "report_sha256": "e" * 64, "receipt_sha256": "f" * 64,
                              "supervisor_completion_sha256": "1" * 64} if job == "final_verification" else None)}


@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
def test_exact_authorization_transport_and_complete_source_inventory(job):
    record = _reviewed_authorization(job)
    text = json.dumps(record, indent=2) + "\n"
    digest = calibration_module.sha256(text.encode("utf-8")).hexdigest()
    assert calibration_module._validate_authorization(text, digest, job, record["approval_reference"], record["expected_revision"], record["run_id"]) == record
    assert len(record["source_files"]) == 19
    normalized = calibration_module._json_bytes(record).decode("utf-8")
    with pytest.raises(PermissionError, match="transport"):
        calibration_module._validate_authorization(normalized, digest, job, record["approval_reference"], record["expected_revision"], record["run_id"])


@pytest.mark.parametrize("change", ["duplicate", "extra", "missing-source", "extra-source", "auxiliary", "nonfinite", "bom", "revision", "lock", "primary"])
def test_authorization_rejects_malformed_or_unresolved_records(change):
    record = _reviewed_authorization()
    if change == "extra": record["extra"] = True
    if change == "missing-source": record["source_files"].pop(next(iter(record["source_files"])))
    if change == "extra-source": record["source_files"]["Thesis_RF/Code/unreviewed.py"] = "a" * 64
    if change == "auxiliary": record["environment"]["auxiliary_versions"] = {}
    if change == "revision": record["expected_revision"] = True
    if change == "lock": record["pretest_lock"] = {}
    if change == "primary": record["primary_evidence"] = {}
    text = json.dumps(record)
    if change == "duplicate": text = text[:-1] + ', "job":"calibration"}'
    if change == "nonfinite": text = text.replace('"pretest_lock": null', '"pretest_lock": NaN')
    if change == "bom": text = "\ufeff" + text
    digest = calibration_module.sha256(text.encode("utf-8")).hexdigest()
    with pytest.raises((ValueError, PermissionError, TypeError)):
        calibration_module._validate_authorization(text, digest, "calibration", "synthetic-owner-approval", "a" * 40, "synthetic-context")


@pytest.mark.parametrize("boundary", ["primary", "inventory", "receipt"])
def test_calibration_private_errors_are_redacted_with_original_priority(tmp_path, monkeypatch, capsys, boundary):
    import traceback
    destination = tmp_path / "private-failure"
    x = _valid_rows(range(4)).loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    y = np.asarray([0, 1, 0, 1], dtype=np.int8)
    base = _synthetic_forest()
    monkeypatch.setattr(calibration_module, "calibrate_frozen_estimator", lambda *a, **k: (_ for _ in ()).throw(ValueError("PRIVATE_CALIBRATION")))
    original_iterdir = Path.iterdir
    if boundary == "inventory":
        monkeypatch.setattr(Path, "iterdir", lambda path: (_ for _ in ()).throw(OSError("PRIVATE_INVENTORY")) if path == destination else original_iterdir(path))
    if boundary == "receipt":
        monkeypatch.setattr(calibration_module, "_write_json_new", lambda *a, **k: (_ for _ in ()).throw(OSError("PRIVATE_RECEIPT")))
    with pytest.raises(RuntimeError, match="ValueError at frozen calibration") as error:
        calibration_module.run_synthetic_calibration_package(base, x, y, destination, provenance={"synthetic": True})
    displayed = "".join(traceback.format_exception(error.type, error.value, error.tb))
    captures = capsys.readouterr()
    receipt = destination / "execution_receipt.json"
    retained = receipt.read_text(encoding="utf-8") if receipt.exists() else ""
    for token in ("PRIVATE_CALIBRATION", "PRIVATE_INVENTORY", "PRIVATE_RECEIPT"):
        assert token not in str(error.value) + displayed + captures.out + captures.err + retained
    assert error.value.__cause__ is None and error.value.__suppress_context__ is True
    assert "During handling of the above exception" not in displayed
    assert "The above exception was the direct cause" not in displayed


def test_cli_context_denial_has_explicit_zero_artifact_counters(monkeypatch, capsys):
    counts = dict.fromkeys(("model", "release", "fit", "serialize", "output"), 0)
    def forbidden(name):
        counts[name] += 1
        pytest.fail("forbidden context-denied artifact operation")
    monkeypatch.setattr(calibration_module.joblib, "load", lambda *a, **k: forbidden("model"))
    monkeypatch.setattr(calibration_module, "load_development_calibration_release", lambda *a, **k: forbidden("release"))
    monkeypatch.setattr(calibration_module, "calibrate_frozen_estimator", lambda *a, **k: forbidden("fit"))
    monkeypatch.setattr(calibration_module.joblib, "dump", lambda *a, **k: forbidden("serialize"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: forbidden("output"))
    with pytest.raises(SystemExit) as error:
        calibration_module.main(["--config", "PRIVATE_ARGUMENT"])
    assert counts == dict.fromkeys(counts, 0)
    assert "PRIVATE_ARGUMENT" not in str(error.value) + capsys.readouterr().err



def _worker_metadata(evidence, handoff, *, launcher_pid=124, worker_pid=125, worker_ppid=124):
    """Complete synthetic claim/OS observations; never an actual process attestation."""
    from datetime import datetime, timedelta, timezone
    start = datetime.fromisoformat(handoff["created_at_utc"])
    stamp = lambda seconds: (start + timedelta(seconds=seconds)).isoformat()
    physical = str(evidence / "synthetic-base-python.exe") if worker_pid != launcher_pid else handoff["executable"]
    command = "synthetic worker command"
    claim = {"schema_version": "phase14a_worker_claim.v1", "job": handoff["job"],
        "run_id": handoff["run_id"], "nonce": handoff["nonce"], "handoff_sha256": calibration_module._sha256_file(evidence / "supervisor_handoff.json"),
        "supervisor_pid": handoff["supervisor_pid"], "worker_pid": worker_pid, "worker_ppid": worker_ppid,
        "logical_executable": handoff["executable"], "physical_executable": physical,
        "worker_created_at_utc": stamp(1 if worker_pid == launcher_pid else 1.2), "command_line": command,
        "claimed_at_utc": stamp(1.3)}
    claim_path = evidence / "worker_claim.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    supervisor = {"pid": handoff["supervisor_pid"], "ppid": 122, "created_at_utc": stamp(-10),
                  "executable": str(evidence / "synthetic-powershell.exe"), "command_line": "synthetic supervisor"}
    launcher = {"pid": launcher_pid, "ppid": handoff["supervisor_pid"], "created_at_utc": stamp(1),
                "executable": handoff["executable"], "command_line": command if worker_pid == launcher_pid else "synthetic launcher"}
    worker = dict(launcher) if worker_pid == launcher_pid else {"pid": worker_pid, "ppid": worker_ppid,
              "created_at_utc": stamp(1.2), "executable": physical, "command_line": command}
    binding = {"schema_version": "phase14a_worker_binding.v1", "claim_sha256": calibration_module._sha256_file(claim_path),
        "claim": claim, "observed_at_utc": stamp(1.4), "observation_source": "Get-CimInstance Win32_Process",
        "supervisor": supervisor, "launcher": launcher, "worker": worker}
    (evidence / "worker_binding.json").write_text(json.dumps(binding), encoding="utf-8")
    binding_hash = calibration_module._binding_hash(binding, evidence)
    process = {"schema_version": "phase14a_process.v2", "run_id": handoff["run_id"], "process_id": launcher_pid,
        "process_ids": [launcher_pid], "process_started_at_utc": stamp(1.1),
        "executable": handoff["executable"], "working_directory": handoff["working_directory"], "worker_binding": binding}
    # Fast workers may be absent from the core scan; the launched PID remains mandatory.
    (evidence / "process.json").write_text(json.dumps(process), encoding="utf-8")
    exited = {"schema_version": "phase14a_worker_exit.v1", "worker_binding_sha256": binding_hash,
        "worker_pid": worker_pid, "worker_created_at_utc": worker["created_at_utc"], "observed_at_utc": stamp(59.5),
        "observation_source": "Get-CimInstance Win32_Process", "same_instance_running": False, "pid_reused": False}
    return process, exited


def _worker_creation_metadata(evidence, handoff, case):
    """Rebind synthetic evidence to isolate creation-time precision and chronology."""
    from datetime import datetime, timedelta
    process, _ = _worker_metadata(evidence, handoff)
    binding = process["worker_binding"]
    claim = binding["claim"]
    base = datetime.fromisoformat(binding["worker"]["created_at_utc"])
    def stamp(value):
        if value == "stale":
            return (base - timedelta(seconds=30)).isoformat()
        if value == "next":
            return (base + timedelta(microseconds=1)).isoformat(timespec="microseconds").replace("+00:00", "0+00:00")
        return base.isoformat(timespec="microseconds").replace("+00:00", value + "+00:00")
    observed, reported, accepted = {
        "equal": ("0", "0", True), "within-forward": ("0", "9", True),
        "within-reverse": ("9", "0", True), "adjacent": ("0", "next", False),
        "boundary-forward": ("9", "next", False), "boundary-reverse": ("next", "9", False),
        "stale": ("stale", "stale", False),
    }[case]
    binding["worker"]["created_at_utc"] = stamp(observed)
    claim["worker_created_at_utc"] = stamp(reported)
    claim_path = evidence / "worker_claim.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    binding["claim_sha256"] = calibration_module._sha256_file(claim_path)
    (evidence / "worker_binding.json").write_text(json.dumps(binding), encoding="utf-8")
    (evidence / "process.json").write_text(json.dumps(process), encoding="utf-8")
    return process, claim, accepted

def _context_fixture(tmp_path, monkeypatch, job):
    import sys
    from datetime import datetime, timedelta, timezone
    repo = tmp_path / "synthetic-checkout"
    code = repo / "Thesis_RF" / "Code"
    for member in calibration_module.SOURCE_FILES:
        path = repo / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source identity: " + member, encoding="utf-8")
    executable = repo / ".venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"synthetic interpreter path anchor; never executed")
    monkeypatch.setattr(calibration_module, "_CODE_ROOT", code)
    monkeypatch.setattr(calibration_module, "PROTOCOL_SHA256", calibration_module._sha256_file(repo / calibration_module.SOURCE_FILES[-1]))
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: code))
    record = _reviewed_authorization(job)
    record["source_files"] = {name: calibration_module._sha256_file(repo / name) for name in calibration_module.SOURCE_FILES}
    config_name = "phase14_setc_calibration_v1.yaml" if job == "calibration" else "phase14_setc_final_test_v1.yaml"
    record["configuration_sha256"] = record["source_files"]["Thesis_RF/Code/config/" + config_name]
    text = json.dumps(record, indent=2) + "\n"
    evidence = code / "output" / "phase14" / "evidence" / record["run_id"]
    evidence.mkdir(parents=True)
    arguments = calibration_module._fixed_child_arguments(job, record["approval_reference"], record["expected_revision"], record["run_id"], record["configuration_sha256"], evidence)
    monkeypatch.setattr(sys, "orig_argv", [str(executable), *arguments])
    output_root = code / "output" / "phase14"
    destination = (output_root / "phase14-setc-rf-61e8f740-r1" / "calibration" / record["run_id"] if job == "calibration"
                   else output_root / "final-test" / record["pretest_lock"]["sha256"][:12] / job.removeprefix("final_") / record["run_id"])
    now = datetime.now(timezone.utc)
    handoff = {"schema_version": "phase14a_supervisor_handoff.v1", "job": job, "run_id": record["run_id"],
        "approval_reference": record["approval_reference"], "expected_revision": record["expected_revision"],
        "authorization_json": text, "authorization_sha256": calibration_module.sha256(text.encode()).hexdigest(),
        "nonce": "a" * 32, "supervisor_pid": os.getppid(), "executable": str(executable), "argv": arguments,
        "working_directory": str(code), "evidence_directory": str(evidence), "destination": str(destination),
        "created_at_utc": (now - timedelta(seconds=3)).isoformat(), "deadline_utc": (now + timedelta(seconds=60)).isoformat()}
    path = evidence / "supervisor_handoff.json"
    path.write_text(json.dumps(handoff), encoding="utf-8")
    process, _ = _worker_metadata(evidence, handoff, launcher_pid=os.getpid(), worker_pid=os.getpid(), worker_ppid=os.getppid())
    claim = process["worker_binding"]["claim"]
    monkeypatch.setattr(calibration_module, "_current_worker_instance", lambda: {key: claim[key] for key in
        ("worker_pid", "worker_ppid", "worker_created_at_utc", "physical_executable", "command_line")})
    monkeypatch.setenv("PHASE14_HANDOFF_PATH", str(path))
    monkeypatch.setenv("PHASE14_SUPERVISOR_NONCE", handoff["nonce"])
    def git(command, **kwargs):
        assert command[1] == "--literal-pathspecs"
        class Result:
            stdout = "a" * 40 + "\n" if "rev-parse" in command else ""
        return Result()
    monkeypatch.setattr(calibration_module.subprocess, "run", git)
    return record, handoff, path, arguments


@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
def test_observed_context_binds_temp_source_environment_and_process(job, tmp_path, monkeypatch):
    record, handoff, path, arguments = _context_fixture(tmp_path, monkeypatch, job)
    context = calibration_module._observe_child_context(job, arguments)
    assert context["source_check"]["source_files"] == record["source_files"]
    assert context["source_check"]["clean"] is True
    assert context["supervisor_handoff"] == {"run_id": record["run_id"], "nonce": handoff["nonce"],
        "sha256": calibration_module._sha256_file(path), "supervisor_pid": os.getppid(), "child_pid": os.getpid()}
    assert context["worker_binding_sha256"] == calibration_module._sha256_file(path.parent / "worker_binding.json")
    assert not Path(handoff["destination"]).exists()
    supplied = arguments[3:] if job == "calibration" else arguments[2:]
    parsed, cli_context, transported = calibration_module._child_cli(supplied, "calibration" if job == "calibration" else None)
    assert parsed.run_id == record["run_id"] and cli_context == context and transported == record


@pytest.mark.parametrize("change", ["nonce", "pid", "job", "run", "argv", "hash", "deadline", "destination", "source", "dirty", "git", "cwd", "interpreter", "environment", "process", "protocol", "config", "auxiliary", "powershell-version", "powershell-edition",
    *("version-" + key for key in ("python", "scikit_learn", "optuna", "joblib", "numpy", "pandas", "scipy", "pyyaml", "platform"))])
def test_observed_context_denies_before_computation_despite_supplied_clean_claim(tmp_path, monkeypatch, change):
    import sys
    record, handoff, path, arguments = _context_fixture(tmp_path, monkeypatch, "calibration")
    counts = dict.fromkeys(("model", "release", "fit", "serialize", "output"), 0)
    def forbidden(name):
        counts[name] += 1
        pytest.fail("forbidden operation during context rejection")
    monkeypatch.setattr(calibration_module.joblib, "load", lambda *a, **k: forbidden("model"))
    monkeypatch.setattr(calibration_module, "load_development_calibration_release", lambda *a, **k: forbidden("release"))
    monkeypatch.setattr(calibration_module, "calibrate_frozen_estimator", lambda *a, **k: forbidden("fit"))
    monkeypatch.setattr(calibration_module.joblib, "dump", lambda *a, **k: forbidden("serialize"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: forbidden("output"))
    mutations = {"nonce": ("nonce", "b" * 32), "pid": ("supervisor_pid", True), "job": ("job", "final_primary"),
        "run": ("run_id", "other-run"), "argv": ("argv", []), "hash": ("authorization_sha256", "0" * 64),
        "deadline": ("deadline_utc", "2000-01-01T00:00:00+00:00"), "destination": ("destination", str(tmp_path / "outside"))}
    if change in mutations: handoff[mutations[change][0]] = mutations[change][1]
    if change.startswith("version-") or change in {"protocol", "config", "auxiliary", "powershell-version", "powershell-edition"}:
        if change.startswith("version-"): record["environment"]["versions"][change.removeprefix("version-")] = "unreviewed-version"
        if change == "protocol": record["protocol_sha256"] = "0" * 64
        if change == "config": record["configuration_sha256"] = "0" * 64
        if change == "auxiliary": record["environment"]["auxiliary_versions"]["rasterio"] = "unreviewed-version"
        if change == "powershell-version": record["environment"]["powershell"]["version"] = "7.6.5"
        if change == "powershell-edition": record["environment"]["powershell"]["edition"] = "Core"
        record["environment"]["lock_sha256"] = calibration_module.sha256(calibration_module._json_bytes(record["environment"]["versions"])).hexdigest()
        record["environment_sha256"] = calibration_module.sha256(calibration_module._json_bytes(record["environment"])).hexdigest()
        handoff["authorization_json"] = json.dumps(record)
        handoff["authorization_sha256"] = calibration_module.sha256(handoff["authorization_json"].encode()).hexdigest()
    path.write_text(json.dumps(handoff), encoding="utf-8")
    if change == "source": (calibration_module._CODE_ROOT / "modules" / "feature_pipeline.py").write_text("changed", encoding="utf-8")
    if change in {"dirty", "git"}:
        original_git = calibration_module.subprocess.run
        def failed_git(command, **kwargs):
            if change == "git": raise OSError("PRIVATE_GIT")
            if "status" in command:
                return type("Result", (), {"stdout": " M source.py"})()
            return original_git(command, **kwargs)
        monkeypatch.setattr(calibration_module.subprocess, "run", failed_git)
    if change == "cwd": monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))
    if change == "interpreter": monkeypatch.setattr(sys, "executable", str(tmp_path / "wrong-python.exe"))
    if change == "environment": monkeypatch.setattr(calibration_module, "_software_versions", lambda: {**record["environment"]["versions"], "platform": "wrong"})
    if change == "process": (path.parent / "process.json").write_text('{"process_id":true}', encoding="utf-8")
    with pytest.raises(PermissionError) as error:
        calibration_module._observe_child_context("calibration", arguments)
    assert counts == dict.fromkeys(counts, 0)
    assert "PRIVATE_GIT" not in str(error.value)
    assert error.value.__cause__ is None and error.value.__suppress_context__ is True



@pytest.mark.parametrize("relative", ["nested_package/injected.py", "modules/nested_package/injected.py", "modules/nested_staged/injected.py"])
def test_nested_runtime_source_denial_reaches_inventory_and_prunes_excluded_trees(tmp_path, monkeypatch, relative):
    record, handoff, path, arguments = _context_fixture(tmp_path, monkeypatch, "calibration")
    code = calibration_module._CODE_ROOT
    injected = code / relative
    injected.parent.mkdir(parents=True)
    injected.write_text("synthetic unreviewed import source", encoding="utf-8")
    excluded = [code / name for name in ("output", "data", "models", "protected", "manuscripts")]
    for directory in excluded:
        directory.mkdir(exist_ok=True)
        (directory / "never_inspect.py").write_text("excluded synthetic source", encoding="utf-8")
    counts = dict.fromkeys(("excluded_traversal", "model", "release", "fit", "serialize", "output"), 0)
    if "nested_staged" in relative:
        original_git = calibration_module.subprocess.run
        relative_source = injected.relative_to(code.parents[1]).as_posix()
        def staged_git(command, **kwargs):
            if "ls-files" in command:
                return type("Result", (), {"stdout": relative_source})()
            if "status" in command and relative_source in command:
                return type("Result", (), {"stdout": "A  " + relative_source})()
            return original_git(command, **kwargs)
        monkeypatch.setattr(calibration_module.subprocess, "run", staged_git)
    inventory_calls = []
    original_inventory = calibration_module._runtime_source_inventory
    original_iterdir = Path.iterdir
    def forbidden(name):
        counts[name] += 1
        pytest.fail(f"forbidden operation during nested source denial: {name}")
    def guarded_iterdir(directory):
        if any(directory == root or root in directory.parents for root in excluded):
            forbidden("excluded_traversal")
        return original_iterdir(directory)
    def inventory(repo):
        inventory_calls.append(repo)
        return original_inventory(repo)
    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    monkeypatch.setattr(calibration_module, "_runtime_source_inventory", inventory)
    monkeypatch.setattr(calibration_module.joblib, "load", lambda *a, **k: forbidden("model"))
    monkeypatch.setattr(calibration_module, "load_development_calibration_release", lambda *a, **k: forbidden("release"))
    monkeypatch.setattr(calibration_module, "calibrate_frozen_estimator", lambda *a, **k: forbidden("fit"))
    monkeypatch.setattr(calibration_module.joblib, "dump", lambda *a, **k: forbidden("serialize"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: forbidden("output"))
    with pytest.raises(PermissionError) as error:
        calibration_module._observe_child_context("calibration", arguments)
    assert inventory_calls == [code.parents[1]]
    assert counts == dict.fromkeys(counts, 0)
    assert error.value.__cause__ is None and error.value.__suppress_context__ is True

@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
@pytest.mark.parametrize("topology", ["direct", "launcher-worker"])
def test_worker_context_supports_only_bound_instances(job, topology, tmp_path, monkeypatch):
    record, handoff, path, arguments = _context_fixture(tmp_path, monkeypatch, job)
    worker_pid = os.getpid()
    launcher_pid = worker_pid if topology == "direct" else os.getppid()
    supervisor_pid = os.getppid() if topology == "direct" else os.getppid() + 10000
    handoff["supervisor_pid"] = supervisor_pid
    path.write_text(json.dumps(handoff), encoding="utf-8")
    process, _ = _worker_metadata(path.parent, handoff, launcher_pid=launcher_pid,
                                 worker_pid=worker_pid, worker_ppid=os.getppid())
    claim = process["worker_binding"]["claim"]
    monkeypatch.setattr(calibration_module, "_current_worker_instance", lambda: {key: claim[key] for key in
        ("worker_pid", "worker_ppid", "worker_created_at_utc", "physical_executable", "command_line")})
    context = calibration_module._observe_child_context(job, arguments)
    assert context["supervisor_handoff"]["supervisor_pid"] == supervisor_pid
    assert context["supervisor_handoff"]["child_pid"] == worker_pid
    assert process["process_id"] == launcher_pid
    assert calibration_module._observe_child_context(job, arguments) == context  # no claim rewrite


@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
@pytest.mark.parametrize("change", ["pid", "parent", "start", "image", "nonce", "claim-missing",
    "binding-missing", "expired", "unavailable", "process-v1", "binding-conflict"])
def test_worker_context_denials_precede_numerical_access(job, change, tmp_path, monkeypatch):
    _, handoff, path, arguments = _context_fixture(tmp_path, monkeypatch, job)
    evidence = path.parent
    claim_path = evidence / "worker_claim.json"
    process = json.loads((evidence / "process.json").read_text())
    claim = json.loads(claim_path.read_text())
    if change == "pid": claim["worker_pid"] += 1
    if change == "parent": claim["worker_ppid"] += 1
    if change == "start": claim["worker_created_at_utc"] = "2000-01-01T00:00:00+00:00"
    if change == "image": claim["physical_executable"] = str(tmp_path / "forged.exe")
    if change == "nonce": claim["nonce"] = "b" * 32
    if change == "expired":
        handoff["deadline_utc"] = "2000-01-01T00:00:00+00:00"
        path.write_text(json.dumps(handoff))
    if change == "process-v1": process["schema_version"] = "phase14a_process.v1"
    if change == "binding-conflict": process["worker_binding"]["worker"]["pid"] += 1
    claim_path.write_text(json.dumps(claim))
    (evidence / "process.json").write_text(json.dumps(process))
    if change == "claim-missing": claim_path.unlink()
    if change == "binding-missing": (evidence / "worker_binding.json").unlink()
    if change == "unavailable":
        monkeypatch.setattr(calibration_module, "_current_worker_instance",
                            lambda: (_ for _ in ()).throw(OSError("PRIVATE_WORKER_OBSERVER")))
    counts = dict.fromkeys(("model", "rows", "fit", "serialize", "output"), 0)
    def forbidden(name):
        counts[name] += 1
        pytest.fail("forbidden worker denial access")
    monkeypatch.setattr(calibration_module.joblib, "load", lambda *a, **k: forbidden("model"))
    monkeypatch.setattr(calibration_module, "load_development_calibration_release", lambda *a, **k: forbidden("rows"))
    monkeypatch.setattr(calibration_module, "calibrate_frozen_estimator", lambda *a, **k: forbidden("fit"))
    monkeypatch.setattr(calibration_module.joblib, "dump", lambda *a, **k: forbidden("serialize"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: forbidden("output"))
    with pytest.raises(PermissionError) as error:
        calibration_module._observe_child_context(job, arguments)
    assert counts == dict.fromkeys(counts, 0)
    assert error.value.__cause__ is None and error.value.__suppress_context__ is True
    import traceback
    displayed = "".join(traceback.format_exception(error.type, error.value, error.tb))
    assert "PRIVATE_WORKER_OBSERVER" not in displayed
    assert "During handling of the above exception" not in displayed
    assert "The above exception was the direct cause" not in displayed


@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
@pytest.mark.parametrize("case", ["equal", "within-forward", "within-reverse", "adjacent",
                                  "boundary-forward", "boundary-reverse", "stale"])
def test_worker_creation_precision_matches_supervisor(job, case, tmp_path, monkeypatch):
    _, handoff, path, _ = _context_fixture(tmp_path, monkeypatch, job)
    process, claim, accepted = _worker_creation_metadata(path.parent, handoff, case)
    arguments = (process, handoff, claim,
                 calibration_module._sha256_file(path.parent / "worker_claim.json"),
                 calibration_module._sha256_file(path))
    if accepted:
        assert calibration_module._validate_worker_binding(*arguments) == process["worker_binding"]
    else:
        with pytest.raises(ValueError, match="Worker startup binding mismatch"):
            calibration_module._validate_worker_binding(*arguments)


@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
@pytest.mark.parametrize("topology", ["direct", "launcher-worker"])
@pytest.mark.parametrize("failure", [None, "source", "environment", "expired-after-checks"])
def test_identity_handshake_precedes_slow_checks_without_releasing_computation(
        job, topology, failure, tmp_path, monkeypatch):
    import time
    from datetime import datetime as RealDatetime
    record, handoff, path, arguments = _context_fixture(tmp_path, monkeypatch, job)
    worker_pid = os.getpid()
    launcher_pid = worker_pid if topology == "direct" else os.getppid()
    handoff["supervisor_pid"] = os.getppid() if topology == "direct" else os.getppid() + 10000
    path.write_text(json.dumps(handoff), encoding="utf-8")
    process, _ = _worker_metadata(path.parent, handoff, launcher_pid=launcher_pid,
                                 worker_pid=worker_pid, worker_ppid=os.getppid())
    claim = process["worker_binding"]["claim"]
    monkeypatch.setattr(calibration_module, "_current_worker_instance", lambda: {key: claim[key] for key in
        ("worker_pid", "worker_ppid", "worker_created_at_utc", "physical_executable", "command_line")})
    events = []
    original_claim = calibration_module._claim_and_await_worker
    original_source = calibration_module._observe_source
    original_environment = calibration_module._verify_environment_lock
    def bind(*args):
        binding = original_claim(*args)
        events.append("binding")
        return binding
    def source(*args):
        assert events == ["binding"]
        events.append("source")
        time.sleep(0.02)
        return original_source(*args)
    def environment(*args):
        assert events == ["binding", "source"]
        events.append("environment")
        time.sleep(0.02)
        return original_environment(*args)
    monkeypatch.setattr(calibration_module, "_claim_and_await_worker", bind)
    monkeypatch.setattr(calibration_module, "_observe_source", source)
    monkeypatch.setattr(calibration_module, "_verify_environment_lock", environment)
    if failure == "source":
        (calibration_module._CODE_ROOT / "modules" / "feature_pipeline.py").write_text("changed synthetic source")
    if failure == "environment":
        monkeypatch.setattr(calibration_module, "_software_versions",
                            lambda: {**record["environment"]["versions"], "platform": "wrong"})
    if failure == "expired-after-checks":
        class Clock(RealDatetime):
            @classmethod
            def now(cls, tz=None):
                return RealDatetime.fromisoformat(handoff["deadline_utc"]) if "environment" in events else RealDatetime.now(tz)
        monkeypatch.setattr(calibration_module, "datetime", Clock)
    counts = dict.fromkeys(("model", "rows", "fit", "serialize", "output", "receipt"), 0)
    def forbidden(name):
        counts[name] += 1
        pytest.fail("forbidden computation before context acceptance")
    monkeypatch.setattr(calibration_module.joblib, "load", lambda *a, **k: forbidden("model"))
    monkeypatch.setattr(calibration_module, "load_development_calibration_release", lambda *a, **k: forbidden("rows"))
    monkeypatch.setattr(calibration_module, "calibrate_frozen_estimator", lambda *a, **k: forbidden("fit"))
    monkeypatch.setattr(calibration_module.joblib, "dump", lambda *a, **k: forbidden("serialize"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: forbidden("output"))
    original_write = calibration_module._write_json_new
    def metadata_only(target, value):
        if target.parent != path.parent or not target.name.startswith("worker_claim-"):
            forbidden("receipt")
        return original_write(target, value)
    monkeypatch.setattr(calibration_module, "_write_json_new", metadata_only)
    if failure is None:
        context = calibration_module._observe_child_context(job, arguments)
        assert context["worker_binding_sha256"] == calibration_module._sha256_file(path.parent / "worker_binding.json")
    else:
        with pytest.raises(PermissionError) as error:
            calibration_module._observe_child_context(job, arguments)
        assert error.value.__cause__ is None and error.value.__suppress_context__ is True
    assert events == (["binding", "source"] if failure == "source" else ["binding", "source", "environment"])
    assert counts == dict.fromkeys(counts, 0)
    assert not Path(handoff["destination"]).exists()
    assert not (Path(handoff["destination"]) / "execution_receipt.json").exists()


@pytest.mark.parametrize("job", ["calibration", "final_primary", "final_verification"])
@pytest.mark.parametrize("failure", ["claim-timeout", "operation-deadline"])
def test_cold_claim_boundary_denies_before_source_environment_and_computation(job, failure, tmp_path, monkeypatch):
    from datetime import datetime as RealDatetime
    _, handoff, path, arguments = _context_fixture(tmp_path, monkeypatch, job)
    counts = dict.fromkeys(("source", "environment", "model", "rows", "fit", "serialize", "output", "receipt"), 0)
    def forbidden(name):
        counts[name] += 1
        pytest.fail("forbidden cold-claim access")
    monkeypatch.setattr(calibration_module, "_observe_source", lambda *a, **k: forbidden("source"))
    monkeypatch.setattr(calibration_module, "_verify_environment_lock", lambda *a, **k: forbidden("environment"))
    monkeypatch.setattr(calibration_module.joblib, "load", lambda *a, **k: forbidden("model"))
    monkeypatch.setattr(calibration_module, "load_development_calibration_release", lambda *a, **k: forbidden("rows"))
    monkeypatch.setattr(calibration_module, "calibrate_frozen_estimator", lambda *a, **k: forbidden("fit"))
    monkeypatch.setattr(calibration_module.joblib, "dump", lambda *a, **k: forbidden("serialize"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: forbidden("output"))
    monkeypatch.setattr(calibration_module, "_write_json_new", lambda *a, **k: forbidden("receipt"))
    sleeps = []
    if failure == "claim-timeout":
        (path.parent / "process.json").unlink()
        ticks = iter((0, 0, 0, 10))
        monkeypatch.setattr(calibration_module.time, "monotonic", lambda: next(ticks, 10))
        monkeypatch.setattr(calibration_module.time, "sleep", sleeps.append)
    else:
        class Clock(RealDatetime):
            calls = 0
            @classmethod
            def now(cls, tz=None):
                cls.calls += 1
                return RealDatetime.now(tz) if cls.calls == 1 else RealDatetime.fromisoformat(handoff["deadline_utc"])
        monkeypatch.setattr(calibration_module, "datetime", Clock)
    with pytest.raises(PermissionError) as error:
        calibration_module._observe_child_context(job, arguments)
    assert "at process handoff" in str(error.value)
    assert error.value.__cause__ is None and error.value.__suppress_context__ is True
    assert counts == dict.fromkeys(counts, 0)
    assert sleeps == ([0.02] if failure == "claim-timeout" else [])
    assert not Path(handoff["destination"]).exists()
    assert not (Path(handoff["destination"]) / "execution_receipt.json").exists()
