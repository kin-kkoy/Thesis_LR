import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from evaluate_rf_final_test import (
    CSV_COLUMNS,
    evaluate_authorized_final_test,
    evaluate_final_test,
    probability_report,
)
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES, TARGET_NAME
from modules.set_c_split_release import _synthetic_test_provenance


class _SyntheticEstimator:
    feature_names_in_ = np.asarray(CANONICAL_FEATURE_NAMES)
    classes_ = np.asarray([1, 0], dtype=np.int8)

    def predict_proba(self, features):
        positive = np.asarray(features)[:, 0]
        return np.column_stack((positive, 1 - positive))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_environment():
    from modules import phase14_calibration as phase14
    versions = {key: "synthetic" for key in ("python", "scikit_learn", "optuna", "joblib", "numpy",
                                            "pandas", "scipy", "pyyaml", "platform")}
    return {"status": "complete_locked", "versions": versions,
            "lock_sha256": hashlib.sha256(phase14._json_bytes(versions)).hexdigest(),
            "interpreter": {"path": sys.executable}, "powershell": {"version": "5.1.26100.9444", "edition": "Desktop"},
            "auxiliary_versions": {"rasterio": "synthetic"}}


def _complete_lock(config, config_hash):
    import evaluate_rf_final_test as evaluator
    from modules import phase14_calibration as phase14
    return {"schema_version": "phase14a_pretest_lock.v1", "lock_id": "synthetic-lock", "created_at_utc": "2026-10-10T00:00:00+00:00",
        "configuration_sha256": config_hash, "evaluation_revision": "b" * 40, "environment": config["environment"],
        **config["identities"], "protocol": {"id": phase14.PROTOCOL_ID, "sha256": phase14.PROTOCOL_SHA256},
        "calibration_revision": "a" * 40, "evaluator_source_sha256": _digest(Path(evaluator.__file__)),
        "phase13": {"run_id": phase14.PHASE13_RUN_ID, "source_revision": phase14.PHASE13_REVISION,
            "protocol_sha256": phase14.PHASE13_PROTOCOL_SHA256, "configuration_sha256": phase14.PHASE13_CONFIG_SHA256,
            "manifest_sha256": phase14.PHASE13_MANIFEST_SHA256, "base_model_sha256": phase14.BASE_MODEL_SHA256,
            "development_attestation_sha256": phase14.DEVELOPMENT_ATTESTATION_SHA256,
            "source_csv_sha256": phase14.SOURCE_CSV_SHA256, "source_manifest_sha256": phase14.SOURCE_MANIFEST_SHA256},
        "calibration": {"run_id": "synthetic-calibration", "approval_reference": "synthetic-calibration-approval",
            "method": "sigmoid", "method_decision_reference": phase14.DECISION_REFERENCE,
            "calibration_csv_sha256": phase14.CALIBRATION_CSV_SHA256,
            **{key: "1" * 64 for key in ("configuration_sha256", "source_sha256", "manifest_sha256", "report_sha256",
                 "receipt_sha256", "supervisor_completion_sha256", "package_review_sha256", "forest_fingerprint_sha256")}},
        "prediction_contract": {"feature_schema_version": phase14.SET_C_SCHEMA_VERSION, "feature_names": list(CANONICAL_FEATURE_NAMES),
            "feature_dtype": "float32", "target_name": TARGET_NAME, "target_version": phase14.TARGET_VERSION,
            "target_dtype": "int8", "required_labels": [0, 1], "positive_label": 1},
        "reporting": {"metrics": ["average_precision", "roc_auc", "brier_score", "log_loss"],
            "brier_definition": "mean_squared_probability_error", "log_loss_labels": [0, 1], "bin_edges": list(phase14.BIN_EDGES),
            "endpoint_rule": "exact_internal_edge_starts_next_bin; p1_in_last_bin", "empty_bin_statistics": None,
            "classification_threshold": None, "verification_rel_tol": 1e-10, "verification_abs_tol": 1e-12}}


def _fixture(root: Path, *, enabled=True):
    protected = root / "protected"
    development = root / "development"
    protected.mkdir()
    development.mkdir()
    rows_path = protected / "final_test_rows.csv"
    model_path = root / "base_rf_calibrated.joblib"
    lock_path = root / "pretest_lock.json"
    config_path = root / "config.yaml"
    output_root = root / "outputs"
    output_root.mkdir()

    values = {name: np.zeros(4, dtype=np.float32) for name in CANONICAL_FEATURE_NAMES}
    values["slope_risk"] = np.asarray([0.1, np.nextafter(np.float32(0.1), np.float32(1)), 0.9, 1.0], dtype=np.float32)
    values["Ignited"] = np.asarray([0, 1, 0, 1], dtype=np.int8)
    values.update({name: ["x"] * 4 for name in CSV_COLUMNS if name not in values and name not in {"Ignited"}})
    values["feature_schema_version"] = ["set_c.v1"] * 4
    values["target_version"] = ["ignition_t_plus_1.v1"] * 4
    pd.DataFrame(values, columns=CSV_COLUMNS).to_csv(rows_path, index=False)
    joblib.dump(_SyntheticEstimator(), model_path)
    row_hash, model_hash = _digest(rows_path), _digest(model_path)

    source_digest = "1" * 64
    provenance = _synthetic_test_provenance(source_digest, 7)
    role_counts = {
        "train": {"components": 1, "rows": 1, "positives": 1, "negatives": 0},
        "validation": {"components": 1, "rows": 1, "positives": 1, "negatives": 0},
        "calibration": {"components": 1, "rows": 1, "positives": 1, "negatives": 0},
        "final_test": {"components": 1, "rows": 4, "positives": 2, "negatives": 2},
    }
    protected_record = {
        "schema_version": "set_c_protected_attestation.v1", "package_schema_version": "set_c_protected_package.v1",
        "mapping_schema_version": "set_c_complete_split_manifest.v1", "final_test_schema_version": "set_c_final_test_rows.v1",
        "policy_version": "set_c_deterministic_allocation_policy.v1", "feature_schema_version": "set_c.v1",
        "target_version": "ignition_t_plus_1.v1", "source_csv_sha256": source_digest,
        "split_manifest_sha256": "2" * 64, "final_test_rows_sha256": row_hash,
        "source_rows": 7, "component_count": 4, "final_test_rows": 4,
        "role_counts": role_counts,
        "source_provenance": provenance, "synthetic": True,
        "publication_state": "protected_only_incomplete_until_paired_release",
    }
    protected_path = protected / "attestation.json"
    protected_path.write_text(json.dumps(protected_record), encoding="utf-8")
    protected_hash = _digest(protected_path)
    development_record = {
        "schema_version": "set_c_development_attestation.v1", "release_schema_version": "set_c_development_projection.v1",
        "source_csv_sha256": source_digest, "policy_version": "set_c_deterministic_allocation_policy.v1",
        "feature_schema_version": "set_c.v1", "target_version": "ignition_t_plus_1.v1",
        "feature_names": list(CANONICAL_FEATURE_NAMES), "roles": ["train", "validation", "calibration"],
        "role_files": {role: {"sha256": "3" * 64, **{key: value for key, value in role_counts[role].items() if key != "components"}}
                       for role in ("train", "validation", "calibration")},
        "synthetic": True, "final_test_used": False, "source_provenance": provenance,
        "publication_receipt": {"schema_version": "set_c_pair_publication.v1", "state": "complete",
                                 "protected_attestation_sha256": protected_hash,
                                 "source_csv_sha256": source_digest, "policy_version": "set_c_deterministic_allocation_policy.v1",
                                 "source_revision": provenance["source_revision"]["revision"]},
    }
    development_path = development / "attestation.json"
    development_path.write_text(json.dumps(development_record), encoding="utf-8")

    config = {
        "schema_version": "phase14_setc_final_test_config.v1",
        "protocol": "phase14a_calibration_final_test_protocol.v1", "execution_enabled": enabled,
        "allow_synthetic": True, "permitted_modes": ["primary", "verification"],
        "environment_status": "complete_locked",
        "environment": _synthetic_environment(),
        "reporting": {"metrics": ["average_precision", "roc_auc", "brier_score", "log_loss"],
                      "reliability_bins": 10, "classification_threshold": None,
                      "verification_rel_tol": 1e-10, "verification_abs_tol": 1e-12},
        "identities": {"model_sha256": model_hash, "final_rows_sha256": row_hash,
                       "protected_attestation_sha256": protected_hash,
                       "development_attestation_sha256": _digest(development_path)},
        "paths": {"model": str(model_path), "final_rows": str(rows_path),
                  "protected_attestation": str(protected_path),
                  "development_attestation": str(development_path),
                  "primary_output_root": str(output_root / "primary"),
                  "verification_output_root": str(output_root / "verification")},
    }
    Path(config["paths"]["primary_output_root"]).mkdir()
    Path(config["paths"]["verification_output_root"]).mkdir()
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    config_hash = _digest(config_path)
    lock = _complete_lock(config, config_hash)
    lock_path.write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    execution = {"source_revision": "b" * 40, "source_clean": True,
                 "configuration_sha256": config_hash, "environment": config["environment"],
                 "run_id": "synthetic-primary", "argv": ["synthetic-evaluator"]}
    authorization = {"primary_approval_reference": "synthetic primary", "verification_approval_reference": "synthetic verification",
                     "pretest_lock_sha256": _digest(lock_path)}
    primary_report = output_root / "primary" / _digest(lock_path)[:12] / "primary" / "synthetic-primary" / "final_test_report.json"
    return config_path, config_hash, lock_path, _digest(lock_path), execution, authorization, primary_report


def _run(fixture, mode="primary", run_id="synthetic-primary"):
    config_path, config_hash, lock_path, lock_hash, execution, authorization, _ = fixture
    execution = {**execution, "run_id": run_id}
    return evaluate_final_test(config_path, expected_config_sha256=config_hash, mode=mode,
                               authorization=authorization, execution_context=execution,
                               lock_path=lock_path, approved_lock_sha256=lock_hash, synthetic=True)


def test_default_config_and_legacy_memory_api_fail_closed(tmp_path):
    config = Path(__file__).parents[1] / "config" / "phase14_setc_final_test_v1.yaml"
    assert yaml.safe_load(config.read_text(encoding="utf-8"))["execution_enabled"] is False
    with pytest.raises(PermissionError, match="Legacy"):
        evaluate_authorized_final_test(_SyntheticEstimator(), pd.DataFrame(), {"enabled": True})
    fixture = _fixture(tmp_path, enabled=False)
    with pytest.raises(PermissionError, match="disabled"):
        _run(fixture)


def test_primary_report_is_redacted_probability_only_and_receipted(tmp_path):
    result = _run(_fixture(tmp_path))
    report = result["report"]
    assert report["evaluation_label"] == "untouched final test"
    assert report["row_count"] == 4 and report["positive_count"] == 2
    assert report["classification_threshold"] is None
    assert len(report["reliability_bins"]) == 10
    assert report["reliability_bins"][1]["count"] == 2
    assert report["reliability_bins"][-1]["count"] == 1
    assert report["metrics"]["average_precision"]["status"] == "defined"
    assert report["metrics"]["log_loss"]["status"] == "defined"
    assert np.isfinite(report["metrics"]["log_loss"]["value"])
    assert not {"run_id", "cell_row", "cell_col", "leakage_component_id"}.intersection(json.dumps(report))
    assert result["receipt"]["computation_status"] == "computation_complete"
    persisted = json.loads((Path(result["output_directory"]) / "execution_receipt.json").read_text(encoding="utf-8"))
    assert persisted == result["receipt"]
    assert persisted["partial_output_inventory"] == ["final_test_report.json"]
    assert set(persisted["finalized_outputs"]) == {"final_test_report.json"}
    started_utc = pd.Timestamp(persisted["computation_started_at_utc"])
    started_local = pd.Timestamp(persisted["computation_started_at_local"])
    ended_utc = pd.Timestamp(persisted["computation_ended_at_utc"])
    ended_local = pd.Timestamp(persisted["computation_ended_at_local"])
    assert started_utc == started_local.tz_convert("UTC")
    assert ended_utc == ended_local.tz_convert("UTC")
    assert started_utc <= ended_utc


def test_reliability_bins_keep_float32_edge_neighbors_and_empty_bins():
    probabilities = np.asarray([
        np.nextafter(np.float32(0.5), np.float32(0.0)),
        np.float32(0.5),
        np.nextafter(np.float32(0.5), np.float32(1.0)),
        np.float32(1.0),
    ], dtype=np.float32)
    rows = pd.DataFrame({name: np.zeros(4, dtype=np.float32) for name in CANONICAL_FEATURE_NAMES})
    rows["slope_risk"] = probabilities
    rows[TARGET_NAME] = np.asarray([0, 1, 0, 1], dtype=np.int8)

    bins = probability_report(_SyntheticEstimator(), rows, "primary")["reliability_bins"]

    assert bins[4]["count"] == 1
    assert bins[5]["count"] == 2
    assert bins[9]["count"] == 1  # p=1 belongs to the final bin.
    assert bins[4]["upper"] == bins[5]["lower"] == 0.5
    for index in (0, 1, 2, 3, 6, 7, 8):
        assert bins[index]["count"] == 0
        assert bins[index]["mean_predicted_probability"] is None
        assert bins[index]["observed_positive_fraction"] is None
        assert bins[index]["status"] == "empty"


def test_hash_mismatch_stops_before_joblib_load(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    config_path, config_hash, lock_path, lock_hash, execution, authorization, _ = fixture
    import evaluate_rf_final_test as evaluator
    monkeypatch.setattr(evaluator.joblib, "load", lambda _: pytest.fail("untrusted model was deserialized"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["identities"]["model_sha256"] = "0" * 64
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    new_hash = _digest(config_path)
    with pytest.raises(PermissionError, match="lock"):
        evaluate_final_test(config_path, expected_config_sha256=new_hash, mode="primary",
                            authorization=authorization, execution_context={**execution, "configuration_sha256": new_hash},
                            lock_path=lock_path, approved_lock_sha256=lock_hash, synthetic=True)


def test_production_path_allowlist_is_exact_without_touching_production_paths():
    import evaluate_rf_final_test as evaluator
    repo = Path(r"C:\Users\gibgib\Thesis-Cellular-Automata")
    phase14 = repo / "Thesis_RF" / "Code" / "output" / "phase14"
    paths = {
        "model": phase14 / "phase14-setc-rf-61e8f740-r1" / "calibration" / "approved-run" / "base_rf_calibrated.joblib",
        "final_rows": repo / "SetC_Protected" / "set_c_v1" / "final_test_rows.csv",
        "protected_attestation": repo / "SetC_Protected" / "set_c_v1" / "attestation.json",
        "development_attestation": repo / "SetC_Development" / "set_c_v1" / "attestation.json",
        "primary_output_root": phase14 / "final-test",
        "verification_output_root": phase14 / "final-test",
    }
    assert evaluator._production_paths_match(paths, repo)
    paths["development_attestation"] = repo / "Elsewhere" / "attestation.json"
    assert not evaluator._production_paths_match(paths, repo)


@pytest.mark.parametrize("damage", ["receipt_schema", "bool_row_count", "missing_dev_role"])
def test_attestation_scalar_contract_rejects_inconsistent_metadata_before_rows(tmp_path, damage):
    fixture = _fixture(tmp_path)
    config_path, _, lock_path, _, _, _, _ = fixture
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protected_path = Path(config["paths"]["protected_attestation"])
    development_path = Path(config["paths"]["development_attestation"])
    protected = json.loads(protected_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if damage == "receipt_schema":
        development["publication_receipt"]["schema_version"] = "wrong"
        development_path.write_text(json.dumps(development), encoding="utf-8")
        config["identities"]["development_attestation_sha256"] = _digest(development_path)
        lock["development_attestation_sha256"] = config["identities"]["development_attestation_sha256"]
    elif damage == "bool_row_count":
        protected["role_counts"]["final_test"]["rows"] = True
        protected_path.write_text(json.dumps(protected), encoding="utf-8")
        config["identities"]["protected_attestation_sha256"] = _digest(protected_path)
        lock["protected_attestation_sha256"] = config["identities"]["protected_attestation_sha256"]
    else:
        development["role_files"].pop("calibration")
        development_path.write_text(json.dumps(development), encoding="utf-8")
        config["identities"]["development_attestation_sha256"] = _digest(development_path)
        lock["development_attestation_sha256"] = config["identities"]["development_attestation_sha256"]
    import evaluate_rf_final_test as evaluator
    paths = {key: Path(value) for key, value in config["paths"].items()}
    with pytest.raises(ValueError):
        evaluator._validate_attestations(config, lock, fixture[1], paths, True)


@pytest.mark.parametrize("location,key", [
    ("top", "configuration_sha256"), ("top", "source_revision"),
    ("top", "pretest_lock_sha256"), ("identities", "source_revision"),
    ("top", "containing_revision"), ("identities", "pretest_lock_sha256"),
    ("identities", "self_sha256"), ("identities", "lock_sha256"),
])
def test_config_rejects_circular_revision_and_lock_fields_before_input_access(tmp_path, location, key, monkeypatch):
    fixture = _fixture(tmp_path)
    config_path, _, lock_path, lock_hash, execution, authorization, _ = fixture
    import evaluate_rf_final_test as evaluator
    monkeypatch.setattr(evaluator.joblib, "load", lambda _: pytest.fail("model accessed before config rejection"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target = config if location == "top" else config["identities"]
    target[key] = "0" * 64
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration schema"):
        evaluate_final_test(config_path, expected_config_sha256=_digest(config_path), mode="primary",
                            authorization=authorization, execution_context=execution,
                            lock_path=lock_path, approved_lock_sha256=lock_hash, synthetic=True)


def test_verification_requires_separate_approval_and_matches_aggregate(tmp_path):
    fixture = _fixture(tmp_path)
    result = _run(fixture)
    _, _, _, _, execution, authorization, primary_path = fixture
    assert primary_path.read_text(encoding="utf-8") == json.dumps(result["report"], sort_keys=True, separators=(",", ":"))
    identities = yaml.safe_load(fixture[0].read_text(encoding="utf-8"))["identities"]
    primary_path = fixture[6]
    verify_context = {**execution, "run_id": "synthetic-verification", "primary_run_id": "synthetic-primary",
                      "primary_report_sha256": _digest(primary_path),
                      "final_rows_sha256": identities["final_rows_sha256"],
                      "model_sha256": identities["model_sha256"],
                      "protected_attestation_sha256": identities["protected_attestation_sha256"],
                      "development_attestation_sha256": identities["development_attestation_sha256"]}
    verify_authorization = {**authorization, "verification_approval_reference": "separate reviewer approval"}
    verification = evaluate_final_test(fixture[0], expected_config_sha256=fixture[1], mode="verification",
                                       authorization=verify_authorization, execution_context=verify_context,
                                       lock_path=fixture[2], approved_lock_sha256=fixture[3], synthetic=True)
    assert verification["receipt"]["computation_status"] == "computation_complete"
    without_approval = dict(verify_authorization)
    without_approval.pop("verification_approval_reference")
    with pytest.raises(PermissionError, match="separate"):
        evaluate_final_test(fixture[0], expected_config_sha256=fixture[1], mode="verification",
                            authorization=without_approval, execution_context={**execution, "run_id": "no-auth"},
                            lock_path=fixture[2], approved_lock_sha256=fixture[3], synthetic=True)


def test_missing_or_tampered_lock_and_no_overwrite_fail_closed(tmp_path):
    fixture = _fixture(tmp_path)
    config_path, config_hash, lock_path, lock_hash, execution, authorization, _ = fixture
    original_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    with pytest.raises(FileNotFoundError):
        evaluate_final_test(config_path, expected_config_sha256=config_hash, mode="primary",
                            authorization=authorization, execution_context=execution,
                            lock_path=tmp_path / "missing.json", approved_lock_sha256=lock_hash, synthetic=True)
    lock_path.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="hash"):
        evaluate_final_test(config_path, expected_config_sha256=config_hash, mode="primary",
                            authorization=authorization, execution_context=execution,
                            lock_path=lock_path, approved_lock_sha256=lock_hash, synthetic=True)
    lock_path.write_text(json.dumps(original_lock), encoding="utf-8")
    fresh_hash = _digest(lock_path)
    authorization["pretest_lock_sha256"] = fresh_hash
    execution = {**execution, "run_id": "no-overwrite"}
    evaluate_final_test(config_path, expected_config_sha256=config_hash, mode="primary",
                        authorization=authorization, execution_context=execution,
                        lock_path=lock_path, approved_lock_sha256=fresh_hash, synthetic=True)
    with pytest.raises(FileExistsError):
        evaluate_final_test(config_path, expected_config_sha256=config_hash, mode="primary",
                            authorization=authorization, execution_context=execution,
                            lock_path=lock_path, approved_lock_sha256=fresh_hash, synthetic=True)


def test_replay_rejects_tampered_primary_before_final_rows_or_inference(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    result = _run(fixture)
    primary_path = fixture[6]
    primary_path.write_text(json.dumps(result["report"]), encoding="utf-8")
    expected_hash = _digest(primary_path)
    primary_path.write_text("private parser detail", encoding="utf-8")
    import evaluate_rf_final_test as evaluator
    final_rows = str(yaml.safe_load(fixture[0].read_text(encoding="utf-8"))["paths"]["final_rows"])
    original_safe_path = evaluator._safe_path
    resolved = []

    def track_path(path, root, **kwargs):
        if str(path) == final_rows:
            pytest.fail("final-row path resolved before primary validation")
        resolved.append(str(path))
        return original_safe_path(path, root, **kwargs)

    monkeypatch.setattr(evaluator, "_safe_path", track_path)
    monkeypatch.setattr(evaluator.pd, "read_csv", lambda *args, **kwargs: pytest.fail("final rows opened before primary validation"))
    execution = {**fixture[4], "run_id": "tampered-replay", "primary_run_id": "synthetic-primary",
                 "primary_report_sha256": expected_hash}
    identities = yaml.safe_load(fixture[0].read_text(encoding="utf-8"))["identities"]
    execution.update({"final_rows_sha256": identities["final_rows_sha256"], "model_sha256": identities["model_sha256"],
                      "protected_attestation_sha256": identities["protected_attestation_sha256"],
                      "development_attestation_sha256": identities["development_attestation_sha256"]})
    with pytest.raises(PermissionError, match=r"^Primary report evidence is invalid \(ValueError\)$") as error:
        evaluate_final_test(fixture[0], expected_config_sha256=fixture[1], mode="verification",
                            authorization=fixture[5], execution_context=execution,
                            lock_path=fixture[2], approved_lock_sha256=fixture[3], synthetic=True)
    assert "private parser detail" not in str(error.value)
    assert error.value.__cause__ is None
    assert not (tmp_path / "outputs" / "verification" / fixture[3][:12]
                / "verification" / "tampered-replay").exists()


def _forbid_replay_access(monkeypatch, fixture, replay_output):
    import builtins
    import io
    import evaluate_rf_final_test as evaluator

    final_rows = Path(yaml.safe_load(fixture[0].read_text(encoding="utf-8"))["paths"]["final_rows"])
    counts = dict.fromkeys(("row_resolution", "row_hash", "row_open", "row_csv",
                            "model_load", "inference", "output_inspection", "output_creation"), 0)
    original_safe_path = evaluator._safe_path
    original_hash = evaluator._hash_file
    original_csv = evaluator.pd.read_csv
    original_mkdir = Path.mkdir

    def forbidden(name):
        counts[name] += 1
        pytest.fail(f"forbidden replay call before primary validation: {name}")

    def safe_path(path, root, **kwargs):
        if Path(path) == final_rows:
            forbidden("row_resolution")
        return original_safe_path(path, root, **kwargs)

    def hash_file(path):
        if Path(path) == final_rows:
            forbidden("row_hash")
        return original_hash(path)

    def guarded_open(original):
        def open_file(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                candidate = Path(path)
                if candidate == final_rows:
                    forbidden("row_open")
                if candidate == replay_output or replay_output in candidate.parents:
                    forbidden("output_creation")
            return original(path, *args, **kwargs)
        return open_file

    def read_csv(path, *args, **kwargs):
        candidate = path if isinstance(path, (str, Path)) else getattr(path, "name", "")
        if Path(candidate) == final_rows:
            forbidden("row_csv")
        return original_csv(path, *args, **kwargs)

    def mkdir(path, *args, **kwargs):
        if path == replay_output or replay_output in path.parents:
            forbidden("output_creation")
        return original_mkdir(path, *args, **kwargs)

    def guarded_inspection(original):
        def inspect(path, *args, **kwargs):
            if path == replay_output or replay_output in path.parents:
                forbidden("output_inspection")
            return original(path, *args, **kwargs)
        return inspect

    monkeypatch.setattr(evaluator, "_safe_path", safe_path)
    monkeypatch.setattr(evaluator, "_hash_file", hash_file)
    monkeypatch.setattr(Path, "open", guarded_open(Path.open))
    monkeypatch.setattr(builtins, "open", guarded_open(builtins.open))
    monkeypatch.setattr(io, "open", guarded_open(io.open))
    monkeypatch.setattr(evaluator.pd, "read_csv", read_csv)
    monkeypatch.setattr(evaluator.joblib, "load", lambda *args, **kwargs: forbidden("model_load"))
    monkeypatch.setattr(evaluator, "predict_positive_probability", lambda *args, **kwargs: forbidden("inference"))
    monkeypatch.setattr(Path, "mkdir", mkdir)
    for method in ("exists", "is_file", "is_dir", "iterdir"):
        monkeypatch.setattr(Path, method, guarded_inspection(getattr(Path, method)))
    return counts


@pytest.mark.parametrize("field_path,value", [
    pytest.param(("schema_version",), "wrong", id="report-schema"),
    pytest.param(("evaluation_scope",), "wrong", id="report-scope"),
    pytest.param(("evaluation_mode",), "verification", id="report-mode"),
    pytest.param(("evaluation_label",), "wrong", id="report-label"),
    pytest.param(("classification_threshold",), 0.5, id="threshold"),
    pytest.param(("unexpected",), "private schema detail", id="extra-report-field"),
    pytest.param(("classification_threshold",), ..., id="missing-report-field"),
    pytest.param(("row_count",), True, id="boolean-row-count"),
    pytest.param(("row_count",), 1, id="small-row-count"),
    pytest.param(("row_count",), 5, id="inconsistent-counts"),
    pytest.param(("positive_count",), 0, id="missing-positive-class"),
    pytest.param(("negative_count",), 0, id="missing-negative-class"),
    pytest.param(("prevalence",), 0.25, id="inconsistent-prevalence"),
    pytest.param(("prevalence",), float("nan"), id="nonfinite-prevalence"),
    pytest.param(("identities", "unexpected"), "wrong", id="extra-identity-field"),
    pytest.param(("identities", "rows_sha256"), ..., id="missing-identity-field"),
    *[pytest.param(("identities", key), "0" * length, id=f"binding-{key}")
      for key, length in (("rows_sha256", 64), ("model_sha256", 64),
                          ("protected_attestation_sha256", 64), ("development_attestation_sha256", 64),
                          ("configuration_sha256", 64), ("evaluation_revision", 40),
                          ("pretest_lock_sha256", 64))],
    pytest.param(("metrics", "unexpected"), {}, id="extra-metric"),
    *[pytest.param(("metrics", name, key), value, id=f"{name}-{case}")
      for name in ("average_precision", "roc_auc", "brier_score", "log_loss")
      for key, value, case in (("status", "wrong", "status"), ("value", None, "null-defined"),
                               ("value", True, "boolean"), ("value", float("nan"), "nonfinite"),
                               ("value", -0.1, "negative"), ("reason", "private metric detail", "defined-reason"),
                               ("reason", ..., "missing-field"), ("unexpected", 0, "extra-field"))],
    *[pytest.param(("metrics", name, "value"), 1.1, id=f"{name}-above-bound")
      for name in ("average_precision", "roc_auc", "brier_score")],
    pytest.param(("metrics", "average_precision"), {"value": 0.5, "status": "undefined", "reason": "non-finite metric"}, id="undefined-with-value"),
    pytest.param(("metrics", "roc_auc"), {"value": None, "status": "undefined", "reason": "wrong"}, id="undefined-reason"),
    pytest.param(("metrics", "brier_score"), {"value": None, "status": "undefined", "reason": "non-finite metric"}, id="undefined-brier"),
    pytest.param(("reliability_bins",), [], id="bin-inventory"),
    pytest.param(("reliability_bins", 0), None, id="bin-schema"),
    pytest.param(("reliability_bins", 0, "unexpected"), 0, id="extra-bin-field"),
    pytest.param(("reliability_bins", 0, "status"), ..., id="missing-bin-field"),
    *[pytest.param(("reliability_bins", index, key), value, id=case)
      for index, key, value, case in (
          (0, "index", 1, "bin-index"), (0, "index", True, "boolean-bin-index"),
          (0, "lower", 0.01, "bin-lower"), (0, "upper", 0.11, "bin-upper"),
          (0, "endpoint_rule", "wrong", "bin-endpoint-rule"),
          (0, "count", -1, "negative-bin-count"), (0, "count", True, "boolean-bin-count"),
          (1, "count", 3, "bin-count-total"), (1, "count", 0, "populated-as-empty-count"),
          (0, "status", "defined", "empty-bin-state"), (1, "status", "empty", "populated-bin-state"),
          (0, "mean_predicted_probability", 0.0, "empty-bin-mean"),
          (0, "observed_positive_fraction", 0.0, "empty-bin-observed"),
          (1, "mean_predicted_probability", None, "populated-bin-null"),
          (1, "mean_predicted_probability", float("nan"), "populated-bin-nonfinite"),
          (1, "mean_predicted_probability", -0.1, "populated-bin-negative"),
          (1, "observed_positive_fraction", 1.1, "populated-bin-above-bound"),
          (1, "observed_positive_fraction", True, "populated-bin-boolean"))],
    pytest.param(None, "parser", id="private-primary-parser"),
    pytest.param(None, "scalar", id="private-primary-scalar"),
])
def test_replay_sanitizes_invalid_primary_before_rows_model_or_output(tmp_path, monkeypatch, field_path, value):
    fixture = _fixture(tmp_path)
    primary_result = _run(fixture)
    primary_path = fixture[6]
    report = json.loads(json.dumps(primary_result["report"]))
    if field_path is not None:
        target = report
        for key in field_path[:-1]:
            target = target[key]
        if value is ...:
            del target[field_path[-1]]
        else:
            target[field_path[-1]] = value
    primary_path.write_text(json.dumps(report), encoding="utf-8")
    execution = {**fixture[4], "run_id": "invalid-primary", "primary_run_id": "synthetic-primary",
                 "primary_report_sha256": _digest(primary_path)}
    import evaluate_rf_final_test as evaluator
    replay_output = tmp_path / "outputs" / "verification" / fixture[3][:12] / "verification" / "invalid-primary"
    original_exists = Path.exists
    counts = _forbid_replay_access(monkeypatch, fixture, replay_output)
    injected = {"parser": 0, "scalar": 0}
    if field_path is None and value == "parser":
        original_load = evaluator.json.load

        def primary_parser(stream, *args, **kwargs):
            if Path(stream.name) == primary_path:
                injected["parser"] += 1
                raise json.JSONDecodeError("PRIVATE_PRIMARY_PARSER", "", 0)
            return original_load(stream, *args, **kwargs)

        monkeypatch.setattr(evaluator.json, "load", primary_parser)
    elif field_path is None and value == "scalar":
        def fail_with_private_detail(_report):
            injected["scalar"] += 1
            raise ValueError("PRIVATE_PRIMARY_SCALAR")
        monkeypatch.setattr(evaluator, "_validate_report_scalars", fail_with_private_detail)
    reason = "JSONDecodeError" if field_path is None and value == "parser" else "ValueError"
    with pytest.raises(PermissionError, match=rf"^Primary report evidence is invalid \({reason}\)$") as error:
        evaluate_final_test(fixture[0], expected_config_sha256=fixture[1], mode="verification",
                            authorization=fixture[5], execution_context=execution,
                            lock_path=fixture[2], approved_lock_sha256=fixture[3], synthetic=True)
    import traceback
    displayed = "".join(traceback.format_exception(error.type, error.value, error.tb))
    for private in ("PRIVATE_PRIMARY_PARSER", "PRIVATE_PRIMARY_SCALAR", "private schema detail", "private metric detail"):
        assert private not in str(error.value)
        assert private not in displayed
    assert "During handling of the above exception" not in displayed
    assert "The above exception was the direct cause" not in displayed
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True
    assert counts == dict.fromkeys(counts, 0)
    assert injected == {"parser": int(field_path is None and value == "parser"),
                        "scalar": int(field_path is None and value == "scalar")}
    assert not original_exists(replay_output)


def test_primary_mode_skips_replay_evidence(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    import evaluate_rf_final_test as evaluator
    calls = []

    def forbidden_primary_validation(*args, **kwargs):
        calls.append("primary_evidence")
        pytest.fail("primary mode invoked replay evidence validation")

    monkeypatch.setattr(evaluator, "_validate_primary_evidence", forbidden_primary_validation)
    assert _run(fixture)["receipt"]["computation_status"] == "computation_complete"
    assert calls == []


def test_synthetic_paths_reject_escape_before_estimator_or_final_row_access(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    config = yaml.safe_load(fixture[0].read_text(encoding="utf-8"))
    config["paths"]["final_rows"] = str(tmp_path.parent / "outside.csv")
    fixture[0].write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    import evaluate_rf_final_test as evaluator
    monkeypatch.setattr(evaluator.joblib, "load", lambda _: pytest.fail("estimator loaded for escaped paths"))
    config_hash = _digest(fixture[0])
    lock = json.loads(fixture[2].read_text(encoding="utf-8"))
    lock["configuration_sha256"] = config_hash
    fixture[2].write_text(json.dumps(lock), encoding="utf-8")
    lock_hash = _digest(fixture[2])
    authorization = {**fixture[5], "pretest_lock_sha256": lock_hash}
    execution = {**fixture[4], "configuration_sha256": config_hash, "run_id": "escaped-path"}
    with pytest.raises(PermissionError, match="outside"):
        evaluate_final_test(fixture[0], expected_config_sha256=config_hash, mode="primary",
                            authorization=authorization, execution_context=execution,
                            lock_path=fixture[2], approved_lock_sha256=lock_hash, synthetic=True)


def test_safe_path_rejects_lexical_escape_before_lstat_and_reparse_components(tmp_path, monkeypatch):
    import evaluate_rf_final_test as evaluator
    outside = tmp_path.parent / "outside-final_test_rows.csv"
    original_lstat = Path.lstat

    def deny_outside_metadata(path):
        if path == outside:
            pytest.fail("outside path metadata was inspected before lexical rejection")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", deny_outside_metadata)
    with pytest.raises(PermissionError, match="outside"):
        evaluator._safe_path(outside, tmp_path)
    monkeypatch.setattr(Path, "lstat", original_lstat)
    reparse = tmp_path / "reparse-root"
    reparse.mkdir()
    candidate = reparse / "final_test_rows.csv"
    candidate.touch()
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == reparse or original_is_symlink(path))
    with pytest.raises(PermissionError, match="reparse"):
        evaluator._safe_path(candidate, tmp_path)


def test_safe_path_confines_existing_temp_file_without_windows_realpath(tmp_path, monkeypatch):
    import evaluate_rf_final_test as evaluator
    candidate = tmp_path / "final_test_rows.csv"
    candidate.write_text("synthetic", encoding="utf-8")

    def unavailable_realpath(*_args, **_kwargs):
        raise PermissionError("synthetic Windows realpath denial")

    monkeypatch.setattr(Path, "resolve", unavailable_realpath)
    assert evaluator._safe_path(candidate, tmp_path) == candidate


def test_redacted_public_failure_receipt_does_not_retain_parser_token(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    import evaluate_rf_final_test as evaluator
    monkeypatch.setattr(evaluator.pd, "read_csv", lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("private row token: NEVER-EXPOSE")))
    with pytest.raises(RuntimeError) as error:
        _run(fixture, run_id="redacted-parser-error")
    assert "NEVER-EXPOSE" not in str(error.value)
    assert error.value.__cause__ is None
    receipt_path = (tmp_path / "outputs" / "primary" / fixture[3][:12] / "primary"
                    / "redacted-parser-error" / "execution_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["computation_status"] == "failed"
    assert "NEVER-EXPOSE" not in json.dumps(receipt)


@pytest.mark.parametrize("change", ["prevalence", "bin_mean", "bin_status", "schema", "label"])
def test_replay_comparison_checks_schema_and_every_aggregate(tmp_path, change):
    fixture = _fixture(tmp_path)
    primary = _run(fixture)["report"]
    verification = json.loads(json.dumps(primary))
    verification["schema_version"] = "phase14a_verification_metrics.v1"
    verification["evaluation_mode"] = "verification"
    if change == "prevalence":
        verification["prevalence"] += 1e-6
    elif change == "bin_mean":
        verification["reliability_bins"][1]["mean_predicted_probability"] += 1e-6
    elif change == "bin_status":
        verification["reliability_bins"][0]["status"] = "defined"
    elif change == "schema":
        verification["schema_version"] = "phase14a_final_test_report.v1"
    elif change == "label":
        verification["evaluation_label"] = "different label"
    import evaluate_rf_final_test as evaluator
    assert not evaluator._reports_match(primary, verification)


def test_replay_comparison_accepts_numerical_differences_within_protocol_tolerances(tmp_path):
    fixture = _fixture(tmp_path)
    primary = _run(fixture)["report"]
    verification = json.loads(json.dumps(primary))
    verification["schema_version"] = "phase14a_verification_metrics.v1"
    verification["evaluation_mode"] = "verification"
    verification["prevalence"] += 1e-13
    verification["metrics"]["average_precision"]["value"] += 1e-12
    verification["reliability_bins"][1]["mean_predicted_probability"] += 1e-13
    import evaluate_rf_final_test as evaluator
    assert evaluator._reports_match(primary, verification)


def _lock_field_paths():
    config = {"environment": _synthetic_environment(), "identities": {key: "1" * 64 for key in
        ("model_sha256", "final_rows_sha256", "protected_attestation_sha256", "development_attestation_sha256")}}
    lock = _complete_lock(config, "2" * 64)
    paths = []
    def visit(value, prefix=()):
        for key, item in value.items():
            paths.append((*prefix, key))
            if isinstance(item, dict): visit(item, (*prefix, key))
    visit(lock)
    return paths


@pytest.mark.parametrize("field_path", _lock_field_paths(), ids=lambda path: ".".join(path))
def test_full_lock_missing_member_denies_before_rows_model_and_output(tmp_path, monkeypatch, field_path):
    fixture = _fixture(tmp_path)
    lock = json.loads(fixture[2].read_text(encoding="utf-8"))
    target = lock
    for key in field_path[:-1]: target = target[key]
    del target[field_path[-1]]
    fixture[2].write_text(json.dumps(lock), encoding="utf-8")
    lock_hash = _digest(fixture[2])
    output = tmp_path / "outputs" / "primary" / lock_hash[:12] / "primary" / "invalid-lock"
    counts = _forbid_replay_access(monkeypatch, fixture, output)
    with pytest.raises(PermissionError, match="Pre-test lock"):
        evaluate_final_test(fixture[0], expected_config_sha256=fixture[1], mode="primary",
            authorization={**fixture[5], "pretest_lock_sha256": lock_hash},
            execution_context={**fixture[4], "run_id": "invalid-lock"},
            lock_path=fixture[2], approved_lock_sha256=lock_hash, synthetic=True)
    assert counts == dict.fromkeys(counts, 0)


@pytest.mark.parametrize("change", ["legacy", "duplicate", "extra", "boolean-label", "nonfinite-tolerance", "threshold", "calibration-method", "calibration-revision", "phase13", "evaluator", "environment"])
def test_complete_lock_rejects_schema_types_policies_and_cross_bindings(tmp_path, monkeypatch, change):
    fixture = _fixture(tmp_path)
    lock = json.loads(fixture[2].read_text(encoding="utf-8"))
    if change == "legacy": lock = {key: lock[key] for key in ("configuration_sha256", "evaluation_revision", "environment", "model_sha256", "final_rows_sha256", "protected_attestation_sha256", "development_attestation_sha256")}
    if change == "extra": lock["calibration"]["future_review"] = "private"
    if change == "boolean-label": lock["prediction_contract"]["positive_label"] = True
    if change == "nonfinite-tolerance": lock["reporting"]["verification_abs_tol"] = float("inf")
    if change == "threshold": lock["reporting"]["classification_threshold"] = 0.5
    if change == "calibration-method": lock["calibration"]["method"] = "isotonic"
    if change == "calibration-revision": lock["calibration_revision"] = True
    if change == "phase13": lock["phase13"]["manifest_sha256"] = "0" * 64
    if change == "evaluator": lock["evaluator_source_sha256"] = "0" * 64
    if change == "environment": lock["environment"]["auxiliary_versions"]["rasterio"] = None
    text = json.dumps(lock)
    if change == "duplicate": text = text[:-1] + ', "lock_id":"duplicate"}'
    fixture[2].write_text(text, encoding="utf-8")
    digest = _digest(fixture[2])
    output = tmp_path / "outputs" / "primary" / digest[:12] / "primary" / "invalid-lock"
    counts = _forbid_replay_access(monkeypatch, fixture, output)
    with pytest.raises(PermissionError, match="Pre-test lock"):
        evaluate_final_test(fixture[0], expected_config_sha256=fixture[1], mode="primary",
            authorization={**fixture[5], "pretest_lock_sha256": digest}, execution_context={**fixture[4], "run_id": "invalid-lock"},
            lock_path=fixture[2], approved_lock_sha256=digest, synthetic=True)
    assert counts == dict.fromkeys(counts, 0)


def _primary_execution_metadata(fixture, root):
    from modules import phase14_calibration as phase14
    code = root / "synthetic-checkout" / "Thesis_RF" / "Code"
    config = yaml.safe_load(fixture[0].read_text(encoding="utf-8"))
    config["environment"]["interpreter"]["path"] = str(code.parents[1] / ".venv" / "Scripts" / "python.exe")
    model = root / "synthetic-calibration" / "base_rf_calibrated.joblib"
    model.parent.mkdir()
    Path(config["paths"]["model"]).rename(model)
    config["paths"]["model"] = str(model)
    final_root = code / "output" / "phase14" / "final-test"
    final_root.mkdir(parents=True)
    config["paths"]["primary_output_root"] = str(final_root)
    config["paths"]["verification_output_root"] = str(final_root)
    fixture[0].write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    config_hash = _digest(fixture[0])
    lock = _complete_lock(config, config_hash)
    fixture[2].write_text(json.dumps(lock, sort_keys=True), encoding="utf-8")
    lock_hash = _digest(fixture[2])
    execution = {**fixture[4], "configuration_sha256": config_hash, "environment": config["environment"]}
    approvals = {**fixture[5], "primary_approval_reference": "synthetic-primary-approval",
                 "pretest_lock_sha256": lock_hash}
    initial_report = Path(config["paths"]["primary_output_root"]) / lock_hash[:12] / "primary" / "synthetic-primary" / "final_test_report.json"
    fixture = fixture[0], config_hash, fixture[2], lock_hash, execution, approvals, initial_report
    result = _run(fixture)
    evidence = code / "output" / "phase14" / "evidence" / "synthetic-primary"
    destination = code / "output" / "phase14" / "final-test" / lock_hash[:12] / "primary" / "synthetic-primary"
    evidence.mkdir(parents=True)
    report_path = destination / "final_test_report.json"
    assert report_path == initial_report
    fixture = (*fixture[:6], report_path)
    source = {"revision": "b" * 40, "clean": True, "source_files": {name: "1" * 64 for name in phase14.SOURCE_FILES},
              "configuration_sha256": config_hash, "protocol_sha256": phase14.PROTOCOL_SHA256}
    source["source_files"]["Thesis_RF/Code/config/phase14_setc_final_test_v1.yaml"] = config_hash
    source["source_files"]["docs/Phase14A_Calibration_Final_Test_Protocol_Proposal.md"] = phase14.PROTOCOL_SHA256
    source["source_files"]["Thesis_RF/Code/evaluate_rf_final_test.py"] = lock["evaluator_source_sha256"]
    arguments = ["-B", "evaluate_rf_final_test.py", "--mode", "primary", "--config",
                 str(code / "config" / "phase14_setc_final_test_v1.yaml"), "--approval-reference",
                 "synthetic-primary-approval", "--expected-revision", "b" * 40, "--run-id",
                 "synthetic-primary", "--expected-config-sha256", config_hash, "--evidence-dir", str(evidence)]
    authorization = {"schema_version": "phase14a_job_authorization.v1", "job": "final_primary",
        "approval_reference": "synthetic-primary-approval", "run_id": "synthetic-primary",
        "expected_revision": "b" * 40, "created_at_utc": "2026-10-10T00:00:00+00:00",
        "configuration_sha256": config_hash, "protocol_sha256": phase14.PROTOCOL_SHA256,
        "source_files": source["source_files"], "environment": config["environment"],
        "environment_sha256": hashlib.sha256(phase14._json_bytes(config["environment"])).hexdigest(),
        "pretest_lock": {"lock_id": "synthetic-lock", "sha256": lock_hash, "owner_approval_reference": "synthetic-lock-approval"},
        "primary_evidence": None}
    authorization_json = json.dumps(authorization)
    transport = {"schema_version": "phase14a_supervisor_handoff.v1", "job": "final_primary",
        "run_id": "synthetic-primary", "approval_reference": "synthetic-primary-approval",
        "expected_revision": "b" * 40, "authorization_json": authorization_json,
        "authorization_sha256": hashlib.sha256(authorization_json.encode()).hexdigest(),
        "nonce": "a" * 32, "supervisor_pid": 123, "executable": config["environment"]["interpreter"]["path"],
        "argv": arguments, "working_directory": str(code), "evidence_directory": str(evidence),
        "destination": str(destination), "created_at_utc": "2026-10-10T00:00:00+00:00",
        "deadline_utc": "2026-10-10T00:30:00+00:00"}
    handoff_path = evidence / "supervisor_handoff.json"
    handoff_path.write_text(json.dumps(transport), encoding="utf-8")
    from test_phase14_calibration_contract import _worker_metadata
    process, worker_exit = _worker_metadata(evidence, transport)
    binding_hash = _digest(evidence / "worker_binding.json")
    handoff = {"run_id": "synthetic-primary", "nonce": "a" * 32, "sha256": _digest(handoff_path), "supervisor_pid": 123, "child_pid": 125}
    receipt = {**result["receipt"], "source_check": source, "supervisor_handoff": handoff, "worker_binding_sha256": binding_hash,
               "working_directory": str(code), "argv": arguments,
               "computation_started_at_utc": "2026-10-10T00:00:02+00:00",
               "computation_ended_at_utc": "2026-10-10T00:00:59+00:00"}
    receipt_path = destination / "execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    completion = {"schema_version": "phase14a_supervisor_completion.v2", "job": "final_primary", "run_id": "synthetic-primary",
        "approval_reference": receipt["approval_reference"], "source_check": source, "supervisor_handoff": handoff,
        "worker_binding_sha256": binding_hash, "worker_exit": worker_exit,
        "pretest_lock_sha256": lock_hash, "child_receipt": {"status": "present", "path": str(receipt_path), "sha256": _digest(receipt_path)},
        "verified_outputs": receipt["finalized_outputs"], "actual_exit_status": 0, "stop_reason": "completed",
        "capture_complete": True, "capture_status": "complete", "capture_error": None,
        "capture_drain_timed_out": False, "capture_drain_timeout_ms": 1000, "process_ids": [124], "remaining_process_ids": [],
        "process_exited": True, "resource_samples": [], "runner_error": None, "cleanup_error": None, "finalization_error": None,
        "started_at_utc": transport["created_at_utc"], "ended_at_utc": "2026-10-10T00:01:00+00:00",
        "deadline_utc": transport["deadline_utc"], "shutdown_limitation": "best-effort; synchronous OS operations may exceed deadline", "overall_status": "PASS"}
    for filename, field in (("stdout.log", "stdout_sha256"), ("stderr.log", "stderr_sha256"), ("operator_resources.jsonl", "resource_sha256")):
        path = evidence / filename
        path.write_bytes(b"")
        completion[field] = _digest(path)
    completion_path = evidence / "supervisor_completion.json"
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    expected = {"run_id": "synthetic-primary", "report_sha256": _digest(report_path), "receipt_sha256": _digest(receipt_path), "supervisor_completion_sha256": _digest(completion_path)}
    context = {**execution, "working_directory": str(code), "source_check": source, "evaluation_mode": "verification",
        "primary_run_id": "synthetic-primary", "primary_report_sha256": expected["report_sha256"], "primary_evidence": expected}
    return fixture, context, receipt_path, completion_path

@pytest.mark.parametrize("change", [None, "receipt-status", "receipt-extra", "supervision", "exit", "capture", "process", "finalization", "missing", "binding",
    "inputs", "receipt-model", "receipt-rows", "approval", "argv", "cwd", "handoff-schema", "handoff-job",
    "handoff-run", "handoff-nonce", "handoff-pid", "handoff-destination", "handoff-argv", "handoff-cwd",
    "authorization-schema", "authorization-config", "authorization-source", "authorization-environment",
    "process-schema", "process-id", "process-executable", "process-cwd", "process-time",
    "authorization-lock-id", "capture-timeout", "capture-timeout-integer",
    "receipt-clean-integer", "completion-clean-integer", "both-clean-integer", "context-clean-integer", "worker-pid", "worker-parent", "worker-start", "worker-image", "worker-nonce", "worker-claim-missing", "worker-binding-missing", "worker-running", "worker-reused", "worker-unavailable", "receipt-v1", "completion-v1"])
@pytest.mark.parametrize("through_caller", [False, True], ids=["direct", "full-caller"])
def test_primary_execution_evidence_is_checked_inside_sanitizer_before_access(tmp_path, monkeypatch, change, through_caller):
    fixture = _fixture(tmp_path)
    fixture, context, receipt_path, completion_path = _primary_execution_metadata(fixture, tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if change == "receipt-status": receipt["computation_status"] = "failed"
    if change == "receipt-extra": receipt["private_extra"] = "PRIVATE_RECEIPT"
    if change == "supervision": completion["overall_status"] = "NEEDS EVIDENCE"
    if change == "exit": completion["actual_exit_status"] = True
    if change == "capture": completion["capture_complete"] = False
    if change == "capture-timeout": completion["capture_drain_timed_out"] = True
    if change == "capture-timeout-integer": completion["capture_drain_timed_out"] = 0
    if change in {"receipt-clean-integer", "both-clean-integer", "context-clean-integer"}:
        receipt["source_check"]["clean"] = 1
    if change in {"completion-clean-integer", "both-clean-integer", "context-clean-integer"}:
        completion["source_check"]["clean"] = 1
    if change == "context-clean-integer": context["source_check"]["clean"] = 1
    if change == "process": completion["remaining_process_ids"] = [124]
    if change == "finalization": completion["finalization_error"] = "PRIVATE_FINALIZATION"
    if change == "binding": context["primary_evidence"]["run_id"] = "other-run"
    if change == "inputs": receipt["inputs"]["model_sha256"] = "0" * 64
    if change == "receipt-model": receipt["model_sha256"] = "0" * 64
    if change == "receipt-rows": receipt["final_rows_sha256"] = "0" * 64
    if change == "approval": receipt["approval_reference"] = "other-approval"
    if change == "argv": receipt["argv"] = []
    if change == "cwd": receipt["working_directory"] = str(tmp_path)
    handoff_path = completion_path.parent / "supervisor_handoff.json"
    transport = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff_mutations = {"handoff-schema": ("schema_version", "wrong"), "handoff-job": ("job", "calibration"),
        "handoff-run": ("run_id", "other-run"), "handoff-nonce": ("nonce", "b" * 32),
        "handoff-pid": ("supervisor_pid", True), "handoff-destination": ("destination", str(tmp_path)),
        "handoff-argv": ("argv", []), "handoff-cwd": ("working_directory", str(tmp_path))}
    if change in handoff_mutations:
        key, value = handoff_mutations[change]
        transport[key] = value
    if change and change.startswith("authorization-"):
        authorization = json.loads(transport["authorization_json"])
        if change == "authorization-schema": authorization["schema_version"] = "wrong"
        if change == "authorization-config": authorization["configuration_sha256"] = "0" * 64
        if change == "authorization-lock-id":
            authorization["pretest_lock"]["lock_id"] = "other-valid-lock"
            assert authorization["pretest_lock"]["sha256"] == fixture[3]
        if change == "authorization-source": authorization["source_files"]["Thesis_RF/Code/modules/feature_pipeline.py"] = "0" * 64
        if change == "authorization-environment":
            authorization["environment"]["versions"]["numpy"] = "other"
            from modules import phase14_calibration as phase14
            authorization["environment"]["lock_sha256"] = hashlib.sha256(phase14._json_bytes(authorization["environment"]["versions"])).hexdigest()
            authorization["environment_sha256"] = hashlib.sha256(phase14._json_bytes(authorization["environment"])).hexdigest()
        transport["authorization_json"] = json.dumps(authorization)
        transport["authorization_sha256"] = hashlib.sha256(transport["authorization_json"].encode()).hexdigest()
    handoff_path.write_text(json.dumps(transport), encoding="utf-8")
    receipt["supervisor_handoff"]["sha256"] = _digest(handoff_path)
    completion["supervisor_handoff"] = receipt["supervisor_handoff"]
    process_path = completion_path.parent / "process.json"
    process_record = json.loads(process_path.read_text(encoding="utf-8"))
    process_mutations = {"process-schema": ("schema_version", "wrong"), "process-id": ("process_id", 125),
        "process-executable": ("executable", "wrong"), "process-cwd": ("working_directory", str(tmp_path)),
        "process-time": ("process_started_at_utc", "2026-10-11T00:00:00+00:00")}
    if change in process_mutations:
        key, value = process_mutations[change]
        process_record[key] = value

    binding_record = process_record["worker_binding"]
    worker_mutations = {"worker-pid": ("pid", 7193), "worker-parent": ("ppid", 7193),
        "worker-start": ("created_at_utc", "2000-01-01T00:00:00+00:00"),
        "worker-image": ("executable", str(tmp_path / "forged.exe"))}
    if change in worker_mutations:
        key, value = worker_mutations[change]
        binding_record["worker"][key] = value
    if change == "worker-nonce": binding_record["claim"]["nonce"] = "b" * 32
    if change == "worker-running": completion["worker_exit"]["same_instance_running"] = True
    if change == "worker-reused": completion["worker_exit"]["pid_reused"] = True
    if change == "worker-unavailable": completion["worker_exit"] = None
    if change == "receipt-v1": receipt["schema_version"] = "phase14a_execution_receipt.v1"
    if change == "completion-v1": completion["schema_version"] = "phase14a_supervisor_completion.v1"
    claim_path = completion_path.parent / "worker_claim.json"
    # Rebind raw hashes so new failures reach semantic validation.
    claim = json.loads(claim_path.read_text())
    claim["handoff_sha256"] = _digest(handoff_path)
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    binding_record["claim"]["handoff_sha256"] = claim["handoff_sha256"]
    binding_record["claim_sha256"] = _digest(claim_path)
    binding_path = completion_path.parent / "worker_binding.json"
    binding_path.write_text(json.dumps(binding_record), encoding="utf-8")
    binding_hash = _digest(binding_path)
    receipt["worker_binding_sha256"] = completion["worker_binding_sha256"] = binding_hash
    if completion["worker_exit"] is not None:
        completion["worker_exit"]["worker_binding_sha256"] = binding_hash
    if change == "worker-claim-missing": claim_path.unlink()
    if change == "worker-binding-missing": binding_path.unlink()
    process_path.write_text(json.dumps(process_record), encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    context["primary_evidence"]["receipt_sha256"] = _digest(receipt_path)
    completion["child_receipt"]["sha256"] = _digest(receipt_path)
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    context["primary_evidence"]["supervisor_completion_sha256"] = _digest(completion_path)
    if change == "missing": completion_path.unlink()  # synthetic fixture mutation only
    import evaluate_rf_final_test as evaluator
    context["run_id"] = "execution-replay"
    config = yaml.safe_load(fixture[0].read_text(encoding="utf-8"))
    output = Path(config["paths"]["verification_output_root"]) / fixture[3][:12] / "verification" / context["run_id"]
    lock = json.loads(fixture[2].read_text(encoding="utf-8"))

    def invoke():
        if through_caller:
            return evaluator.evaluate_final_test(fixture[0], expected_config_sha256=fixture[1], mode="verification",
                authorization=fixture[5], execution_context=context, lock_path=fixture[2],
                approved_lock_sha256=fixture[3], synthetic=True)
        return evaluator._validate_primary_evidence(fixture[6], context, fixture[1], fixture[3], lock)

    if change is None and through_caller:
        result = invoke()
        assert result["report"]["evaluation_mode"] == "verification"
        assert evaluator._reports_match(json.loads(fixture[6].read_text(encoding="utf-8")), result["report"])
        assert result["receipt"]["computation_status"] == "computation_complete"
        return
    counts = _forbid_replay_access(monkeypatch, fixture, output)
    if change is None:
        assert invoke()["evaluation_mode"] == "primary"
    else:
        with pytest.raises(PermissionError, match="Primary report evidence is invalid") as error:
            invoke()
        import traceback
        displayed = "".join(traceback.format_exception(error.type, error.value, error.tb))
        assert "PRIVATE_RECEIPT" not in displayed and "PRIVATE_FINALIZATION" not in displayed
        assert "During handling of the above exception" not in displayed
        assert "The above exception was the direct cause" not in displayed
        assert error.value.__cause__ is None and error.value.__suppress_context__ is True
    assert counts == dict.fromkeys(counts, 0)


def test_final_cli_denies_private_context_before_any_numerical_access(monkeypatch, capsys):
    import evaluate_rf_final_test as evaluator
    counts = dict.fromkeys(("model", "rows", "inference", "output"), 0)
    def forbidden(name):
        counts[name] += 1
        pytest.fail("forbidden operation before final CLI context validation")
    monkeypatch.setattr(evaluator.joblib, "load", lambda *a, **k: forbidden("model"))
    monkeypatch.setattr(evaluator.pd, "read_csv", lambda *a, **k: forbidden("rows"))
    monkeypatch.setattr(evaluator, "predict_positive_probability", lambda *a, **k: forbidden("inference"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: forbidden("output"))
    with pytest.raises(PermissionError) as error:
        evaluator.main(["--mode", "primary", "--config", "PRIVATE_CLI"])
    assert counts == dict.fromkeys(counts, 0)
    assert "PRIVATE_CLI" not in str(error.value) + capsys.readouterr().err
    assert error.value.__cause__ is None and error.value.__suppress_context__ is True
