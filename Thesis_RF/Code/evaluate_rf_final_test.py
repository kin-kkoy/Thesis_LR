"""Default-deny, probability-only Phase 14 final-test evaluator."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

from modules import phase14_calibration as phase14

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from modules.feature_pipeline import (
    AUTHORITATIVE_CSV_COLUMNS,
    CANONICAL_FEATURE_NAMES,
    FINAL_TEST_ROLE,
    POSITIVE_LABEL,
    SET_C_SCHEMA_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
    predict_positive_probability,
    validate_feature_values,
    validate_model_feature_schema,
)
from modules.set_c_split_release import (
    APPROVED_POLICY_VERSION,
    ATTESTATION_SCHEMA_VERSION,
    FINAL_TEST_SCHEMA_VERSION,
    MAPPING_SCHEMA_VERSION,
    PAIR_PUBLICATION_SCHEMA_VERSION,
    PROTECTED_ATTESTATION_SCHEMA_VERSION,
    PROTECTED_SCHEMA_VERSION,
    RELEASE_SCHEMA_VERSION,
    _validate_bound_provenance,
)

PROTOCOL_ID = "phase14a_calibration_final_test_protocol.v1"
REPORT_SCHEMA = "phase14a_final_test_report.v1"
VERIFICATION_SCHEMA = "phase14a_verification_metrics.v1"
RECEIPT_SCHEMA = "phase14a_execution_receipt.v2"
BIN_EDGES = tuple(np.linspace(0.0, 1.0, 11))
CSV_COLUMNS = (*AUTHORITATIVE_CSV_COLUMNS, "leakage_component_id")
METADATA_FIELDS = {
    "protected": {
        "schema_version", "package_schema_version", "mapping_schema_version",
        "final_test_schema_version", "policy_version", "feature_schema_version",
        "target_version", "source_csv_sha256", "split_manifest_sha256",
        "final_test_rows_sha256", "source_rows", "component_count",
        "final_test_rows", "role_counts", "source_provenance", "synthetic",
        "publication_state",
    },
    "development": {
        "schema_version", "release_schema_version", "source_csv_sha256",
        "policy_version", "feature_schema_version", "target_version",
        "feature_names", "roles", "role_files", "synthetic", "final_test_used",
        "source_provenance", "publication_receipt",
    },
}
CONFIG_IDENTITIES = {
    "model_sha256", "final_rows_sha256", "protected_attestation_sha256",
    "development_attestation_sha256",
}
FORBIDDEN_CONFIG_IDENTITIES = {
    "configuration_sha256", "config_sha256", "self_sha256",
    "pretest_lock_sha256", "source_revision", "evaluation_revision",
    "containing_revision", "revision", "commit_revision", "source_commit",
    "evaluation_commit", "containing_commit", "self_hash", "own_sha256",
    "lock_sha256", "pretest_lock_hash",
}


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe_path(path: Path, root: Path, *, must_exist: bool = True) -> Path:
    """Check lexical containment before inspecting components, then reject reparse points."""
    root = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(path if path.is_absolute() else root / path))
    if not candidate.is_relative_to(root):
        raise PermissionError("Evaluator path is outside its approved root")
    probe = candidate
    while True:
        try:
            if probe.is_symlink() or getattr(probe.lstat(), "st_file_attributes", 0) & 0x400:
                raise PermissionError("Evaluator path contains a reparse point")
        except PermissionError:
            raise
        except FileNotFoundError:
            pass
        except OSError:
            raise PermissionError("Evaluator path cannot be inspected") from None
        if probe == probe.parent:
            break
        probe = probe.parent
    if must_exist and not candidate.exists():
        raise FileNotFoundError("Approved evaluator path is missing")
    # The absolute lexical check plus component-by-component reparse rejection
    # establishes confinement without relying on Windows handle-based realpath.
    return candidate


def _validate_paths(config_path: Path, config: dict, lock_path: Path, synthetic: bool,
                    keys: set[str] | None = None) -> dict[str, Path]:
    paths = config.get("paths")
    required = {"model", "final_rows", "protected_attestation", "development_attestation",
                "primary_output_root", "verification_output_root"}
    if not isinstance(paths, dict) or set(paths) != required:
        raise ValueError("Final-test path schema is invalid")
    _validate_narrow_path_names(paths, set(paths))
    if synthetic:
        import tempfile
        root = Path(tempfile.gettempdir()).resolve(strict=True)
        resolved = {"config": _safe_path(config_path, root), "lock": _safe_path(lock_path, root), "root": root}
        for key in keys or set():
            if paths.get(key) is None:
                raise PermissionError("Synthetic evaluator path is unresolved")
            resolved[key] = _safe_path(Path(paths[key]), root)
        return resolved
    expected_config = Path(__file__).resolve().parent / "config" / "phase14_setc_final_test_v1.yaml"
    if _safe_path(config_path, expected_config.parent) != expected_config:
        raise PermissionError("Production configuration path is not approved")
    repo = Path(__file__).resolve().parents[2]
    model_lock_root = repo / "Thesis_RF" / "Code" / "output" / "phase14" / "model-lock"
    configured = {key: Path(value) for key, value in paths.items()}
    if not _production_paths_match(configured, repo):
        raise PermissionError("Production evaluator paths are outside the approved Phase 14 structure")
    lock = _safe_path(lock_path, model_lock_root)
    if (lock.parent != model_lock_root or lock.suffix.casefold() != ".json"
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", lock.stem) is None):
        raise PermissionError("Production pre-test lock path is not approved")
    resolved = {"config": expected_config, "lock": lock, "root": repo}
    if not keys:
        return resolved
    _validate_narrow_path_names(paths, keys)
    values = {key: Path(paths[key]) for key in keys}
    if not _production_paths_match(values, repo):
        raise PermissionError("Production evaluator paths are outside the approved Phase 14 structure")
    resolved.update({key: _safe_path(value, repo) for key, value in values.items()})
    return resolved


def _production_paths_match(paths: dict[str, Path], repo: Path) -> bool:
    phase14 = repo / "Thesis_RF" / "Code" / "output" / "phase14"
    model_root = phase14 / "phase14-setc-rf-61e8f740-r1" / "calibration"
    model = paths["model"]
    try:
        model_parts = model.relative_to(model_root).parts
    except ValueError:
        return False
    return (paths["final_rows"] == repo / "SetC_Protected" / "set_c_v1" / "final_test_rows.csv"
            and paths["protected_attestation"] == repo / "SetC_Protected" / "set_c_v1" / "attestation.json"
            and paths["development_attestation"] == repo / "SetC_Development" / "set_c_v1" / "attestation.json"
            and len(model_parts) == 2
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", model_parts[0]) is not None
            and model_parts[1] == "base_rf_calibrated.joblib"
            and paths["primary_output_root"] == phase14 / "final-test"
            and paths["verification_output_root"] == phase14 / "final-test")


def _validate_narrow_path_names(paths: dict, keys: set[str]) -> None:
    forbidden = {"canonical.csv", "split_manifest.csv", "mapping.csv", "complement.csv",
                 "train.csv", "validation.csv", "calibration.csv", "staging"}
    for key in keys:
        value = paths.get(key)
        if value is None:
            raise PermissionError("Evaluator path remains unresolved")
        parts = Path(value).parts
        if any(part.casefold() in forbidden for part in parts) or any(
                any(token in part.casefold() for token in ("mapping", "complement", "split_manifest", "staging"))
                for part in parts):
            raise PermissionError("Configured path names an excluded or staging artifact")
    if (Path(paths["final_rows"]).name != "final_test_rows.csv"
            or Path(paths["protected_attestation"]).name != "attestation.json"
            or Path(paths["development_attestation"]).name != "attestation.json"
            or Path(paths["model"]).name != "base_rf_calibrated.joblib"):
        raise PermissionError("Configured input filename is outside the narrow evaluator contract")


def _load_config(path: Path, expected_hash: str) -> tuple[dict, str]:
    try:
        if not _is_hash(expected_hash) or _hash_file(path) != expected_hash:
            raise PermissionError("Final-test configuration identity check failed")
        import yaml
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except Exception as exc:
        raise PermissionError(f"Final-test configuration could not be validated ({type(exc).__name__})") from None
    identities = config.get("identities") if isinstance(config, dict) else None
    if (not isinstance(config, dict)
            or config.get("schema_version") != "phase14_setc_final_test_config.v1"
            or config.get("protocol") != PROTOCOL_ID
            or FORBIDDEN_CONFIG_IDENTITIES.intersection(config)
            or not isinstance(identities, dict)
            or set(identities) != CONFIG_IDENTITIES
            or FORBIDDEN_CONFIG_IDENTITIES.intersection(identities)
            or config.get("permitted_modes") != ["primary", "verification"]
            or config.get("allow_synthetic") not in (True, False)
            or config.get("reporting") != {"metrics": ["average_precision", "roc_auc", "brier_score", "log_loss"],
                "reliability_bins": 10, "classification_threshold": None,
                "verification_rel_tol": 1e-10, "verification_abs_tol": 1e-12}):
        raise ValueError("Final-test configuration schema is invalid")
    return config, expected_hash


def _validate_gate(config: dict, config_hash: str, mode: str, authorization: dict,
                   execution_context: dict, lock_path: Path, approved_lock_hash: str,
                   paths: dict[str, Path]) -> tuple[dict, Path]:
    """All mode, config, source, environment and lock checks precede final-row access."""
    if config.get("execution_enabled") is not True:
        raise PermissionError("Final-test execution is disabled")
    if mode not in {"primary", "verification"} or mode not in config.get("permitted_modes", []):
        raise PermissionError("Final-test mode is not separately authorized")
    approval_key = f"{mode}_approval_reference"
    if not isinstance(authorization.get(approval_key), str) or not authorization[approval_key].strip():
        raise PermissionError("A separate procedural owner authorization is required")
    if authorization.get("pretest_lock_sha256") != approved_lock_hash or not _is_hash(approved_lock_hash):
        raise PermissionError("Externally approved pre-test lock identity is required")
    if not isinstance(execution_context, dict) or execution_context.get("source_clean") is not True:
        raise PermissionError("An independently reviewed clean evaluation source is required")
    revision = execution_context.get("source_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise PermissionError("A full evaluation revision is required")
    if execution_context.get("configuration_sha256") != config_hash:
        raise PermissionError("Runtime configuration identity does not match")
    if config.get("environment_status") != "complete_locked" or config.get("environment") != execution_context.get("environment"):
        raise PermissionError("The owner-reviewed runtime environment lock is incomplete or mismatched")
    lock_path = paths["lock"]
    try:
        lock_digest = _hash_file(lock_path)
    except OSError as exc:
        raise PermissionError(f"Pre-test lock could not be validated ({type(exc).__name__})") from None
    if lock_digest != approved_lock_hash:
        raise PermissionError("Pre-test lock hash mismatch")
    try:
        with lock_path.open("r", encoding="utf-8") as stream:
            lock = json.load(stream, object_pairs_hook=phase14._strict_pairs,
                             parse_constant=phase14._reject_json_constant)
    except Exception as exc:
        raise PermissionError(f"Pre-test lock could not be validated ({type(exc).__name__})") from None
    _validate_full_lock(lock, config, config_hash, execution_context)
    if lock.get("environment") != execution_context["environment"]:
        raise PermissionError("Pre-test lock environment does not match")
    if any(not _is_hash(lock.get(key)) for key in (
            "model_sha256", "final_rows_sha256", "protected_attestation_sha256",
            "development_attestation_sha256")):
        raise PermissionError("Pre-test lock contains an unresolved input identity")
    run_id = execution_context.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
        raise PermissionError("A safe unique run ID is required")
    output_key = "primary_output_root" if mode == "primary" else "verification_output_root"
    output_root = _safe_path(Path(config["paths"][output_key]), paths["root"], must_exist=True)
    paths[output_key] = output_root
    paths["primary_output_root"] = _safe_path(Path(config["paths"]["primary_output_root"]), paths["root"], must_exist=True)
    output = _safe_path(output_root / approved_lock_hash[:12] / mode / run_id,
                        paths["root"], must_exist=False)
    if mode == "verification":
        primary_run_id = execution_context.get("primary_run_id")
        if not isinstance(primary_run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", primary_run_id):
            raise PermissionError("Externally identified primary run is required")
        primary_report = paths["primary_output_root"] / approved_lock_hash[:12] / "primary" / primary_run_id / "final_test_report.json"
        paths["primary_report"] = _safe_path(primary_report, paths["root"])
    if not output.is_relative_to(output_root):
        raise PermissionError("Run destination is outside its approved output root")
    return lock, output


def _validate_full_lock(lock: dict, config: dict, config_hash: str, context: dict) -> None:
    try:
        phase14._exact_object(lock, {"schema_version", "lock_id", "created_at_utc", "configuration_sha256",
            "evaluation_revision", "environment", "model_sha256", "final_rows_sha256",
            "protected_attestation_sha256", "development_attestation_sha256", "protocol",
            "calibration_revision", "evaluator_source_sha256", "phase13", "calibration", "prediction_contract", "reporting"})
        phase14._safe_identifier(lock["lock_id"])
        phase14._aware_timestamp(lock["created_at_utc"])
        phase14._validate_environment_envelope(lock["environment"])
        expected_phase13 = {"run_id": phase14.PHASE13_RUN_ID, "source_revision": phase14.PHASE13_REVISION,
            "protocol_sha256": phase14.PHASE13_PROTOCOL_SHA256, "configuration_sha256": phase14.PHASE13_CONFIG_SHA256,
            "manifest_sha256": phase14.PHASE13_MANIFEST_SHA256, "base_model_sha256": phase14.BASE_MODEL_SHA256,
            "development_attestation_sha256": phase14.DEVELOPMENT_ATTESTATION_SHA256,
            "source_csv_sha256": phase14.SOURCE_CSV_SHA256, "source_manifest_sha256": phase14.SOURCE_MANIFEST_SHA256}
        expected_prediction = {"feature_schema_version": SET_C_SCHEMA_VERSION, "feature_names": list(CANONICAL_FEATURE_NAMES),
            "feature_dtype": "float32", "target_name": TARGET_NAME, "target_version": TARGET_VERSION,
            "target_dtype": "int8", "required_labels": [0, 1], "positive_label": POSITIVE_LABEL}
        expected_reporting = {"metrics": ["average_precision", "roc_auc", "brier_score", "log_loss"],
            "brier_definition": "mean_squared_probability_error", "log_loss_labels": [0, 1],
            "bin_edges": list(phase14.BIN_EDGES), "endpoint_rule": "exact_internal_edge_starts_next_bin; p1_in_last_bin",
            "empty_bin_statistics": None, "classification_threshold": None,
            "verification_rel_tol": 1e-10, "verification_abs_tol": 1e-12}
        calibration = phase14._exact_object(lock["calibration"], {"run_id", "approval_reference", "method",
            "method_decision_reference", "configuration_sha256", "source_sha256", "calibration_csv_sha256",
            "manifest_sha256", "report_sha256", "receipt_sha256", "supervisor_completion_sha256",
            "package_review_sha256", "forest_fingerprint_sha256"})
        phase14._safe_identifier(calibration["run_id"])
        phase14._safe_identifier(calibration["approval_reference"])
        if (lock["schema_version"] != "phase14a_pretest_lock.v1"
                or lock["configuration_sha256"] != config_hash
                or lock["evaluation_revision"] != context["source_revision"]
                or lock["environment"] != context["environment"] or lock["environment"] != config["environment"]
                or lock["protocol"] != {"id": PROTOCOL_ID, "sha256": phase14.PROTOCOL_SHA256}
                or _json_bytes(lock["phase13"]) != _json_bytes(expected_phase13)
                or _json_bytes(lock["prediction_contract"]) != _json_bytes(expected_prediction)
                or _json_bytes(lock["reporting"]) != _json_bytes(expected_reporting)
                or calibration["method"] != "sigmoid" or calibration["method_decision_reference"] != phase14.DECISION_REFERENCE
                or calibration["calibration_csv_sha256"] != phase14.CALIBRATION_CSV_SHA256
                or re.fullmatch(r"[0-9a-f]{40}", lock["calibration_revision"] or "") is None
                or any(not _is_hash(calibration[key]) for key in calibration if key.endswith("sha256"))
                or any(not _is_hash(lock[key]) for key in (*CONFIG_IDENTITIES, "evaluator_source_sha256"))
                or any(lock[key] != config["identities"][key] for key in CONFIG_IDENTITIES)):
            raise ValueError("Full lock identity or policy mismatch")
        source = context.get("source_check")
        if source is not None:
            if (lock["evaluator_source_sha256"] != source["source_files"]["Thesis_RF/Code/evaluate_rf_final_test.py"]
                    or Path(config["paths"]["model"]).parent.name != calibration["run_id"]):
                raise ValueError("Lock does not bind observed evaluator and accepted calibration run")
        elif lock["evaluator_source_sha256"] != _hash_file(Path(__file__)):
            raise ValueError("Synthetic lock evaluator identity mismatch")
    except Exception as exc:
        raise PermissionError(f"Pre-test lock is invalid ({type(exc).__name__})") from None


def _validate_primary_execution_evidence(path: Path, primary: dict, context: dict,
                                       config_hash: str, lock_hash: str, lock_id: str) -> None:
    expected = context.get("primary_evidence")
    phase14._exact_object(expected, {"run_id", "report_sha256", "receipt_sha256", "supervisor_completion_sha256"})
    if expected["run_id"] != context["primary_run_id"] or expected["report_sha256"] != context["primary_report_sha256"]:
        raise ValueError("Primary authorization bindings differ")
    receipt_path = _safe_path(path.parent / "execution_receipt.json", path.parent)
    evidence_root = Path(context["working_directory"]) / "output" / "phase14" / "evidence" / expected["run_id"]
    completion_path = _safe_path(evidence_root / "supervisor_completion.json", evidence_root)
    if (_hash_file(receipt_path) != expected["receipt_sha256"]
            or _hash_file(completion_path) != expected["supervisor_completion_sha256"]):
        raise ValueError("Primary execution evidence identity mismatch")
    receipt = phase14._strict_json(receipt_path.read_text(encoding="utf-8"))
    completion = phase14._strict_json(completion_path.read_text(encoding="utf-8"))
    phase14._exact_object(completion, {"schema_version", "job", "run_id", "approval_reference", "source_check",
        "supervisor_handoff", "pretest_lock_sha256", "child_receipt", "verified_outputs", "actual_exit_status", "stop_reason",
        "capture_complete", "capture_status", "capture_error", "stdout_sha256", "stderr_sha256", "resource_sha256",
        "capture_drain_timed_out", "capture_drain_timeout_ms", "process_ids", "remaining_process_ids", "process_exited",
        "resource_samples", "runner_error", "cleanup_error", "finalization_error", "worker_binding_sha256", "worker_exit", "started_at_utc", "ended_at_utc",
        "deadline_utc", "shutdown_limitation", "overall_status"})
    handoff = phase14._exact_object(receipt.get("supervisor_handoff"), {"run_id", "nonce", "sha256", "supervisor_pid", "child_pid"})
    if (handoff["run_id"] != expected["run_id"] or re.fullmatch(r"[0-9a-f]{32}", handoff["nonce"] or "") is None
            or not _is_hash(handoff["sha256"]) or any(type(handoff[k]) is not int or handoff[k] <= 0 for k in ("supervisor_pid", "child_pid"))):
        raise ValueError("Primary handoff evidence is invalid")
    handoff_path = _safe_path(evidence_root / "supervisor_handoff.json", evidence_root)
    if _hash_file(handoff_path) != handoff["sha256"]:
        raise ValueError("Primary handoff digest mismatch")
    transport = phase14._exact_object(phase14._strict_json(handoff_path.read_text(encoding="utf-8")),
                                      phase14.HANDOFF_FIELDS)
    authorization = phase14._validate_authorization(
        transport["authorization_json"], transport["authorization_sha256"], "final_primary",
        receipt["approval_reference"], context["source_revision"], expected["run_id"])
    code = Path(context["working_directory"])
    executable = str(code.parents[1] / ".venv" / "Scripts" / "python.exe")
    arguments = ["-B", "evaluate_rf_final_test.py", "--mode", "primary", "--config",
                 str(code / "config" / "phase14_setc_final_test_v1.yaml"),
                 "--approval-reference", receipt["approval_reference"], "--expected-revision",
                 context["source_revision"], "--run-id", expected["run_id"],
                 "--expected-config-sha256", config_hash, "--evidence-dir", str(evidence_root)]
    destination = code / "output" / "phase14" / "final-test" / lock_hash[:12] / "primary" / expected["run_id"]
    expected_source = {"revision": authorization["expected_revision"], "clean": True,
                       "source_files": authorization["source_files"],
                       "configuration_sha256": authorization["configuration_sha256"],
                       "protocol_sha256": authorization["protocol_sha256"]}
    start = phase14._aware_timestamp(transport["created_at_utc"])
    deadline = phase14._aware_timestamp(transport["deadline_utc"])
    process = phase14._strict_json(_safe_path(evidence_root / "process.json", evidence_root).read_text(encoding="utf-8"))
    claim_path = _safe_path(evidence_root / "worker_claim.json", evidence_root)
    claim = phase14._strict_json(claim_path.read_text(encoding="utf-8"))
    binding = phase14._validate_worker_binding(process, transport, claim, _hash_file(claim_path), _hash_file(handoff_path))
    process_started = phase14._aware_timestamp(process["process_started_at_utc"])
    ended = phase14._aware_timestamp(completion["ended_at_utc"])
    binding_hash = phase14._binding_hash(binding, evidence_root)
    phase14._validate_worker_exit(completion["worker_exit"], binding, ended, binding_hash)
    if (receipt.get("worker_binding_sha256") != binding_hash
            or completion["worker_binding_sha256"] != binding_hash
            or handoff["child_pid"] != binding["worker"]["pid"]
            or phase14._aware_timestamp(binding["observed_at_utc"]) > phase14._aware_timestamp(receipt["computation_started_at_utc"])):
        raise ValueError("Primary worker evidence mismatch")
    computation_start = phase14._aware_timestamp(receipt["computation_started_at_utc"])
    computation_end = phase14._aware_timestamp(receipt["computation_ended_at_utc"])
    process_ids = process["process_ids"]
    if (transport["schema_version"] != "phase14a_supervisor_handoff.v1"
            or transport["job"] != "final_primary" or transport["run_id"] != expected["run_id"]
            or transport["approval_reference"] != receipt["approval_reference"]
            or completion["approval_reference"] != receipt["approval_reference"]
            or transport["expected_revision"] != context["source_revision"]
            or authorization["configuration_sha256"] != config_hash
            or authorization["pretest_lock"]["sha256"] != lock_hash
            or authorization["pretest_lock"]["lock_id"] != lock_id
            or authorization["environment"] != context["environment"]
            or authorization["source_files"] != context["source_check"]["source_files"]
            or expected_source != context["source_check"]
            or transport["nonce"] != handoff["nonce"]
            or type(transport["supervisor_pid"]) is not int
            or transport["supervisor_pid"] != handoff["supervisor_pid"]
            or transport["executable"] != executable
            or authorization["environment"]["interpreter"]["path"] != executable
            or transport["argv"] != arguments or receipt["argv"] != arguments
            or transport["working_directory"] != str(code) or receipt["working_directory"] != str(code)
            or transport["evidence_directory"] != str(evidence_root)
            or transport["destination"] != str(destination) or path.parent != destination
            or (deadline - start).total_seconds() != 1800
            or completion["started_at_utc"] != transport["created_at_utc"]
            or completion["deadline_utc"] != transport["deadline_utc"]
            or not start <= process_started <= computation_start <= computation_end <= ended < deadline
            or process["schema_version"] != "phase14a_process.v2" or process["run_id"] != expected["run_id"]
            or type(process["process_id"]) is not int or process["process_id"] != binding["launcher"]["pid"]
            or not isinstance(process_ids, list) or not process_ids
            or any(type(pid) is not int or pid <= 0 for pid in process_ids)
            or len(set(process_ids)) != len(process_ids) or process["process_id"] not in process_ids
            or not isinstance(completion["process_ids"], list)
            or any(type(pid) is not int or pid <= 0 for pid in completion["process_ids"])
            or len(set(completion["process_ids"])) != len(completion["process_ids"])
            or not set(process_ids) <= set(completion["process_ids"]) or process["executable"] != executable
            or process["working_directory"] != str(code)
            or receipt["inputs"] != {key: primary["identities"][key] for key in
                                    ("rows_sha256", "model_sha256", "protected_attestation_sha256", "development_attestation_sha256")}
            or receipt["partial_output_inventory"] != ["final_test_report.json"]):
        raise ValueError("Primary execution context mismatch")
    if "updated_at_utc" in process:
        if not process_started <= phase14._aware_timestamp(process["updated_at_utc"]) <= ended:
            raise ValueError("Primary process update timestamp mismatch")
    if "stop_reason" in process and (not isinstance(process["stop_reason"], str) or process["stop_reason"] != "completed"):
        raise ValueError("Primary process stop state mismatch")
    for filename, field in (("stdout.log", "stdout_sha256"), ("stderr.log", "stderr_sha256"), ("operator_resources.jsonl", "resource_sha256")):
        if not _is_hash(completion.get(field)) or _hash_file(_safe_path(evidence_root / filename, evidence_root)) != completion[field]:
            raise ValueError("Primary supervisory capture evidence mismatch")
    required_receipt = {"schema_version", "run_id", "approval_reference", "argv", "working_directory",
        "source_revision", "source_clean", "source_check", "supervisor_handoff", "worker_binding_sha256", "configuration_sha256",
        "model_sha256", "final_rows_sha256", "environment", "inputs", "computation_started_at_utc",
        "computation_ended_at_utc", "computation_started_at_local", "computation_ended_at_local", "local_timezone",
        "computation_status", "failure_reason", "finalized_outputs", "partial_output_inventory"}
    phase14._exact_object(receipt, required_receipt)
    if (receipt["schema_version"] != RECEIPT_SCHEMA or receipt["run_id"] != expected["run_id"]
            or receipt["computation_status"] != "computation_complete" or receipt["failure_reason"] is not None
            or receipt["configuration_sha256"] != config_hash or receipt["environment"] != context["environment"]
            or receipt["source_revision"] != context["source_revision"] or receipt["source_clean"] is not True
            or receipt["finalized_outputs"] != {"final_test_report.json": expected["report_sha256"]}
            or receipt["model_sha256"] != primary["identities"]["model_sha256"]
            or receipt["final_rows_sha256"] != primary["identities"]["rows_sha256"]
            or receipt["source_check"]["clean"] is not True
            or receipt["source_check"] != context["source_check"]
            or completion.get("schema_version") != "phase14a_supervisor_completion.v2"
            or completion.get("job") != "final_primary" or completion.get("run_id") != expected["run_id"]
            or completion.get("overall_status") != "PASS" or type(completion.get("actual_exit_status")) is not int
            or completion["actual_exit_status"] != 0 or completion.get("stop_reason") != "completed"
            or completion.get("capture_complete") is not True or completion.get("capture_status") != "complete"
            or completion["capture_drain_timed_out"] is not False
            or completion.get("process_exited") is not True or completion.get("remaining_process_ids") != []
            or any(completion.get(key) is not None for key in ("runner_error", "cleanup_error", "capture_error", "finalization_error"))
            or completion.get("child_receipt") != {"status": "present", "path": str(receipt_path), "sha256": expected["receipt_sha256"]}
            or completion.get("verified_outputs") != receipt["finalized_outputs"]
            or completion.get("pretest_lock_sha256") != lock_hash
            or completion["source_check"]["clean"] is not True
            or completion.get("source_check") != receipt["source_check"]
            or completion.get("supervisor_handoff") != receipt["supervisor_handoff"]):
        raise ValueError("Primary computation or supervision is incomplete")


def _validate_primary_evidence(path: Path, context: dict, config_hash: str,
                               lock_hash: str, lock: dict) -> dict | None:
    if context.get("evaluation_mode") != "verification":
        return None
    expected_hash = context.get("primary_report_sha256")
    try:
        if not _is_hash(expected_hash) or not path.is_file() or _hash_file(path) != expected_hash:
            raise ValueError("primary hash mismatch")
        with path.open("r", encoding="utf-8") as stream:
            primary = json.load(stream, object_pairs_hook=phase14._strict_pairs,
                                parse_constant=phase14._reject_json_constant)
        identities = primary.get("identities") if isinstance(primary, dict) else None
        if (not isinstance(primary, dict) or primary.get("schema_version") != REPORT_SCHEMA
                or set(primary) != {"schema_version", "evaluation_scope", "evaluation_label", "evaluation_mode",
                    "row_count", "positive_count", "negative_count", "prevalence", "metrics", "reliability_bins",
                    "classification_threshold", "identities"}
                or primary.get("evaluation_scope") != FINAL_TEST_ROLE
                or primary.get("evaluation_label") != "untouched final test"
                or primary.get("evaluation_mode") != "primary"
                or primary.get("classification_threshold") is not None
                or type(primary.get("row_count")) is not int or primary["row_count"] < 2
                or type(primary.get("positive_count")) is not int or primary["positive_count"] < 1
                or type(primary.get("negative_count")) is not int or primary["negative_count"] < 1
                or primary["positive_count"] + primary["negative_count"] != primary["row_count"]
                or not _close_scalar(primary.get("prevalence"), primary["positive_count"] / primary["row_count"])
                or not isinstance(primary.get("metrics"), dict)
                or set(primary["metrics"]) != {"average_precision", "roc_auc", "brier_score", "log_loss"}
                or not isinstance(primary.get("reliability_bins"), list)
                or len(primary["reliability_bins"]) != 10
                or not isinstance(identities, dict)
                or set(identities) != {"rows_sha256", "model_sha256", "protected_attestation_sha256",
                    "development_attestation_sha256", "configuration_sha256", "evaluation_revision",
                    "pretest_lock_sha256"}
                or identities.get("rows_sha256") != lock.get("final_rows_sha256")
                or identities.get("model_sha256") != lock.get("model_sha256")
                or identities.get("protected_attestation_sha256") != lock.get("protected_attestation_sha256")
                or identities.get("development_attestation_sha256") != lock.get("development_attestation_sha256")
                or identities.get("configuration_sha256") != config_hash
                or identities.get("evaluation_revision") != context.get("source_revision")
                or identities.get("pretest_lock_sha256") != lock_hash):
            raise ValueError("Primary report schema or external binding is invalid")
        _validate_report_scalars(primary)
        if context.get("primary_evidence") is not None:
            _validate_primary_execution_evidence(path, primary, context, config_hash, lock_hash, lock["lock_id"])
        return primary
    except Exception as exc:
        raise PermissionError(f"Primary report evidence is invalid ({type(exc).__name__})") from None


def _close_scalar(value: Any, expected: float) -> bool:
    return (type(value) in (int, float) and math.isfinite(value)
            and math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12))


def _validate_report_scalars(report: dict) -> None:
    if not isinstance(report.get("reliability_bins"), list) or len(report["reliability_bins"]) != 10:
        raise ValueError("Primary report reliability-bin schema is invalid")
    metric_bounds = {"average_precision": (0.0, 1.0), "roc_auc": (0.0, 1.0),
                     "brier_score": (0.0, 1.0), "log_loss": (0.0, None)}
    for name, bounds in metric_bounds.items():
        metric = report["metrics"].get(name)
        if not isinstance(metric, dict) or set(metric) != {"value", "status", "reason"}:
            raise ValueError("Primary report metric schema is invalid")
        value = metric.get("value")
        if metric.get("status") == "defined":
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or metric.get("reason") is not None or value < bounds[0]
                    or (bounds[1] is not None and value > bounds[1])):
                raise ValueError("Primary report metric value is invalid")
        elif (metric.get("status") != "undefined" or value is not None
              or metric.get("reason") not in {"non-finite metric", "metric undefined for these labels"}
              or name == "brier_score"):
            raise ValueError("Primary report undefined metric state is invalid")
    expected_rule = "lower inclusive; upper exclusive except bin 9 includes 1.0"
    for index, item in enumerate(report["reliability_bins"]):
        if (not isinstance(item, dict)
                or set(item) != {"index", "lower", "upper", "endpoint_rule", "count",
                                 "mean_predicted_probability", "observed_positive_fraction", "status"}
                or type(item.get("index")) is not int or item["index"] != index
                or type(item.get("count")) is not int or item["count"] < 0
                or item.get("lower") != float(BIN_EDGES[index])
                or item.get("upper") != float(BIN_EDGES[index + 1])
                or item.get("endpoint_rule") != expected_rule):
            raise ValueError("Primary report reliability-bin schema is invalid")
        means = (item.get("mean_predicted_probability"), item.get("observed_positive_fraction"))
        if item["count"] == 0:
            if item.get("status") != "empty" or means != (None, None):
                raise ValueError("Primary report empty-bin values are invalid")
        elif (item.get("status") != "defined"
              or any(type(value) not in (int, float) or not math.isfinite(value)
                     or not 0.0 <= value <= 1.0 for value in means)):
            raise ValueError("Primary report populated-bin values are invalid")
    if sum(item["count"] for item in report["reliability_bins"]) != report["row_count"]:
        raise ValueError("Primary report reliability-bin counts are invalid")


def _metadata(path: Path, expected_fields: set[str]) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("Publication metadata schema is invalid")
    return value


def _validate_attestations(config: dict, lock: dict, config_hash: str,
                           paths: dict[str, Path], synthetic: bool) -> tuple[dict, dict, dict]:
    pins = config.get("identities", {})
    for name, value in pins.items():
        if not _is_hash(value):
            raise PermissionError(f"Required {name} identity remains unresolved")
    expected = {
        "model_sha256": pins.get("model_sha256"),
        "final_rows_sha256": pins.get("final_rows_sha256"),
        "protected_attestation_sha256": pins.get("protected_attestation_sha256"),
        "development_attestation_sha256": pins.get("development_attestation_sha256"),
        "configuration_sha256": config_hash,
        "environment": config.get("environment"),
    }
    if any(lock.get(key) != value for key, value in expected.items()):
        raise PermissionError("Pre-test lock input identities do not match configuration")
    protected_path, development_path = paths["protected_attestation"], paths["development_attestation"]
    if _hash_file(protected_path) != pins["protected_attestation_sha256"] or _hash_file(development_path) != pins["development_attestation_sha256"]:
        raise ValueError("Publication attestation hash mismatch")
    protected = _metadata(protected_path, METADATA_FIELDS["protected"])
    development = _metadata(development_path, METADATA_FIELDS["development"])
    counts_by_role = protected.get("role_counts")
    receipt = development.get("publication_receipt")
    provenance = protected.get("source_provenance")
    try:
        _validate_bound_provenance(provenance)
    except Exception:
        raise ValueError("Publication source provenance is invalid") from None
    roles = ("train", "validation", "calibration", FINAL_TEST_ROLE)
    if (not isinstance(counts_by_role, dict) or set(counts_by_role) != set(roles)
            or any(not isinstance(counts_by_role.get(role), dict)
                   or set(counts_by_role[role]) != {"components", "rows", "positives", "negatives"}
                   or any(type(counts_by_role[role].get(field)) is not int
                          for field in ("components", "rows", "positives", "negatives"))
                   or counts_by_role[role]["components"] < 1
                   or counts_by_role[role]["rows"] < 1
                   or counts_by_role[role]["positives"] < 0
                   or counts_by_role[role]["negatives"] < 0
                   or counts_by_role[role]["positives"] + counts_by_role[role]["negatives"] != counts_by_role[role]["rows"]
                   for role in roles)
            or type(protected.get("source_rows")) is not int or protected["source_rows"] < 1
            or type(protected.get("component_count")) is not int or protected["component_count"] < 1
            or protected["source_rows"] != sum(counts_by_role[role]["rows"] for role in roles)
            or protected["component_count"] != sum(counts_by_role[role]["components"] for role in roles)):
        raise ValueError("Protected allocation totals are invalid")
    counts = counts_by_role[FINAL_TEST_ROLE]
    dev_role_files = development.get("role_files")
    if not isinstance(dev_role_files, dict) or set(dev_role_files) != set(roles[:-1]):
        raise ValueError("Development role inventory is invalid")
    if any(not isinstance(record, dict) or set(record) != {"sha256", "rows", "positives", "negatives"}
           or not _is_hash(record.get("sha256"))
           or any(type(record.get(field)) is not int for field in ("rows", "positives", "negatives"))
           or record["rows"] < 1 or record["positives"] < 0 or record["negatives"] < 0
           or record["positives"] + record["negatives"] != record["rows"]
           or any(record[field] != counts_by_role[role][field]
                  for field in ("rows", "positives", "negatives"))
           for role, record in dev_role_files.items()):
        raise ValueError("Development scalar role totals are invalid")
    if (not isinstance(receipt, dict) or set(receipt) != {
            "schema_version", "state", "protected_attestation_sha256", "source_csv_sha256",
            "policy_version", "source_revision"}):
        raise ValueError("Paired publication receipt schema is invalid")
    source_manifest = provenance.get("source_manifest", {})
    source_revision = provenance.get("source_revision", {}).get("revision")
    if (protected.get("schema_version") != PROTECTED_ATTESTATION_SCHEMA_VERSION
            or protected.get("package_schema_version") != PROTECTED_SCHEMA_VERSION
            or protected.get("mapping_schema_version") != MAPPING_SCHEMA_VERSION
            or protected.get("final_test_schema_version") != FINAL_TEST_SCHEMA_VERSION
            or protected.get("publication_state") != "protected_only_incomplete_until_paired_release"
            or protected.get("feature_schema_version") != SET_C_SCHEMA_VERSION
            or protected.get("target_version") != TARGET_VERSION
            or development.get("schema_version") != "set_c_development_attestation.v1"
            or development.get("release_schema_version") != "set_c_development_projection.v1"
            or development.get("feature_schema_version") != SET_C_SCHEMA_VERSION
            or development.get("target_version") != TARGET_VERSION
            or development.get("feature_names") != list(CANONICAL_FEATURE_NAMES)
            or development.get("roles") != ["train", "validation", "calibration"]
            or protected.get("policy_version") != "set_c_deterministic_allocation_policy.v1"
            or development.get("policy_version") != "set_c_deterministic_allocation_policy.v1"
            or protected.get("final_test_rows_sha256") != pins["final_rows_sha256"]
            or not _is_hash(protected.get("source_csv_sha256"))
            or not _is_hash(protected.get("split_manifest_sha256"))
            or not _is_hash(protected.get("final_test_rows_sha256"))
            or protected.get("source_rows") != source_manifest.get("row_count")
            or type(protected.get("final_test_rows")) is not int or protected["final_test_rows"] < 1
            or counts.get("rows") != protected.get("final_test_rows")
            or protected.get("source_csv_sha256") != source_manifest.get("source_csv_sha256")
            or protected.get("source_csv_sha256") != development.get("source_csv_sha256")
            or protected.get("source_provenance") != development.get("source_provenance")
            or type(protected.get("synthetic")) is not bool
            or type(development.get("synthetic")) is not bool
            or development.get("schema_version") != ATTESTATION_SCHEMA_VERSION
            or development.get("release_schema_version") != RELEASE_SCHEMA_VERSION
            or development.get("final_test_used") is not False
            or receipt.get("schema_version") != PAIR_PUBLICATION_SCHEMA_VERSION
            or receipt.get("state") != "complete"
            or receipt.get("protected_attestation_sha256") != pins["protected_attestation_sha256"]
            or receipt.get("source_csv_sha256") != protected.get("source_csv_sha256")
            or receipt.get("policy_version") != protected.get("policy_version")
            or receipt.get("source_revision") != source_revision):
        raise ValueError("Published final rows lack matching scalar publication attestation")
    if (protected.get("synthetic") is not development.get("synthetic")
            or protected.get("synthetic") is True and not (synthetic and config.get("allow_synthetic"))
            or protected.get("synthetic") is not True and synthetic):
        raise PermissionError("Synthetic publication/runtime mode mismatch")
    return protected, development, {"rows_sha256": pins["final_rows_sha256"], "model_sha256": pins["model_sha256"],
                          "protected_attestation_sha256": pins["protected_attestation_sha256"],
                          "development_attestation_sha256": pins["development_attestation_sha256"]}


def _validate_inputs(config: dict, paths: dict[str, Path], inputs: dict,
                     protected: dict, mode: str) -> tuple[pd.DataFrame, object]:
    model_path, rows_path = paths["model"], paths["final_rows"]
    if not model_path.is_file() or _hash_file(model_path) != inputs["model_sha256"]:
        raise ValueError("Locked estimator hash mismatch")
    model = joblib.load(model_path)
    validate_model_feature_schema(model)
    if not rows_path.is_file() or _hash_file(rows_path) != inputs["rows_sha256"]:
        raise ValueError("Pinned final-row hash mismatch")
    features_dtype = {name: np.float64 for name in CANONICAL_FEATURE_NAMES}
    features_dtype[TARGET_NAME] = str
    frame = pd.read_csv(rows_path, dtype=features_dtype)
    if tuple(frame.columns) != CSV_COLUMNS:
        raise ValueError("Published final-row schema is invalid")
    if frame.empty or frame["feature_schema_version"].ne(SET_C_SCHEMA_VERSION).any() or frame["target_version"].ne(TARGET_VERSION).any():
        raise ValueError("Final rows are empty or use an unsupported schema version")
    raw_labels = pd.to_numeric(frame[TARGET_NAME], errors="raise").to_numpy(dtype=np.float64)
    if (not np.isfinite(raw_labels).all() or not np.equal(raw_labels, np.floor(raw_labels)).all()
            or not np.isin(raw_labels, (0, 1)).all()):
        raise ValueError("Final-test labels must contain both integer classes 0 and 1")
    labels = raw_labels.astype(np.int8)
    if set(labels) != {0, 1}:
        raise ValueError("Final-test labels must contain both integer classes 0 and 1")
    raw_features = frame.loc[:, list(CANONICAL_FEATURE_NAMES)].to_numpy(dtype=np.float64)
    validate_feature_values(raw_features)
    features = raw_features.astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Final-test features overflow float32")
    validate_feature_values(features)
    frame.loc[:, list(CANONICAL_FEATURE_NAMES)] = features
    frame[TARGET_NAME] = labels
    counts = protected["role_counts"][FINAL_TEST_ROLE]
    if (protected.get("final_test_rows") != len(frame) or counts.get("rows") != len(frame)
            or counts.get("positives") != int(labels.sum())
            or counts.get("negatives") != int(len(frame) - labels.sum())):
        raise ValueError("Final rows do not match the approved scalar row counts")
    return frame, model


def _metric(fn: Any, y_true: np.ndarray, probabilities: np.ndarray, **kwargs: Any) -> dict:
    try:
        value = float(fn(y_true, probabilities, **kwargs))
        if not math.isfinite(value):
            return {"value": None, "status": "undefined", "reason": "non-finite metric"}
        return {"value": value, "status": "defined", "reason": None}
    except (ValueError, ZeroDivisionError):
        return {"value": None, "status": "undefined", "reason": "metric undefined for these labels"}


def probability_report(model: object, rows: pd.DataFrame, mode: str) -> dict:
    labels = rows[TARGET_NAME].to_numpy(dtype=np.int8)
    if labels.size == 0 or not np.array_equal(np.unique(labels), np.array([0, 1], dtype=np.int8)):
        raise ValueError("Final-test report requires both integer classes")
    features = rows.loc[:, list(CANONICAL_FEATURE_NAMES)].astype(np.float32)
    probabilities = predict_positive_probability(model, features)
    if probabilities.ndim != 1 or probabilities.shape != labels.shape:
        raise ValueError("Prediction probabilities must match the binary label vector")
    metrics = {
        "average_precision": _metric(average_precision_score, labels, probabilities),
        "roc_auc": _metric(roc_auc_score, labels, probabilities),
        "brier_score": {"value": float(np.mean(np.square(probabilities - labels))), "status": "defined", "reason": None},
        "log_loss": _metric(log_loss, labels, probabilities, labels=[0, 1]),
    }
    bins = np.minimum(np.searchsorted(BIN_EDGES, probabilities, side="right") - 1, 9)
    reliability = []
    for index in range(10):
        mask = bins == index
        count = int(mask.sum())
        reliability.append({"index": index, "lower": float(BIN_EDGES[index]), "upper": float(BIN_EDGES[index + 1]),
                            "endpoint_rule": "lower inclusive; upper exclusive except bin 9 includes 1.0",
                            "count": count, "mean_predicted_probability": float(probabilities[mask].mean()) if count else None,
                            "observed_positive_fraction": float(labels[mask].mean()) if count else None,
                            "status": "defined" if count else "empty"})
    return {"schema_version": VERIFICATION_SCHEMA if mode == "verification" else REPORT_SCHEMA,
            "evaluation_scope": FINAL_TEST_ROLE, "evaluation_label": "untouched final test",
            "evaluation_mode": mode, "row_count": int(labels.size), "positive_count": int(labels.sum()),
            "negative_count": int(labels.size - labels.sum()), "prevalence": float(labels.mean()),
            "metrics": metrics, "reliability_bins": reliability, "classification_threshold": None}


def _receipt(context: dict, config_hash: str, inputs: dict, start: datetime, status: str,
             failure: str | None, partial: list[str], finalized: dict | None = None) -> dict:
    end = datetime.now(timezone.utc)
    local_start = start.astimezone()
    local_end = end.astimezone()
    return {"schema_version": RECEIPT_SCHEMA, "run_id": context["run_id"],
            "approval_reference": context.get("approval_reference"), "argv": context.get("argv", sys.argv),
            "working_directory": str(Path.cwd()), "source_revision": context.get("source_revision"),
            "source_clean": context.get("source_clean"), "configuration_sha256": config_hash,
            "source_check": context.get("source_check"), "supervisor_handoff": context.get("supervisor_handoff"),
            "worker_binding_sha256": context.get("worker_binding_sha256"),
            "model_sha256": inputs.get("model_sha256"), "final_rows_sha256": inputs.get("rows_sha256"),
            "environment": context.get("environment"), "inputs": dict(inputs),
            "computation_started_at_utc": start.isoformat(),
            "computation_ended_at_utc": end.isoformat(),
            "computation_started_at_local": local_start.isoformat(),
            "computation_ended_at_local": local_end.isoformat(),
            "local_timezone": str(local_start.tzinfo), "computation_status": status,
            "failure_reason": failure, "finalized_outputs": dict(finalized or {}),
            "partial_output_inventory": list(partial)}


def evaluate_final_test(config_path: str | Path, *, expected_config_sha256: str, mode: str,
                        authorization: dict, execution_context: dict, lock_path: str | Path,
                        approved_lock_sha256: str, synthetic: bool = False) -> dict:
    """Run one guarded aggregate pass; context values are supplied evidence, not proofs of supervision or human approval."""
    config_path = Path(config_path)
    if synthetic:
        import tempfile
        config_path = _safe_path(config_path, Path(tempfile.gettempdir()))
    else:
        expected_config = Path(__file__).resolve().parent / "config" / "phase14_setc_final_test_v1.yaml"
        if _safe_path(config_path, expected_config.parent) != expected_config:
            raise PermissionError("Production configuration path is not approved")
    config, config_hash = _load_config(config_path, expected_config_sha256)
    if not synthetic:
        observed = phase14._observe_child_context("final_" + mode, execution_context.get("argv", []))
        if any(execution_context.get(key) != observed[key] for key in observed):
            raise PermissionError("Evaluator context differs from independently observed evidence")
        if mode == "verification" and execution_context.get("primary_evidence") is None:
            raise PermissionError("Reviewed primary execution evidence is required")
        phase14._safe_identifier(authorization.get("lock_owner_approval_reference"))
    if synthetic and config.get("allow_synthetic") is not True:
        raise PermissionError("Synthetic execution is disabled by configuration")
    execution_context = dict(execution_context)
    paths = _validate_paths(Path(config_path), config, Path(lock_path), synthetic)
    execution_context["evaluation_mode"] = mode
    lock, output = _validate_gate(config, config_hash, mode, authorization, execution_context,
                                  Path(lock_path), approved_lock_sha256, paths)
    if not synthetic and Path(lock_path).stem != lock["lock_id"]:
        raise PermissionError("Externally approved lock identifier does not match its fixed path")
    primary = (_validate_primary_evidence(paths["primary_report"], execution_context,
                                          config_hash, approved_lock_sha256, lock)
               if mode == "verification" else None)
    output = _safe_path(output, paths["root"], must_exist=False)
    try:
        exists = output.exists()
    except OSError:
        raise PermissionError("Run destination cannot be inspected") from None
    if exists:
        raise FileExistsError("Run destination already exists")
    start = datetime.now(timezone.utc)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError("Run destination already exists") from None
    except OSError as exc:
        raise PermissionError(f"Run destination could not be created ({type(exc).__name__})") from None
    execution_context["approval_reference"] = authorization[f"{mode}_approval_reference"]
    inputs: dict = {}
    partial: list[str] = []
    stage = "input path authorization"
    try:
        paths.update(_validate_paths(config_path, config, Path(lock_path), synthetic,
                                     {"model", "final_rows", "protected_attestation", "development_attestation",
                                      "primary_output_root", "verification_output_root"}))
        stage = "publication attestations"
        protected, development, inputs = _validate_attestations(config, lock, config_hash, paths, synthetic)
        stage = "locked estimator and final rows"
        rows, model = _validate_inputs(config, paths, inputs, protected, mode)
        report = probability_report(model, rows, mode)
        report["identities"] = {**inputs, "configuration_sha256": config_hash,
                                "evaluation_revision": execution_context["source_revision"],
                                "pretest_lock_sha256": approved_lock_sha256}
        report_path = output / ("verification_metrics.json" if mode == "verification" else "final_test_report.json")
        with report_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        partial.append(report_path.name)
        status, failure = "computation_complete", None
        if mode == "verification":
            stage = "verification comparison"
            if not _reports_match(primary, report):
                status, failure = "needs_evidence", "verification aggregates differ from primary report"
        receipt = _receipt(execution_context, config_hash, inputs, start, status, failure, partial,
                           {report_path.name: _hash_file(report_path)})
        with (output / "execution_receipt.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {"report": report, "receipt": receipt, "output_directory": str(output)}
    except Exception as exc:
        partial = [name for name in ("final_test_report.json", "verification_metrics.json")
                   if (output / name).is_file()]
        if not (output / "execution_receipt.json").exists():
            receipt = _receipt(execution_context, config_hash, inputs, start, "failed",
                               f"{type(exc).__name__} at {stage}", partial)
            try:
                with (output / "execution_receipt.json").open("x", encoding="utf-8") as stream:
                    json.dump(receipt, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            except Exception:
                pass
        raise RuntimeError(f"Final-test evaluator failed ({type(exc).__name__} at {stage})") from None


def _reports_match(primary: dict, verification: dict) -> bool:
    report_fields = {"schema_version", "evaluation_scope", "evaluation_label", "evaluation_mode",
                     "row_count", "positive_count", "negative_count", "prevalence", "metrics",
                     "reliability_bins", "classification_threshold", "identities"}
    identity_fields = {"rows_sha256", "model_sha256", "protected_attestation_sha256",
                       "development_attestation_sha256", "configuration_sha256",
                       "evaluation_revision", "pretest_lock_sha256"}
    if (not isinstance(primary, dict) or not isinstance(verification, dict)
            or set(primary) != report_fields or set(verification) != report_fields
            or primary.get("schema_version") != REPORT_SCHEMA
            or primary.get("evaluation_mode") != "primary"
            or verification.get("schema_version") != VERIFICATION_SCHEMA
            or verification.get("evaluation_mode") != "verification"):
        return False
    try:
        _validate_report_scalars(primary)
        _validate_report_scalars(verification)
    except (KeyError, TypeError, ValueError):
        return False
    for report in (primary, verification):
        identities = report.get("identities")
        if (type(report.get("row_count")) is not int or report["row_count"] < 2
                or type(report.get("positive_count")) is not int or report["positive_count"] < 1
                or type(report.get("negative_count")) is not int or report["negative_count"] < 1
                or report["positive_count"] + report["negative_count"] != report["row_count"]
                or not _close_scalar(report.get("prevalence"), report["positive_count"] / report["row_count"])
                or report.get("classification_threshold") is not None
                or not isinstance(identities, dict) or set(identities) != identity_fields
                or any(not _is_hash(identities.get(key)) for key in identity_fields - {"evaluation_revision"})
                or not isinstance(identities.get("evaluation_revision"), str)
                or re.fullmatch(r"[0-9a-f]{40}", identities["evaluation_revision"]) is None):
            return False
    exact = ("row_count", "positive_count", "negative_count", "classification_threshold",
             "evaluation_scope", "evaluation_label", "identities")
    if (primary.get("evaluation_scope") != FINAL_TEST_ROLE
            or primary.get("evaluation_label") != "untouched final test"
            or verification.get("evaluation_scope") != FINAL_TEST_ROLE
            or verification.get("evaluation_label") != "untouched final test"
            or any(primary.get(field) != verification.get(field) for field in exact)):
        return False
    def compare(left: Any, right: Any) -> bool:
        if isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                return False
            return all(compare(left[key], right[key]) for key in left)
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(compare(a, b) for a, b in zip(left, right))
        if type(left) in (int, bool, str) or left is None:
            return left == right
        if (type(left) in (int, float) and type(right) in (int, float)
                and math.isfinite(left) and math.isfinite(right)):
            return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)
        return left == right
    bins_match = len(primary["reliability_bins"]) == len(verification["reliability_bins"])
    if bins_match:
        for left, right in zip(primary["reliability_bins"], verification["reliability_bins"]):
            exact_bin = ("index", "lower", "upper", "endpoint_rule", "count", "status")
            if any(left.get(field) != right.get(field) for field in exact_bin):
                bins_match = False
                break
            for field in ("mean_predicted_probability", "observed_positive_fraction"):
                left_value, right_value = left.get(field), right.get(field)
                if left_value is None or right_value is None:
                    if left_value is not right_value:
                        bins_match = False
                        break
                elif not _close_scalar(left_value, right_value):
                    bins_match = False
                    break
            if not bins_match:
                break
    return (compare(primary.get("metrics"), verification.get("metrics"))
            and _close_scalar(primary.get("prevalence"), verification.get("prevalence"))
            and bins_match)


def evaluate_authorized_final_test(model: object, protected_rows: pd.DataFrame,
                                  authorization: dict) -> dict:
    """Legacy in-memory entry is deliberately fail-closed; use guarded file contract."""
    raise PermissionError("Legacy in-memory final-test evaluation is disabled")


def main(argv: list[str] | None = None) -> int:
    try:
        arguments, context, record = phase14._child_cli(argv, None)
        locked = record["pretest_lock"]
        lock_path = phase14._CODE_ROOT / "output" / "phase14" / "model-lock" / (locked["lock_id"] + ".json")
        if record["primary_evidence"] is not None:
            evidence = record["primary_evidence"]
            context.update(primary_run_id=evidence["run_id"], primary_report_sha256=evidence["report_sha256"], primary_evidence=evidence)
        authorization = {arguments.mode + "_approval_reference": arguments.approval_reference,
                         "pretest_lock_sha256": locked["sha256"], "lock_owner_approval_reference": locked["owner_approval_reference"]}
        result = evaluate_final_test(arguments.config, expected_config_sha256=arguments.expected_config_sha256,
            mode=arguments.mode, authorization=authorization, execution_context=context,
            lock_path=lock_path, approved_lock_sha256=locked["sha256"])
        return 0 if result["receipt"]["computation_status"] == "computation_complete" else 1
    except BaseException as exc:
        raise PermissionError(f"Final-test CLI is disabled or failed ({type(exc).__name__})") from None


if __name__ == "__main__":
    raise SystemExit(main())
