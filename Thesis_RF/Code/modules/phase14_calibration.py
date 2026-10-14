"""Phase 14 frozen-model sigmoid calibration contracts.

The command-line entry point remains disabled until supervised Phase 14
execution has a separately reviewed adapter and authorization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from modules.feature_pipeline import (
    CANONICAL_FEATURE_NAMES,
    POSITIVE_LABEL,
    SET_C_SCHEMA_VERSION,
    TARGET_NAME,
    TARGET_VERSION,
    predict_positive_probability,
    validate_feature_values,
    validate_model_feature_schema,
)
from modules.set_c_split_release import load_development_calibration_release


PROTOCOL_ID = "phase14a_calibration_final_test_protocol.v1"
PROTOCOL_SHA256 = "c7791fe57672541bbcde375b7858fe6b73cbdfc35eb41975730b49937abe6efc"
DECISION_REFERENCE = "D-077/D-079"
PHASE13_RUN_ID = "phase13-setc-rf-61e8f740-r2"
PHASE13_REVISION = "61e8f74031d61c9be22a9b6408aec3ad74f4e892"
PHASE13_PROTOCOL_SHA256 = "69c39d1d97430ebb903ec2d856e6205a2c1b8ad77641798955d6ad2bdb671637"
PHASE13_CONFIG_SHA256 = "3800fe006463392f102a85ca3c491a6386b5340aeb8d2211eb31ae079a63bfc8"
PHASE13_MANIFEST_SHA256 = "b0d0d976a51d8559db73f9ee0db8365c0133c4414de5242cd328d0eb7ff96f3c"
BASE_MODEL_SHA256 = "7c3b431dcad6554d9ea80e0e6b0d13826245f8fcf3de9534ccfde78bd06b4ce1"
DEVELOPMENT_ATTESTATION_SHA256 = "4fcf286156892c9da0289e56712210e90cccdcd608641c7ee573a8db1dcea23f"
SOURCE_CSV_SHA256 = "7f0f1867445141f365687d127e449117f234348cec571990d70bc2b226ba40f6"
SOURCE_MANIFEST_SHA256 = "b700f1b9f196893f23e7190daf1bc3c68b0f569f99a977ec9e919f0de73e9e3c"
CALIBRATION_CSV_SHA256 = "e424865be9143823f5a15fce27a3eb6df74abe802be84f142505f1209e3a77dc"
CALIBRATION_ATTESTATION_SHA256 = DEVELOPMENT_ATTESTATION_SHA256
CALIBRATION_COUNTS = (2060, 153, 1907)
CALIBRATED_MODEL_FILENAME = "base_rf_calibrated.joblib"
MANIFEST_FILENAME = "calibration_manifest.json"
REPORT_FILENAME = "calibration_report.json"
RECEIPT_FILENAME = "execution_receipt.json"
BIN_EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_CODE_ROOT = Path(__file__).resolve().parents[1]
_PHASE13_ROOT = _CODE_ROOT / "output" / "phase13" / "set_c_rf_v1"
_PHASE14_CONFIG = _CODE_ROOT / "config" / "phase14_setc_calibration_v1.yaml"


def _load_phase14_config(path: Path, expected_sha256: str, *, synthetic: bool) -> tuple[dict, str]:
    """Hash-bind and validate the reviewed policy before touching any inputs."""
    path = _reject_reparse_path(Path(path))
    if not path.is_file():
        raise FileNotFoundError("Phase 14 configuration is missing")
    digest = _sha256_file(path)
    if not expected_sha256 or digest != expected_sha256:
        raise ValueError("Phase 14 configuration hash mismatch")
    import yaml
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Phase 14 configuration must be a mapping")
    calibration = config.get("calibration", {})
    inputs = config.get("inputs", {})
    expected_inputs = {
        "phase13_run_id": PHASE13_RUN_ID,
        "phase13_revision": PHASE13_REVISION,
        "phase13_protocol_sha256": PHASE13_PROTOCOL_SHA256,
        "phase13_config_sha256": PHASE13_CONFIG_SHA256,
        "phase13_manifest_sha256": PHASE13_MANIFEST_SHA256,
        "base_model_filename": "base_rf.joblib",
        "base_model_sha256": BASE_MODEL_SHA256,
        "release_attestation_sha256": CALIBRATION_ATTESTATION_SHA256,
        "source_csv_sha256": SOURCE_CSV_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "calibration_csv_sha256": CALIBRATION_CSV_SHA256,
    }
    if (
        config.get("schema_version") != "phase14_setc_calibration_config.v1"
        or config.get("protocol") != {"id": PROTOCOL_ID, "sha256": PROTOCOL_SHA256}
        or calibration.get("method") != "sigmoid"
        or calibration.get("role") != "calibration"
        or calibration.get("n_jobs") != 1
        or calibration.get("weighting") != "equal_rows"
        or calibration.get("sample_weight", "unset") is not None
        or calibration.get("rows") != CALIBRATION_COUNTS[0]
        or calibration.get("positives") != CALIBRATION_COUNTS[1]
        or calibration.get("negatives") != CALIBRATION_COUNTS[2]
        or calibration.get("feature_names") != list(CANONICAL_FEATURE_NAMES)
        or calibration.get("feature_dtype") != "float32"
        or calibration.get("target_dtype") != "int8"
        or calibration.get("target_name") != TARGET_NAME
        or calibration.get("feature_schema_version") != SET_C_SCHEMA_VERSION
        or calibration.get("target_version") != TARGET_VERSION
        or calibration.get("positive_label") != POSITIVE_LABEL
        or calibration.get("required_labels") != [0, 1]
        or calibration.get("require_nonconstant_base_probabilities") is not True
        or any(inputs.get(key) != value for key, value in expected_inputs.items())
        or config.get("reporting", {}).get("classification_threshold", "unset") is not None
        or config.get("reporting", {}).get("reliability_bins") != 10
        or config.get("resources", {}).get("supervised_stop_seconds") != 1800
        or config.get("resources", {}).get("min_available_physical_memory_bytes") != 4 * 1024**3
        or config.get("resources", {}).get("min_free_output_volume_bytes") != 2 * 1024**3
        or config.get("resources", {}).get("sampling_interval_seconds") != 1
        or config.get("paths", {}).get("calibrated_model") != CALIBRATED_MODEL_FILENAME
        or config.get("paths", {}).get("manifest") != MANIFEST_FILENAME
        or config.get("paths", {}).get("report") != REPORT_FILENAME
        or config.get("paths", {}).get("child_receipt") != RECEIPT_FILENAME
    ):
        raise ValueError("Phase 14 configuration does not match the accepted protocol contract")
    if not synthetic:
        env = config.get("environment", {})
        if config.get("execution_enabled") is not True:
            raise PermissionError("Phase 14 production execution is disabled by configuration")
        if env.get("status") != "complete_locked":
            raise PermissionError("A complete hash-bound Phase 14 environment lock is required")
        _verify_environment_lock(env)
        if Path(path) != _PHASE14_CONFIG:
            raise PermissionError("Production must use the approved Phase 14 configuration path")
    return config, digest


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def _verify_environment_lock(environment: dict, *, observed: dict | None = None) -> str:
    _validate_environment_envelope(environment)
    versions = environment.get("versions") if isinstance(environment, dict) else None
    required = {"python", "scikit_learn", "optuna", "joblib", "numpy", "pandas", "scipy", "pyyaml", "platform"}
    if not isinstance(versions, dict) or set(versions) != required or any(not isinstance(v, str) or not v for v in versions.values()):
        raise PermissionError("The Phase 14 environment lock is incomplete")
    digest = sha256(_json_bytes(versions)).hexdigest()
    if not _is_sha256(environment.get("lock_sha256")) or environment["lock_sha256"] != digest:
        raise PermissionError("The Phase 14 environment lock digest is invalid")
    fixed_baseline = {
        "python": "3.14.3", "scikit_learn": "1.8.0", "optuna": "4.8.0",
        "joblib": "1.5.3", "numpy": "2.4.4",
    }
    if any(versions.get(key) != value for key, value in fixed_baseline.items()):
        raise PermissionError("The Phase 14 environment differs from the accepted runtime baseline")
    current = _software_versions() if observed is None else observed
    if versions != current:
        raise PermissionError("The Phase 14 runtime does not match its locked environment")
    return digest


def _validate_production_context(config: dict, execution_context: dict, code_root: Path) -> Path:
    required_context = ("source_revision", "source_clean", "approval_reference", "argv", "working_directory")
    if any(key not in execution_context or execution_context[key] is None or execution_context[key] == "" for key in required_context):
        raise PermissionError("Reviewed production execution context is required")
    if execution_context["source_clean"] is not True:
        raise PermissionError("Production source revision must be independently verified clean")
    if not isinstance(execution_context["source_revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", execution_context["source_revision"]):
        raise PermissionError("Production source revision must be a full commit identity")
    if not isinstance(execution_context["argv"], list) or not execution_context["argv"] or any(not isinstance(arg, str) for arg in execution_context["argv"]):
        raise PermissionError("Production command context is invalid")
    run_id = execution_context.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id) or run_id in (".", ".."):
        raise PermissionError("A safe separately approved run ID is required")
    if not isinstance(execution_context["approval_reference"], str) or not execution_context["approval_reference"].strip():
        raise PermissionError("A procedural owner approval reference is required")
    if Path(execution_context["working_directory"]).resolve() != Path(code_root).resolve():
        raise PermissionError("Production working directory must be the approved RF code directory")
    paths = config.get("paths", {})
    if paths.get("phase13_root") != "output/phase13/set_c_rf_v1" or paths.get("calibration_release") != "../../SetC_Development/set_c_v1":
        raise PermissionError("Production input paths differ from the approved fixed paths")
    base = Path(code_root) / paths.get("output_root", "")
    expected_base = Path(code_root) / "output/phase14/phase14-setc-rf-61e8f740-r1/calibration"
    if base != expected_base:
        raise PermissionError("Production destination base differs from the approved path")
    destination = base / run_id
    if destination.exists():
        raise PermissionError("Production run destination already exists")
    return _reject_reparse_path(destination)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _normal(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _normal(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normal(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported non-deterministic estimator parameter: {type(value).__name__}")


def forest_fingerprint(estimator: object) -> str:
    """Hash estimator parameters and every fitted tree structure/value array."""
    state = getattr(estimator, "__dict__", {})
    trees = state.get("estimators_") if isinstance(state, dict) else None
    get_params = getattr(estimator, "get_params", None)
    if trees is None or not callable(get_params) or not len(trees):
        raise TypeError("Expected a fitted forest with estimators_ and get_params()")
    digest = sha256(_json_bytes(_normal(get_params(deep=False))))
    for tree_estimator in trees:
        tree = getattr(tree_estimator, "tree_", None)
        if tree is None:
            raise TypeError("Every forest member must expose fitted tree_ state")
        for name in (
            "children_left", "children_right", "feature", "threshold", "value",
            "impurity", "n_node_samples", "weighted_n_node_samples",
        ):
            array = np.ascontiguousarray(getattr(tree, name))
            digest.update(name.encode("ascii"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(_json_bytes(list(array.shape)))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_calibration_inputs(features: pd.DataFrame, labels: object) -> np.ndarray:
    if not isinstance(features, pd.DataFrame) or tuple(features.columns) != CANONICAL_FEATURE_NAMES:
        raise ValueError("Calibration features must use the canonical ordered Set C schema")
    if features.empty or any(dtype != np.dtype("float32") for dtype in features.dtypes):
        raise ValueError("Calibration features must be nonempty float32 data")
    validate_feature_values(features)
    y = np.asarray(labels)
    if y.ndim != 1 or y.shape[0] != len(features) or y.dtype != np.dtype("int8"):
        raise ValueError("Calibration labels must be aligned one-dimensional int8 values")
    if not np.isfinite(y).all() or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Calibration labels must contain both binary classes 0 and 1")
    return y


def calibrate_frozen_estimator(estimator: object, features: pd.DataFrame, labels: object) -> tuple[object, dict]:
    """Fit one sigmoid calibrator while freezing the supplied fitted forest."""
    y = _validate_calibration_inputs(features, labels)
    if not callable(getattr(estimator, "predict_proba", None)) or not hasattr(estimator, "classes_"):
        raise TypeError("Base estimator must expose classes_ and callable predict_proba()")
    if tuple(validate_model_feature_schema(estimator)) != CANONICAL_FEATURE_NAMES:
        raise ValueError("Base estimator feature schema does not match Set C")
    before = forest_fingerprint(estimator)
    base_probabilities = predict_positive_probability(estimator, features)
    if base_probabilities.size != y.size or not np.isfinite(base_probabilities).all():
        raise ValueError("Base estimator returned invalid calibration scores")
    if np.unique(base_probabilities).size < 2:
        raise ValueError("Calibration suitability requires nonconstant base probabilities")

    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
    except ImportError as exc:
        raise RuntimeError("This scikit-learn runtime lacks the accepted FrozenEstimator API") from exc

    # sklearn 1.8's FrozenEstimator path uses CV prediction plumbing, but its
    # fit is a no-op; the unchanged base predicts every row and the sigmoid
    # calibrator is fitted once on the complete supplied calibration role.
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(estimator), method="sigmoid", n_jobs=1
    )
    calibrated.fit(features, y)
    after = forest_fingerprint(estimator)
    if before != after:
        raise RuntimeError("Frozen calibration changed the accepted forest")
    validate_model_feature_schema(calibrated)
    if not callable(getattr(calibrated, "predict_proba", None)) or not hasattr(calibrated, "classes_"):
        raise TypeError("Calibrated estimator lost the generic probability interface")
    positive_index = _positive_index(calibrated.classes_)
    wrapped_forest = _find_forest(calibrated)
    embedded = forest_fingerprint(wrapped_forest)
    if embedded != before:
        raise RuntimeError("Calibrated artifact does not contain the unchanged forest")
    return calibrated, {
        "before_sha256": before,
        "after_sha256": after,
        "embedded_sha256": embedded,
        "positive_class_index": positive_index,
    }


def _positive_index(classes: object) -> int:
    values = np.asarray(classes)
    matches = [
        index for index, value in enumerate(values.tolist())
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        and int(value) == POSITIVE_LABEL
    ]
    if values.ndim != 1 or len(matches) != 1:
        raise ValueError("Estimator classes_ must contain integer positive label 1 exactly once")
    return matches[0]


def _find_forest(root: object) -> object:
    pending = [root]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        state = getattr(current, "__dict__", {})
        if (
            isinstance(state, dict)
            and state.get("estimators_") is not None
            and callable(getattr(current, "get_params", None))
        ):
            return current
        values = state
        if isinstance(values, dict):
            for value in values.values():
                if isinstance(value, (list, tuple)):
                    pending.extend(value)
                elif isinstance(value, dict):
                    pending.extend(value.values())
                elif hasattr(value, "__dict__"):
                    pending.append(value)
    raise TypeError("Calibrated estimator does not contain a fitted forest")


def probability_report(estimator: object, features: pd.DataFrame, labels: object) -> dict:
    """Create probability-only metrics and fixed ten-bin reliability summaries."""
    y = _validate_calibration_inputs(features, labels)
    probabilities = np.asarray(predict_positive_probability(estimator, features), dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.shape != y.shape:
        raise ValueError("Probability and label shapes do not match")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite values in [0, 1]")
    results = {
        "average_precision": _metric(average_precision_score, y, probabilities),
        "roc_auc": _metric(roc_auc_score, y, probabilities),
        "brier_score": {"value": float(np.mean(np.square(probabilities - y))), "status": "defined", "reason": None},
        "log_loss": _metric(lambda target, score: log_loss(target, score, labels=[0, 1]), y, probabilities),
    }
    bins = np.searchsorted(BIN_EDGES, probabilities, side="right") - 1
    bins = np.minimum(bins, 9)
    reliability = []
    for index in range(10):
        selected = bins == index
        count = int(np.count_nonzero(selected))
        reliability.append({
            "index": index,
            "lower": float(BIN_EDGES[index]),
            "upper": float(BIN_EDGES[index + 1]),
            "endpoint_rule": "lower inclusive; upper exclusive except bin 9 includes 1.0",
            "count": count,
            "mean_predicted_probability": float(probabilities[selected].mean()) if count else None,
            "observed_positive_fraction": float(y[selected].mean()) if count else None,
            "status": "defined" if count else "empty",
        })
    return {
        "schema_version": "phase14a_calibration_report.v1",
        "diagnostic_scope": "in-sample calibration-fit diagnostics",
        "metrics": results,
        "row_count": int(y.size),
        "positive_count": int(np.count_nonzero(y == 1)),
        "negative_count": int(np.count_nonzero(y == 0)),
        "prevalence": float(np.mean(y == 1)),
        "classification_threshold": None,
        "reliability_bins": reliability,
    }


def _metric(function, labels: np.ndarray, probabilities: np.ndarray) -> dict:
    try:
        value = float(function(labels, probabilities))
        if not np.isfinite(value):
            return {"value": None, "status": "undefined", "reason": "metric returned a non-finite value"}
        return {"value": value, "status": "defined", "reason": None}
    except (ValueError, ZeroDivisionError) as exc:
        return {"value": None, "status": "undefined", "reason": str(exc)}


def _reject_reparse_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            stat_result = current.lstat()
        except FileNotFoundError:
            continue
        if current.is_symlink() or getattr(stat_result, "st_file_attributes", 0) & 0x400:
            raise PermissionError(f"Reparse path is not allowed: {current}")
    return absolute


def load_verified_base_estimator(
    *,
    allow_synthetic: bool = False,
    synthetic_path: Path | None = None,
    synthetic_expected_sha256: str | None = None,
) -> tuple[object, dict]:
    """Verify all pinned Phase 13 identities before loading its estimator.

    Production paths are fixed and hash-checked. Tests may supply only a
    synthetic model inside the OS temporary tree and explicit synthetic proof.
    """
    if allow_synthetic:
        if synthetic_path is None:
            raise PermissionError("Synthetic model loading requires an explicit temporary path")
        path = _reject_reparse_path(Path(synthetic_path))
        temp_root = Path(os.path.abspath(__import__("tempfile").gettempdir()))
        if not path.is_relative_to(temp_root) or not path.is_file():
            raise PermissionError("Synthetic model paths must be files under OS temporary storage")
        digest = _sha256_file(path)
        if synthetic_expected_sha256 is None or digest != synthetic_expected_sha256:
            raise ValueError("Synthetic model hash must be verified before deserialization")
        return joblib.load(path), {"synthetic": True, "model_sha256": digest}

    root = _reject_reparse_path(_PHASE13_ROOT)
    model_path = _reject_reparse_path(root / "base_rf.joblib")
    manifest_path = _reject_reparse_path(root / "manifest.json")
    config_path = _reject_reparse_path(_CODE_ROOT / "config" / "phase13_set_c_rf_v1.yaml")
    identities = {
        config_path: PHASE13_CONFIG_SHA256,
        manifest_path: PHASE13_MANIFEST_SHA256,
        model_path: BASE_MODEL_SHA256,
    }
    if any(not item.is_file() or _sha256_file(item) != expected for item, expected in identities.items()):
        raise ValueError("Phase 13 input identity mismatch; model deserialization denied")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    model_record = manifest.get("models", {}).get("base_model", {})
    training_source = manifest.get("training_source", {})
    schema = manifest.get("schema_population", {})
    manifest_inputs = manifest.get("inputs", {})
    if (
        manifest.get("schema_version") != "set_c_rf_phase13_run.v1"
        or manifest.get("run_id") != PHASE13_RUN_ID
        or training_source.get("revision") != PHASE13_REVISION
        or manifest.get("protocol_sha256") != PHASE13_PROTOCOL_SHA256
        or manifest.get("artifact_kind") != "set_c_rf_uncalibrated_candidate_run"
        or manifest.get("completion_status") != "complete"
        or training_source.get("configuration_sha256") != PHASE13_CONFIG_SHA256
        or manifest_inputs.get("attestation_sha256") != DEVELOPMENT_ATTESTATION_SHA256
        or manifest_inputs.get("source_csv_sha256") != SOURCE_CSV_SHA256
        or manifest_inputs.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or schema.get("feature_schema_version") != SET_C_SCHEMA_VERSION
        or schema.get("target_version") != TARGET_VERSION
        or schema.get("feature_names") != list(CANONICAL_FEATURE_NAMES)
        or schema.get("feature_dtype") != "float32"
        or schema.get("target_name") != TARGET_NAME
        or schema.get("target_dtype") != "int8"
        or schema.get("positive_label") != POSITIVE_LABEL
        or model_record.get("filename") != "base_rf.joblib"
        or model_record.get("training_roles_ordered") != ["train", "validation"]
        or model_record.get("feature_names_in") != list(CANONICAL_FEATURE_NAMES)
        or model_record.get("classes") != [0, 1]
        or model_record.get("calibrated") is not False
    ):
        raise ValueError("Phase 13 manifest contract mismatch; model deserialization denied")
    model = joblib.load(model_path)
    validate_model_feature_schema(model)
    _positive_index(model.classes_)
    if forest_fingerprint(model) is None:
        raise ValueError("Loaded base model failed forest fingerprint validation")
    return model, {
        "synthetic": False,
        "model_sha256": BASE_MODEL_SHA256,
        "manifest_sha256": PHASE13_MANIFEST_SHA256,
        "source_revision": PHASE13_REVISION,
    }


def _write_json_new(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _build_calibration_package(
    estimator: object,
    features: pd.DataFrame,
    labels: object,
    destination: Path,
    *,
    provenance: dict,
    run_id: str,
    configuration_sha256: str,
    execution_context: dict,
    expected_policy: dict | None = None,
) -> dict:
    """Shared computation/package builder; supervision remains an external stage."""
    target = Path(os.path.abspath(destination))
    if target.exists():
        raise FileExistsError("Calibration run destination already exists")
    _reject_reparse_path(target.parent).mkdir(parents=True, exist_ok=True)
    target.mkdir()
    started_dt = datetime.now().astimezone()
    started = datetime.now(timezone.utc).isoformat()
    finalized: dict[str, str] = {}
    stage = "calibration inputs"
    try:
        if not isinstance(provenance, dict) or provenance.get("synthetic") not in (True, False):
            raise PermissionError("Package requires explicit validated loader provenance")
        if (execution_context.get("execution_scope") == "synthetic_test_only") != (provenance.get("synthetic") is True):
            raise PermissionError("Execution scope and input provenance disagree")
        stage = "frozen calibration"
        calibrated, fingerprints = calibrate_frozen_estimator(estimator, features, labels)
        stage = "probability report"
        report = probability_report(calibrated, features, labels)
        stage = "model serialization and verification"
        model_path = target / CALIBRATED_MODEL_FILENAME
        joblib.dump(calibrated, model_path)
        round_trip = joblib.load(model_path)
        round_trip_forest = forest_fingerprint(_find_forest(round_trip))
        if round_trip_forest != fingerprints["before_sha256"]:
            raise RuntimeError("Serialized calibrated artifact changed the forest")
        finalized[model_path.name] = _sha256_file(model_path)
        stage = "report publication"
        report_path = target / REPORT_FILENAME
        _write_json_new(report_path, report)
        finalized[report_path.name] = _sha256_file(report_path)
        manifest = {
            "schema_version": "phase14a_calibrated_model_manifest.v1",
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": PROTOCOL_SHA256,
            "run_id": run_id,
            "source_revision": execution_context["source_revision"],
            "source_clean": execution_context["source_clean"],
            "execution_scope": execution_context["execution_scope"],
            "approval_reference": execution_context["approval_reference"],
            "command": execution_context["argv"],
            "configuration_sha256": configuration_sha256,
            "working_directory": execution_context["working_directory"],
            "actual_base_model_identity": execution_context.get("actual_base_model_identity"),
            "expected_policy": expected_policy,
            "output_paths": {
                "calibrated_model": CALIBRATED_MODEL_FILENAME,
                "manifest": MANIFEST_FILENAME,
                "report": REPORT_FILENAME,
                "execution_receipt": RECEIPT_FILENAME,
            },
            "phase13_expected_policy": {
                "run_id": PHASE13_RUN_ID,
                "source_revision": PHASE13_REVISION,
                "protocol_sha256": PHASE13_PROTOCOL_SHA256,
                "config_sha256": PHASE13_CONFIG_SHA256,
                "manifest_sha256": PHASE13_MANIFEST_SHA256,
                "base_model_sha256": BASE_MODEL_SHA256,
            },
            "method": "sigmoid",
            "method_decision_reference": DECISION_REFERENCE,
            "feature_schema_version": SET_C_SCHEMA_VERSION,
            "feature_names": list(CANONICAL_FEATURE_NAMES),
            "target_name": TARGET_NAME,
            "target_version": TARGET_VERSION,
            "positive_label": POSITIVE_LABEL,
            "input_role": "calibration",
            "input": _approved_provenance(provenance),
            "software": _software_versions(),
            "forest_fingerprint": fingerprints,
            "calibrated_model": {"filename": CALIBRATED_MODEL_FILENAME, "sha256": _sha256_file(model_path)},
            "report": {"filename": REPORT_FILENAME, "sha256": _sha256_file(report_path)},
            "status": "complete",
            "started_at_utc": started,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "computation_started_at_local": started_dt.isoformat(),
            "computation_ended_at_local": datetime.now().astimezone().isoformat(),
            "local_timezone": str(datetime.now().astimezone().tzinfo),
        }
        stage = "manifest publication"
        manifest_path = target / MANIFEST_FILENAME
        _write_json_new(manifest_path, manifest)
        finalized[manifest_path.name] = _sha256_file(manifest_path)
        receipt = _receipt(
            run_id, started, finalized, provenance, status="complete", failure=None,
            partial_inventory=sorted(finalized),
            configuration_sha256=configuration_sha256, execution_context=execution_context,
            started_local=started_dt.isoformat(),
        )
        stage = "execution receipt publication"
        _write_json_new(target / RECEIPT_FILENAME, receipt)
        return {"destination": str(target), "manifest": manifest, "report": report, "receipt": receipt}
    except BaseException as error:
        partial_inventory = []
        try:
            partial_inventory = sorted(
                path.name for path in target.iterdir()
                if path.name != RECEIPT_FILENAME
            )
        except BaseException as inventory_error:
            try:
                sys.stderr.write(f"Phase 14 partial inventory unavailable ({type(inventory_error).__name__})\n")
                sys.stderr.flush()
            except BaseException:
                pass
        try:
            _write_json_new(
                target / RECEIPT_FILENAME,
                _receipt(
                    run_id, started, finalized, provenance, status="failed",
                    failure=f"{type(error).__name__} at {stage}",
                    partial_inventory=partial_inventory,
                    configuration_sha256=configuration_sha256, execution_context=execution_context,
                    started_local=started_dt.isoformat(),
                ),
            )
        except BaseException as receipt_error:
            try:
                sys.stderr.write(f"Phase 14 failure receipt write failed ({type(receipt_error).__name__})\n")
                sys.stderr.flush()
            except BaseException:
                pass
        raise RuntimeError(f"Phase 14 calibration failed ({type(error).__name__} at {stage})") from None


def run_synthetic_calibration_package(
    estimator: object,
    features: pd.DataFrame,
    labels: object,
    destination: Path,
    *,
    provenance: dict,
    run_id: str = "synthetic-phase14-calibration",
) -> dict:
    """Exercise package lifecycle with explicitly synthetic OS-temp fixtures."""
    target = Path(os.path.abspath(destination))
    temp_root = Path(os.path.abspath(__import__("tempfile").gettempdir()))
    if not target.is_relative_to(temp_root) or target == temp_root:
        raise PermissionError("Synthetic calibration outputs must be under OS temporary storage")
    if provenance.get("synthetic") is not True:
        raise PermissionError("Synthetic package requires explicit synthetic loader provenance")
    config, config_digest = _load_phase14_config(
        _PHASE14_CONFIG, _sha256_file(_PHASE14_CONFIG), synthetic=True
    )
    del config
    now = datetime.now().astimezone().isoformat()
    context = {
        "source_revision": "synthetic-fixture-not-applicable",
        "source_clean": "not_applicable_synthetic_fixture",
        "execution_scope": "synthetic_test_only",
        "approval_reference": "not_applicable_synthetic_fixture",
        "argv": ["synthetic-test-only"],
        "working_directory": str(Path.cwd()),
        "started_at_local": now,
        "actual_base_model_identity": {
            "kind": "synthetic_fixture_forest_fingerprint",
            "sha256": forest_fingerprint(estimator),
        },
    }
    return _build_calibration_package(
        estimator, features, labels, target, provenance=provenance, run_id=run_id,
        configuration_sha256=config_digest, execution_context=context,
        expected_policy={"protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256},
    )


def run_synthetic_calibration_from_release(
    estimator: object,
    release_dir: Path,
    destination: Path,
    *,
    expected_attestation_sha256: str,
    run_id: str = "synthetic-phase14-calibration",
) -> dict:
    """Exercise the accepted calibration-only loader and synthetic package path."""
    features, labels, provenance = load_development_calibration_release(
        release_dir,
        expected_attestation_sha256=expected_attestation_sha256,
        allow_synthetic=True,
    )
    return run_synthetic_calibration_package(
        estimator, features, labels, destination, provenance=provenance, run_id=run_id
    )


def run_calibration_job(
    *,
    config_path: Path,
    expected_config_sha256: str,
    execution_context: dict,
) -> dict:
    """Production-shaped adapter contract; current configuration fails closed.

    The callable is not a CLI or supervisor. It requires an externally validated,
    clean source identity, a procedural approval reference, and a locked config.
    """
    config, config_digest = _load_phase14_config(
        config_path, expected_config_sha256, synthetic=False
    )
    destination = _validate_production_context(config, execution_context, _CODE_ROOT)
    observed_context = _observe_child_context("calibration", execution_context["argv"])
    if any(execution_context.get(key) != observed_context.get(key) for key in observed_context):
        raise PermissionError("Calibration context differs from independently observed evidence")
    paths = config["paths"]
    source_release = _reject_reparse_path(_CODE_ROOT / paths["calibration_release"])
    model, model_evidence = load_verified_base_estimator()
    features, labels, provenance = load_development_calibration_release(
        source_release,
        expected_attestation_sha256=CALIBRATION_ATTESTATION_SHA256,
        allow_synthetic=False,
    )
    if provenance.get("synthetic") is not False:
        raise PermissionError("Production calibration loader returned synthetic provenance")
    actual_input = _approved_provenance(provenance)
    execution = dict(execution_context)
    execution.update(execution_scope="production_calibration", source_clean=True)
    execution["actual_base_model_identity"] = {
        "kind": "phase13_base_model_file",
        "sha256": model_evidence["model_sha256"],
        "manifest_sha256": model_evidence["manifest_sha256"],
    }
    return _build_calibration_package(
        model, features, labels, destination, provenance=actual_input,
        run_id=execution_context["run_id"],
        configuration_sha256=config_digest, execution_context=execution,
        expected_policy={
            "protocol_id": PROTOCOL_ID, "protocol_sha256": PROTOCOL_SHA256,
            "phase13_model_sha256": BASE_MODEL_SHA256,
            "phase13_manifest_sha256": PHASE13_MANIFEST_SHA256,
        },
    )


def _approved_provenance(provenance: dict) -> dict:
    keys = (
        "attestation_sha256", "calibration_csv_sha256", "rows", "positives",
        "negatives", "source_csv_sha256", "source_revision", "synthetic",
    )
    if set(provenance) != set(keys):
        raise ValueError("Calibration provenance must contain exactly the approved eight fields")
    return {key: provenance[key] for key in keys}


def _receipt(
    run_id: str,
    started: str,
    finalized: dict,
    provenance: dict,
    *,
    status: str,
    failure: str | None,
    partial_inventory: list[str],
    configuration_sha256: str,
    execution_context: dict,
    started_local: str,
) -> dict:
    allowed = {
        "attestation_sha256", "calibration_csv_sha256", "rows", "positives",
        "negatives", "source_csv_sha256", "source_revision", "synthetic",
    }
    safe_provenance = {key: value for key, value in provenance.items() if key in allowed} if isinstance(provenance, dict) else {}
    return {
        "schema_version": "phase14a_execution_receipt.v2",
        "run_id": run_id,
        "approval_reference": execution_context["approval_reference"],
        "argv": execution_context["argv"],
        "source_revision": execution_context["source_revision"],
        "source_clean": execution_context["source_clean"],
        "source_check": execution_context.get("source_check"),
        "supervisor_handoff": execution_context.get("supervisor_handoff"),
        "worker_binding_sha256": execution_context.get("worker_binding_sha256"),
        "execution_scope": execution_context["execution_scope"],
        "working_directory": execution_context["working_directory"],
        "configuration_sha256": configuration_sha256,
        "inputs": {"role": "calibration", "synthetic": safe_provenance.get("synthetic"), "provenance": safe_provenance},
        "finalized_outputs": dict(sorted(finalized.items())),
        "environment": _software_versions(),
        "computation_started_at_utc": started,
        "computation_ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "computation_started_at_local": started_local,
        "computation_ended_at_local": datetime.now().astimezone().isoformat(),
        "local_timezone": str(datetime.now().astimezone().tzinfo),
        "computation_status": status,
        "failure_reason": failure,
        "partial_output_inventory": partial_inventory,
    }


def _software_versions() -> dict:
    import sklearn
    import scipy
    import optuna
    import yaml

    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "optuna": optuna.__version__,
        "joblib": joblib.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyyaml": yaml.__version__,
        "platform": platform.platform(),
    }


SOURCE_FILES = (
    "Thesis_RF/Code/run_phase14_supervised.ps1",
    "Thesis_RF/Code/evaluate_rf_final_test.py",
    *("Thesis_RF/Code/modules/" + name for name in (
        "__init__.py", "phase_supervision.psm1", "phase14_calibration.py",
        "feature_pipeline.py", "wind_convention.py", "ca_state.py", "data_loader.py",
        "automata_engine.py", "set_c_split_release.py", "set_c_leakage.py",
        "set_c_publication.py", "set_c_collector.py", "transition_observer.py")),
    *("Thesis_RF/Code/config/" + name for name in (
        "phase13_set_c_rf_v1.yaml", "phase14_setc_calibration_v1.yaml", "phase14_setc_final_test_v1.yaml")),
    "docs/Phase14A_Calibration_Final_Test_Protocol_Proposal.md",
)
AUTHORIZATION_FIELDS = {"schema_version", "job", "approval_reference", "run_id", "expected_revision",
                        "created_at_utc", "configuration_sha256", "protocol_sha256", "source_files",
                        "environment", "environment_sha256", "pretest_lock", "primary_evidence"}
HANDOFF_FIELDS = {"schema_version", "job", "run_id", "approval_reference", "expected_revision",
                  "authorization_json", "authorization_sha256", "nonce", "supervisor_pid",
                  "executable", "argv", "working_directory", "evidence_directory", "destination",
                  "created_at_utc", "deadline_utc"}


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON member")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant")


def _strict_json(text: str) -> dict:
    # Encode first to reject malformed transport text; never normalize reviewed bytes.
    text.encode("utf-8", errors="strict")
    value = json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON record must be an object")
    return value


def _exact_object(value: object, fields: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Record member inventory is invalid")
    return value


def _safe_identifier(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value) is None:
        raise ValueError("Invalid procedural identifier")
    return value


def _aware_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Invalid timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("UTC timestamp required")
    return parsed


def _validate_environment_envelope(environment: dict) -> None:
    _exact_object(environment, {"status", "versions", "lock_sha256", "interpreter", "powershell", "auxiliary_versions"})
    _exact_object(environment["versions"], {"python", "scikit_learn", "optuna", "joblib", "numpy", "pandas", "scipy", "pyyaml", "platform"})
    _exact_object(environment["interpreter"], {"path"})
    _exact_object(environment["powershell"], {"version", "edition"})
    _exact_object(environment["auxiliary_versions"], {"rasterio"})
    if (environment["status"] != "complete_locked"
            or any(not isinstance(v, str) or not v for v in environment["versions"].values())
            or not isinstance(environment["auxiliary_versions"]["rasterio"], str)
            or not environment["auxiliary_versions"]["rasterio"]
            or not isinstance(environment["interpreter"]["path"], str) or not environment["interpreter"]["path"]
            or environment["powershell"] != {"version": "5.1.26100.9711", "edition": "Desktop"}):
        raise PermissionError("Environment envelope is incomplete or mismatched")
    if environment["lock_sha256"] != sha256(_json_bytes(environment["versions"])).hexdigest():
        raise PermissionError("The Phase 14 environment lock digest is invalid")


def _validate_authorization(text: str, digest: str, job: str, approval: str, revision: str, run_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", digest or "") or sha256(text.encode("utf-8")).hexdigest() != digest:
        raise PermissionError("Exact authorization transport digest mismatch")
    record = _exact_object(_strict_json(text), AUTHORIZATION_FIELDS)
    _safe_identifier(approval)
    _safe_identifier(run_id)
    if (record["schema_version"] != "phase14a_job_authorization.v1"
            or job not in {"calibration", "final_primary", "final_verification"}
            or (record["job"], record["approval_reference"], record["expected_revision"], record["run_id"])
            != (job, approval, revision, run_id)
            or re.fullmatch(r"[0-9a-f]{40}", revision or "") is None
            or record["protocol_sha256"] != PROTOCOL_SHA256
            or re.fullmatch(r"[0-9a-f]{64}", record["configuration_sha256"] or "") is None):
        raise PermissionError("Authorization fields disagree with fixed job")
    _aware_timestamp(record["created_at_utc"])
    source = _exact_object(record["source_files"], set(SOURCE_FILES))
    if any(not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in source.values()):
        raise ValueError("Source manifest contains unresolved identities")
    config_member = "Thesis_RF/Code/config/" + ("phase14_setc_calibration_v1.yaml" if job == "calibration" else "phase14_setc_final_test_v1.yaml")
    if (source["docs/Phase14A_Calibration_Final_Test_Protocol_Proposal.md"] != PROTOCOL_SHA256
            or source[config_member] != record["configuration_sha256"]):
        raise PermissionError("Source manifest config/protocol cross-binding mismatch")
    _validate_environment_envelope(record["environment"])
    if record["environment_sha256"] != sha256(_json_bytes(record["environment"])).hexdigest():
        raise PermissionError("Environment envelope digest mismatch")
    lock, primary = record["pretest_lock"], record["primary_evidence"]
    if job == "calibration":
        if lock is not None or primary is not None:
            raise ValueError("Calibration cannot bind final evidence")
    else:
        _exact_object(lock, {"lock_id", "sha256", "owner_approval_reference"})
        _safe_identifier(lock["lock_id"])
        _safe_identifier(lock["owner_approval_reference"])
        if re.fullmatch(r"[0-9a-f]{64}", lock["sha256"] or "") is None:
            raise ValueError("Unresolved external lock identity")
        if job == "final_primary" and primary is not None:
            raise ValueError("Primary cannot bind future evidence")
        if job == "final_verification":
            _exact_object(primary, {"run_id", "report_sha256", "receipt_sha256", "supervisor_completion_sha256"})
            _safe_identifier(primary["run_id"])
            if primary["run_id"] == run_id or any(
                    not isinstance(primary[k], str) or re.fullmatch(r"[0-9a-f]{64}", primary[k]) is None
                    for k in ("report_sha256", "receipt_sha256", "supervisor_completion_sha256")):
                raise ValueError("Verification requires distinct finalized primary evidence")
    return record


def _runtime_source_inventory(repo: Path) -> list[str]:
    # Prune names before inspecting entries: excluded trees are never traversed.
    excluded = {"tests", "output", "data", "datafiles", "processeddata", "dataset", "datasets",
                "model", "models", "trainedmodels", "trained_models", "release", "releases",
                "protected", "setc_protected", "setc_development", "artifacts", "results",
                "ground_truth", "groundtruth", "thesispaper", "manuscript", "manuscripts",
                ".git", ".venv", "__pycache__"}
    pending = [_reject_reparse_path(repo / "Thesis_RF" / "Code")]
    sources = []
    inspected = 0
    while pending:
        for path in pending.pop().iterdir():
            if path.name.casefold() in excluded:
                continue
            inspected += 1
            if inspected > 4096:
                raise PermissionError("Runtime source inventory bound exceeded")
            metadata = path.lstat()
            if (stat.S_ISLNK(metadata.st_mode)
                    or getattr(metadata, "st_file_attributes", 0) & 0x400):
                raise PermissionError("Runtime source reparse entry denied")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(path)
            elif path.suffix.casefold() in {".py", ".ps1", ".psm1"}:
                sources.append(path.relative_to(repo).as_posix())
    return sorted(sources)


def _observe_source(record: dict, repo: Path) -> dict:
    def git(*arguments: str) -> str:
        result = subprocess.run(["git", "--literal-pathspecs", "-C", str(repo), *arguments], capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    revision = git("rev-parse", "HEAD")
    if revision != record["expected_revision"] or git("status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_FILES):
        raise PermissionError("Runtime source revision is dirty or mismatched")
    inventory = _runtime_source_inventory(repo)
    tracked = set(git("ls-files", "--cached", "--", *inventory).splitlines()) if inventory else set()
    additional = [relative for relative in inventory if relative not in SOURCE_FILES]
    if (any(relative not in tracked for relative in additional)
            or (additional and git("status", "--porcelain=v1", "--untracked-files=all", "--", *additional))):
        raise PermissionError("Unreviewed local import source addition")
    observed = {}
    for relative in SOURCE_FILES:
        path = _reject_reparse_path(repo / relative)
        observed[relative] = _sha256_file(path)
    if observed != record["source_files"]:
        raise PermissionError("Reviewed source manifest bytes differ")
    return {"revision": revision, "clean": True, "source_files": observed,
            "configuration_sha256": record["configuration_sha256"], "protocol_sha256": PROTOCOL_SHA256}


def _fixed_child_arguments(job: str, approval: str, revision: str, run_id: str, config_hash: str, evidence: Path) -> list[str]:
    config_name = "phase14_setc_calibration_v1.yaml" if job == "calibration" else "phase14_setc_final_test_v1.yaml"
    entry = ["-B", "-m", "modules.phase14_calibration"] if job == "calibration" else ["-B", "evaluate_rf_final_test.py", "--mode", job.removeprefix("final_")]
    return entry + ["--config", str(_CODE_ROOT / "config" / config_name), "--approval-reference", approval,
                    "--expected-revision", revision, "--run-id", run_id, "--expected-config-sha256", config_hash,
                    "--evidence-dir", str(evidence)]



CLAIM_FIELDS = {"schema_version", "job", "run_id", "nonce", "handoff_sha256", "supervisor_pid",
                "worker_pid", "worker_ppid", "logical_executable", "physical_executable",
                "worker_created_at_utc", "command_line", "claimed_at_utc"}
INSTANCE_FIELDS = {"pid", "ppid", "created_at_utc", "executable", "command_line"}
BINDING_FIELDS = {"schema_version", "claim_sha256", "claim", "observed_at_utc",
                  "observation_source", "supervisor", "launcher", "worker"}
PROCESS_FIELDS = {"schema_version", "run_id", "process_id", "process_ids", "process_started_at_utc",
                  "executable", "working_directory", "worker_binding"}
EXIT_FIELDS = {"schema_version", "worker_binding_sha256", "worker_pid", "worker_created_at_utc",
               "observed_at_utc", "observation_source", "same_instance_running", "pid_reused"}


def _current_worker_instance() -> dict:
    """Worker claims are corroborated by the supervisor; they are not OS attestations."""
    import ctypes
    from ctypes import wintypes
    from datetime import timedelta
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    times = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), *(ctypes.byref(t) for t in times)):
        raise ctypes.WinError(ctypes.get_last_error())
    ticks = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    created = datetime(1601, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)
    kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel.GetModuleFileNameW.restype = wintypes.DWORD
    image = ctypes.create_unicode_buffer(32768)
    size = kernel.GetModuleFileNameW(None, image, len(image))
    if not 0 < size < len(image):
        raise OSError("Worker image unavailable or truncated")
    kernel.GetCommandLineW.restype = wintypes.LPWSTR
    command = kernel.GetCommandLineW()
    if not command:
        raise OSError("Worker command line unavailable")
    return {"worker_pid": os.getpid(), "worker_ppid": os.getppid(),
            "worker_created_at_utc": created.isoformat(), "physical_executable": image.value,
            "command_line": command}


def _binding_hash(binding: dict, evidence: Path) -> str:
    path = _reject_reparse_path(evidence / "worker_binding.json")
    if _strict_json(path.read_text(encoding="utf-8")) != binding:
        raise ValueError("Immutable worker binding differs")
    return _sha256_file(path)


def _validate_worker_binding(process: dict, handoff: dict, claim: dict, claim_hash: str,
                             handoff_hash: str) -> dict:
    fields = PROCESS_FIELDS | (set(process) & {"stop_reason", "updated_at_utc"})
    _exact_object(process, fields)
    binding = _exact_object(process["worker_binding"], BINDING_FIELDS)
    _exact_object(claim, CLAIM_FIELDS)
    snapshots = {name: _exact_object(binding[name], INSTANCE_FIELDS)
                 for name in ("supervisor", "launcher", "worker")}
    for instance in snapshots.values():
        if (any(type(instance[k]) is not int or instance[k] <= 0 for k in ("pid", "ppid"))
                or any(not isinstance(instance[k], str) or not instance[k] for k in
                       ("executable", "command_line"))):
            raise ValueError("Worker OS instance schema mismatch")
    supervisor, launcher, worker = (snapshots[k] for k in ("supervisor", "launcher", "worker"))
    started = _aware_timestamp(handoff["created_at_utc"])
    deadline = _aware_timestamp(handoff["deadline_utc"])
    launch_recorded = _aware_timestamp(process["process_started_at_utc"])
    observed = _aware_timestamp(binding["observed_at_utc"])
    claimed = _aware_timestamp(claim["claimed_at_utc"])
    super_start, launch_start, worker_start = (
        _aware_timestamp(snapshots[k]["created_at_utc"]) for k in ("supervisor", "launcher", "worker"))
    if (process["schema_version"] != "phase14a_process.v2"
            or binding["schema_version"] != "phase14a_worker_binding.v1"
            or claim["schema_version"] != "phase14a_worker_claim.v1"
            or binding["observation_source"] != "Get-CimInstance Win32_Process"
            or binding["claim"] != claim or binding["claim_sha256"] != claim_hash
            or not _is_sha256(claim_hash) or claim["handoff_sha256"] != handoff_hash
            or any(claim[k] != handoff[k] for k in ("job", "run_id", "nonce", "supervisor_pid"))
            or process["run_id"] != handoff["run_id"] or process["executable"] != handoff["executable"]
            or process["working_directory"] != handoff["working_directory"]
            or claim["logical_executable"] != handoff["executable"]
            or type(process["process_id"]) is not int or process["process_id"] != launcher["pid"]
            or any(type(claim[k]) is not int for k in ("worker_pid", "worker_ppid", "supervisor_pid"))
            or supervisor["pid"] != handoff["supervisor_pid"] or launcher["ppid"] != supervisor["pid"]
            or supervisor["pid"] in {launcher["pid"], worker["pid"]}
            or worker["pid"] != claim["worker_pid"] or worker["ppid"] != claim["worker_ppid"]
            or (worker["pid"] != launcher["pid"] and worker["ppid"] != launcher["pid"])
            or (worker["pid"] == launcher["pid"] and worker != launcher)
            or os.path.normcase(launcher["executable"]) != os.path.normcase(handoff["executable"])
            or os.path.normcase(worker["executable"]) != os.path.normcase(claim["physical_executable"])
            or worker["command_line"] != claim["command_line"]
            or worker_start != _aware_timestamp(claim["worker_created_at_utc"])
            or not super_start <= started <= launch_start <= launch_recorded <= observed < deadline
            or not launch_start <= worker_start <= claimed <= observed
            or (observed - launch_recorded).total_seconds() > 10
            or not isinstance(process["process_ids"], list) or not process["process_ids"]
            or any(type(pid) is not int or pid <= 0 for pid in process["process_ids"])
            or len(set(process["process_ids"])) != len(process["process_ids"])
            or launcher["pid"] not in process["process_ids"]):
        raise ValueError("Worker startup binding mismatch")
    return binding


def _validate_worker_exit(value: dict, binding: dict, ended: datetime, binding_hash: str) -> None:
    _exact_object(value, EXIT_FIELDS)
    if (value["schema_version"] != "phase14a_worker_exit.v1"
            or value["worker_binding_sha256"] != binding_hash
            or type(value["worker_pid"]) is not int or value["worker_pid"] != binding["worker"]["pid"]
            or value["worker_created_at_utc"] != binding["worker"]["created_at_utc"]
            or value["observation_source"] != "Get-CimInstance Win32_Process"
            or value["same_instance_running"] is not False or value["pid_reused"] is not False
            or not _aware_timestamp(binding["observed_at_utc"]) <= _aware_timestamp(value["observed_at_utc"]) <= ended):
        raise ValueError("Worker completion observation unavailable or inconsistent")


def _claim_and_await_worker(handoff_path: Path, handoff: dict) -> dict:
    root = handoff_path.parent
    claim_path = _reject_reparse_path(root / "worker_claim.json")
    identity = {"schema_version": "phase14a_worker_claim.v1", "job": handoff["job"],
                "run_id": handoff["run_id"], "nonce": handoff["nonce"],
                "handoff_sha256": _sha256_file(handoff_path), "supervisor_pid": handoff["supervisor_pid"],
                "logical_executable": str(Path(os.path.abspath(sys.executable))), **_current_worker_instance()}
    if claim_path.exists():
        claim = _exact_object(_strict_json(claim_path.read_text(encoding="utf-8")), CLAIM_FIELDS)
        if any(claim[k] != v for k, v in identity.items()):
            raise ValueError("Existing worker claim conflicts")
    else:
        claim = {**identity, "claimed_at_utc": datetime.now(timezone.utc).isoformat()}
        pending = _reject_reparse_path(root / ("worker_claim-" + handoff["nonce"] + ".pending.json"))
        _write_json_new(pending, claim)
        # Windows rename is exclusive; a conflicting claim is never replaced.
        pending.rename(claim_path)
    process_path = _reject_reparse_path(root / "process.json")
    limit = min(time.monotonic() + 10, time.monotonic() +
                (_aware_timestamp(handoff["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds())
    while not process_path.is_file() and time.monotonic() < limit:
        time.sleep(0.02)
    if datetime.now(timezone.utc) >= _aware_timestamp(handoff["deadline_utc"]):
        raise ValueError("Worker startup deadline expired")
    process = _strict_json(process_path.read_text(encoding="utf-8"))
    if _aware_timestamp(process["worker_binding"]["observed_at_utc"]) > datetime.now(timezone.utc):
        raise ValueError("Worker observation is in the future")
    return _validate_worker_binding(process, handoff, claim, _sha256_file(claim_path),
                                    _sha256_file(handoff_path))

def _observe_child_context(job: str, child_arguments: list[str]) -> dict:
    stage = "supervised context"
    try:
        code = _reject_reparse_path(_CODE_ROOT)
        repo = code.parents[1]
        executable = _reject_reparse_path(repo / ".venv" / "Scripts" / "python.exe")
        if Path(os.path.abspath(sys.executable)) != executable or Path(os.path.abspath(Path.cwd())) != code:
            raise PermissionError("Fixed child interpreter/cwd mismatch")
        handoff_path = Path(os.environ.get("PHASE14_HANDOFF_PATH", ""))
        evidence_root = code / "output" / "phase14" / "evidence"
        if (not handoff_path.is_absolute() or not handoff_path.is_relative_to(evidence_root)
                or handoff_path.name != "supervisor_handoff.json" or handoff_path.parent.parent != evidence_root):
            raise PermissionError("Fixed handoff path required")
        handoff_path = _reject_reparse_path(handoff_path)
        handoff = _exact_object(_strict_json(handoff_path.read_text(encoding="utf-8")), HANDOFF_FIELDS)
        record = _validate_authorization(handoff["authorization_json"], handoff["authorization_sha256"], job,
                                         handoff["approval_reference"], handoff["expected_revision"], handoff["run_id"])
        if (record["environment"]["interpreter"] != {"path": str(executable)}
                or handoff["schema_version"] != "phase14a_supervisor_handoff.v1"
                or handoff["job"] != job or handoff["run_id"] != handoff_path.parent.name
                or re.fullmatch(r"[0-9a-f]{32}", handoff["nonce"] or "") is None
                or os.environ.get("PHASE14_SUPERVISOR_NONCE") != handoff["nonce"]
                or type(handoff["supervisor_pid"]) is not int or handoff["supervisor_pid"] <= 0
                or handoff["executable"] != str(executable) or handoff["working_directory"] != str(code)
                or handoff["evidence_directory"] != str(handoff_path.parent)):
            raise PermissionError("Supervisor handoff cross-binding mismatch")
        started, deadline = _aware_timestamp(handoff["created_at_utc"]), _aware_timestamp(handoff["deadline_utc"])
        now = datetime.now(timezone.utc)
        if not started <= now < deadline or not 0 < (deadline - started).total_seconds() <= 1800:
            raise PermissionError("Supervisor operation deadline is invalid or expired")
        expected_args = _fixed_child_arguments(job, record["approval_reference"], record["expected_revision"],
                                               record["run_id"], record["configuration_sha256"], handoff_path.parent)
        output_root = code / "output" / "phase14"
        destination = (output_root / "phase14-setc-rf-61e8f740-r1" / "calibration" / record["run_id"]
                       if job == "calibration" else output_root / "final-test" / record["pretest_lock"]["sha256"][:12]
                       / job.removeprefix("final_") / record["run_id"])
        if handoff["destination"] != str(_reject_reparse_path(destination)):
            raise PermissionError("Fixed destination handoff mismatch")
        if (handoff["argv"] != expected_args or child_arguments != expected_args
                or list(sys.orig_argv[1:]) != expected_args or not sys.flags.dont_write_bytecode):
            raise PermissionError("Fixed child command mismatch")
        stage = "process handoff"
        binding = _claim_and_await_worker(handoff_path, handoff)
        stage = "source observation"
        source_check = _observe_source(record, repo)
        stage = "environment observation"
        _verify_environment_lock(record["environment"])
        import rasterio
        if record["environment"]["auxiliary_versions"] != {"rasterio": rasterio.__version__}:
            raise PermissionError("Auxiliary runtime version mismatch")
        if datetime.now(timezone.utc) >= deadline:
            raise PermissionError("Supervisor operation deadline expired during context checks")
        return {"run_id": record["run_id"], "approval_reference": record["approval_reference"],
                "source_revision": source_check["revision"], "source_clean": True, "source_check": source_check,
                "supervisor_handoff": {"run_id": record["run_id"], "nonce": handoff["nonce"],
                    "sha256": _sha256_file(handoff_path), "supervisor_pid": handoff["supervisor_pid"], "child_pid": os.getpid()},
                "worker_binding_sha256": _binding_hash(binding, handoff_path.parent),
                "configuration_sha256": record["configuration_sha256"], "environment": record["environment"],
                "working_directory": str(code), "argv": expected_args}
    except Exception as exc:
        raise PermissionError(f"Phase 14 child context is invalid ({type(exc).__name__} at {stage})") from None


def _child_cli(argv: list[str] | None, job: str | None) -> tuple[Any, dict, dict]:
    import argparse
    class SafeParser(argparse.ArgumentParser):
        def error(self, message: str) -> None:
            raise ValueError("Fixed child argument contract is invalid")
    parser = SafeParser(allow_abbrev=False)
    if job is None:
        parser.add_argument("--mode", choices=("primary", "verification"), required=True)
    for flag in ("config", "approval-reference", "expected-revision", "run-id", "expected-config-sha256", "evidence-dir"):
        parser.add_argument("--" + flag, required=True)
    supplied = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(supplied)
    selected = job or "final_" + arguments.mode
    command = (["-B", "-m", "modules.phase14_calibration"] if selected == "calibration"
               else ["-B", "evaluate_rf_final_test.py"]) + supplied
    context = _observe_child_context(selected, command)
    if (arguments.approval_reference != context["approval_reference"] or arguments.run_id != context["run_id"]
            or arguments.expected_revision != context["source_revision"]
            or arguments.expected_config_sha256 != context["configuration_sha256"]
            or Path(arguments.evidence_dir) != _CODE_ROOT / "output" / "phase14" / "evidence" / arguments.run_id):
        raise PermissionError("CLI and observed context disagree")
    handoff = _strict_json((_CODE_ROOT / "output" / "phase14" / "evidence" / arguments.run_id / "supervisor_handoff.json").read_text(encoding="utf-8"))
    authorization = _strict_json(handoff["authorization_json"])
    expected = _fixed_child_arguments(selected, arguments.approval_reference, arguments.expected_revision,
                                      arguments.run_id, arguments.expected_config_sha256, Path(arguments.evidence_dir))
    if command != expected:
        raise PermissionError("Only the fixed child command is permitted")
    return arguments, context, authorization


def main(argv: list[str] | None = None) -> int:
    try:
        arguments, context, authorization = _child_cli(argv, "calibration")
        del authorization
        run_calibration_job(config_path=Path(arguments.config),
                            expected_config_sha256=arguments.expected_config_sha256,
                            execution_context=context)
        return 0
    except BaseException as exc:
        raise SystemExit(f"Phase 14 calibration execution is disabled or failed ({type(exc).__name__})") from None


if __name__ == "__main__":
    raise SystemExit(main())
