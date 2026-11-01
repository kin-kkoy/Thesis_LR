"""Candidate-only, default-deny sigmoid calibration of the accepted frozen RF."""
import argparse
from collections.abc import Callable
from copy import deepcopy
import hashlib
import io
import math
import os
from pathlib import Path
import struct
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import yaml

import train_candidate_rf as training
from modules import phase14_calibration as historical
from modules import feature_pipeline as features

CODE = Path(__file__).resolve().parent
CONFIG = CODE / "config/candidate_cropped_rf_calibration_v1.yaml"
OUTPUT = CODE / "output"
DATASET = training.DATASET
PACKAGE = OUTPUT / (DATASET + "-package")
BASE = OUTPUT / (DATASET + "-rf-estimator-r1")
CALIBRATION_ROWS = 181265
SCHEMA = "candidate_cropped_rf_sigmoid.v1"
PRESERVATION = "candidate_forest_named_state.v1"
BASE_PINS = {
    "model": dict(bytes=1089577, sha256="46e5929bdf6669d4cd6a5b1635d07a7aa5d2e81044e9ed3db3e577a35f8be90d"),
    "manifest": dict(bytes=3881, sha256="5675a588d7b6905194e9c351d0cfd5d589af24cde40e443f7bdaee9eab665452"),
    "configuration": dict(bytes=2808, sha256="32de2d6a31ab85a2177fa4593ea270f4db27c694aa81bf711007c6058befa970"),
}
PUBLIC_PIN = dict(training.PUBLIC_PIN)
PROTECTED = {
    "train_candidate_rf.py": "414d8c714650ad3c4f8c8b0468a25ec12aed6c6d0de67f6514b4a9602bf9ba85",
    "modules/phase14_calibration.py": "4ca4e4c78888d09917c11c58f7035cbbd35f0a3190cd2f28817e59abb00847af",
    "modules/feature_pipeline.py": "b404e82cac952f3111defedf857ecf0b7891214afe4f0876a3bf7617680daeb1",
    "modules/set_c_pilot.py": "69ce00a3cb866c06a6adc9cc4756f9b8007e319865c54e765938195b5c3b3c4a",
}
NODE_FIELDS = dict(left_child="i8", right_child="i8", feature="i8", threshold="f8",
    impurity="f8", n_node_samples="i8", weighted_n_node_samples="f8", missing_go_to_left="u1")
canonical = training.canonical
identity_bytes = training.identity_bytes
strict_json = training.strict_json
ordinary = training.ordinary
file_identity = training.file_identity
write_json = training.write_json


def default_configuration() -> dict:
    return dict(schema_version=SCHEMA, proposal_sha256="440f9bf5a70b3c9b48ee864eb9936e314d29ada8f4ad9673cfce79824ff897bf",
        enabled=False, authorization_reference=None,
        dataset_id=DATASET, package_root=str(PACKAGE), base_package=str(BASE),
        base=deepcopy(BASE_PINS), public_marker=deepcopy(PUBLIC_PIN),
        calibration=dict(rows=CALIBRATION_ROWS, manifest=None, csv=None),
        teacher_id="exploratory_additive_exposure_teacher.r2",
        numerical_contract_id="additive_exposure_quantized_f32_f64.r2",
        feature_names=list(features.CANONICAL_FEATURE_NAMES), feature_dtype="float32",
        target_name="Ignited", target_dtype="int8", positive_label=1,
        procedure=dict(method="sigmoid", cv=None, ensemble="auto", n_jobs=1,
            sample_weight=None, forest_refit=False, method_search=False, threshold=None,
            input_score="raw_probability", optimizer_status="unavailable_through_reused_api"),
        roundtrip=dict(relative_tolerance=1e-10, absolute_tolerance=1e-12),
        environment=dict(training.VERSIONS), resources={k: None for k in training.LIMIT_KEYS},
        destination=None, evidence_dir=None)


def validate_configuration(config: dict, *, active: bool = False) -> None:
    expected, compared = default_configuration(), deepcopy(config)
    if not isinstance(compared, dict) or set(compared) != set(expected):
        raise ValueError("Unknown/missing candidate calibration fields")
    if active:
        if (compared["enabled"] is not True or type(compared["authorization_reference"]) is not str
                or not compared["authorization_reference"].strip()):
            raise PermissionError("Separate calibration execution reference required")
        for name in ("manifest", "csv"):
            pin = compared["calibration"][name]
            if (not isinstance(pin, dict) or set(pin) != {"bytes", "sha256"}
                    or type(pin["bytes"]) is not int or pin["bytes"] <= 0
                    or type(pin["sha256"]) is not str or len(pin["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in pin["sha256"])):
                raise PermissionError("Unresolved calibration identity: " + name)
            compared["calibration"][name] = None
        limits = compared["resources"]
        if not isinstance(limits, dict) or set(limits) != set(training.LIMIT_KEYS):
            raise PermissionError("Exact resource fields required")
        for name, value in limits.items():
            kinds = (int,) if name.endswith("_bytes") else (int, float)
            if type(value) not in kinds or not math.isfinite(value) or value <= 0:
                raise PermissionError("Unresolved/invalid resource: " + name)
        for name in ("enabled", "authorization_reference", "resources", "destination", "evidence_dir"):
            compared[name] = expected[name]
    if canonical(compared) != canonical(expected):
        raise ValueError("Fixed candidate calibration contract changed")


def read_bound(name: str, pin: dict, check: Callable[[str], None]) -> bytes:
    paths = {"model": BASE / "rf_uncalibrated.joblib", "manifest": BASE / "training_manifest.json",
        "configuration": BASE / "resolved_configuration.json", "public": PACKAGE / "COMMITTED.json",
        "calibration_manifest": PACKAGE / "calibration/manifest.json", "calibration_csv": PACKAGE / "calibration/data.csv"}
    if name not in paths:
        raise PermissionError("Input outside candidate calibration allowlist")
    ordinary(paths[name])
    check("before_input")
    raw = paths[name].read_bytes()
    if identity_bytes(raw) != pin:
        raise ValueError("Input identity mismatch: " + name)
    check("after_input")
    return raw


def load_inputs(config: dict, check: Callable[[str], None]) -> tuple[bytes, pd.DataFrame, np.ndarray]:
    manifest = strict_json(read_bound("manifest", config["base"]["manifest"], check))
    resolved = strict_json(read_bound("configuration", config["base"]["configuration"], check))
    if (manifest.get("schema_version") != training.SCHEMA or manifest.get("calibrated") is not False
            or manifest.get("training_roles") != ["train"] or manifest.get("estimator") != training.PARAMS
            or manifest.get("feature_names") != config["feature_names"] or manifest.get("feature_dtype") != "float32"
            or manifest.get("target_dtype") != "int8" or manifest.get("positive_label") != 1
            or manifest.get("dataset", {}).get("id") != DATASET
            or manifest.get("teacher_id") != config["teacher_id"]
            or manifest.get("numerical_contract_id") != config["numerical_contract_id"]
            or manifest.get("files", {}).get("rf_uncalibrated.joblib") != config["base"]["model"]
            or manifest.get("files", {}).get("resolved_configuration.json") != config["base"]["configuration"]
            or manifest.get("configuration_sha256") != hashlib.sha256(canonical(resolved)).hexdigest()
            or resolved.get("protocol") != training.default_configuration()["protocol"]
            or resolved.get("estimator") != training.PARAMS or resolved.get("feature_names") != config["feature_names"]
            or resolved.get("dataset", {}).get("id") != DATASET):
        raise ValueError("Accepted train-only base metadata mismatch")
    marker = strict_json(read_bound("public", config["public_marker"], check))
    if marker != dict(schema_version=training.RELEASE_SCHEMA, manifest_sha256=training.CUSTODIAN_SHA):
        raise ValueError("Dataset commit marker mismatch")
    scoped = strict_json(read_bound("calibration_manifest", config["calibration"]["manifest"], check))
    if (scoped.get("schema_version") != training.RELEASE_SCHEMA or scoped.get("role") != "calibration"
            or type(scoped.get("rows")) is not int or scoped["rows"] != CALIBRATION_ROWS
            or scoped.get("teacher_id") != config["teacher_id"]
            or scoped.get("numerical_contract_id") != config["numerical_contract_id"]
            or scoped.get("files", {}).get("data.csv") != config["calibration"]["csv"]):
        raise ValueError("Calibration role metadata mismatch")
    raw = read_bound("calibration_csv", config["calibration"]["csv"], check)
    header = (",".join((*features.CANONICAL_FEATURE_NAMES, "Ignited")) + "\n").encode("ascii")
    if not raw.startswith(header) or not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("Exact LF-only calibration CSV required")
    matrix, y = np.empty((CALIBRATION_ROWS, 11), dtype=np.float32), np.empty(CALIBRATION_ROWS, dtype=np.int8)
    count = 0
    for line in io.StringIO(raw[len(header):].decode("ascii")):
        columns = line[:-1].split(",")
        if count >= CALIBRATION_ROWS or len(columns) != 12 or columns[-1] not in ("0", "1"):
            raise ValueError("Invalid row/binary label before int8 narrowing")
        with np.errstate(over="raise", invalid="raise"):
            matrix[count] = np.asarray(columns[:11], dtype=np.float32)
        y[count] = int(columns[-1])
        count += 1
        if count % 256 == 0:
            check("input_row")
    if count != CALIBRATION_ROWS or set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("Calibration count/both-class requirement failed")
    features.validate_feature_values(matrix)
    if (not np.all(matrix[:, 2] == 1) or not np.all(np.isin(matrix[:, 4], range(1, 6)))
            or not np.all(np.isin(matrix[:, 8], range(1, 9)))):
        raise ValueError("Candidate all-eligible domain mismatch")
    x = pd.DataFrame(matrix, columns=features.CANONICAL_FEATURE_NAMES, copy=False)
    return read_bound("model", config["base"]["model"], check), x, y


def _normal(value: object) -> object:
    if isinstance(value, np.generic):
        return _normal(value.item())
    if isinstance(value, dict):
        return {k: _normal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("Unsupported parameter representation")


def supplementary_fingerprint(forest: object) -> str:
    """Named logical state, not struct padding/spare capacity/object identity.

    sklearn 1.8 Tree.__getstate__/__reduce__ define the fields below. Each
    primitive field is hashed separately as C-order little-endian bytes, with
    length-framed JSON name/dtype/shape headers. Float bits are not rounded.
    capacity and record offsets/padding are allocation layout, not logical
    prediction state; round-trips may normalize them. Unknown fields fail.
    """
    digest = hashlib.sha256()
    def put(name, value):
        raw = canonical([name, value])
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    def array(name, value):
        a = np.asarray(value)
        if a.dtype.fields is not None or a.dtype.kind not in "iufb":
            raise TypeError("Primitive fitted-state array required: " + name)
        normalized = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
        put(name, [name, normalized.dtype.str, list(normalized.shape), normalized.nbytes])
        digest.update(normalized.tobytes(order="C"))
    put("schema", PRESERVATION)
    put("forest_type", [type(forest).__module__, type(forest).__qualname__])
    put("forest_parameters", _normal(forest.get_params(deep=False)))
    put("feature_names", list(forest.feature_names_in_))
    for name in ("n_features_in_", "n_outputs_", "n_classes_"):
        put("forest." + name, _normal(getattr(forest, name)))
    array("forest.classes_", forest.classes_)
    put("tree_count", len(forest.estimators_))
    for index, estimator in enumerate(forest.estimators_):
        prefix = "tree." + str(index)
        put(prefix + ".type", [type(estimator).__module__, type(estimator).__qualname__])
        put(prefix + ".parameters", _normal(estimator.get_params(deep=False)))
        for name in ("n_features_in_", "n_outputs_", "n_classes_"):
            put(prefix + "." + name, _normal(getattr(estimator, name)))
        array(prefix + ".classes_", estimator.classes_)
        tree, state = estimator.tree_, estimator.tree_.__getstate__()
        if set(state) != {"max_depth", "node_count", "nodes", "values"}:
            raise ValueError("Unsupported Tree state keys")
        for name in ("n_features", "n_outputs", "max_n_classes", "max_depth", "node_count"):
            put(prefix + ".tree." + name, int(getattr(tree, name)))
        array(prefix + ".tree.n_classes", tree.n_classes)
        n = int(tree.node_count)
        nodes, values = state["nodes"], state["values"]
        if (n <= 0 or state["node_count"] != n or state["max_depth"] != tree.max_depth
                or nodes.shape != (n,) or set(nodes.dtype.names or ()) != set(NODE_FIELDS)
                or values.shape != (n, tree.n_outputs, tree.max_n_classes)
                or values.dtype != np.dtype("float64")):
            raise ValueError("Unsupported/inconsistent logical Tree state")
        for name, kind in NODE_FIELDS.items():
            field = nodes[name]
            if field.dtype.kind != kind[0] or field.dtype.itemsize != int(kind[1:]):
                raise ValueError("Unsupported Tree field dtype: " + name)
            array(prefix + ".node." + name, field)
        array(prefix + ".values", values)
    return digest.hexdigest()


def preservation(forest: object) -> dict:
    if type(forest) is not RandomForestClassifier:
        raise TypeError("Expected the accepted RandomForestClassifier")
    training.model_contract(forest)
    return dict(historical=historical.forest_fingerprint(forest),
        supplementary_schema=PRESERVATION, supplementary=supplementary_fingerprint(forest))


def embedded_forest(calibrated: object) -> object:
    features.validate_model_feature_schema(calibrated)
    classes = np.asarray(calibrated.classes_)
    if (classes.shape != (2,) or classes.tolist() != [0, 1]
            or any(isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)) for c in classes)):
        raise ValueError("Calibrated integer class interface mismatch")
    if (type(calibrated) is not CalibratedClassifierCV or calibrated.method != "sigmoid"
            or calibrated.n_jobs != 1 or calibrated.cv is not None or calibrated.ensemble != "auto"
            or type(calibrated.estimator) is not FrozenEstimator
            or len(calibrated.calibrated_classifiers_) != 1):
        raise ValueError("Expected one frozen sigmoid wrapper")
    member = calibrated.calibrated_classifiers_[0]
    if (member.estimator is not calibrated.estimator or len(member.calibrators) != 1
            or not np.array_equal(member.classes, calibrated.classes_)):
        raise ValueError("Calibrated wrapper references diverged")
    coefficients = (member.calibrators[0].a_, member.calibrators[0].b_)
    if not all(math.isfinite(float(v)) for v in coefficients):
        raise ValueError("Nonfinite sigmoid coefficient")
    return calibrated.estimator.estimator


def probability_report(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    y, p = np.asarray(labels), np.asarray(probabilities)
    if (y.dtype != np.dtype("int8") or p.dtype != np.dtype("float64") or y.ndim != 1
            or p.shape != y.shape or set(np.unique(y).tolist()) != {0, 1}
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Invalid diagnostic labels/probabilities")
    def metric(function):
        try:
            value = float(function(y, p))
            if not math.isfinite(value):
                raise ValueError("Nonfinite metric")
            return dict(value=value, status="defined", reason=None)
        except (ValueError, ZeroDivisionError) as error:
            return dict(value=None, status="undefined", reason=str(error))
    bins = np.minimum(np.searchsorted(historical.BIN_EDGES, p, side="right") - 1, 9)
    reliability = []
    for index in range(10):
        chosen = bins == index
        count = int(np.count_nonzero(chosen))
        reliability.append(dict(index=index, lower=historical.BIN_EDGES[index], upper=historical.BIN_EDGES[index+1],
            count=count, mean_predicted_probability=float(p[chosen].mean()) if count else None,
            observed_positive_fraction=float(y[chosen].mean()) if count else None,
            status="defined" if count else "empty"))
    return dict(schema_version=SCHEMA, diagnostic_scope="in-sample calibration-fit diagnostics",
        rows=len(y), positives=int(y.sum()), negatives=len(y)-int(y.sum()), prevalence=float(y.mean()),
        calibrated=True, threshold=None, optimizer_status="unavailable_through_reused_api",
        raw_base_metrics_status="not_computed_here; existing base metrics remain program-reported",
        metrics=dict(average_precision=metric(average_precision_score), roc_auc=metric(roc_auc_score),
            brier_mean_squared_error=metric(lambda target, score: np.mean((score-target)**2)),
            log_loss=metric(lambda target, score: log_loss(target, score, labels=[0, 1]))),
        endpoint_rule="lower inclusive; upper exclusive except bin 9 includes 1.0", reliability_bins=reliability)


def write_audit(path: Path, labels: np.ndarray, probabilities: np.ndarray, check: Callable[[str], None]) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("xb") as stream:
        for index, (label, probability) in enumerate(zip(labels, probabilities, strict=True)):
            raw = canonical(dict(row_index=index, label=int(label),
                probability_f64_le=struct.pack("<d", float(probability)).hex())) + b"\n"
            if stream.write(raw) != len(raw):
                raise OSError("Short audit write")
            digest.update(raw)
            size += len(raw)
            if (index+1) % 256 == 0:
                check("input_row")
        stream.flush()
        os.fsync(stream.fileno())
    expected = dict(bytes=size, sha256=digest.hexdigest())
    if file_identity(path) != expected:
        raise OSError("Audit readback mismatch")
    return expected


def source_identities() -> dict:
    paths = {p: CODE / p for p in PROTECTED}
    paths.update({"calibrate_candidate_rf.py": Path(__file__), "config/candidate_cropped_rf_calibration_v1.yaml": CONFIG})
    result = {p: file_identity(path) for p, path in paths.items()}
    if (any(result[p]["sha256"] != pin for p, pin in PROTECTED.items())
            or Path(training.__file__).resolve() != CODE / "train_candidate_rf.py"
            or Path(historical.__file__).resolve() != CODE / "modules/phase14_calibration.py"
            or Path(features.__file__).resolve() != CODE / "modules/feature_pipeline.py"):
        raise RuntimeError("Protected source identity/origin mismatch")
    return result


def destinations(config: dict) -> tuple[Path, Path, Path]:
    import re
    destination, evidence = (Path(config[k]) for k in ("destination", "evidence_dir"))
    if (any(not p.is_absolute() or ".." in p.parts or p.parent != OUTPUT for p in (destination, evidence))
            or not re.fullmatch(re.escape(BASE.name) + r"-sigmoid-r[1-9][0-9]*", destination.name)
            or evidence != Path(str(destination) + "-evidence")):
        raise PermissionError("Explicit local versioned calibration destinations required")
    stage = Path(str(destination) + "-payload")
    ordinary(OUTPUT, True)
    ordinary(evidence, True)
    if {p.name for p in evidence.iterdir()} != {"worker.stdout.txt", "worker.stderr.txt", "wrapper.receipt.json"}:
        raise ValueError("Fresh wrapper capture reservations required")
    for name in ("worker.stdout.txt", "worker.stderr.txt", "wrapper.receipt.json"):
        ordinary(evidence / name)
    if any(os.path.lexists(p) for p in (stage, destination)):
        raise FileExistsError("Refusing existing calibration stage/package")
    if os.name != "nt" or evidence.stat().st_dev != OUTPUT.stat().st_dev:
        raise PermissionError("Windows same-volume exclusive commit required")
    return stage, destination, evidence


class CalibrationFailure(RuntimeError):
    def __init__(self, report: dict) -> None:
        super().__init__("Candidate calibration incomplete; retain partial evidence")
        self.report = report


def _problem(error: BaseException) -> dict:
    return dict(type=type(error).__name__, message=str(error), traceback="".join(traceback.format_exception(error)))


def run(config: dict, *, execution_authorized: bool = False, expected_configuration_sha256: str | None = None) -> dict:
    report = dict(complete=False, committed=None, original_failure=None, incidents=[], resources=[], stage=None,
        preservation={}, declared_forest_refits=0, calibration_attempts=0)
    guard, original = None, None
    try:
        if execution_authorized is not True:
            raise PermissionError("Separate owner calibration execution authority required")
        validate_configuration(config, active=True)
        digest = hashlib.sha256(canonical(config)).hexdigest()
        if expected_configuration_sha256 != digest:
            raise PermissionError("Independent resolved configuration commitment required")
        report["configuration_sha256"] = digest
        stage, destination, evidence = destinations(config)
        guard = training.Resources(config["resources"], (destination, evidence))
        guard.check("admission")
        report["environment"], report["source_before"] = training.runtime_identity(), source_identities()
        model_bytes, x, y = load_inputs(config, guard.check)
        guard.check("before_model_load")
        base = joblib.load(io.BytesIO(model_bytes))
        del model_bytes
        before = preservation(base)
        report["preservation"]["before"] = before
        report["inputs"] = dict(base=config["base"], public_marker=config["public_marker"],
            calibration=config["calibration"], positives=int(y.sum()), negatives=len(y)-int(y.sum()))
        stage.mkdir()
        report["stage"] = str(stage)
        files = {"resolved_configuration.json": write_json(stage / "resolved_configuration.json", config)}
        guard.check("before_calibration")
        calibration_error = None
        try:
            report["calibration_attempts"] += 1
            calibrated, historical_checks = historical.calibrate_frozen_estimator(base, x, y)
        except BaseException as error:
            calibration_error = error
            raise
        finally:
            try:
                report["preservation"]["after"] = preservation(base)
                if report["preservation"]["after"] != before:
                    raise RuntimeError("Frozen forest state changed during calibration")
            except BaseException as error:
                if calibration_error is None:
                    raise
                report["incidents"].append(_problem(error))
        report["historical_checks"] = historical_checks
        report["preservation"]["embedded"] = preservation(embedded_forest(calibrated))
        if report["preservation"]["embedded"] != before:
            raise RuntimeError("Embedded frozen forest state changed")
        guard.check("after_calibration")
        probabilities = features.predict_positive_probability(calibrated, x)
        diagnostics = probability_report(y, probabilities)
        report["diagnostics"] = diagnostics
        coefficients = calibrated.calibrated_classifiers_[0].calibrators[0]
        report["coefficients_f64_le"] = {name: struct.pack("<d", float(getattr(coefficients, name))).hex()
            for name in ("a_", "b_")}
        files["calibration_predictions.jsonl"] = write_audit(stage / "calibration_predictions.jsonl", y, probabilities, guard.check)
        files["calibration_report.json"] = write_json(stage / "calibration_report.json", diagnostics)
        guard.check("before_serialization")
        model_path = stage / "rf_sigmoid.joblib"
        with model_path.open("xb") as stream:
            joblib.dump(calibrated, stream, compress=0, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        files[model_path.name] = file_identity(model_path)
        raw = model_path.read_bytes()
        if identity_bytes(raw) != files[model_path.name] or not raw:
            raise OSError("Serialized wrapper readback mismatch")
        roundtrip = joblib.load(io.BytesIO(raw))
        del raw
        report["preservation"]["roundtrip"] = preservation(embedded_forest(roundtrip))
        report["preservation"]["original_after_roundtrip"] = preservation(base)
        if any(report["preservation"][name] != before for name in ("roundtrip", "original_after_roundtrip")):
            raise RuntimeError("Serialization changed frozen forest state")
        restored = features.predict_positive_probability(roundtrip, x)
        bit_differences = int(np.count_nonzero(np.ascontiguousarray(probabilities).view(np.uint64)
            != np.ascontiguousarray(restored).view(np.uint64)))
        report["roundtrip_probability_check"] = dict(bit_differences=bit_differences,
            max_absolute_difference=float(np.max(np.abs(probabilities-restored))), **config["roundtrip"])
        if not np.allclose(probabilities, restored, rtol=config["roundtrip"]["relative_tolerance"],
                           atol=config["roundtrip"]["absolute_tolerance"], equal_nan=False):
            raise ValueError("Round-trip probability mismatch")
        if file_identity(BASE / "rf_uncalibrated.joblib") != config["base"]["model"]:
            raise ValueError("Original accepted model bytes changed")
        report["source_after"] = source_identities()
        if report["source_before"] != report["source_after"]:
            raise ValueError("Calibration/helper source changed")
        manifest = dict(schema_version=SCHEMA, proposal_sha256=config["proposal_sha256"],
            status="prepared_for_commit", calibrated=True, threshold=None,
            dataset_id=DATASET, package_root=str(PACKAGE), base_package=str(BASE), base=config["base"],
            inputs=report["inputs"], configuration_sha256=digest, procedure=config["procedure"],
            feature_names=config["feature_names"], feature_dtype="float32", target_dtype="int8", positive_label=1,
            teacher_id=config["teacher_id"], numerical_contract_id=config["numerical_contract_id"],
            preservation=report["preservation"], coefficients_f64_le=report["coefficients_f64_le"],
            roundtrip_probability_check=report["roundtrip_probability_check"], environment=report["environment"],
            sources=report["source_before"], audit_schema="row_index,label,probability_f64_le.v1",
            audit_row_order="calibration/data.csv data rows, zero-based", files=files,
            raw_base_metrics_status="not_computed_here; existing base metrics remain program-reported")
        manifest_id = write_json(stage / "calibration_manifest.json", manifest)
        guard.check("before_commit")
        if os.path.lexists(destination):
            raise FileExistsError("Calibration destination appeared before commit")
        os.rename(stage, destination)
        report["committed"] = dict(path=str(destination), manifest=manifest_id, model=files["rf_sigmoid.joblib"])
        report["complete"] = True
    except BaseException as error:
        original = error
        report["original_failure"] = _problem(error)
    finally:
        if guard is not None:
            try:
                guard.check("final_observation")
            except BaseException as error:
                failure = _problem(error)
                report["incidents"].append(failure)
                if original is None:
                    original, report["original_failure"] = error, failure
            try:
                report["resources"] = deepcopy(guard.samples)
                times = [s["elapsed_seconds"] for s in guard.samples]
                report["max_actual_gap_seconds"] = max((b-a for a,b in zip([0.0]+times, times)), default=None)
                report["resource_policy"] = dict(coverage_end="last attempted final_observation; excludes console/outer receipt work",
                    enforcement="cooperative; bootstrap, blocked calls, native peaks and parent memory are not continuously covered")
            except BaseException as error:
                failure = _problem(error)
                report["incidents"].append(failure)
                if original is None:
                    original, report["original_failure"] = error, failure
        if report["incidents"]:
            report["complete"] = False
    if not report["complete"]:
        raise CalibrationFailure(report) from original
    return report


def load_configuration(path: Path, expected_sha256: str) -> dict:
    if Path(path).absolute() != CONFIG:
        raise PermissionError("Only the assigned disabled calibration configuration may be read")
    ordinary(CONFIG)
    raw = CONFIG.read_bytes()
    if identity_bytes(raw)["sha256"] != expected_sha256:
        raise PermissionError("Calibration configuration identity mismatch")
    config = yaml.load(raw.decode("utf-8"), Loader=training.UniqueLoader)
    validate_configuration(config)
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--authorize-execution", action="store_true")
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--calibration-manifest-json", required=True)
    parser.add_argument("--calibration-csv-json", required=True)
    parser.add_argument("--resources-json", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--expected-resolved-sha256", required=True)
    args = parser.parse_args(argv)
    config = load_configuration(CONFIG, args.expected_file_sha256)
    config.update(enabled=True, authorization_reference=args.approval_reference,
        resources=strict_json(args.resources_json), destination=args.destination, evidence_dir=args.evidence_dir)
    config["calibration"].update(manifest=strict_json(args.calibration_manifest_json), csv=strict_json(args.calibration_csv_json))
    try:
        report = run(config, execution_authorized=args.authorize_execution,
                     expected_configuration_sha256=args.expected_resolved_sha256)
    except CalibrationFailure as error:
        try:
            print(canonical(error.report).decode("ascii"), flush=True)
        except BaseException as reporting:
            error.add_note("Secondary console reporting failure: " + repr(reporting))
        raise
    try:
        print(canonical(report).decode("ascii"), flush=True)
    except BaseException as error:
        report["complete"] = False
        report["incidents"].append(_problem(error))
        raise CalibrationFailure(report) from error


if __name__ == "__main__":
    main()
