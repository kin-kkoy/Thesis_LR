"""Disabled-by-default, train-only RF for the fixed cropped candidate release."""
import argparse
from collections.abc import Callable, Iterable
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
import yaml

from modules import feature_pipeline as features
from modules.set_c_pilot import capture_resource_snapshot

CODE = Path(__file__).resolve().parent
CONFIG = CODE / "config/candidate_cropped_rf_v1.yaml"
OUTPUT = CODE / "output"
DATASET = "candidate-r2-cropped-1024-v1-g364-r1"
PACKAGE = OUTPUT / (DATASET + "-package")
SCHEMA = "candidate_cropped_rf_initial.v1"
RELEASE_SCHEMA = "candidate_teacher_shared_dataset_release.v1"
MASTER_SHA = "c7a92327eb7916e1e1f8ed99b7e178537875c8fe8aecd045ec97055b6738a729"
CUSTODIAN_SHA = "44875949f38d06090c902bb129205ea61c5e7f77e0baa47fe1d0084bfe0a22ca"
PUBLIC_PIN = dict(bytes=150, sha256="76acce9bb029f699766dfc5a2d5ba737806076fafd90a7ff90e5599fdf7f9b10")
ROLE_PINS = {
    "train": dict(rows=227576, manifest=dict(bytes=626,
        sha256="778ded55e5f2b6a7eaa02e8012c9c1422f17315dcb34fd0cdf8c8046e9d94ab4"),
        csv=dict(bytes=16413270, sha256="591ab4b3f7e228654f6359a736df9876265c88ae7cb2b07e1647df93535160cd")),
    "validation": dict(rows=136341, manifest=dict(bytes=630,
        sha256="a85c29a919c892f5a3f2740e94ce887829aab2bba18164f419f28fdaa673966f"),
        csv=dict(bytes=9620367, sha256="6bdd0f1855360c5bb4ff6af50cce774bb46e3a2a5583d4717742d4ee336e2079")),
}
PARAMS = dict(n_estimators=200, max_depth=5, min_samples_split=20, min_samples_leaf=1,
    max_features="sqrt", class_weight=None, criterion="gini", bootstrap=True,
    random_state=42, n_jobs=2, oob_score=False, warm_start=False, max_samples=None,
    max_leaf_nodes=None, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0,
    ccp_alpha=0.0, monotonic_cst=None, verbose=0)
VERSIONS = dict(python="3.14.3", sklearn="1.8.0", numpy="2.4.4", pandas="3.0.2",
    scipy="1.17.1", joblib="1.5.3", pyyaml="6.0.3")
LIMIT_KEYS = ("max_process_rss_bytes", "min_available_physical_memory_bytes",
    "min_free_disk_bytes", "max_wall_seconds", "observation_interval_seconds")
ALLOWED_INPUTS = frozenset({"COMMITTED.json"} | {
    f"development/{role}/{name}" for role in ("train", "validation")
    for name in ("manifest.json", "data.csv")})


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def identity_bytes(data: bytes) -> dict:
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def unique_pairs(items: Iterable[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        if not isinstance(key, str) or key in result:
            raise ValueError("Duplicate/non-string mapping key")
        result[key] = value
    return result


def strict_json(data: bytes | str) -> object:
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(data, object_pairs_hook=unique_pairs, parse_constant=invalid)


class UniqueLoader(yaml.SafeLoader):
    pass


def yaml_mapping(loader: UniqueLoader, node: object) -> dict:
    return unique_pairs((loader.construct_object(k, deep=True),
                         loader.construct_object(v, deep=True)) for k, v in node.value)


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, yaml_mapping)


def default_configuration() -> dict:
    return dict(schema_version=SCHEMA, enabled=False, authorization_reference=None,
        dataset=dict(id=DATASET, package_root=str(PACKAGE), master_sha256=MASTER_SHA,
            custodian_manifest_sha256=CUSTODIAN_SHA, public_marker=deepcopy(PUBLIC_PIN),
            roles=deepcopy(ROLE_PINS)),
        teacher_id="exploratory_additive_exposure_teacher.r2",
        numerical_contract_id="additive_exposure_quantized_f32_f64.r2",
        feature_schema_version=features.SET_C_SCHEMA_VERSION,
        feature_names=list(features.CANONICAL_FEATURE_NAMES), feature_dtype="float32",
        target_name="Ignited", target_version="ignition_t_plus_1.v1", target_dtype="int8",
        positive_label=1, estimator=dict(PARAMS), environment=dict(VERSIONS),
        protocol=dict(fits=1, fit_role="train", prediction_role="validation", search_trials=0,
            refit=False, preprocessing="none", calibration=False, threshold=None,
            metrics=["average_precision", "roc_auc", "brier_mean_squared_error", "log_loss"]),
        resources={key: None for key in LIMIT_KEYS}, destination=None, evidence_dir=None)


def validate_configuration(config: dict, *, active: bool = False) -> None:
    expected = default_configuration()
    compared = deepcopy(config)
    if not isinstance(compared, dict) or set(compared) != set(expected):
        raise ValueError("Unknown/missing configuration fields")
    if active:
        if (compared["enabled"] is not True or type(compared["authorization_reference"]) is not str
                or not compared["authorization_reference"].strip()):
            raise PermissionError("Separate execution reference required")
        limits = compared["resources"]
        if not isinstance(limits, dict) or set(limits) != set(LIMIT_KEYS):
            raise ValueError("Exact resource fields required")
        for key, value in limits.items():
            kinds = (int,) if key.endswith("_bytes") else (int, float)
            if type(value) not in kinds or not math.isfinite(value) or value <= 0:
                raise PermissionError("Unresolved/invalid resource limit: " + key)
        for key in ("enabled", "authorization_reference", "resources", "destination", "evidence_dir"):
            compared[key] = expected[key]
    if canonical(compared) != canonical(expected):
        raise ValueError("Fixed candidate RF contract changed")


def ordinary(path: Path, directory: bool = False) -> None:
    for current in (path, *path.parents):
        s = current.lstat()
        if stat.S_ISLNK(s.st_mode) or getattr(s, "st_file_attributes", 0) & 0x400:
            raise PermissionError("Reparse/symlink path rejected")
        is_directory = directory if current == path else True
        if not (stat.S_ISDIR(s.st_mode) if is_directory else stat.S_ISREG(s.st_mode)):
            raise ValueError("Ordinary file/directory required")


def read_input(relative: str, expected: dict, check: Callable[[str], None]) -> bytes:
    if relative not in ALLOWED_INPUTS:
        raise PermissionError("Input is outside the two-role allowlist")
    path = PACKAGE / relative
    ordinary(path)
    check("before_input_read")
    data = path.read_bytes()
    if identity_bytes(data) != expected:
        raise ValueError("Input identity mismatch: " + relative)
    check("after_input_read")
    return data


def load_development(config: dict, check: Callable[[str], None]) -> tuple[dict, dict]:
    marker = strict_json(read_input("COMMITTED.json", config["dataset"]["public_marker"], check))
    if marker != dict(schema_version=RELEASE_SCHEMA, manifest_sha256=CUSTODIAN_SHA):
        raise ValueError("Public commit marker mismatch")
    partitions, observed = {}, {}
    header = (",".join((*features.CANONICAL_FEATURE_NAMES, "Ignited")) + "\n").encode("ascii")
    for role in ("train", "validation"):
        pin = config["dataset"]["roles"][role]
        manifest = strict_json(read_input(f"development/{role}/manifest.json", pin["manifest"], check))
        if (manifest.get("schema_version") != RELEASE_SCHEMA or manifest.get("role") != role
                or manifest.get("teacher_id") != config["teacher_id"]
                or manifest.get("numerical_contract_id") != config["numerical_contract_id"]
                or type(manifest.get("rows")) is not int or manifest["rows"] != pin["rows"]
                or manifest.get("files", {}).get("data.csv") != pin["csv"]):
            raise ValueError("Role manifest mismatch: " + role)
        data = read_input(f"development/{role}/data.csv", pin["csv"], check)
        if not data.startswith(header) or not data.endswith(b"\n") or b"\r" in data:
            raise ValueError("Exact LF-only twelve-column CSV required")
        matrix = np.empty((pin["rows"], 11), dtype=np.float32)
        labels = np.empty(pin["rows"], dtype=np.int8)
        count = 0
        # The pinned format is unquoted numeric ASCII; reject alternate encodings.
        for line in io.StringIO(data[len(header):].decode("ascii")):
            if count >= pin["rows"]:
                raise ValueError("Unexpected extra CSV row")
            parts = line[:-1].split(",")
            if len(parts) != 12 or parts[-1] not in ("0", "1"):
                raise ValueError("Invalid columns/binary label before int8 narrowing")
            with np.errstate(over="raise", invalid="raise"):
                matrix[count] = np.asarray(parts[:11], dtype=np.float32)
            labels[count] = int(parts[-1])
            count += 1
            if count % 256 == 0:
                check("input_row")
        if count != pin["rows"] or set(map(int, np.unique(labels))) != {0, 1}:
            raise ValueError("Row count/both-class requirement failed")
        features.validate_feature_values(matrix)
        if (not np.all(matrix[:, 2] == 1) or not np.all(np.isin(matrix[:, 4], range(1, 6)))
                or not np.all(np.isin(matrix[:, 8], range(1, 9)))):
            raise ValueError("Candidate all-eligible feature domain mismatch")
        x = pd.DataFrame(matrix, columns=features.CANONICAL_FEATURE_NAMES, copy=False)
        y = pd.Series(labels, name="Ignited", copy=False)
        if x.dtypes.ne(np.dtype("float32")).any() or y.dtype != np.dtype("int8"):
            raise ValueError("Named float32/int8 representation changed")
        partitions[role] = (x, y)
        observed[role] = dict(rows=count, positives=int(labels.sum()), negatives=count-int(labels.sum()),
            csv=pin["csv"], manifest=pin["manifest"])
        del data, matrix, labels
        check("after_role_validation")
    return partitions, observed


class Resources:
    """Cooperative checkpoints, not continuous or preventive enforcement."""
    def __init__(self, limits: dict, destinations: tuple[Path, ...]) -> None:
        self.limits, self.destinations = limits, destinations
        self.started, self.last = time.monotonic(), None
        self.samples, self.failure = [], None

    def check(self, boundary: str) -> None:
        if self.failure is not None:
            raise RuntimeError("Resource failure is latched") from self.failure
        try:
            elapsed = time.monotonic() - self.started
            if not math.isfinite(elapsed) or not 0 <= elapsed <= self.limits["max_wall_seconds"]:
                raise RuntimeError("Worker wall/clock limit")
            if (boundary == "input_row" and self.last is not None
                    and elapsed-self.last < self.limits["observation_interval_seconds"]):
                return
            for destination in self.destinations:
                s = capture_resource_snapshot(destination)
                values = (s.process_rss_bytes, s.available_memory_bytes, s.free_disk_bytes)
                elapsed = time.monotonic() - self.started
                if (any(type(v) is not int or v <= 0 for v in values) or not math.isfinite(elapsed)
                        or not 0 <= elapsed <= self.limits["max_wall_seconds"]):
                    raise RuntimeError("Invalid resource observation/wall limit")
                self.samples.append(dict(boundary=boundary, elapsed_seconds=elapsed,
                    destination=str(destination), rss_bytes=values[0], available_bytes=values[1], free_disk_bytes=values[2]))
                self.last = elapsed
                if (values[0] > self.limits["max_process_rss_bytes"]
                        or values[1] < self.limits["min_available_physical_memory_bytes"]
                        or values[2] < self.limits["min_free_disk_bytes"]):
                    raise RuntimeError("Resource ceiling/floor breached")
        except BaseException as error:
            self.failure = error
            raise


def runtime_identity() -> dict:
    versions = dict(python=".".join(map(str, sys.version_info[:3])), sklearn=sklearn.__version__,
        numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
        joblib=joblib.__version__, pyyaml=yaml.__version__)
    if sys.implementation.name != "cpython" or versions != VERSIONS:
        raise RuntimeError("Pinned training runtime mismatch")
    return dict(versions=versions, executable=sys.executable)


def file_identity(path: Path) -> dict:
    ordinary(path)
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
            size += len(block)
    return dict(bytes=size, sha256=h.hexdigest())


def source_identities() -> dict:
    paths = {"train_candidate_rf.py": Path(__file__), "config/candidate_cropped_rf_v1.yaml": CONFIG,
             "modules/feature_pipeline.py": Path(features.__file__),
             "modules/set_c_pilot.py": CODE / "modules/set_c_pilot.py"}
    result = {name: file_identity(path) for name, path in paths.items()}
    if (Path(features.__file__).resolve() != CODE / "modules/feature_pipeline.py"
            or result["modules/feature_pipeline.py"]["sha256"] != "b404e82cac952f3111defedf857ecf0b7891214afe4f0876a3bf7617680daeb1"
            or result["modules/set_c_pilot.py"]["sha256"] != "69ce00a3cb866c06a6adc9cc4756f9b8007e319865c54e765938195b5c3b3c4a"):
        raise RuntimeError("Protected helper identity/origin mismatch")
    return result


def destinations(config: dict) -> tuple[Path, Path, Path]:
    destination, evidence = (Path(config[k]) for k in ("destination", "evidence_dir"))
    for p in (destination, evidence):
        if not p.is_absolute() or ".." in p.parts or p.parent != OUTPUT:
            raise PermissionError("Explicit local versioned output paths required")
    if (not re.fullmatch(re.escape(DATASET) + r"-rf-estimator-r[1-9][0-9]*", destination.name)
            or evidence != Path(str(destination) + "-evidence")):
        raise PermissionError("Versioned model/evidence identity mismatch")
    stage = Path(str(destination) + "-payload")
    ordinary(OUTPUT, True)
    ordinary(evidence, True)
    names = {p.name for p in evidence.iterdir()}
    if names != {"worker.stdout.txt", "worker.stderr.txt", "wrapper.receipt.json"}:
        raise ValueError("Fresh wrapper capture reservations required")
    for name in names:
        ordinary(evidence / name)
    if any(os.path.lexists(p) for p in (stage, destination)):
        raise FileExistsError("Refusing existing stage/model package")
    if os.name != "nt" or evidence.stat().st_dev != OUTPUT.stat().st_dev:
        raise PermissionError("Windows same-volume exclusive commit required")
    return stage, destination, evidence


def model_contract(model: object) -> None:
    features.validate_model_feature_schema(model)
    classes = np.asarray(getattr(model, "classes_", None), dtype=object)
    if (classes.shape != (2,) or any(not isinstance(c, (int, np.integer)) or isinstance(c, (bool, np.bool_)) for c in classes)
            or set(map(int, classes)) != {0, 1} or getattr(model, "n_features_in_", None) != 11
            or not callable(getattr(model, "predict_proba", None))
            or canonical(model.get_params(deep=False)) != canonical(PARAMS)):
        raise ValueError("Exact binary RF interface/settings required")
    features.positive_class_index(classes)


def write_json(path: Path, value: object) -> dict:
    data = canonical(value) + b"\n"
    with path.open("xb") as stream:
        if stream.write(data) != len(data):
            raise OSError("Short report write")
        stream.flush()
        os.fsync(stream.fileno())
    if file_identity(path) != identity_bytes(data):
        raise OSError("Report readback mismatch")
    return identity_bytes(data)


class TrainingFailure(RuntimeError):
    def __init__(self, report: dict) -> None:
        super().__init__("Candidate RF training incomplete; retain partial evidence")
        self.report = report


def run(config: dict, *, execution_authorized: bool = False, expected_configuration_sha256: str | None = None) -> dict:
    report = dict(complete=False, committed=None, original_failure=None, incidents=[], resources=[],
                  stage=None, configuration_sha256=None, fit_calls=0, prediction_calls=0)
    guard = None
    original = None
    try:
        if execution_authorized is not True:
            raise PermissionError("Separate owner execution authority required")
        validate_configuration(config, active=True)
        digest = hashlib.sha256(canonical(config)).hexdigest()
        if expected_configuration_sha256 != digest:
            raise PermissionError("Independent resolved configuration commitment required")
        report["configuration_sha256"] = digest
        stage, destination, evidence = destinations(config)
        guard = Resources(config["resources"], (destination, evidence))
        guard.check("admission")
        report["environment"], report["source_before"] = runtime_identity(), source_identities()
        partitions, observed = load_development(config, guard.check)
        report["inputs"] = observed
        guard.check("before_stage")
        stage.mkdir()
        report["stage"] = str(stage)
        files = {"resolved_configuration.json": write_json(stage / "resolved_configuration.json", config)}
        model = RandomForestClassifier(**config["estimator"])
        if hasattr(model, "classes_"):
            raise ValueError("A fresh unfitted estimator is required")
        x_train, y_train = partitions["train"]
        x_validation, y_validation = partitions["validation"]
        guard.check("before_fit")
        report["fit_calls"] += 1
        if model.fit(x_train, y_train) is not model:
            raise ValueError("RF fit returned another estimator")
        guard.check("after_fit")
        model_contract(model)
        report["prediction_calls"] += 1
        probabilities = features.predict_positive_probability(model, x_validation)
        guard.check("after_validation_prediction")
        target = y_validation.to_numpy(dtype=np.float64)
        metrics = dict(role="validation", calibrated=False, threshold=None,
            rows=len(target), positives=int(y_validation.sum()), negatives=len(target)-int(y_validation.sum()),
            positive_fraction=float(np.mean(target)), average_precision=float(average_precision_score(target, probabilities)),
            roc_auc=float(roc_auc_score(target, probabilities)),
            brier_mean_squared_error=float(np.mean((probabilities-target)**2)),
            log_loss=float(log_loss(target, probabilities, labels=[0, 1])))
        if any(not math.isfinite(metrics[k]) for k in ("average_precision", "roc_auc", "brier_mean_squared_error", "log_loss")):
            raise ValueError("Undefined/nonfinite validation metric")
        report["metrics"] = metrics
        guard.check("before_serialization")
        with (stage / "rf_uncalibrated.joblib").open("xb") as stream:
            joblib.dump(model, stream, compress=0, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        files["rf_uncalibrated.joblib"] = file_identity(stage / "rf_uncalibrated.joblib")
        if files["rf_uncalibrated.joblib"]["bytes"] == 0:
            raise OSError("Empty serialized model")
        guard.check("after_serialization")
        files["validation_metrics.json"] = write_json(stage / "validation_metrics.json", metrics)
        report["source_after"] = source_identities()
        if report["source_after"] != report["source_before"]:
            raise ValueError("Training/helper source changed")
        manifest = dict(schema_version=SCHEMA, calibrated=False, training_roles=["train"],
            configuration_sha256=digest, dataset=config["dataset"], inputs=observed,
            upstream_identity_basis="Owner-accepted master/custodian identities; neither file opened",
            estimator=dict(PARAMS), feature_names=list(features.CANONICAL_FEATURE_NAMES),
            feature_dtype="float32", target_dtype="int8", positive_label=1,
            teacher_id=config["teacher_id"], numerical_contract_id=config["numerical_contract_id"],
            protocol=config["protocol"], environment=report["environment"], sources=report["source_before"], files=files)
        manifest_id = write_json(stage / "training_manifest.json", manifest)
        guard.check("before_completion")
        write_json(stage / "completion.json", dict(schema_version=SCHEMA, training_manifest=manifest_id,
            model=files["rf_uncalibrated.joblib"], status="prepared_for_commit",
            not_final_completion_evidence=True))
        guard.check("before_commit")
        if os.path.lexists(destination):
            raise FileExistsError("Destination appeared before commit")
        os.rename(stage, destination)
        report["committed"] = dict(path=str(destination), training_manifest=manifest_id,
                                    model=files["rf_uncalibrated.joblib"])
        report["complete"] = True
    except BaseException as error:
        original = error
        report["original_failure"] = dict(type=type(error).__name__, message=str(error),
            traceback="".join(traceback.format_exception(error)))
    finally:
        if guard is not None:
            try:
                guard.check("final_observation")
            except BaseException as error:
                failure = dict(type=type(error).__name__, message=str(error),
                    traceback="".join(traceback.format_exception(error)))
                report["incidents"].append(failure)
                if original is None:
                    original = error
                    report["original_failure"] = failure
            report["resources"] = deepcopy(guard.samples)
            report["resource_policy"] = dict(coverage_start="after authorization/configuration/destination checks",
                coverage_end="last attempted final_observation; excludes console/outer receipt work",
                enforcement="cooperative; native/blocked calls and transient peaks are not continuously covered",
                nominal_interval_seconds=config["resources"]["observation_interval_seconds"])
            times = [s["elapsed_seconds"] for s in guard.samples]
            report["max_actual_gap_seconds"] = max((b-a for a,b in zip([0.0]+times,times)), default=None)
        if report["incidents"]:
            report["complete"] = False
    if not report["complete"]:
        raise TrainingFailure(report) from original
    return report


def load_configuration(path: Path, expected_sha256: str) -> dict:
    if Path(path).absolute() != CONFIG:
        raise PermissionError("Only the assigned disabled configuration may be read")
    ordinary(CONFIG)
    data = CONFIG.read_bytes()
    if identity_bytes(data)["sha256"] != expected_sha256:
        raise PermissionError("Configuration file identity mismatch")
    config = yaml.load(data.decode("utf-8"), Loader=UniqueLoader)
    validate_configuration(config)
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--authorize-execution", action="store_true")
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--resources-json", required=True)
    parser.add_argument("--expected-resolved-sha256", required=True)
    args = parser.parse_args(argv)
    config = load_configuration(args.config, args.expected_file_sha256)
    config.update(enabled=True, authorization_reference=args.approval_reference,
        destination=args.destination, evidence_dir=args.evidence_dir, resources=strict_json(args.resources_json))
    try:
        report = run(config, execution_authorized=args.authorize_execution,
                     expected_configuration_sha256=args.expected_resolved_sha256)
    except TrainingFailure as error:
        try:
            print(canonical(error.report).decode("ascii"), flush=True)
        except BaseException as reporting:
            error.add_note("Secondary console reporting failure: " + repr(reporting))
        raise
    try:
        print(canonical(report).decode("ascii"), flush=True)
    except BaseException as error:
        report["complete"] = False
        report["incidents"].append(dict(type=type(error).__name__, message=str(error)))
        raise TrainingFailure(report) from error


if __name__ == "__main__":
    main()
