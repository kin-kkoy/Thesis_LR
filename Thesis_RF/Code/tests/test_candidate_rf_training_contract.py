"""Synthetic contracts only: recording estimators and byte-writing substitutes."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import train_candidate_rf as rf


def pin(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"


def active_config():
    config = rf.default_configuration()
    destination = rf.OUTPUT / (rf.DATASET + "-rf-estimator-r1")
    config.update(enabled=True, authorization_reference="SYNTHETIC-NOT-EXECUTION-AUTHORITY",
        destination=str(destination), evidence_dir=str(destination) + "-evidence",
        resources=dict(max_process_rss_bytes=4096, min_available_physical_memory_bytes=2,
            min_free_disk_bytes=2, max_wall_seconds=60, observation_interval_seconds=1.0))
    return config


def invoke(config):
    return rf.run(config, execution_authorized=True,
                  expected_configuration_sha256=hashlib.sha256(rf.canonical(config)).hexdigest())


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    package, output = tmp_path / "package", tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(rf, "PACKAGE", package)
    monkeypatch.setattr(rf, "OUTPUT", output)
    state = dict(models=[], serializations=[])
    role_pins = {}
    header = ",".join((*rf.features.CANONICAL_FEATURE_NAMES, "Ignited")) + "\n"
    for role, proximity in (("train", "0.3"), ("validation", "0.7")):
        folder = package / "development" / role
        folder.mkdir(parents=True)
        lines = [f"-0,{proximity},1,0.95,1,10,0,1,1,0.95,0.25,{label}\n" for label in (0, 1, 0, 1)]
        data = (header + "".join(lines)).encode("ascii")
        (folder / "data.csv").write_bytes(data)
        manifest = dict(schema_version=rf.RELEASE_SCHEMA, role=role, rows=4,
            teacher_id="exploratory_additive_exposure_teacher.r2",
            numerical_contract_id="additive_exposure_quantized_f32_f64.r2",
            custody="synthetic substitute", files={"data.csv": pin(data),
                "membership.jsonl": {"path": "../../custodian/final_test/data.csv"},
                "probabilities.jsonl": {"path": "../../calibration/data.csv"}})
        raw = encoded(manifest)
        (folder / "manifest.json").write_bytes(raw)
        role_pins[role] = dict(rows=4, csv=pin(data), manifest=pin(raw))
    marker = encoded(dict(schema_version=rf.RELEASE_SCHEMA, manifest_sha256=rf.CUSTODIAN_SHA))
    (package / "COMMITTED.json").write_bytes(marker)
    monkeypatch.setattr(rf, "PUBLIC_PIN", pin(marker))
    monkeypatch.setattr(rf, "ROLE_PINS", role_pins)
    destination = output / (rf.DATASET + "-rf-estimator-r1")
    evidence = Path(str(destination) + "-evidence")
    evidence.mkdir()
    for name in ("worker.stdout.txt", "worker.stderr.txt", "wrapper.receipt.json"):
        (evidence / name).touch()

    class RecordingRF:
        def __init__(self, **kwargs):
            self.params, self.fits, self.predictions = dict(kwargs), [], []
            state["models"].append(self)

        def fit(self, x, y):
            self.fits.append((x.copy(), y.copy()))
            self.feature_names_in_ = np.asarray(x.columns, dtype=object)
            self.n_features_in_ = 11
            self.classes_ = np.array([1, 0], dtype=np.int8)
            return self

        def get_params(self, deep=False):
            return dict(self.params)

        def predict_proba(self, x):
            self.predictions.append(x.copy())
            positive = np.array([0.25, 0.75, 0.1, 0.9], dtype=np.float64)
            return np.column_stack((positive, 1-positive))

    def dump(model, stream, *, compress, protocol):
        state["serializations"].append((model, compress, protocol))
        stream.write(b"synthetic-estimator-substitute\n")

    monkeypatch.setattr(rf, "RandomForestClassifier", RecordingRF)
    monkeypatch.setattr(rf.joblib, "dump", dump)
    monkeypatch.setattr(rf, "runtime_identity", lambda: {"synthetic_substitute": True})
    monkeypatch.setattr(rf, "capture_resource_snapshot", lambda _: SimpleNamespace(
        process_rss_bytes=1024, available_memory_bytes=8192, free_disk_bytes=8192))
    state.update(package=package, output=output, destination=destination,
                 stage=Path(str(destination) + "-payload"), evidence=evidence, estimator=RecordingRF)
    return state


def replace_csv(prepared, role, data):
    folder = prepared["package"] / "development" / role
    (folder / "data.csv").write_bytes(data)
    manifest = json.loads((folder / "manifest.json").read_bytes())
    manifest["files"]["data.csv"] = pin(data)
    raw = encoded(manifest)
    (folder / "manifest.json").write_bytes(raw)
    rf.ROLE_PINS[role].update(csv=pin(data), manifest=pin(raw))


def test_disabled_yaml_and_independent_fixed_estimator_contract():
    config = yaml.load(rf.CONFIG.read_text(encoding="utf-8"), Loader=rf.UniqueLoader)
    rf.validate_configuration(config)
    assert config == rf.default_configuration()
    assert config["enabled"] is False and config["authorization_reference"] is None
    assert config["destination"] is None and config["evidence_dir"] is None
    assert all(value is None for value in config["resources"].values())
    assert config["estimator"] == dict(n_estimators=200, max_depth=5, min_samples_split=20,
        min_samples_leaf=1, max_features="sqrt", class_weight=None, criterion="gini", bootstrap=True,
        random_state=42, n_jobs=2, oob_score=False, warm_start=False, max_samples=None,
        max_leaf_nodes=None, min_weight_fraction_leaf=0.0, min_impurity_decrease=0.0,
        ccp_alpha=0.0, monotonic_cst=None, verbose=0)
    assert config["protocol"]["fits"] == 1 and config["protocol"]["search_trials"] == 0
    assert config["protocol"]["refit"] is False and config["protocol"]["calibration"] is False
    assert config["protocol"]["preprocessing"] == "none" and config["protocol"]["threshold"] is None


@pytest.mark.parametrize("fault", ["unauthorized", "disabled", "reference", "commitment", "resource_missing",
    "resource_bool", "resource_unknown", "contract", "unexpected_field", "path", "collision"])
def test_execution_gates_precede_input_access(prepared, monkeypatch, fault):
    config = active_config()
    if fault == "disabled":
        config["enabled"] = False
    elif fault == "reference":
        config["authorization_reference"] = None
    elif fault == "resource_missing":
        config["resources"]["max_wall_seconds"] = None
    elif fault == "resource_bool":
        config["resources"]["max_process_rss_bytes"] = True
    elif fault == "resource_unknown":
        config["resources"]["extra"] = 1
    elif fault == "contract":
        config["estimator"]["n_jobs"] = -1
    elif fault == "unexpected_field":
        config["search"] = {}
    elif fault == "path":
        config["destination"] = str(prepared["output"].parent / "outside")
    elif fault == "collision":
        prepared["destination"].mkdir()
    calls = []
    def forbidden(*args):
        calls.append(args)
        raise AssertionError("Input access before gates")
    monkeypatch.setattr(rf, "read_input", forbidden)
    expected = "0"*64 if fault == "commitment" else hashlib.sha256(rf.canonical(config)).hexdigest()
    with pytest.raises(rf.TrainingFailure):
        rf.run(config, execution_authorized=fault != "unauthorized", expected_configuration_sha256=expected)
    assert not calls and not prepared["models"] and not prepared["stage"].exists()


def test_only_five_inputs_preserve_named_float32_and_int8(prepared, monkeypatch):
    seen, original = [], rf.read_input
    filesystem_reads, original_open = [], Path.open
    def guarded_open(path, *args, **kwargs):
        if path.is_relative_to(prepared["package"]):
            relative = path.relative_to(prepared["package"]).as_posix()
            assert relative in rf.ALLOWED_INPUTS
            filesystem_reads.append(relative)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    def tracking(relative, expected, check):
        seen.append(relative)
        return original(relative, expected, check)
    monkeypatch.setattr(rf, "read_input", tracking)
    partitions, observed = rf.load_development(active_config(), lambda _: None)
    assert seen == ["COMMITTED.json", "development/train/manifest.json", "development/train/data.csv",
                    "development/validation/manifest.json", "development/validation/data.csv"]
    assert filesystem_reads == seen
    for role, (x, y) in partitions.items():
        assert x.shape == (4, 11) and tuple(x.columns) == rf.features.CANONICAL_FEATURE_NAMES
        assert x.dtypes.eq(np.dtype("float32")).all() and y.dtype == np.dtype("int8")
        assert np.signbit(x.iloc[0, 0]) and y.tolist() == [0, 1, 0, 1]
        assert observed[role]["positives"] == observed[role]["negatives"] == 2
    assert not prepared["models"]


@pytest.mark.parametrize("relative", ["development/train/membership.jsonl", "development/train/probabilities.jsonl",
    "development/validation/membership.jsonl", "development/validation/probabilities.jsonl",
    "calibration/data.csv", "custodian/final_test/data.csv", "custodian/master.csv"])
def test_forbidden_inputs_reject_before_filesystem(monkeypatch, relative):
    def forbidden(*args):
        raise AssertionError("Filesystem reached for forbidden input")
    monkeypatch.setattr(rf, "ordinary", forbidden)
    with pytest.raises(PermissionError, match="allowlist"):
        rf.read_input(relative, {}, forbidden)


@pytest.mark.parametrize("role", ["train", "validation"])
@pytest.mark.parametrize("name", ["manifest.json", "data.csv"])
def test_independent_input_pin_rejects_modified_bytes(prepared, role, name):
    path = prepared["package"] / "development" / role / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        rf.load_development(active_config(), lambda _: None)


def test_manifest_reference_cannot_replace_independent_csv_pin(prepared):
    folder = prepared["package"] / "development/train"
    raw = (folder / "data.csv").read_bytes() + b"\n"
    (folder / "data.csv").write_bytes(raw)
    manifest = json.loads((folder / "manifest.json").read_bytes())
    manifest["files"]["data.csv"] = pin(raw)
    changed = encoded(manifest)
    (folder / "manifest.json").write_bytes(changed)
    rf.ROLE_PINS["train"]["manifest"] = pin(changed)
    with pytest.raises(ValueError, match="manifest mismatch"):
        rf.load_development(active_config(), lambda _: None)


@pytest.mark.parametrize("fault", ["label_fraction", "label_wrap", "label_bool", "nan", "overflow", "material",
    "nonreceiving", "zero_neighbours", "missing_row", "extra_row", "blank_row", "wrong_header", "extra_column",
    "crlf", "bom", "one_class"])
def test_malformed_rows_are_rejected_without_filtering(prepared, fault):
    path = prepared["package"] / "development/train/data.csv"
    rows = path.read_text(encoding="ascii").splitlines()
    columns = rows[1].split(",")
    if fault in ("label_fraction", "label_wrap", "label_bool"):
        columns[-1] = {"label_fraction": "1.5", "label_wrap": "256", "label_bool": "True"}[fault]
    elif fault in ("nan", "overflow"):
        columns[1] = "nan" if fault == "nan" else "1e999"
    elif fault == "material":
        columns[4] = "1.5"
    elif fault == "nonreceiving":
        for i in (2, 3, 4, 9):
            columns[i] = "0"
    elif fault == "zero_neighbours":
        columns[8] = "0"
    elif fault == "extra_column":
        columns.append("0")
    rows[1] = ",".join(columns)
    if fault == "missing_row":
        rows.pop()
    elif fault == "extra_row":
        rows.append(rows[1])
    elif fault == "blank_row":
        rows.insert(2, "")
    elif fault == "wrong_header":
        rows[0] = rows[0].replace("slope_risk,proximity_risk", "proximity_risk,slope_risk")
    elif fault == "one_class":
        rows[1:] = [line.rsplit(",", 1)[0] + ",0" for line in rows[1:]]
    data = ("\n".join(rows) + "\n").encode("ascii")
    if fault == "crlf":
        data = data.replace(b"\n", b"\r\n")
    elif fault == "bom":
        data = b"\xef\xbb\xbf" + data
    replace_csv(prepared, "train", data)
    with pytest.raises((ValueError, FloatingPointError)):
        rf.load_development(active_config(), lambda _: None)
    assert not prepared["models"]


@pytest.mark.parametrize("field", ["teacher_id", "numerical_contract_id", "role", "rows", "schema_version", "marker"])
def test_manifest_and_marker_semantics_are_required(prepared, field):
    if field == "marker":
        raw = encoded(dict(schema_version=rf.RELEASE_SCHEMA, manifest_sha256="0"*64))
        (prepared["package"] / "COMMITTED.json").write_bytes(raw)
        rf.PUBLIC_PIN.update(pin(raw))
    else:
        path = prepared["package"] / "development/train/manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest[field] = 5 if field == "rows" else "wrong"
        raw = encoded(manifest)
        path.write_bytes(raw)
        rf.ROLE_PINS["train"]["manifest"] = pin(raw)
    with pytest.raises(ValueError):
        rf.load_development(active_config(), lambda _: None)


def test_one_train_fit_one_validation_prediction_and_bound_artifacts(prepared):
    config = active_config()
    report = invoke(config)
    assert report["complete"] is True and report["fit_calls"] == report["prediction_calls"] == 1
    assert len(prepared["models"]) == 1
    model = prepared["models"][0]
    assert len(model.fits) == len(model.predictions) == 1
    x, y = model.fits[0]
    assert x.shape == (4, 11) and y.tolist() == [0, 1, 0, 1]
    assert np.all(x["proximity_risk"].to_numpy() == np.float32(0.3))
    assert np.all(model.predictions[0]["proximity_risk"].to_numpy() == np.float32(0.7))
    assert np.signbit(x.iloc[0, 0]) and x.dtypes.eq(np.dtype("float32")).all()
    assert y.dtype == np.dtype("int8") and prepared["serializations"] == [(model, 0, 5)]
    metrics = report["metrics"]
    assert metrics["average_precision"] == metrics["roc_auc"] == 1.0
    assert metrics["brier_mean_squared_error"] == pytest.approx(0.03625)
    assert metrics["log_loss"] == pytest.approx(-(math.log(0.75)+math.log(0.9))/2)
    assert metrics["threshold"] is None and metrics["calibrated"] is False
    root = prepared["destination"]
    assert {p.name for p in root.iterdir()} == {"rf_uncalibrated.joblib", "training_manifest.json",
        "validation_metrics.json", "resolved_configuration.json", "completion.json"}
    manifest = json.loads((root / "training_manifest.json").read_bytes())
    for name, identity in manifest["files"].items():
        assert pin((root / name).read_bytes()) == identity
    assert manifest["training_roles"] == ["train"] and manifest["protocol"]["refit"] is False
    assert manifest["estimator"]["class_weight"] is None and manifest["calibrated"] is False
    assert manifest["configuration_sha256"] == hashlib.sha256(rf.canonical(config)).hexdigest()
    completion = json.loads((root / "completion.json").read_bytes())
    assert completion["training_manifest"] == pin((root / "training_manifest.json").read_bytes())
    assert completion["not_final_completion_evidence"] is True
    assert completion["model"] == pin(b"synthetic-estimator-substitute\n")
    assert not prepared["stage"].exists()


@pytest.mark.parametrize("fault", ["classes_float", "classes_bool", "classes_missing", "classes_duplicate",
    "features", "feature_count", "params", "no_predict"])
def test_invalid_binary_estimator_interfaces_fail(prepared, monkeypatch, fault):
    original = prepared["estimator"].fit
    def fit(model, x, y):
        original(model, x, y)
        if fault.startswith("classes_"):
            model.classes_ = {"classes_float": [1.0, 0.0], "classes_bool": [True, False],
                "classes_missing": [0], "classes_duplicate": [1, 1]}[fault]
        elif fault == "features":
            model.feature_names_in_ = model.feature_names_in_[::-1]
        elif fault == "feature_count":
            model.n_features_in_ = 12
        elif fault == "params":
            model.params["warm_start"] = True
        else:
            model.predict_proba = None
        return model
    monkeypatch.setattr(prepared["estimator"], "fit", fit)
    with pytest.raises(rf.TrainingFailure) as caught:
        invoke(active_config())
    assert caught.value.report["prediction_calls"] == 0
    assert not prepared["destination"].exists() and not prepared["serializations"]


@pytest.mark.parametrize("fault", ["nan", "range", "shape", "rows"])
def test_invalid_prediction_results_fail(prepared, monkeypatch, fault):
    original = prepared["estimator"].predict_proba
    def predict(model, x):
        result = original(model, x)
        if fault == "nan":
            result[0, 0] = np.nan
        elif fault == "range":
            result[0, 0] = 1.1
        elif fault == "shape":
            result = result[:, 0]
        else:
            result = result[:-1]
        return result
    monkeypatch.setattr(prepared["estimator"], "predict_proba", predict)
    with pytest.raises(rf.TrainingFailure):
        invoke(active_config())
    assert not prepared["destination"].exists() and not prepared["serializations"]


@pytest.mark.parametrize("boundary", ["fit", "dump", "rename", "final_observation"])
def test_failures_retain_partial_outputs_and_committed_identity(prepared, monkeypatch, boundary):
    error = OSError("injected " + boundary)
    if boundary == "fit":
        def fit(*args):
            raise error
        monkeypatch.setattr(prepared["estimator"], "fit", fit)
    elif boundary == "dump":
        def dump(model, stream, **kwargs):
            stream.write(b"partial")
            raise error
        monkeypatch.setattr(rf.joblib, "dump", dump)
    elif boundary == "rename":
        def rename(*args):
            raise error
        monkeypatch.setattr(rf.os, "rename", rename)
    else:
        original = rf.Resources.check
        def check(guard, label):
            original(guard, label)
            if label == "final_observation":
                raise error
        monkeypatch.setattr(rf.Resources, "check", check)
    with pytest.raises(rf.TrainingFailure) as caught:
        invoke(active_config())
    report = caught.value.report
    assert caught.value.__cause__ is error and report["complete"] is False
    assert report["original_failure"]["message"] == str(error)
    if boundary == "final_observation":
        assert report["committed"] is not None and prepared["destination"].is_dir()
        assert report["resources"][-1]["boundary"] == "final_observation"
        assert not prepared["stage"].exists()
    else:
        assert report["committed"] is None and prepared["stage"].is_dir()
        assert (prepared["stage"] / "resolved_configuration.json").is_file()
        assert not prepared["destination"].exists()
    if boundary == "dump":
        assert (prepared["stage"] / "rf_uncalibrated.joblib").read_bytes() == b"partial"
    assert {p.name for p in prepared["evidence"].iterdir()} == {"worker.stdout.txt", "worker.stderr.txt", "wrapper.receipt.json"}


def test_original_failure_has_priority_over_final_observation_failure(prepared, monkeypatch):
    first, later = OSError("serialization original"), RuntimeError("observation secondary")
    def dump(model, stream, **kwargs):
        stream.write(b"retained")
        raise first
    original = rf.Resources.check
    def check(guard, label):
        if label == "final_observation":
            raise later
        return original(guard, label)
    monkeypatch.setattr(rf.joblib, "dump", dump)
    monkeypatch.setattr(rf.Resources, "check", check)
    with pytest.raises(rf.TrainingFailure) as caught:
        invoke(active_config())
    assert caught.value.__cause__ is first
    assert caught.value.report["original_failure"]["message"] == str(first)
    assert caught.value.report["incidents"][-1]["message"] == str(later)
    assert (prepared["stage"] / "rf_uncalibrated.joblib").read_bytes() == b"retained"


def test_unassigned_configuration_path_rejects_before_filesystem(monkeypatch, tmp_path):
    def forbidden(*args):
        raise AssertionError("Unassigned configuration access")
    monkeypatch.setattr(rf, "ordinary", forbidden)
    with pytest.raises(PermissionError):
        rf.load_configuration(tmp_path / "unassigned.yaml", "0"*64)


@pytest.mark.parametrize("fault", ["invalid", "rss", "available", "disk", "wall"])
def test_resource_observations_fail_closed_and_latch(monkeypatch, tmp_path, fault):
    limits = dict(max_process_rss_bytes=4096, min_available_physical_memory_bytes=2,
        min_free_disk_bytes=2, max_wall_seconds=60, observation_interval_seconds=1.0)
    values = dict(process_rss_bytes=1024, available_memory_bytes=8192, free_disk_bytes=8192)
    if fault == "invalid":
        values["process_rss_bytes"] = None
    elif fault == "rss":
        values["process_rss_bytes"] = 8192
    elif fault == "available":
        values["available_memory_bytes"] = 1
    elif fault == "disk":
        values["free_disk_bytes"] = 1
    probes = []
    def probe(path):
        probes.append(path)
        return SimpleNamespace(**values)
    monkeypatch.setattr(rf, "capture_resource_snapshot", probe)
    guard = rf.Resources(limits, (tmp_path,))
    if fault == "wall":
        guard.started -= 1000
    with pytest.raises(RuntimeError):
        guard.check("synthetic")
    first, count = guard.failure, len(probes)
    with pytest.raises(RuntimeError, match="latched") as caught:
        guard.check("another")
    assert guard.failure is first and caught.value.__cause__ is first and len(probes) == count
