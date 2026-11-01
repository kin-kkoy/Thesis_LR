"""Synthetic contracts only; no prepared test has been executed."""
from copy import deepcopy
from functools import wraps
import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator

import calibrate_candidate_rf as cal


def encoded(value: object) -> bytes:
    return cal.canonical(value) + b"\n"


def commitment(config: dict) -> str:
    return hashlib.sha256(cal.canonical(config)).hexdigest()


def invoke(config: dict) -> dict:
    return cal.run(config, execution_authorized=True, expected_configuration_sha256=commitment(config))


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Fabricated metadata and 80 rows; no estimator construction or model load."""
    output = tmp_path / "output"
    package, base = output / "synthetic-package", output / "synthetic-rf-estimator-r1"
    for path in (package / "calibration", base):
        path.mkdir(parents=True)
    for name, value in (("OUTPUT", output), ("PACKAGE", package), ("BASE", base), ("CALIBRATION_ROWS", 80)):
        monkeypatch.setattr(cal, name, value)
    names = list(cal.features.CANONICAL_FEATURE_NAMES)
    matrix = np.tile(np.array([0.2, 0, 1, 0.95, 1, 17.7, -0.707, 0.707, 1, 0.95, 0.25],
                              dtype=np.float32), (80, 1))
    matrix[:, 1] = np.linspace(0, 1, 80, dtype=np.float32)
    matrix[0, 0] = np.float32(-0.0)
    x, y = pd.DataFrame(matrix, columns=names), np.array([0]*40 + [1]*40, dtype=np.int8)
    csv = (",".join(names + ["Ignited"]) + "\n" + "".join(
        ",".join([*(str(float(v)) for v in row), str(int(label))]) + "\n"
        for row, label in zip(matrix, y, strict=True))).encode("ascii")
    (package / "calibration/data.csv").write_bytes(csv)
    marker = dict(schema_version=cal.training.RELEASE_SCHEMA, manifest_sha256=cal.training.CUSTODIAN_SHA)
    (package / "COMMITTED.json").write_bytes(encoded(marker))
    monkeypatch.setattr(cal, "PUBLIC_PIN", cal.identity_bytes(encoded(marker)))
    (base / "rf_uncalibrated.joblib").write_bytes(b"unloaded synthetic model placeholder")
    resolved = cal.training.default_configuration()
    (base / "resolved_configuration.json").write_bytes(encoded(resolved))
    manifest = dict(schema_version=cal.training.SCHEMA, calibrated=False, training_roles=["train"],
        estimator=cal.training.PARAMS, feature_names=names, feature_dtype="float32", target_dtype="int8",
        positive_label=1, dataset=dict(id=cal.DATASET), teacher_id=resolved["teacher_id"],
        numerical_contract_id=resolved["numerical_contract_id"], configuration_sha256=commitment(resolved), files={
            "rf_uncalibrated.joblib": cal.file_identity(base / "rf_uncalibrated.joblib"),
            "resolved_configuration.json": cal.file_identity(base / "resolved_configuration.json")})
    (base / "training_manifest.json").write_bytes(encoded(manifest))
    pins = {key: cal.file_identity(base / name) for key, name in (
        ("model", "rf_uncalibrated.joblib"), ("manifest", "training_manifest.json"),
        ("configuration", "resolved_configuration.json"))}
    monkeypatch.setattr(cal, "BASE_PINS", pins)
    config = cal.default_configuration()
    config.update(enabled=True, authorization_reference="SYNTHETIC-NOT-OPERATIONAL",
        destination=str(output / (base.name + "-sigmoid-r1")),
        evidence_dir=str(output / (base.name + "-sigmoid-r1-evidence")),
        resources=dict(max_process_rss_bytes=4096, min_available_physical_memory_bytes=2,
            min_free_disk_bytes=2, max_wall_seconds=60, observation_interval_seconds=1))
    scope = dict(schema_version=cal.training.RELEASE_SCHEMA, role="calibration", rows=80,
        teacher_id=config["teacher_id"], numerical_contract_id=config["numerical_contract_id"],
        files={"data.csv": cal.identity_bytes(csv)}, membership="FORBIDDEN-REFERENCE-ONLY",
        probability_sidecar="FORBIDDEN-REFERENCE-ONLY")
    (package / "calibration/manifest.json").write_bytes(encoded(scope))
    config["calibration"].update(manifest=cal.identity_bytes(encoded(scope)), csv=cal.identity_bytes(csv))
    evidence = Path(config["evidence_dir"])
    evidence.mkdir()
    for name in ("worker.stdout.txt", "worker.stderr.txt", "wrapper.receipt.json"):
        (evidence / name).write_bytes(b"")
    return SimpleNamespace(config=config, x=x, y=y, package=package, base=base,
        stage=Path(config["destination"] + "-payload"), destination=Path(config["destination"]))


def rebind_csv(fixture: SimpleNamespace, raw: bytes) -> None:
    (fixture.package / "calibration/data.csv").write_bytes(raw)
    fixture.config["calibration"]["csv"] = cal.identity_bytes(raw)
    path = fixture.package / "calibration/manifest.json"
    scope = cal.strict_json(path.read_bytes())
    scope["files"]["data.csv"] = cal.identity_bytes(raw)
    path.write_bytes(encoded(scope))
    fixture.config["calibration"]["manifest"] = cal.identity_bytes(encoded(scope))


@pytest.fixture
def tiny(inputs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """One real tiny 200-tree fit and initial dump, solely on 80 synthetic rows."""
    forest = RandomForestClassifier(**cal.training.PARAMS)
    forest.fit(inputs.x, inputs.y)
    with (inputs.base / "rf_uncalibrated.joblib").open("wb") as stream:
        joblib.dump(forest, stream, compress=0, protocol=5)
    model_pin = cal.file_identity(inputs.base / "rf_uncalibrated.joblib")
    path = inputs.base / "training_manifest.json"
    manifest = cal.strict_json(path.read_bytes())
    manifest["files"]["rf_uncalibrated.joblib"] = model_pin
    path.write_bytes(encoded(manifest))
    pins = dict(cal.BASE_PINS, model=model_pin, manifest=cal.file_identity(path))
    monkeypatch.setattr(cal, "BASE_PINS", pins)
    inputs.config["base"] = deepcopy(pins)
    monkeypatch.setattr(cal, "source_identities", lambda: {"synthetic_source_substitution": True})
    monkeypatch.setattr(cal.training, "runtime_identity", lambda: {"synthetic_runtime_substitution": True})
    monkeypatch.setattr(cal.training, "capture_resource_snapshot", lambda _: SimpleNamespace(
        process_rss_bytes=1024, available_memory_bytes=8192, free_disk_bytes=8192))
    inputs.forest = forest
    return inputs


def test_disabled_configuration_and_unresolved_activation() -> None:
    config = cal.yaml.load(cal.CONFIG.read_text(encoding="utf-8"), Loader=cal.training.UniqueLoader)
    assert cal.canonical(config) == cal.canonical(cal.default_configuration())
    cal.validate_configuration(config)
    assert not config["enabled"] and config["authorization_reference"] is None
    assert config["calibration"]["manifest"] is None and config["calibration"]["csv"] is None
    assert all(v is None for v in config["resources"].values())
    assert config["destination"] is None and config["evidence_dir"] is None
    with pytest.raises(PermissionError):
        cal.validate_configuration(config, active=True)


@pytest.mark.parametrize("field", ["enabled", "authorization_reference", "manifest", "csv", "resources", "method"])
def test_activation_guards_precede_io(inputs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    config = deepcopy(inputs.config)
    if field in ("manifest", "csv"):
        config["calibration"][field] = None
    elif field == "resources":
        config["resources"]["max_process_rss_bytes"] = None
    elif field == "method":
        config["procedure"]["method"] = "isotonic"
    else:
        config[field] = False if field == "enabled" else None
    monkeypatch.setattr(cal, "destinations", lambda _: pytest.fail("I/O reached"))
    with pytest.raises(cal.CalibrationFailure) as caught:
        invoke(config)
    assert not caught.value.report["complete"] and caught.value.report["committed"] is None


@pytest.mark.parametrize("name", ["master", "validation", "final_test", "probabilities", "membership", "../model"])
def test_input_allowlist_rejects_before_filesystem(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setattr(cal, "ordinary", lambda *args: pytest.fail("filesystem reached"))
    with pytest.raises(PermissionError):
        cal.read_bound(name, {}, lambda _: pytest.fail("callback reached"))


def test_selective_loader_preserves_order_dtype_and_signed_zero(inputs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    accessed, original = [], Path.read_bytes
    def read(path):
        accessed.append(path)
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    raw, x, y = cal.load_inputs(inputs.config, lambda _: None)
    assert raw == b"unloaded synthetic model placeholder"
    assert list(x.columns) == list(cal.features.CANONICAL_FEATURE_NAMES)
    assert all(dtype == np.dtype("float32") for dtype in x.dtypes)
    assert y.dtype == np.dtype("int8") and np.array_equal(y, inputs.y)
    assert np.array_equal(x.to_numpy().view(np.uint32), inputs.x.to_numpy().view(np.uint32))
    assert np.signbit(x.iloc[0, 0])
    assert accessed == [inputs.base / "training_manifest.json", inputs.base / "resolved_configuration.json",
        inputs.package / "COMMITTED.json", inputs.package / "calibration/manifest.json",
        inputs.package / "calibration/data.csv", inputs.base / "rf_uncalibrated.joblib"]


@pytest.mark.parametrize("fault", ["fractional_label", "overflow_label", "nan", "material", "one_class", "missing_row", "header"])
def test_malformed_rows_reject_without_model_read(inputs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    lines = (inputs.package / "calibration/data.csv").read_text(encoding="ascii").splitlines()
    row = lines[1].split(",")
    if fault == "fractional_label":
        row[-1] = "0.5"
    elif fault == "overflow_label":
        row[-1] = "256"
    elif fault == "nan":
        row[0] = "nan"
    elif fault == "material":
        row[4] = "1.5"
    elif fault == "one_class":
        lines[1:] = [line.rsplit(",", 1)[0] + ",0" for line in lines[1:]]
    elif fault == "missing_row":
        lines.pop()
    else:
        lines[0] = lines[0].replace("slope_risk", "wrong")
    if fault in ("fractional_label", "overflow_label", "nan", "material"):
        lines[1] = ",".join(row)
    rebind_csv(inputs, ("\n".join(lines) + "\n").encode("ascii"))
    original = cal.read_bound
    def read(name, pin, check):
        assert name != "model", "model read before input rejection"
        return original(name, pin, check)
    monkeypatch.setattr(cal, "read_bound", read)
    with pytest.raises((ValueError, FloatingPointError)):
        cal.load_inputs(inputs.config, lambda _: None)


def test_metadata_and_independent_hash_rejection(inputs: SimpleNamespace) -> None:
    inputs.config["calibration"]["csv"]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="metadata"):
        cal.load_inputs(inputs.config, lambda _: None)
    with pytest.raises(ValueError, match="identity"):
        cal.read_bound("model", dict(bytes=1, sha256="0"*64), lambda _: None)


def test_single_sigmoid_audit_roundtrip_and_acyclic_bindings(tiny: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    counts = dict(frozen=0, predict=0, dump=0, load=0, sigmoid=0)
    frozen_fit, predict, dump, load = FrozenEstimator.fit, RandomForestClassifier.predict_proba, joblib.dump, joblib.load
    helper = cal.historical.calibrate_frozen_estimator
    def forbidden_fit(*args, **kwargs):
        pytest.fail("forest refit attempted")
    def no_op(self, *args, **kwargs):
        counts["frozen"] += 1
        return frozen_fit(self, *args, **kwargs)
    @wraps(predict)
    def probability(self, *args, **kwargs):
        counts["predict"] += 1
        return predict(self, *args, **kwargs)
    def dumping(*args, **kwargs):
        counts["dump"] += 1
        return dump(*args, **kwargs)
    def loading(*args, **kwargs):
        counts["load"] += 1
        return load(*args, **kwargs)
    def calibrating(*args, **kwargs):
        counts["sigmoid"] += 1
        return helper(*args, **kwargs)
    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden_fit)
    monkeypatch.setattr(FrozenEstimator, "fit", no_op)
    monkeypatch.setattr(RandomForestClassifier, "predict_proba", probability)
    monkeypatch.setattr(joblib, "dump", dumping)
    monkeypatch.setattr(joblib, "load", loading)
    monkeypatch.setattr(cal.historical, "calibrate_frozen_estimator", calibrating)
    original = (tiny.base / "rf_uncalibrated.joblib").read_bytes()
    report = invoke(tiny.config)
    assert report["complete"] and report["committed"] and report["declared_forest_refits"] == 0
    assert counts == dict(frozen=6, predict=8, dump=1, load=2, sigmoid=1)
    assert (tiny.base / "rf_uncalibrated.joblib").read_bytes() == original
    assert len({v["supplementary"] for v in report["preservation"].values()}) == 1
    expected = {"resolved_configuration.json", "rf_sigmoid.joblib", "calibration_predictions.jsonl",
                "calibration_report.json", "calibration_manifest.json"}
    assert {p.name for p in tiny.destination.iterdir()} == expected
    manifest = cal.strict_json((tiny.destination / "calibration_manifest.json").read_bytes())
    assert set(manifest["files"]) == expected - {"calibration_manifest.json"}
    assert manifest["threshold"] is None and manifest["calibrated"] is True
    for name, pin in manifest["files"].items():
        assert cal.file_identity(tiny.destination / name) == pin
    audit = [cal.strict_json(line) for line in (tiny.destination / "calibration_predictions.jsonl").read_bytes().splitlines()]
    assert [v["row_index"] for v in audit] == list(range(80))
    labels = np.array([v["label"] for v in audit], dtype=np.int8)
    p = np.array([struct.unpack("<d", bytes.fromhex(v["probability_f64_le"]))[0] for v in audit])
    assert np.array_equal(labels, tiny.y)
    diagnostic = report["diagnostics"]
    assert diagnostic["metrics"]["brier_mean_squared_error"]["value"] == float(np.mean((p-labels)**2))
    manual_log = -np.mean(labels*np.log(np.clip(p, np.finfo(float).eps, 1)) +
                          (1-labels)*np.log(np.clip(1-p, np.finfo(float).eps, 1)))
    assert diagnostic["metrics"]["log_loss"]["value"] == pytest.approx(float(manual_log))
    bins = [min(int(np.searchsorted(cal.historical.BIN_EDGES, v, side="right"))-1, 9) for v in p]
    assert [sum(v == i for v in bins) for i in range(10)] == [v["count"] for v in diagnostic["reliability_bins"]]
    assert diagnostic["optimizer_status"] == "unavailable_through_reused_api" and not tiny.stage.exists()


def mutate_missing(forest: object) -> None:
    nodes = forest.estimators_[0].tree_.__getstate__()["nodes"]
    old = nodes["missing_go_to_left"].copy()
    nodes["missing_go_to_left"][0] ^= np.uint8(1)
    assert np.count_nonzero(old != nodes["missing_go_to_left"]) == 1


def test_only_omitted_state_mutation_rejects_before_diagnostics(tiny: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = cal.historical.calibrate_frozen_estimator
    def mutate(base, x, y):
        result = helper(base, x, y)
        old, extra = cal.historical.forest_fingerprint(base), cal.supplementary_fingerprint(base)
        mutate_missing(base)
        assert cal.historical.forest_fingerprint(base) == old
        assert cal.supplementary_fingerprint(base) != extra
        return result
    monkeypatch.setattr(cal.historical, "calibrate_frozen_estimator", mutate)
    monkeypatch.setattr(cal, "probability_report", lambda *args: pytest.fail("diagnostics reached"))
    monkeypatch.setattr(joblib, "dump", lambda *args, **kwargs: pytest.fail("serialization reached"))
    raw = (tiny.base / "rf_uncalibrated.joblib").read_bytes()
    with pytest.raises(cal.CalibrationFailure) as caught:
        invoke(tiny.config)
    report = caught.value.report
    assert not report["complete"] and report["committed"] is None
    assert "state changed" in report["original_failure"]["message"]
    assert {p.name for p in tiny.stage.iterdir()} == {"resolved_configuration.json"}
    assert (tiny.base / "rf_uncalibrated.joblib").read_bytes() == raw


@pytest.mark.parametrize("where", ["embedded", "roundtrip"])
def test_embedded_and_roundtrip_omitted_state_checked(tiny: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, where: str) -> None:
    if where == "embedded":
        helper = cal.historical.calibrate_frozen_estimator
        def substitute(base, x, y):
            wrapper, checks = helper(base, x, y)
            changed = deepcopy(wrapper)
            mutate_missing(cal.embedded_forest(changed))
            return changed, checks
        monkeypatch.setattr(cal.historical, "calibrate_frozen_estimator", substitute)
    else:
        original, calls = joblib.load, []
        def loading(*args, **kwargs):
            value = original(*args, **kwargs)
            calls.append(value)
            if len(calls) == 2:
                mutate_missing(cal.embedded_forest(value))
            return value
        monkeypatch.setattr(joblib, "load", loading)
    with pytest.raises(cal.CalibrationFailure) as caught:
        invoke(tiny.config)
    assert not caught.value.report["complete"] and caught.value.report["committed"] is None
    assert "forest state" in caught.value.report["original_failure"]["message"]


def test_named_state_ignores_padding_covers_metadata_and_never_predicts(tiny: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RandomForestClassifier, "predict_proba", lambda *args: pytest.fail("prediction reached"))
    monkeypatch.setattr(RandomForestClassifier, "fit", lambda *args: pytest.fail("fit reached"))
    forest = tiny.forest
    before = cal.supplementary_fingerprint(forest)
    nodes = forest.estimators_[0].tree_.__getstate__()["nodes"]
    meaningful = set()
    for dtype, offset, *rest in nodes.dtype.fields.values():
        meaningful.update(range(offset, offset+dtype.itemsize))
    padding = [i for i in range(nodes.dtype.itemsize) if i not in meaningful]
    assert padding, "Pinned Node padding expectation requires installed-runtime verification"
    bytes_view = nodes.view(np.uint8).reshape(len(nodes), nodes.dtype.itemsize)
    bytes_view[:, padding] ^= np.uint8(1)
    assert cal.supplementary_fingerprint(forest) == before
    forest.estimators_[0].classes_ = forest.estimators_[0].classes_[::-1].copy()
    assert cal.supplementary_fingerprint(forest) != before


def test_reliability_endpoints_empty_bins_and_undefined_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    y, p = np.array([0, 1, 0, 1], dtype=np.int8), np.array([0.0, 0.1, 0.9, 1.0])
    report = cal.probability_report(y, p)
    assert [v["count"] for v in report["reliability_bins"]] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 2]
    assert report["reliability_bins"][2]["mean_predicted_probability"] is None
    monkeypatch.setattr(cal, "roc_auc_score", lambda *args: (_ for _ in ()).throw(ValueError("synthetic undefined")))
    assert cal.probability_report(y, p)["metrics"]["roc_auc"]["status"] == "undefined"


@pytest.mark.parametrize("failure", ["helper_and_integrity", "partial_dump", "commit", "after_commit"])
def test_failure_priority_partial_retention_and_committed_identity(tiny: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    if failure == "helper_and_integrity":
        def failing(base, x, y):
            mutate_missing(base)
            raise RuntimeError("ORIGINAL-helper-failure")
        monkeypatch.setattr(cal.historical, "calibrate_frozen_estimator", failing)
    elif failure == "partial_dump":
        def dumping(value, stream, **kwargs):
            stream.write(b"RETAIN-PARTIAL")
            stream.flush()
            raise OSError("ORIGINAL-dump-failure")
        monkeypatch.setattr(joblib, "dump", dumping)
    elif failure == "commit":
        monkeypatch.setattr(cal.os, "rename", lambda *args: (_ for _ in ()).throw(OSError("ORIGINAL-commit-failure")))
    else:
        original = cal.training.Resources.check
        def checking(self, boundary):
            original(self, boundary)
            if boundary == "final_observation":
                raise RuntimeError("ORIGINAL-final-observation-failure")
        monkeypatch.setattr(cal.training.Resources, "check", checking)
    with pytest.raises(cal.CalibrationFailure) as caught:
        invoke(tiny.config)
    report = caught.value.report
    assert not report["complete"] and report["original_failure"]["message"].startswith("ORIGINAL-")
    assert caught.value.__cause__ is not None
    if failure == "helper_and_integrity":
        assert report["incidents"] and "state changed" in report["incidents"][0]["message"]
    elif failure == "partial_dump":
        assert (tiny.stage / "rf_sigmoid.joblib").read_bytes() == b"RETAIN-PARTIAL"
    elif failure == "commit":
        assert len(list(tiny.stage.iterdir())) == 5 and report["committed"] is None
    else:
        assert report["committed"] and report["incidents"]
        assert cal.file_identity(tiny.destination / "rf_sigmoid.joblib") == report["committed"]["model"]
        assert len(list(tiny.destination.iterdir())) == 5
        assert report["resources"][-1]["boundary"] == "final_observation"


def test_destination_collision_preserves_existing_bytes(inputs: SimpleNamespace) -> None:
    inputs.stage.mkdir()
    marker = inputs.stage / "retained"
    marker.write_bytes(b"do-not-touch")
    with pytest.raises(FileExistsError):
        cal.destinations(inputs.config)
    assert marker.read_bytes() == b"do-not-touch"


def test_authority_and_resolved_commitment_precede_io(inputs: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cal, "destinations", lambda _: pytest.fail("I/O reached"))
    for authority, digest in ((False, commitment(inputs.config)), (True, "0"*64)):
        with pytest.raises(cal.CalibrationFailure) as caught:
            cal.run(inputs.config, execution_authorized=authority, expected_configuration_sha256=digest)
        assert caught.value.report["original_failure"]["type"] == "PermissionError"
        assert not caught.value.report["complete"]


def test_integer_class_one_routing_and_boolean_rejection() -> None:
    model = SimpleNamespace(feature_names_in_=np.array(cal.features.CANONICAL_FEATURE_NAMES),
        classes_=np.array([1, 0]), predict_proba=lambda x: np.array([[0.8, 0.2], [0.3, 0.7]]))
    assert np.array_equal(cal.features.predict_positive_probability(model, np.zeros((2, 11))), [0.8, 0.3])
    for classes in ([False, True], [0.0, 1.0], [0, 2]):
        with pytest.raises(ValueError):
            cal.features.positive_class_index(classes)
