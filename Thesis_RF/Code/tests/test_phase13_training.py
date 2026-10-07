from datetime import datetime, timedelta, timezone
from io import StringIO
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path
import tempfile

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest

import train_rf_optuna as phase13
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES


class _FakeTrial:
    def __init__(self, number, fixed_params=None):
        self.number = number
        self.params = {}
        self.user_attrs = {}
        self.system_attrs = {}
        self.state = optuna.trial.TrialState.RUNNING
        self.value = None
        self.datetime_start = datetime.now(timezone.utc)
        self.datetime_complete = None
        self.duration = None
        self._fixed_params = fixed_params or {}

    def suggest_categorical(self, name, choices):
        value = self._fixed_params.get(name, choices[0])
        assert value in choices
        self.params[name] = value
        return value

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class _FakeStudy:
    def __init__(self, completed_count=60):
        self.trials = []
        self.queued = []
        self.completed_count = completed_count

    def enqueue_trial(self, params):
        self.queued.append(params)

    def optimize(self, objective, *, n_trials, callbacks, **_kwargs):
        for number in range(self.completed_count):
            trial = _FakeTrial(number, self.queued[number] if number < len(self.queued) else None)
            try:
                trial.value = objective(trial)
            except BaseException:
                trial.state = optuna.trial.TrialState.FAIL
                trial.datetime_complete = datetime.now(timezone.utc)
                trial.duration = timedelta(0)
                self.trials.append(trial)
                raise
            trial.state = optuna.trial.TrialState.COMPLETE
            trial.datetime_complete = datetime.now(timezone.utc)
            trial.duration = timedelta(0)
            self.trials.append(trial)
            for callback in callbacks:
                callback(self, trial)


class _FakeEstimator:
    instances = []
    fail_on_fit = None

    def __init__(self, params):
        self.params = params
        self.fit_rows = []
        self.__class__.instances.append(self)

    def fit(self, features, labels):
        rows = len(features)
        self.fit_rows.append(rows)
        if self.__class__.fail_on_fit == rows:
            raise RuntimeError("synthetic fit failure")
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.classes_ = np.asarray([0, 1])
        return self

    def get_params(self, deep=False):
        return dict(self.params)

    def predict_proba(self, features):
        return np.full((len(features), 2), 0.5, dtype=np.float64)


def _synthetic_executor(tmp_path, monkeypatch, *, completed_count=60, fail_on_fit=None):
    _FakeEstimator.instances = []
    _FakeEstimator.fail_on_fit = fail_on_fit
    config = phase13._load_phase13_config(phase13.PHASE13_CONFIG_PATH)
    config["study"] = {**config["study"], "n_trials": 60, "timeout_seconds": 1}
    config["resource_limits"] = {**config["resource_limits"], "monitor_interval_seconds": 3600}
    destination = tmp_path / "set_c_rf_v1"
    evidence_dir = tmp_path / "evidence" / "synthetic-run"
    x_train = pd.DataFrame(np.zeros((6, len(CANONICAL_FEATURE_NAMES)), dtype=np.float32),
                           columns=CANONICAL_FEATURE_NAMES)
    y_train = pd.Series([0, 1, 0, 1, 0, 1], dtype=np.int8)
    x_validation = pd.DataFrame(np.zeros((4, len(CANONICAL_FEATURE_NAMES)), dtype=np.float32),
                                columns=CANONICAL_FEATURE_NAMES)
    y_validation = pd.Series([0, 1, 0, 1], dtype=np.int8)
    attestation = {"source_provenance": {"source_manifest": {
        "source_csv_sha256": "s" * 64,
    }}}
    preflight = {
        "config": config,
        "destination": destination,
        "evidence_dir": evidence_dir,
        "partitions": {"X_train": x_train, "y_train": y_train,
                       "X_validation": x_validation, "y_validation": y_validation},
        "attestation": attestation,
        "observed_role_counts": {
            "train": {"rows": 6, "positives": 3, "negatives": 3},
            "validation": {"rows": 4, "positives": 2, "negatives": 2},
            "refit": {"rows": 10, "positives": 5, "negatives": 5},
        },
        "source": {"revision": "a" * 40, "dirty": False},
        "environment": {"versions": {"optuna": optuna.__version__}},
        "resource": {"timestamp_utc": "synthetic", "available_physical_memory_bytes": 9,
                     "free_c_bytes": 9},
    }
    study = _FakeStudy(completed_count=completed_count)
    monkeypatch.setattr(phase13.optuna, "create_study", lambda **_kwargs: study)
    monkeypatch.setattr(phase13, "_estimator", lambda params, _config: _FakeEstimator(params))
    monkeypatch.setattr(phase13.joblib, "dump", lambda _model, path: Path(path).write_bytes(b"synthetic model"))
    monkeypatch.setattr(phase13, "_resource_snapshot", lambda: {
        "timestamp_utc": "synthetic", "available_physical_memory_bytes": 9,
        "free_c_bytes": 9,
    })
    return preflight, study


def _synthetic_supervisor_evidence(tmp_path, monkeypatch, *, run_id="synthetic-run",
                                   process_record=True, resources="valid"):
    code_dir = tmp_path / "synthetic_code"
    code_dir.mkdir()
    script_path = code_dir / "train_rf_optuna.py"
    script_path.write_text("# synthetic path anchor\n", encoding="utf-8")
    monkeypatch.setattr(phase13, "__file__", str(script_path))
    output_root = code_dir / "output" / "phase13"
    evidence_parent = output_root / "evidence"
    evidence_dir = evidence_parent / run_id
    evidence_dir.mkdir(parents=True)
    nonce = "a" * 32
    monkeypatch.setenv("PHASE13_SUPERVISOR_NONCE", nonce)
    now = datetime.now(timezone.utc)
    (evidence_dir / "stdout.log").write_bytes(b"")
    (evidence_dir / "stderr.log").write_bytes(b"")
    (evidence_dir / "supervisor_handoff.json").write_text(json.dumps({
        "schema_version": "phase13_set_c_rf_supervisor_handoff.v1",
        "run_id": run_id, "nonce": nonce, "created_at_utc": now.isoformat(),
        "executable": str(Path(sys.executable).resolve()),
    }), encoding="utf-8")
    if process_record:
        (evidence_dir / "process.json").write_text(json.dumps({
            "schema_version": "phase13_set_c_rf_process.v1", "run_id": run_id,
            "process_id": os.getpid(), "process_ids": [os.getpid()],
            "process_started_at_utc": now.isoformat(), "executable": str(Path(sys.executable).resolve()),
            "working_directory": str(code_dir.resolve()),
        }), encoding="utf-8")
    if resources != "absent":
        sample_time = now if resources != "stale" else datetime(2000, 1, 1, tzinfo=timezone.utc)
        sample = {
            "run_id": run_id, "timestamp_utc": sample_time.isoformat(), "elapsed_seconds": 0.0,
            "available_physical_memory_bytes": 8 * 1024**3, "free_c_bytes": 4 * 1024**3,
        }
        if resources == "empty":
            content = ""
        elif resources == "malformed":
            content = "not-json\n"
        else:
            content = json.dumps(sample) + "\n"
        (evidence_dir / "operator_resources.jsonl").write_text(content, encoding="utf-8")
    config = {
        "destination": str(output_root / "set_c_rf_v1"),
        "evidence_parent": str(evidence_parent),
        "resource_limits": {
            "min_available_physical_memory_bytes": 4 * 1024**3,
            "min_free_c_bytes": 2 * 1024**3,
        },
    }
    return config, evidence_dir


def test_pinned_configuration_and_ordered_baselines():
    config = phase13._load_phase13_config(phase13.PHASE13_CONFIG_PATH)
    assert config["protocol_id"] == phase13.PHASE13_PROTOCOL_ID
    assert config["protocol_sha256"] == phase13.PHASE13_PROTOCOL_SHA256
    assert tuple(config["feature_names"]) == CANONICAL_FEATURE_NAMES
    assert [row["class_weight"] for row in config["baselines"]] == [
        "none", "balanced", "balanced_subsample"
    ]
    assert config["study"] == {
        "direction": "maximize", "n_trials": 60, "n_jobs": 1,
        "timeout_seconds": 5400, "exact_tie": "smallest_trial_number",
    }


@pytest.mark.parametrize(
    ("resources", "process_record"),
    [("absent", False), ("empty", False), ("valid", True)],
)
def test_execution_guard_accepts_only_fresh_bound_supervisor_evidence(
    tmp_path, monkeypatch, resources, process_record
):
    config, evidence = _synthetic_supervisor_evidence(
        tmp_path, monkeypatch, resources=resources, process_record=process_record,
    )
    destination, resolved_evidence = phase13._validate_destination(
        config, execution=True, evidence_dir=evidence,
    )
    assert destination == Path(config["destination"]).resolve()
    assert resolved_evidence == evidence.resolve()


@pytest.mark.parametrize(
    ("tamper", "exception"),
    [
        ("foreign_handoff", PermissionError),
        ("foreign_nonce", PermissionError),
        ("stale_handoff", PermissionError),
        ("foreign_process", PermissionError),
        ("unexpected", FileExistsError),
        ("malformed_resources", ValueError),
        ("stale_resources", PermissionError),
        ("nonempty_redirect", FileExistsError),
    ],
)
def test_execution_guard_rejects_stale_foreign_or_unexpected_evidence(
    tmp_path, monkeypatch, tamper, exception
):
    config, evidence = _synthetic_supervisor_evidence(tmp_path, monkeypatch)
    if tamper == "foreign_handoff":
        handoff_path = evidence / "supervisor_handoff.json"
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        handoff["run_id"] = "another-run"
        handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    elif tamper == "foreign_nonce":
        monkeypatch.setenv("PHASE13_SUPERVISOR_NONCE", "b" * 32)
    elif tamper == "stale_handoff":
        handoff_path = evidence / "supervisor_handoff.json"
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        handoff["created_at_utc"] = "2000-01-01T00:00:00+00:00"
        handoff_path.write_text(json.dumps(handoff), encoding="utf-8")
    elif tamper == "foreign_process":
        process_path = evidence / "process.json"
        record = json.loads(process_path.read_text(encoding="utf-8"))
        record["process_id"] = os.getpid() + 1
        process_path.write_text(json.dumps(record), encoding="utf-8")
    elif tamper == "unexpected":
        (evidence / "unreviewed.txt").write_text("unexpected", encoding="utf-8")
    elif tamper == "malformed_resources":
        (evidence / "operator_resources.jsonl").write_text("not-json\n", encoding="utf-8")
    elif tamper == "stale_resources":
        (evidence / "operator_resources.jsonl").write_text(
            json.dumps({
                "run_id": evidence.name, "timestamp_utc": "2000-01-01T00:00:00+00:00",
                "elapsed_seconds": 0.0, "available_physical_memory_bytes": 8 * 1024**3,
                "free_c_bytes": 4 * 1024**3,
            }) + "\n",
            encoding="utf-8",
        )
    elif tamper == "nonempty_redirect":
        (evidence / "stdout.log").write_text("foreign output", encoding="utf-8")

    with pytest.raises(exception):
        phase13._validate_destination(config, execution=True, evidence_dir=evidence)


def test_execution_guard_keeps_default_deny_and_approved_path_checks(tmp_path, monkeypatch):
    config, evidence = _synthetic_supervisor_evidence(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError, match="must be absent for preflight"):
        phase13._validate_destination(config, execution=False, evidence_dir=evidence)
    config["destination"] = str(tmp_path / "unapproved" / "output")
    with pytest.raises(FileExistsError, match="absent approved versioned path"):
        phase13._validate_destination(config, execution=True, evidence_dir=evidence)


def test_execution_guard_rejects_reparse_evidence_entries(tmp_path, monkeypatch):
    config, evidence = _synthetic_supervisor_evidence(tmp_path, monkeypatch)
    real_lstat = Path.lstat
    reparse_log = evidence / "stdout.log"

    def reparse_lstat(path):
        if path == reparse_log:
            return SimpleNamespace(st_file_attributes=0x400)
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", reparse_lstat)
    with pytest.raises(PermissionError, match="reparse entry"):
        phase13._validate_destination(config, execution=True, evidence_dir=evidence)


def test_sampler_pruner_and_baseline_contract():
    config = phase13._load_phase13_config(phase13.PHASE13_CONFIG_PATH)
    sampler, pruner = phase13._make_sampler_pruner(config)
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert isinstance(pruner, optuna.pruners.NopPruner)
    assert sampler._n_startup_trials == 10
    assert [baseline["class_weight"] for baseline in config["baselines"]] == [
        "none", "balanced", "balanced_subsample"
    ]
    assert all({key: value for key, value in baseline.items() if key != "class_weight"} == {
        "n_estimators": 200, "max_depth": 15, "min_samples_split": 2,
        "min_samples_leaf": 1, "max_features": "sqrt",
    } for baseline in config["baselines"])
    provenance = phase13._sampler_provenance(sampler, config)
    assert provenance["optuna_version"] == optuna.__version__
    expected_settings = {
        "n_ei_candidates", "gamma", "weights", "consider_prior", "prior_weight",
        "consider_magic_clip", "consider_endpoints", "n_startup_trials", "seed",
        "multivariate", "group", "warn_independent_sampling", "constant_liar",
        "constraints_func", "categorical_distance_func",
    }
    assert set(provenance["constructor_options"]) == expected_settings
    assert set(provenance["resolved_options"]) == expected_settings
    assert provenance["resolved_options"]["n_ei_candidates"] == 24
    assert provenance["resolved_options"]["consider_prior"] is True
    assert provenance["resolved_options"]["prior_weight"] == 1.0
    assert provenance["resolved_options"]["consider_magic_clip"] is True
    assert provenance["resolved_options"]["consider_endpoints"] is False
    assert provenance["resolved_options"]["gamma"]
    assert provenance["constructor_options"]["weights"] is None
    assert provenance["constructor_options"]["warn_independent_sampling"] is None
    assert provenance["resolved_options"]["warn_independent_sampling"] is True
    assert provenance["resolved_options"]["weights"] == phase13._callable_identity(
        sampler._parzen_estimator_parameters.weights
    )
    assert provenance["resolved_options"]["weights"]
    assert provenance["resolved_options"]["constraints_func"] is None
    assert provenance["resolved_options"]["categorical_distance_func"] is None


def test_search_uses_ordered_categorical_choices():
    config = phase13._load_phase13_config(phase13.PHASE13_CONFIG_PATH)

    class FakeTrial:
        def __init__(self):
            self.calls = []

        def suggest_categorical(self, name, choices):
            self.calls.append((name, choices))
            return choices[0]

    trial = FakeTrial()
    params = phase13._phase13_trial_params(trial, config)
    assert list(params) == list(config["search"])
    assert trial.calls == list(config["search"].items())


def test_exact_ap_tie_selects_smallest_trial_number():
    complete = optuna.trial.TrialState.COMPLETE
    trials = [
        SimpleNamespace(state=complete, value=0.75, number=8),
        SimpleNamespace(state=complete, value=0.75, number=3),
        SimpleNamespace(state=complete, value=0.74, number=1),
    ]
    assert phase13._select_best_completed_trial(trials).number == 3
    with pytest.raises(ValueError, match="No completed"):
        phase13._select_best_completed_trial([
            SimpleNamespace(state=optuna.trial.TrialState.FAIL, value=None, number=0)
        ])


def test_tiny_synthetic_forest_keeps_generic_estimator_interface():
    config = phase13._load_phase13_config(phase13.PHASE13_CONFIG_PATH)
    features = pd.DataFrame(
        np.zeros((6, len(CANONICAL_FEATURE_NAMES)), dtype=np.float32),
        columns=CANONICAL_FEATURE_NAMES,
    )
    labels = pd.Series([0, 1, 0, 1, 0, 1], dtype=np.int8)
    model = phase13._estimator(config["baselines"][0], config)
    model.fit(features, labels)

    with tempfile.TemporaryDirectory(prefix="phase13-synthetic-") as temp_dir:
        artifact = Path(temp_dir) / "synthetic.joblib"
        joblib.dump(model, artifact)
        loaded = joblib.load(artifact)
    assert callable(loaded.predict_proba)
    assert tuple(loaded.feature_names_in_) == CANONICAL_FEATURE_NAMES
    assert tuple(loaded.classes_) == (0, 1)
    assert loaded.predict_proba(features).shape == (6, 2)
    assert loaded.phase13_calibrated is False


def test_cli_requires_explicit_mode_and_has_no_release_path_override():
    with pytest.raises(SystemExit):
        phase13._parse_args([])
    with pytest.raises(SystemExit):
        phase13._parse_args([
            "--preflight-only", "--source-revision", "a" * 40,
            "--expected-config-sha256", phase13.PHASE13_CONFIG_SHA256,
            "--release-dir", "C:/synthetic-override",
        ])


def test_resource_floor_gate_rejects_low_samples():
    config = phase13._load_phase13_config(phase13.PHASE13_CONFIG_PATH)
    phase13._assert_resource_floors({
        "available_physical_memory_bytes": 5 * 1024**3,
        "free_c_bytes": 3 * 1024**3,
    }, config)
    with pytest.raises(RuntimeError, match="memory"):
        phase13._assert_resource_floors({
            "available_physical_memory_bytes": 3 * 1024**3,
            "free_c_bytes": 3 * 1024**3,
        }, config)


def test_trial_failure_is_recorded_without_refit_or_publication(tmp_path, monkeypatch):
    preflight, study = _synthetic_executor(tmp_path, monkeypatch, completed_count=60, fail_on_fit=6)
    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        phase13._execute_phase13(
            preflight, approval_reference="synthetic", run_id="trial-fail",
            package_version="set_c_rf_v1", command=["synthetic"],
        )
    stage = tmp_path / ".set_c_rf_v1.staging-trial-fail"
    records = [json.loads(line) for line in (stage / "trials.jsonl").read_text().splitlines()]
    diagnostics = json.loads((stage / "run_diagnostics.json").read_text())
    assert records[0]["state"] == "FAIL"
    assert diagnostics["stop_reason"].startswith("RuntimeError: synthetic fit failure")
    assert "Traceback (most recent call last)" in diagnostics["traceback"]
    assert "synthetic fit failure" in diagnostics["traceback"]
    assert len(_FakeEstimator.instances) == 1
    assert not preflight["destination"].exists()


def test_trial_record_retry_and_diagnostic_write_failure_preserve_original_and_stderr(
    tmp_path, monkeypatch
):
    preflight, _study = _synthetic_executor(tmp_path, monkeypatch)
    writes = 0

    def failed_trial_write(_path, _study):
        nonlocal writes
        writes += 1
        if writes == 1:
            raise RuntimeError("original trial-record failure")
        raise OSError("secondary trial-record failure")

    real_write_text = Path.write_text
    diagnostic_attempts = []

    def failed_diagnostic_write(path, *args, **kwargs):
        if path.name == "run_diagnostics.json":
            diagnostic_attempts.append(path)
            raise OSError("diagnostic disk failure")
        return real_write_text(path, *args, **kwargs)

    stderr = StringIO()
    monkeypatch.setattr(phase13, "_write_trial_records", failed_trial_write)
    monkeypatch.setattr(Path, "write_text", failed_diagnostic_write)
    monkeypatch.setattr(phase13.sys, "stderr", stderr)
    with pytest.raises(RuntimeError, match="original trial-record failure"):
        phase13._execute_phase13(
            preflight, approval_reference="synthetic", run_id="evidence-fail",
            package_version="set_c_rf_v1", command=["synthetic"],
        )
    assert writes == 2
    assert diagnostic_attempts
    assert "original trial-record failure" in stderr.getvalue()
    assert "secondary trial-record failure" in stderr.getvalue()
    assert "diagnostics file write failed" in stderr.getvalue()
    assert "diagnostic disk failure" in stderr.getvalue()
    assert "Traceback (most recent call last)" in stderr.getvalue()
    assert "Traceback (most recent call last)" in stderr.getvalue()


def test_fewer_than_sixty_completed_trials_never_refits_or_publishes(tmp_path, monkeypatch):
    preflight, _study = _synthetic_executor(tmp_path, monkeypatch, completed_count=59)
    with pytest.raises(RuntimeError, match="study incomplete"):
        phase13._execute_phase13(
            preflight, approval_reference="synthetic", run_id="short-study",
            package_version="set_c_rf_v1", command=["synthetic"],
        )
    assert len(_FakeEstimator.instances) == 59
    assert not preflight["destination"].exists()
    assert not (tmp_path / ".set_c_rf_v1.staging-short-study" / "selection_freeze.json").exists()


def test_selection_freeze_precedes_one_refit_and_success_publishes_once(tmp_path, monkeypatch):
    preflight, study = _synthetic_executor(tmp_path, monkeypatch)
    real_factory = phase13._estimator
    checks = []

    def checked_factory(params, config):
        if len(_FakeEstimator.instances) == 60:
            freeze = tmp_path / ".set_c_rf_v1.staging-success" / "selection_freeze.json"
            checks.append(json.loads(freeze.read_text()))
        return real_factory(params, config)

    monkeypatch.setattr(phase13, "_estimator", checked_factory)
    output = phase13._execute_phase13(
        preflight, approval_reference="synthetic", run_id="success",
        package_version="set_c_rf_v1", command=["synthetic"],
    )
    assert output == preflight["destination"]
    assert len(study.trials) == 60
    assert len(_FakeEstimator.instances) == 61
    assert [model.fit_rows for model in _FakeEstimator.instances].count([10]) == 1
    assert checks and checks[0]["selected_trial_number"] == 0
    freeze_path = output / "selection_freeze.json"
    assert freeze_path.is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["study"]["sampler"]["resolved_options"]["n_ei_candidates"] == 24
    assert manifest["schema_population"]["observed_role_counts"]["refit"]["rows"] == 10
    assert manifest["selection_freeze"]["path"] == freeze_path.name
    assert manifest["selection_freeze"]["sha256"] == phase13._file_sha256(freeze_path)
    assert manifest["files"][freeze_path.name]["sha256"] == manifest["selection_freeze"]["sha256"]


def test_selection_freeze_write_failure_prevents_refit_and_publication(tmp_path, monkeypatch):
    preflight, _study = _synthetic_executor(tmp_path, monkeypatch)

    def fail_selection_write(*_args, **_kwargs):
        raise OSError("selection freeze write failure")

    monkeypatch.setattr(phase13, "_write_selection_freeze", fail_selection_write)
    with pytest.raises(OSError, match="selection freeze write failure"):
        phase13._execute_phase13(
            preflight, approval_reference="synthetic", run_id="selection-freeze-fail",
            package_version="set_c_rf_v1", command=["synthetic"],
        )
    stage = tmp_path / ".set_c_rf_v1.staging-selection-freeze-fail"
    assert len(_FakeEstimator.instances) == 60
    assert not (stage / "selection_freeze.json").exists()
    assert "selection freeze write failure" in json.loads(
        (stage / "run_diagnostics.json").read_text()
    )["traceback"]
    assert not preflight["destination"].exists()


def test_refit_failure_preserves_frozen_selection_in_incomplete_stage(tmp_path, monkeypatch):
    preflight, _study = _synthetic_executor(tmp_path, monkeypatch, fail_on_fit=10)
    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        phase13._execute_phase13(
            preflight, approval_reference="synthetic", run_id="refit-fail",
            package_version="set_c_rf_v1", command=["synthetic"],
        )
    stage = tmp_path / ".set_c_rf_v1.staging-refit-fail"
    freeze = json.loads((stage / "selection_freeze.json").read_text())
    assert freeze["selected_trial_number"] == 0
    assert json.loads((stage / "run_diagnostics.json").read_text())["stop_reason"].startswith(
        "RuntimeError: synthetic fit failure"
    )
    assert not preflight["destination"].exists()


@pytest.mark.parametrize("refusal", ["destination", "staging"])
def test_executor_refuses_existing_destination_or_interrupted_staging(tmp_path, monkeypatch, refusal):
    preflight, _study = _synthetic_executor(tmp_path, monkeypatch)
    if refusal == "destination":
        preflight["destination"].mkdir()
    else:
        (preflight["destination"].parent / ".set_c_rf_v1.staging-existing").mkdir()
    with pytest.raises(FileExistsError):
        phase13._execute_phase13(
            preflight, approval_reference="synthetic", run_id="refusal",
            package_version="set_c_rf_v1", command=["synthetic"],
        )
    assert not _FakeEstimator.instances
