from datetime import datetime, timedelta, timezone
from io import StringIO
import json
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
