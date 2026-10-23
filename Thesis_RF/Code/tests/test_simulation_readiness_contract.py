"""Synthetic source only; owner must separately release collection/execution."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import builtins
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import joblib
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import yaml

import main
import orchestrator as o
from modules.automata_engine import FireAutomata
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES


class SyntheticEstimator:
    feature_names_in_ = np.asarray(CANONICAL_FEATURE_NAMES)
    classes_ = np.array([1, 0], dtype=np.int64)

    def predict_proba(self, features):
        return np.tile([0.0, 1.0], (len(features), 1))


def independent_digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


@pytest.fixture
def ready(tmp_path, monkeypatch):
    template = Path(o.__file__).parent / "config/phase15_setc_rf_ca_v1.yaml"
    config = o._phase15_yaml(template.read_bytes())
    code = tmp_path / "Thesis_RF/Code"
    root = code / "output/phase15"
    root.mkdir(parents=True)
    inputs = code / "inputs"
    inputs.mkdir()
    transform = from_origin(100, 200, 1, 1)
    grids = {
        "slope": [[1, 1, -9999], [1, 1, 1]],
        "proximity": [[0, 0, -9999], [0, 0, 0]],
        "buildings": [[10, 0, -9999], [10, 10, 10]],
        "materials": [[1, 0, -9999], [1, 1, 1]],
    }
    config["environment"].update(raster_dir=str(inputs), nodata_value=-9999)
    for name, values in grids.items():
        path = inputs / (name + ".tif")
        with rasterio.open(path, "w", driver="GTiff", height=2, width=3, count=1,
                           dtype="float32", crs="EPSG:32651", transform=transform, nodata=-9999) as dst:
            dst.write(np.array(values, dtype=np.float32), 1)
        config["environment"][name + "_file"] = path.name
        config["environment"]["input_sha256"][name] = sha256(path.read_bytes()).hexdigest()
    buffer = BytesIO()
    joblib.dump(SyntheticEstimator(), buffer)
    payload = buffer.getvalue()
    model = code / "synthetic.joblib"
    model.write_bytes(payload)
    monkeypatch.setattr(o, "PHASE15_CODE_DIR", code)
    monkeypatch.setattr(o, "PHASE15_MODEL", str(model))
    monkeypatch.setattr(o, "PHASE15_MODEL_SHA256", sha256(payload).hexdigest())
    config["ml_model"].update(model_path=str(model), model_sha256=sha256(payload).hexdigest(), inference_chunk_size=2)
    config["phase15"].update(authorization_reference="synthetic-owner-approval", run_id="synthetic1", scenario_id="s1")
    config["phase15"]["resources"].update(max_process_rss_bytes=1000, min_system_available_bytes=100,
        min_free_output_bytes=100, max_wall_seconds=100, sample_interval_seconds=1)
    config["simulation"].update(seed=7, max_timesteps=1, ignition_points=[[100.5, 199.5]])
    config["placeholder_transition"].update(T_3_to_4=1, T_4_to_5=1, wind_weight=0.2)
    config["wind"].update(speed_kmh=0, direction_deg=0)
    config["flammability_weights"].update(base_weight=0.5, slope_weight=0.3, proximity_weight=0.2)
    source = {"repository_root": tmp_path.as_posix(), "head": "a" * 40, "dirty": True,
              "files": {key: "b" * 64 for key in o.PHASE15_SOURCES}, "absent_paths": list(o.PHASE15_ABSENT)}
    runtime = {"executable": (code / "synthetic-python.exe").as_posix(), "implementation": "CPython", "python_version": "synthetic",
               "cache_tag": "synthetic", "pointer_bits": 64, "platform": "synthetic",
               "packages": {key: "synthetic" for key in ("numpy", "scipy", "pandas", "rasterio", "scikit-learn", "joblib", "PyYAML")}}
    source_paths = {tmp_path / name: digest for name, digest in source["files"].items()}
    real_hash, real_no_reparse = o._sha256_file, o._phase15_no_reparse
    source_reads, git_calls = [], []
    def source_hash(path, on_chunk=None):
        if Path(path) in source_paths:
            source_reads.append(Path(path))
            return source_paths[Path(path)]
        return real_hash(path, on_chunk)
    def git(args, **kwargs):
        git_calls.append((args, kwargs))
        assert args[:1] == ["git"] and kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        if args[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout=source["head"] + "\n")
        assert args[-3:] == ["status", "--porcelain", "--untracked-files=normal"]
        return SimpleNamespace(stdout=" M synthetic.py\n" if source["dirty"] else "")
    monkeypatch.setattr(o, "_sha256_file", source_hash)
    monkeypatch.setattr(o, "_phase15_no_reparse", lambda path: None if Path(path) in source_paths else real_no_reparse(path))
    monkeypatch.setattr(o, "subprocess", SimpleNamespace(run=git))
    monkeypatch.setattr(o, "sys", SimpleNamespace(executable=runtime["executable"], version=runtime["python_version"],
        implementation=SimpleNamespace(cache_tag=runtime["cache_tag"]), argv=["synthetic-launch"], stderr=sys.stderr))
    monkeypatch.setattr(o, "platform", SimpleNamespace(python_implementation=lambda: runtime["implementation"], platform=lambda: runtime["platform"]))
    monkeypatch.setattr(o, "struct", SimpleNamespace(calcsize=lambda _format: 8))
    monkeypatch.setattr(o, "importlib", SimpleNamespace(metadata=SimpleNamespace(version=lambda name: runtime["packages"][name])))
    now = [0.0]
    monkeypatch.setattr(o.time, "monotonic", lambda: now[0])
    snapshot = SimpleNamespace(process_rss_bytes=64, available_memory_bytes=10000,
                               free_disk_bytes=10000, captured_at_utc="2026-10-20T00:00:00+00:00")
    monkeypatch.setattr(o, "capture_resource_snapshot", lambda _path: snapshot)
    path = code / "release.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    active = o._phase15_normalize(config)
    binding = o._phase15_binding(active, path, sha256(path.read_bytes()).hexdigest())
    attempts, audit = [], {"active": False}
    def attempted(kind, path):
        if audit["active"]:
            path = Path(path).absolute()
            attempts.append((kind, path))
            destination = root / "synthetic1"
            assert (kind == "mkdir" and path == destination) or (path.parent == destination and path.name in o.PHASE15_OUTPUTS)
    real_builtin_open, real_io_open = builtins.open, io.open
    real_raster_open, real_mkdir, real_link = rasterio.open, Path.mkdir, o.os.link
    def opened(delegate, path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            attempted("open", path)
        return delegate(path, mode, *args, **kwargs)
    def mkdir(path, *args, **kwargs):
        attempted("mkdir", path)
        return real_mkdir(path, *args, **kwargs)
    def link(source_path, target_path, *args, **kwargs):
        attempted("link_source", source_path)
        attempted("link_target", target_path)
        return real_link(source_path, target_path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", lambda path, mode="r", *args, **kwargs: opened(real_builtin_open, path, mode, *args, **kwargs))
    monkeypatch.setattr(io, "open", lambda path, mode="r", *args, **kwargs: opened(real_io_open, path, mode, *args, **kwargs))
    monkeypatch.setattr(rasterio, "open", lambda path, mode="r", *args, **kwargs: opened(real_raster_open, path, mode, *args, **kwargs))
    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(o.os, "link", link)
    # These spies observe Python/native-writer entrypoints, not every GDAL/OS-internal write attempt.
    return SimpleNamespace(config=config, path=path, active=active, binding=binding,
                           expected=independent_digest(binding), root=root, destination=root / "synthetic1",
                           now=now, snapshot=snapshot, model=model, payload=payload, source=source, runtime=runtime,
                           source_reads=source_reads, git_calls=git_calls, attempts=attempts, audit=audit)


def run(ready, **overrides):
    kwargs = dict(explicitly_authorized=True, approval_reference="synthetic-owner-approval",
                  expected_binding_sha256=ready.expected, config_path=ready.path)
    kwargs.update(overrides)
    ready.attempts.clear()
    ready.audit["active"] = True
    try:
        return o.run_simulation(ready.config, **kwargs)
    finally:
        ready.audit["active"] = False


def record(ready):
    return json.loads((ready.destination / "termination.json").read_text())


def rebind(ready):
    ready.path.write_text(yaml.safe_dump(ready.config), encoding="utf-8")
    ready.active = o._phase15_normalize(ready.config)
    ready.binding = o._phase15_binding(ready.active, ready.path, sha256(ready.path.read_bytes()).hexdigest())
    ready.expected = independent_digest(ready.binding)


@pytest.mark.parametrize("raw", [
    b"phase15: {}\nphase15: {}", b"phase15: {run_id: a, run_id: b}",
    b"phase15: {1: a, '1': b}", b"phase15: {true: a}",
    b"phase15: {<<: {run_id: a}, run_id: b}", b"phase15: {? [a,b]: c}",
])
def test_yaml_rejects_duplicate_nonstring_and_merge_keys_before_loss(raw):
    with pytest.raises((ValueError, yaml.YAMLError)):
        o._phase15_yaml(raw)


def test_nonstring_direct_keys_reject_before_canonical_coercion():
    with pytest.raises(TypeError, match="keys must be strings"):
        o._phase15_normalize({1: "a", "1": "b"})


def test_canonical_semantics_and_exact_payload(ready):
    assert independent_digest(ready.binding) == o.canonical_metadata_hash(ready.binding)
    assert set(ready.binding) == {"schema_version", "approval_reference", "configuration", "source", "runtime", "model", "inputs", "phase14_reconciliation_sha256", "run"}
    assert set(ready.binding["source"]["files"]) == set(o.PHASE15_SOURCES)
    assert "expected_binding_sha256" not in json.dumps(ready.binding)
    assert independent_digest({"a": 1, "b": 2}) == independent_digest({"b": 2, "a": 1})
    assert independent_digest([1, 2]) != independent_digest([2, 1])
    assert ready.active["phase15"]["enabled"] is True
    assert ready.config["phase15"]["enabled"] is False


@pytest.mark.parametrize("override", [
    {"explicitly_authorized": False}, {"approval_reference": "technical-assessment-only"},
    {"expected_binding_sha256": None}, {"expected_binding_sha256": "A" * 64},
    {"expected_binding_sha256": "0" * 64}, {"config_path": None},
])
def test_authorization_rejects_before_artifact_access_or_run_writes(ready, monkeypatch, override):
    for name in ("_phase15_environment", "_phase15_load_model", "load_model"):
        monkeypatch.setattr(o, name, lambda *_: pytest.fail("premature artifact access"))
    monkeypatch.setattr(FireAutomata, "load_model", lambda *_: pytest.fail("premature deserialization"))
    with pytest.raises((ValueError, PermissionError)):
        run(ready, **override)
    assert not ready.destination.exists()
    assert ready.attempts == []


@pytest.mark.parametrize("section,key", [
    ("configuration", "disk_sha256"), ("configuration", "activated_effective_sha256"),
    ("source", "head"), ("source", "dirty"), ("source", "files"),
    ("runtime", "python_version"), ("runtime", "packages"), ("model", "sha256"),
    ("inputs", "slope"), ("run", "run_id"), ("run", "scenario_id"), ("run", "destination"),
])
def test_each_binding_identity_is_covered_before_allocation(ready, monkeypatch, section, key):
    altered = deepcopy(ready.binding)
    altered[section][key] = "changed"
    monkeypatch.setattr(o, "_phase15_binding", lambda *_: altered)
    with pytest.raises(PermissionError, match="binding"):
        run(ready)
    assert not ready.destination.exists()


def test_disk_byte_change_and_direct_config_disagreement_reject(ready):
    ready.path.write_bytes(ready.path.read_bytes() + b"\n# byte change\n")
    with pytest.raises(PermissionError, match="binding"):
        run(ready)
    ready.config["simulation"]["seed"] += 1
    with pytest.raises(ValueError, match="disagree"):
        run(ready)
    assert not ready.destination.exists()


@pytest.mark.parametrize("section,key,value", [
    ("ml_model", "threshold", 0.5), ("ml_model", "inference_threshold", 0.5),
    ("ml_model", "classification_threshold", 0.5), ("simulation", "resume_checkpoint", "x"),
    ("simulation", "seed", True), ("simulation", "max_timesteps", None),
    ("ml_model", "inference_chunk_size", 0), ("phase15", "run_id", "../escape"),
    ("phase15", "run_id", "CON"), ("phase15", "expected_binding_sha256", "0" * 64),
])
def test_required_null_pending_and_type_rules(ready, section, key, value):
    config = deepcopy(ready.config)
    config[section][key] = value
    with pytest.raises((TypeError, ValueError)):
        o._phase15_normalize(config)


def test_disabled_template_retains_pending_values_without_activation():
    config = o.load_config(str(Path(o.__file__).parent / "config/phase15_setc_rf_ca_v1.yaml"))
    assert config["phase15"]["enabled"] is False
    assert config["simulation"]["seed"] is None
    assert config["simulation"]["resume_checkpoint"] is None
    with pytest.raises(ValueError):
        o._phase15_normalize(config)


def test_verified_buffer_survives_model_path_substitution(ready, monkeypatch):
    original = FireAutomata.load_model
    received = []

    def substitute(self, stream):
        # This deliberate adversarial fixture mutation is not a candidate output write.
        ready.audit["active"] = False
        try:
            ready.model.write_bytes(b"substituted bytes must never be deserialized")
        finally:
            ready.audit["active"] = True
        received.append(stream.getvalue())
        return original(self, stream)

    monkeypatch.setattr(FireAutomata, "load_model", substitute)
    run(ready)
    assert received == [ready.payload]
    assert record(ready)["output_valid"] is True


def test_initial_model_hash_mismatch_never_deserializes(ready, monkeypatch):
    ready.model.write_bytes(b"wrong initial bytes")
    monkeypatch.setattr(FireAutomata, "load_model", lambda *_: pytest.fail("deserialized unverified bytes"))
    with pytest.raises(ValueError, match="Model SHA-256 mismatch"):
        run(ready)
    assert record(ready)["output_valid"] is False
    assert not (ready.destination / "final_state.tif").exists()


def test_valid_pair_immutable_start_and_exact_environment_mask(ready):
    run(ready)
    result = record(ready)
    start_bytes = (ready.destination / "start.json").read_bytes()
    start = json.loads(start_bytes)
    assert result["start_record_sha256"] == sha256(start_bytes).hexdigest()
    assert start["model"]["observed_sha256"] is None
    assert start["scenario"]["converted_ignitions"] is None
    assert result["observed_bindings"]["converted_ignitions"] == [[0, 0]]
    assert result["termination_reason"] == "inactive" and result["last_completed_timestep"] == 1
    assert result["output_valid"] is True
    assert {path.name for path in ready.destination.iterdir()} == o.PHASE15_OUTPUTS
    assert {path.name for kind, path in ready.attempts if kind != "mkdir"} == o.PHASE15_OUTPUTS
    assert any(kind == "mkdir" for kind, _ in ready.attempts)
    with rasterio.open(ready.destination / "simulation_environment_valid.tif") as src:
        assert src.dtypes == ("uint8",) and src.nodata is None
        assert src.read(1).tolist() == [[1, 1, 0], [1, 1, 1]]
    with rasterio.open(ready.destination / "final_state.tif") as src:
        state = src.read(1)
        assert src.dtypes == ("int8",) and src.nodata is None
        assert state[0, 1] == state[0, 2] == 1
    for output in result["outputs"]:
        assert sha256((ready.destination / output["filename"]).read_bytes()).hexdigest() == output["sha256"]
    assert result["resource_observations"]["valid_count"] > 0


@pytest.mark.parametrize("cancellation", [False, True])
def test_timer_mutating_failed_step_never_exports_state(ready, monkeypatch, cancellation):
    def fail(engine):
        engine.ignition_timers[:] = 23
        engine.grid[:] = 5
        raise KeyboardInterrupt() if cancellation else ArithmeticError("mutated failed step")

    monkeypatch.setattr(FireAutomata, "step", fail)
    with pytest.raises((ArithmeticError, KeyboardInterrupt)):
        run(ready)
    result = record(ready)
    assert result["last_completed_timestep"] == 0
    assert result["state_counts"]["blazing"] == 1
    assert result["termination_reason"] == ("cancelled" if cancellation else "step_failure")
    assert not (ready.destination / "final_state.tif").exists()


def test_last_step_resource_and_wall_breach_precede_extinction(ready, monkeypatch):
    original = FireAutomata.step
    def step(engine):
        original(engine)
        ready.now[0] = 100
        ready.snapshot.process_rss_bytes = 1001
    monkeypatch.setattr(FireAutomata, "step", step)
    with pytest.raises(o._Phase15Stop, match="resource_limit"):
        run(ready)
    result = record(ready)
    assert result["last_completed_timestep"] == 1 and result["active_cell_count"] == 0
    assert result["violations"] == ["process_rss_bytes", "wall_time_limit"]
    assert not (ready.destination / "final_state.tif").exists()


@pytest.mark.parametrize("value", [None, -1, True, "100"])
def test_unavailable_invalid_telemetry_fails_before_allocation(ready, value):
    ready.snapshot.process_rss_bytes = value
    with pytest.raises(o._Phase15Stop, match="resource_measurement_failure"):
        run(ready)
    assert not ready.destination.exists()


def test_backend_exception_and_unsupported_supervision_fail_closed(ready, monkeypatch):
    config = deepcopy(ready.config)
    config["phase15"]["resources"]["enforcement"] = "external_supervision"
    with pytest.raises(ValueError, match="between_steps"):
        o._phase15_normalize(config)
    def unavailable(_path):
        raise OSError("telemetry unavailable")
    monkeypatch.setattr(o, "capture_resource_snapshot", unavailable)
    with pytest.raises(o._Phase15Stop, match="resource_measurement_failure"):
        run(ready)
    assert not ready.destination.exists()


def test_cadence_records_actual_gaps_and_sampled_extrema(ready):
    monitor = o._Phase15Resources(ready.active["phase15"]["resources"], ready.root, 0)
    monitor.check("first")
    ready.now[0] = 0.5
    monitor.check("not_due", force=False)
    ready.now[0] = 3
    monitor.check("late_boundary", force=False)
    summary = monitor.summary()
    assert summary["attempted_count"] == summary["valid_count"] == 2
    assert summary["max_gap_seconds"] == 3
    assert summary["sampled_max_rss_bytes"] == 64


@pytest.mark.parametrize("entrypoint", [o.simulation_main, main.main, o.run_simulation])
def test_shared_cli_preserves_clock_across_configuration_and_output_work(ready, monkeypatch, entrypoint):
    original_load, original_raster = o.load_config, o._phase15_raster
    original_binding, original_environment, original_model = o._phase15_binding, o._phase15_environment, o._phase15_load_model
    def load(path):
        ready.now[0] += 3
        return original_load(path)
    def raster(*args):
        ready.now[0] += 2
        return original_raster(*args)
    def binding(*args):
        ready.now[0] += 2
        return original_binding(*args)
    def environment(*args):
        ready.now[0] += 5
        return original_environment(*args)
    def model(*args):
        ready.now[0] += 7
        return original_model(*args)
    monkeypatch.setattr(o, "load_config", load)
    monkeypatch.setattr(o, "_phase15_raster", raster)
    monkeypatch.setattr(o, "_phase15_binding", binding)
    monkeypatch.setattr(o, "_phase15_environment", environment)
    monkeypatch.setattr(o, "_phase15_load_model", model)
    if entrypoint is o.run_simulation:
        run(ready)
    else:
        assert entrypoint(["--config", str(ready.path), "--execute-approved-phase15-pilot",
                           "--approval-reference", "synthetic-owner-approval", "--expected-binding-sha256", ready.expected]) == 0
    assert record(ready)["decision_elapsed_seconds"] == (18 if entrypoint is o.run_simulation else 21)


@pytest.mark.parametrize("entrypoint", [o.simulation_main, main.main])
def test_cli_without_independent_binding_cannot_allocate(ready, entrypoint):
    assert entrypoint(["--config", str(ready.path), "--execute-approved-phase15-pilot",
                       "--approval-reference", "synthetic-owner-approval"]) == 1
    assert not ready.destination.exists()


def test_completion_decision_precedes_its_own_publication_tail(ready, monkeypatch):
    original = o._phase15_publish
    def publish(*args):
        ready.now[0] = 200
        return original(*args)
    monkeypatch.setattr(o, "_phase15_publish", publish)
    run(ready)
    assert record(ready)["decision_elapsed_seconds"] == 0
    assert record(ready)["clock_policy"]["termination_publication_excluded"] is True


@pytest.mark.parametrize("target", ["start.json", "final_state.tif", "simulation_environment_valid.tif", "termination.json.partial", "termination.json"])
def test_publication_failures_never_establish_success_or_retry(ready, monkeypatch, target):
    write, raster, finalize = o._phase15_write_json, o._phase15_raster, o._phase15_finalize
    attempts = []
    def fail_write(path, value):
        attempts.append(path.name)
        if path.name == target:
            raise OSError("synthetic record write failure")
        return write(path, value)
    def fail_raster(destination, name, *args):
        if name == target:
            raise OSError("synthetic raster failure")
        return raster(destination, name, *args)
    def fail_finalize(partial, final):
        if final.name == target:
            raise OSError("synthetic finalization failure")
        return finalize(partial, final)
    monkeypatch.setattr(o, "_phase15_write_json", fail_write)
    monkeypatch.setattr(o, "_phase15_raster", fail_raster)
    monkeypatch.setattr(o, "_phase15_finalize", fail_finalize)
    with pytest.raises(OSError):
        run(ready)
    assert attempts.count("termination.json.partial") <= 1
    if (ready.destination / "termination.json").exists():
        assert record(ready)["output_valid"] is False
    assert {path.name for path in ready.destination.iterdir()} <= o.PHASE15_OUTPUTS


@pytest.mark.parametrize("postcommit", [False, True])
def test_precommit_reporting_failure_versus_postcommit_outer_incident(ready, monkeypatch, postcommit):
    committed_bytes = {}
    def report(message, **kwargs):
        if ("committed" in message) == postcommit:
            if postcommit:
                committed_bytes.update({name: (ready.destination / name).read_bytes() for name in
                    ("termination.json", "final_state.tif", "simulation_environment_valid.tif")})
            raise BrokenPipeError("synthetic console failure")
    monkeypatch.setattr(o, "print", report, raising=False)
    with pytest.raises((BrokenPipeError, RuntimeError)) as error:
        run(ready)
    result = record(ready)
    assert result["output_valid"] is postcommit
    if postcommit:
        assert "post_commit_reporting_failure" in str(error.value)
        assert result["exit_code"] == 0
        assert len(committed_bytes) == 3
        assert all((ready.destination / name).read_bytes() == payload for name, payload in committed_bytes.items())
    else:
        assert result["termination_reason"] == "reporting_failure"


def test_existing_destination_and_exclusive_finalization_never_overwrite(ready):
    ready.destination.mkdir()
    with pytest.raises(FileExistsError):
        run(ready)
    assert list(ready.destination.iterdir()) == []
    partial, final = ready.root / "source", ready.root / "existing"
    partial.write_bytes(b"new")
    final.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        o._phase15_finalize(partial, final)
    assert final.read_bytes() == b"preserved"


@pytest.mark.parametrize("defect", ["hash", "transform", "materials", "nodata", "companion"])
def test_input_identity_alignment_domain_and_mask_failures(ready, defect):
    path = Path(ready.binding["inputs"]["materials"]["path"])
    if defect == "hash":
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        with rasterio.open(path, "r+") as dst:
            if defect == "transform":
                dst.transform = from_origin(101, 200, 1, 1)
            elif defect == "materials":
                values = dst.read(1)
                values[0, 0] = 1.5
                dst.write(values, 1)
            elif defect == "nodata":
                dst.nodata = -8
            else:
                dst.write_mask(np.zeros((2, 3), dtype=np.uint8))
        ready.config["environment"]["input_sha256"]["materials"] = sha256(path.read_bytes()).hexdigest()
        rebind(ready)
    with pytest.raises(ValueError):
        run(ready)
    assert record(ready)["output_valid"] is False
    assert not (ready.destination / "final_state.tif").exists()


@pytest.mark.parametrize("points", [
    [[99, 201]], [[101.5, 199.5]], [[102.5, 199.5]], [[100.5, 199.5], [100.75, 199.75]],
])
def test_ignitions_reject_bounds_nonbuilding_invalid_and_duplicate_cells(ready, points):
    ready.config["simulation"]["ignition_points"] = points
    rebind(ready)
    with pytest.raises(ValueError):
        run(ready)
    assert record(ready)["last_completed_timestep"] is None
    assert not (ready.destination / "final_state.tif").exists()


def test_active_timestep_exhaustion_is_censored(ready):
    ready.config["placeholder_transition"]["T_4_to_5"] = 2
    rebind(ready)
    run(ready)
    result = record(ready)
    assert result["termination_reason"] == "timestep_limit"
    assert result["status"] == "censored" and result["active_cell_count"] == 1


def test_limit_after_verified_outputs_prevents_success_commit(ready, monkeypatch):
    original = o._phase15_raster
    def export(destination, name, *args):
        output = original(destination, name, *args)
        if name == "simulation_environment_valid.tif":
            ready.now[0] = 100
        return output
    monkeypatch.setattr(o, "_phase15_raster", export)
    with pytest.raises(o._Phase15Stop, match="wall_time_limit"):
        run(ready)
    result = record(ready)
    assert len(result["outputs"]) == 2
    assert result["output_valid"] is False
    assert result["termination_reason"] == "wall_time_limit"


def test_raster_round_trip_corruption_prevents_publication(ready, monkeypatch):
    original = o._Phase15Resources.check
    def check(monitor, boundary, force=True):
        if boundary == "before_verify_final_state.tif":
            with rasterio.open(ready.destination / "final_state.tif.partial", "r+") as dst:
                values = dst.read(1)
                values[0, 0] = 2
                dst.write(values, 1)
        return original(monitor, boundary, force)
    monkeypatch.setattr(o._Phase15Resources, "check", check)
    with pytest.raises(ValueError, match="round trip"):
        run(ready)
    assert not (ready.destination / "final_state.tif").exists()
    assert record(ready)["output_valid"] is False


def test_original_step_exception_survives_failure_reporting_error(ready, monkeypatch):
    primary = ArithmeticError("primary step error")
    def fail_step(_engine):
        raise primary
    def fail_report(*_args):
        raise OSError("secondary reporting error")
    monkeypatch.setattr(FireAutomata, "step", fail_step)
    monkeypatch.setattr(o, "_phase15_publish", fail_report)
    with pytest.raises(ArithmeticError) as error:
        run(ready)
    assert error.value is primary
    assert not (ready.destination / "termination.json").exists()


@pytest.mark.parametrize("controls", [
    {"explicitly_authorized": True}, {"approval_reference": "synthetic-owner-approval"},
    {"expected_binding_sha256": "0" * 64}, {"config_path": Path("phase15.yaml")}, {"config_path": ""},
])
def test_legacy_observer_dispatch_and_pilot_rejection_remain_separate(monkeypatch, controls):
    calls = []
    config, observer, provenance = {}, lambda value: None, {"synthetic": True}
    monkeypatch.setattr(o, "_run_legacy_simulation", lambda value, **kwargs: calls.append((value, kwargs)))
    o.run_simulation(config, transition_observer=observer, transition_provenance=provenance)
    assert calls == [(config, {"transition_observer": observer, "transition_provenance": provenance})]
    with pytest.raises(PermissionError):
        o.run_simulation(config, **controls)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="does not collect"):
        o.run_simulation({"phase15": {}}, transition_observer=observer)


@pytest.mark.parametrize("name,value,accepted", [
    ("materials", 1.00000001, False), ("buildings", 10.00000001, False),
    ("materials", 1.0, True), ("buildings", 10.0, True),
])
def test_native_float64_categories_precede_loader_narrowing(ready, monkeypatch, name, value, accepted):
    path = Path(ready.binding["inputs"][name]["path"])
    with rasterio.open(path) as src:
        values, profile = src.read(1).astype(np.float64), src.profile
    values[0, 0] = value
    profile.update(dtype="float64")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
    ready.config["environment"]["input_sha256"][name] = sha256(path.read_bytes()).hexdigest()
    rebind(ready)
    if accepted:
        run(ready)
        assert record(ready)["output_valid"] is True
    else:
        assert value != float(np.float32(value))
        assert float(np.float32(value)) in (1.0, 10.0)
        monkeypatch.setattr(o.EnvironmentManager, "load_rasters", lambda *_: pytest.fail("narrowed before validation"))
        with pytest.raises(ValueError, match="original precision"):
            run(ready)
        assert record(ready)["output_valid"] is False


def test_real_binding_adapters_use_synthetic_dependencies(ready):
    assert o._phase15_source() == ready.source
    assert o._phase15_runtime() == ready.runtime
    assert {path.relative_to(Path(ready.source["repository_root"])).as_posix() for path in ready.source_reads} == set(o.PHASE15_SOURCES)
    assert ready.git_calls
    assert all(kwargs["check"] and kwargs["capture_output"] and kwargs["text"] for _, kwargs in ready.git_calls)
    assert all(f"safe.directory={ready.source['repository_root']}" in args for args, _ in ready.git_calls)


def test_shared_trainer_probability_caller_uses_unfitted_estimator():
    import pandas as pd
    from modules.model_trainer import ModelTrainer

    features = pd.DataFrame(np.zeros((2, 11), dtype=np.float32), columns=CANONICAL_FEATURE_NAMES)
    trainer = object.__new__(ModelTrainer)
    trainer.model = SyntheticEstimator()
    trainer.get_partition = lambda _role: (features, np.array([0, 1], dtype=np.int8))
    metrics = trainer.evaluate()
    assert metrics["observation_count"] == 2
    assert metrics["brier_score"] == pytest.approx(0.5)


def test_attempted_write_spy_rejects_undeclared_path_before_creation(ready, monkeypatch):
    def undeclared(destination, *_args):
        with (destination / "undeclared.tmp").open("wb"):
            pass
    monkeypatch.setattr(o, "_phase15_raster", undeclared)
    with pytest.raises(AssertionError):
        run(ready)
    assert any(path.name == "undeclared.tmp" for _, path in ready.attempts)
    assert not (ready.destination / "undeclared.tmp").exists()
    assert record(ready)["output_valid"] is False


@pytest.mark.parametrize("target,damage", [
    ("start.json", "partial"), ("termination.json.partial", "partial"),
    ("termination.json.partial", "corrupt"),
])
def test_partial_and_corrupt_record_writes_never_commit_or_retry(ready, monkeypatch, target, damage):
    original_open, writes = Path.open, []
    class DamagedWriter:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            self.stream.__enter__()
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def __getattr__(self, name):
            return getattr(self.stream, name)
        def write(self, payload):
            writes.append(target)
            self.stream.write(payload[:max(1, len(payload) // 2)] if damage == "partial" else b"corrupt")
            if damage == "partial":
                raise OSError("partial record write")
            return len(payload)
    def opened(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return DamagedWriter(stream) if path.name == target and mode == "xb" else stream
    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(OSError):
        run(ready)
    assert writes == [target]
    assert (ready.destination / target).read_bytes()
    if target == "start.json":
        assert record(ready)["output_valid"] is False
        assert record(ready)["start_record_sha256"] is None
    else:
        assert not (ready.destination / "termination.json").exists()
