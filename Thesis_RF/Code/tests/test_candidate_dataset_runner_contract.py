"""Bounded synthetic controls; no real-input runner or full campaign."""
from copy import deepcopy
import gc
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import weakref

import numpy as np
import pytest
import yaml

import run_candidate_dataset as runner
from modules import candidate_dataset_collection as collection
from modules import candidate_dataset_release as release
from test_candidate_dataset_collection_contract import (
    TracedCandidateGenerator, assert_candidate_pair, candidate_reference_pair, context,
)


def authority_fixture(tmp_path):
    prepared, config, spec, source, construct = context()
    source["source_sha256"] = dict.fromkeys(runner.SOURCE_FILES, "c" * 64)
    authority = dict(execution_approval_reference=config["authorization_reference"],
        publication_approval_reference="synthetic publication substitute", source=source,
        runtime={"synthetic": "explicit runtime substitute"},
        run_bindings={p["run_id"]: "a" * 64 for p in collection.planned_runs(config["inventory"])},
        destinations=dict(evidence=str(tmp_path / "evidence"), package=str(tmp_path / "package")),
        custody=dict(approval_reference="synthetic custody substitute", families={
            f["scenario_family_id"]: dict(eligible=False, prior_use=f["prior_use"],
                assessment_reference="synthetic development-only assessment")
            for f in config["inventory"]}), allocation_node_budget=10000,
        disk_configuration_sha256="d" * 64)
    execution = release.digest_bytes(release.json_bytes(runner.execution_payload(config, authority)))
    publication = release.digest_bytes(release.json_bytes(runner.publication_payload(execution, authority)))
    kwargs = dict(execution_authorized=True, publication_authorized=True,
                  expected_execution_sha256=execution, expected_publication_sha256=publication)
    return prepared, config, spec, source, construct, authority, kwargs


def engine_snapshot(engine):
    return dict(grid=engine.grid.copy(), ignition=engine.ignition_timers.copy(),
        blazing=engine.blazing_timers.copy(), rng=deepcopy(engine.rng.bit_generator.state),
        row_count=engine.row_count, completed=deepcopy(engine.last_completed),
        failed=engine.failed, incomplete=deepcopy(engine.incomplete_transition),
        capture=engine._capture, pending=engine._pending, timestep=engine.timestep)


def assert_engine_preserved(engine, before):
    assert np.array_equal(engine.grid, before["grid"])
    assert np.array_equal(engine.ignition_timers, before["ignition"])
    assert np.array_equal(engine.blazing_timers, before["blazing"])
    assert engine.rng.bit_generator.state == before["rng"]
    assert engine.row_count == before["row_count"]
    assert engine.last_completed == before["completed"]
    assert engine.failed == before["failed"] and engine.incomplete_transition == before["incomplete"]
    assert engine._capture is before["capture"] and engine._pending is before["pending"]
    assert engine.timestep == before["timestep"]


def test_successful_acknowledgement_releases_only_engine_references():
    engine = context()[-1]()
    first = engine.step()
    snapshot = engine.batches
    engine.failed = True
    engine.incomplete_transition = dict(collection_completed=False, synthetic=True)
    before = engine_snapshot(engine)
    engine.acknowledge_batches(snapshot)
    assert engine.batches == ()
    assert snapshot[0] is first
    assert_engine_preserved(engine, before)


@pytest.mark.parametrize("kind", ["empty", "substituted", "partial", "stale", "list"])
def test_bad_acknowledgement_is_identity_based_and_nonmutating(kind):
    engine = context()[-1]()
    engine.step()
    old = engine.batches
    engine.step()
    current = engine.batches
    supplied = {"empty": (), "substituted": tuple(list(current)), "partial": current[:1],
                "stale": old, "list": list(current)}[kind]
    before = engine_snapshot(engine)
    with pytest.raises(ValueError, match="Exact current nonempty"):
        engine.acknowledge_batches(supplied)
    assert engine.batches is current
    assert_engine_preserved(engine, before)


def test_failed_persistence_never_acknowledges_and_retains_completed_evidence():
    engine = context()[-1]()
    engine.step()
    snapshot, before = engine.batches, engine_snapshot(engine)
    class FailingStore:
        def persist(self, binding, batches):
            assert batches is snapshot
            raise OSError("injected persistence/readback failure")
    with pytest.raises(OSError, match="persistence"):
        runner.persist_and_acknowledge(engine, FailingStore())
    assert engine.batches is snapshot
    assert_engine_preserved(engine, before)


def test_real_store_persists_reconciles_then_acknowledges(tmp_path):
    engine = context()[-1]()
    engine.step()
    before = engine_snapshot(engine)
    store = runner.EvidenceStore(tmp_path / "evidence")
    runner.persist_and_acknowledge(engine, store)
    store.close()
    assert engine.batches == ()
    assert store.rows == engine.row_count
    assert_engine_preserved(engine, before)
    retained = json.loads((tmp_path / "evidence/transitions.jsonl").read_bytes())
    assert retained["envelope"]["bindings"] == [
        {k: r[k] for k in ("observation_id", "q32_bits", "row_sha256")} for r in retained["rows"]]


def test_draining_preserves_cumulative_cap_and_last_completed(monkeypatch):
    engine = context()[-1]()
    engine.step()
    engine.acknowledge_batches(engine.batches)
    rows, completed = engine.row_count, deepcopy(engine.last_completed)
    monkeypatch.setattr(collection, "MAX_RUN_ROWS", rows - 1)
    with pytest.raises(RuntimeError, match="row cap"):
        engine.step()
    assert engine.row_count == rows and engine.last_completed == completed
    assert engine.batches == () and engine.failed
    assert engine.incomplete_transition["collection_completed"] is False


@pytest.mark.parametrize("fault", ["execution", "publication", "inventory", "resource",
                                   "authorization", "capability", "run_binding", "digest", "custody"])
def test_activation_rejection_precedes_inputs_resources_and_writes(tmp_path, monkeypatch, fault):
    _, config, _, _, _, authority, kwargs = authority_fixture(tmp_path)
    def forbidden(*args, **kw):
        pytest.fail("input/resource/write access after rejected authorization")
    monkeypatch.setattr(runner, "read_inputs", forbidden)
    monkeypatch.setattr(runner, "Resources", forbidden)
    monkeypatch.setattr(runner, "EvidenceStore", forbidden)
    if fault in ("execution", "publication"):
        kwargs[fault + "_authorized"] = False
    elif fault == "inventory":
        config["inventory"] = None
    elif fault == "resource":
        config["resources"]["max_wall_seconds"] = None
    elif fault == "authorization":
        config["authorization_reference"] = None
    elif fault == "capability":
        config["reader_capability"]["controls_verified"] = False
    elif fault == "run_binding":
        authority["run_bindings"].pop(next(iter(authority["run_bindings"])))
    elif fault == "digest":
        kwargs["expected_execution_sha256"] = "0" * 64
    else:
        authority["custody"]["families"].clear()
    with pytest.raises(runner.CandidateFailure) as captured:
        runner.run(config, authority, **kwargs)
    assert captured.value.report["evidence"] is None
    assert not (tmp_path / "evidence").exists()
    assert captured.value.report["original_failure"] is not None


def test_real_source_runtime_adapter_with_synthetic_hash_and_dependency_observations(tmp_path, monkeypatch):
    import importlib
    _, _, _, source, _, authority, _ = authority_fixture(tmp_path)
    modules = {name: SimpleNamespace(__version__="synthetic-version", __file__=str(tmp_path / (name + ".py")))
               for name in ("numpy", "scipy", "pandas", "rasterio", "joblib", "yaml")}
    modules["rasterio"].__gdal_version__ = "synthetic-gdal"
    monkeypatch.setattr(importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(Path, "is_symlink", lambda _: False)
    monkeypatch.setattr(Path, "is_junction", lambda _: False)
    monkeypatch.setattr(Path, "is_file", lambda _: True)
    monkeypatch.setattr(runner, "file_identity", lambda path, check: dict(
        bytes=1, sha256="e" * 64 if str(path) == sys.executable else "c" * 64))
    runtime = dict(python=list(sys.version_info[:3]), executable=str(Path(sys.executable).resolve()),
        executable_sha256="e" * 64, gdal="synthetic-gdal",
        venv_configuration=str(Path(sys.prefix) / "pyvenv.cfg"), venv_configuration_sha256="c" * 64)
    runtime.update({name: dict(version="synthetic-version", origin=str(Path(m.__file__).resolve()))
                    for name, m in modules.items()})
    authority["runtime"] = runtime
    assert runner.verify_sources_runtime(authority, lambda _: None) == runtime
    source["source_sha256"][runner.SOURCE_FILES[0]] = "0" * 64
    with pytest.raises(PermissionError, match="source identity"):
        runner.verify_sources_runtime(authority, lambda _: None)


@pytest.mark.parametrize("fault", ["rss", "available", "disk", "wall", "missing", "probe"])
def test_resource_fail_closed_latches_original_observation(fault, tmp_path):
    limits = dict(max_process_rss_bytes=100, min_available_physical_memory_bytes=50,
        min_free_disk_bytes=50, max_wall_seconds=10, observation_interval_seconds=1)
    def probe(_):
        if fault == "probe":
            raise OSError("synthetic telemetry failure")
        return SimpleNamespace(process_rss_bytes=101 if fault == "rss" else 1,
            available_memory_bytes=49 if fault == "available" else 100,
            free_disk_bytes=None if fault == "missing" else 49 if fault == "disk" else 100)
    guard = runner.Resources(limits, [tmp_path], 0, probe=probe,
                             clock=lambda: 11 if fault == "wall" else 1)
    with pytest.raises((RuntimeError, OSError)):
        guard.check("synthetic")
    original = guard.failure
    with pytest.raises(RuntimeError, match="latched"):
        guard.check("again")
    assert guard.failure is original


def bounded_runner(tmp_path, monkeypatch):
    prepared, config, spec, source, construct, authority, kwargs = authority_fixture(tmp_path)
    binding = collection.execution_binding(config, prepared, spec, source)
    authority["run_bindings"][spec["run_id"]] = release.digest_bytes(release.json_bytes(binding))
    monkeypatch.setattr(runner, "validate_authority", lambda *a, **kw: ([spec], {"synthetic": True}))
    monkeypatch.setattr(runner, "verify_sources_runtime", lambda *a: {})
    monkeypatch.setattr(runner, "read_inputs", lambda *a, **kw: prepared)
    class Guard:
        def __init__(self, *a):
            self.samples = []
        def check(self, name):
            pass
    monkeypatch.setattr(runner, "Resources", Guard)
    monkeypatch.setattr(runner, "load_completed_evidence", lambda *a: ([], []))
    monkeypatch.setattr(release, "components", lambda *a: [])
    def publish(*a, **kw):
        return dict(committed=True, package=authority["destinations"]["package"])
    monkeypatch.setattr(release, "publish", publish)
    return config, authority, kwargs, spec


@pytest.mark.parametrize("stage", ["step", "callback", "close", "after_commit", "cap"])
def test_bounded_runner_failures_retain_evidence_without_full_campaign(tmp_path, monkeypatch, stage):
    config, authority, kwargs, _ = bounded_runner(tmp_path, monkeypatch)
    if stage == "step":
        def failed(self):
            self.incomplete_transition = {"collection_completed": False}
            raise ValueError("synthetic underlying step error")
        original_factory = collection.CandidateAutomata
        def factory(*a, **kw):
            engine = original_factory(*a, **kw)
            engine.step = lambda: failed(engine)
            return engine
        monkeypatch.setattr(collection, "CandidateAutomata", factory)
    elif stage == "cap":
        original_factory = collection.CandidateAutomata
        def capped_factory(*a, **kw):
            engine = original_factory(*a, **kw)
            engine.timestep = collection.MAX_UPDATES  # Synthetic cap state, no 300-step replay.
            return engine
        monkeypatch.setattr(collection, "CandidateAutomata", capped_factory)
    elif stage == "close":
        original_close = runner.EvidenceStore.close
        def broken_close(self):
            original_close(self)
            raise OSError("synthetic close failure")
        monkeypatch.setattr(runner.EvidenceStore, "close", broken_close)
    def emit(record):
        if stage == "callback" and record["kind"] == "completed_transition":
            raise RuntimeError("synthetic callback failure")
        if stage == "after_commit" and record["kind"] == "package_committed":
            raise RuntimeError("synthetic postcommit console failure")
    with pytest.raises(runner.CandidateFailure) as captured:
        runner.run(config, authority, emit=emit, **kwargs)
    report = captured.value.report
    assert not report["complete"] and report["original_failure"] is not None
    assert (tmp_path / "evidence/runs.jsonl").exists()
    assert (tmp_path / "evidence/outcome.json").exists()
    assert report["last_completed"] is not None
    if stage == "step":
        assert report["original_failure"]["type"] == "ValueError"
        assert report["incomplete_transition"]["collection_completed"] is False
    if stage == "after_commit":
        assert report["committed"]["committed"] is True


def test_final_console_failure_preserves_committed_record_and_original_error(monkeypatch):
    report = dict(complete=True, committed=dict(committed=True), original_failure=None,
                  finalization_failures=[])
    monkeypatch.setattr("builtins.print", lambda *a, **kw: (_ for _ in ()).throw(OSError("console lost")))
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    with pytest.raises(runner.CandidateFailure) as captured:
        runner.print_final_report(report)
    assert captured.value.report["committed"] == {"committed": True}
    assert report["complete"] is False and "console lost" in report["finalization_failures"][0]


@pytest.mark.parametrize("text", ["a: 1\na: 2\n", "1: value\n"])
def test_yaml_duplicate_and_nonstring_keys_reject_before_normalization(text):
    with pytest.raises(ValueError):
        yaml.load(text, Loader=runner.UniqueLoader)


def test_empty_initial_acknowledgement_rejects_without_changes():
    engine = context()[-1]()
    before = engine_snapshot(engine)
    with pytest.raises(ValueError):
        engine.acknowledge_batches(engine.batches)
    assert engine.batches == ()
    assert_engine_preserved(engine, before)


def test_missing_or_partial_completed_records_block_reconciliation(tmp_path):
    root = tmp_path / "evidence"
    store = runner.EvidenceStore(root)
    store.close()
    _, config, _, _, _, authority, _ = authority_fixture(tmp_path)
    with pytest.raises(ValueError, match="initialization"):
        runner.load_completed_evidence(root, config, authority, lambda _: None)
    with (root / "transitions.jsonl").open("ab") as stream:
        stream.write(b'{"partial":')
    with pytest.raises(ValueError):
        runner.load_completed_evidence(root, config, authority, lambda _: None)


def test_package_row_cap_prevents_persistence_and_acknowledgement(tmp_path, monkeypatch):
    engine = context()[-1]()
    engine.step()
    snapshot = engine.batches
    store = runner.EvidenceStore(tmp_path / "evidence")
    monkeypatch.setattr(collection, "MAX_PACKAGE_ROWS", engine.row_count - 1)
    with pytest.raises(RuntimeError, match="Package row cap"):
        runner.persist_and_acknowledge(engine, store)
    store.close()
    assert engine.batches is snapshot and store.rows == 0
    assert (tmp_path / "evidence/transitions.jsonl").read_bytes() == b""


def test_real_reconciliation_of_one_bounded_inactive_run(tmp_path, monkeypatch):
    _, config, spec, _, construct = context()
    engine = construct()
    root = tmp_path / "evidence"
    store = runner.EvidenceStore(root)
    store.append("runs", dict(kind="initialization", spec=spec,
                              completed=deepcopy(engine.last_completed)))
    for _ in range(64):
        if not engine.is_active():
            break
        engine.step()
        runner.persist_and_acknowledge(engine, store)
    assert not engine.is_active()
    store.append("runs", dict(kind="inactive", spec=spec,
                              completed=deepcopy(engine.last_completed)))
    store.close()
    # Only the reconciliation plan is substituted; no 364-run campaign is launched.
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    authority = {"run_bindings": {spec["run_id"]: engine.run_identity_sha256}}
    rows, runs = runner.load_completed_evidence(root, config, authority, lambda _: None)
    assert len(rows) == engine.row_count and len(runs) == 2
    runs[-1]["kind"] = "capped"
    (root / "runs.jsonl").write_bytes(b"".join(release.json_bytes(r) + b"\n" for r in runs))
    with pytest.raises(ValueError, match="completeness"):
        runner.load_completed_evidence(root, config, authority, lambda _: None)


@pytest.mark.parametrize("reason", ["infeasible", "node_budget", "resource"])
def test_runner_preserves_distinct_allocation_outcomes(tmp_path, monkeypatch, reason):
    config, authority, kwargs, _ = bounded_runner(tmp_path, monkeypatch)
    def allocation_failure(*args, **options):
        options["check"]("allocation_search")
        if reason == "infeasible":
            raise release.AllocationInfeasible("synthetic exhaustive infeasibility")
        if reason == "node_budget":
            raise release.AllocationIncomplete("synthetic node budget exhaustion")
        raise RuntimeError("synthetic allocation resource stop")
    monkeypatch.setattr(release, "publish", allocation_failure)
    with pytest.raises(runner.CandidateFailure) as captured:
        runner.run(config, authority, **kwargs)
    expected = "allocation_infeasible" if reason == "infeasible" else "allocation_incomplete"
    assert captured.value.report["allocation_status"] == expected
    assert captured.value.report["committed"] is None


def final_observation_fixture(tmp_path, monkeypatch, *, fail_final=False):
    config, authority, kwargs, _ = bounded_runner(tmp_path, monkeypatch)
    guards, writes = [], []
    marker = b'{"synthetic_committed":true}\n'
    package = Path(authority["destinations"]["package"])
    committed = dict(committed=True, package=str(package),
                     manifest=dict(bytes=len(marker), sha256=release.digest_bytes(marker)))

    def engine(*args, **options):
        return SimpleNamespace(last_completed=dict(state_timestep=0, committed_rows=0,
            state_counts=dict(ignited=0, blazing=0)), incomplete_transition=None,
            batches=(), row_count=0, timestep=0, is_active=lambda: False,
            validate_handoff=lambda *, expected_context: None,
            latch_handoff_failure=lambda: None)

    class Guard:
        def __init__(self, *args):
            self.samples, self.calls, self.snapshot_bytes = [], [], None
            guards.append(self)

        def check(self, boundary):
            self.calls.append(boundary)
            if boundary == "after_outcome_publication":
                self.snapshot_bytes = (tmp_path / "evidence/outcome.json").read_bytes()
                elapsed = 100.0
            else:
                elapsed = float(len(self.samples) + 1)
            self.samples.append(dict(boundary=boundary, elapsed_seconds=elapsed,
                rss_bytes=1, available_bytes=2, free_disk_bytes=3))
            if boundary == "after_outcome_publication" and fail_final:
                raise OSError("synthetic final observation failure")

    def publish(*args, **options):
        # Synthetic commit bytes only; the real publisher is not exercised here.
        package.mkdir()
        with (package / "COMMITTED.json").open("xb") as stream:
            stream.write(marker)
        return deepcopy(committed)

    original_write = release.write_new
    def write_new(path, chunks, check):
        writes.append(Path(path).name)
        return original_write(path, chunks, check)

    monkeypatch.setattr(collection, "CandidateAutomata", engine)
    monkeypatch.setattr(runner, "Resources", Guard)
    monkeypatch.setattr(release, "publish", publish)
    monkeypatch.setattr(release, "write_new", write_new)
    return config, authority, kwargs, guards, writes, committed, marker


def test_final_observation_largest_gap_updates_returned_and_console_report(tmp_path, monkeypatch, capsys):
    config, authority, kwargs, guards, writes, committed, marker = final_observation_fixture(
        tmp_path, monkeypatch)
    report = runner.run(config, authority, **kwargs)
    guard = guards[0]
    snapshot = json.loads(guard.snapshot_bytes)
    times = [sample["elapsed_seconds"] for sample in report["resources"]]
    gaps = [b - a for a, b in zip([0.0] + times, times)]
    assert gaps[-1] > max(gaps[:-1])
    assert report["resource_policy"]["max_actual_gap_seconds"] == gaps[-1]
    assert snapshot["resource_policy"]["max_actual_gap_seconds"] == max(gaps[:-1])
    assert snapshot["report_stage"] == "outcome_snapshot"
    assert snapshot["resource_policy"]["coverage_boundary"] == "before_outcome_publication"
    assert snapshot["outcome_snapshot"]["not_final_completion_evidence"] is True
    assert snapshot["resources"] == report["resources"][:-1]
    assert report["resources"] == guard.samples and report["resources"] is not guard.samples
    assert report["resource_policy"]["resource_sample_count"] == len(report["resources"])
    assert report["resource_policy"]["last_resource_boundary"] == "after_outcome_publication"
    assert report["report_stage"] == "final_report" and report["complete"] is True
    assert report["committed"] == committed
    assert writes == ["outcome.json"]
    assert (tmp_path / "evidence/outcome.json").read_bytes() == guard.snapshot_bytes
    assert (Path(committed["package"]) / "COMMITTED.json").read_bytes() == marker
    runner.print_final_report(report)
    console = json.loads(capsys.readouterr().out)
    assert console["resources"] == report["resources"]
    assert console["resource_policy"] == report["resource_policy"]
    assert guard.calls[-1] == "after_outcome_publication"
    assert guard.calls.count("after_outcome_publication") == 1


@pytest.mark.parametrize("earlier_failure", [False, True])
def test_failing_final_observation_preserves_snapshot_primary_failure_and_commit(
        tmp_path, monkeypatch, capsys, earlier_failure):
    config, authority, kwargs, guards, writes, committed, marker = final_observation_fixture(
        tmp_path, monkeypatch, fail_final=True)
    def emit(record):
        if earlier_failure and record["kind"] == "package_committed":
            raise RuntimeError("original postcommit callback failure")
    with pytest.raises(runner.CandidateFailure) as captured:
        runner.run(config, authority, emit=emit, **kwargs)
    report, guard = captured.value.report, guards[0]
    snapshot = json.loads(guard.snapshot_bytes)
    assert report["complete"] is False and report["report_stage"] == "final_report"
    assert snapshot["complete"] is (not earlier_failure)
    assert snapshot["report_stage"] == "outcome_snapshot"
    assert snapshot["finalization_failures"] == []
    assert any("final resource observation" in item and "synthetic final observation failure" in item
               for item in report["finalization_failures"])
    assert report["resources"] == guard.samples and snapshot["resources"] == guard.samples[:-1]
    assert report["resource_policy"]["max_actual_gap_seconds"] == (
        guard.samples[-1]["elapsed_seconds"] - guard.samples[-2]["elapsed_seconds"])
    assert report["resource_policy"]["coverage_boundary"] == "after_last_attempted_resource_observation"
    assert report["original_failure"] == snapshot["original_failure"]
    if earlier_failure:
        assert report["original_failure"]["message"] == "original postcommit callback failure"
        assert isinstance(captured.value.__cause__, RuntimeError)
    else:
        assert report["original_failure"] is None and captured.value.__cause__ is None
    assert report["committed"] == committed
    assert writes == ["outcome.json"]
    assert (tmp_path / "evidence/outcome.json").read_bytes() == guard.snapshot_bytes
    assert (Path(committed["package"]) / "COMMITTED.json").read_bytes() == marker
    runner.print_final_report(report)
    console = json.loads(capsys.readouterr().out)
    assert console["finalization_failures"] == report["finalization_failures"]
    assert console["resource_policy"] == report["resource_policy"]
    assert guard.calls[-1] == "after_outcome_publication"
    assert guard.calls.count("after_outcome_publication") == 1


def test_constructed_completed_evidence_handoff_to_actual_publisher(tmp_path):
    from dataclasses import replace
    from test_candidate_dataset_release_contract import publication_fixture

    seed_rows, _, options = publication_fixture(tmp_path)
    provenance = options["authority"]
    config, authority = provenance["configuration"], provenance["execution_authority"]
    prepared = replace(context()[0], full_grid_shape=tuple(config["grid"]["shape"]))
    plan = collection.planned_runs(config["inventory"])
    authority["run_bindings"] = {}
    store = runner.EvidenceStore(tmp_path / "constructed-completed-evidence")
    authority["destinations"]["evidence"] = str(store.root)
    options["payload"] = store.root / "payload"
    expected_rows = []
    # Construct bounded metadata/batches, not CA histories or a generation campaign.
    # Probabilities and inactive states here are synthetic fixture declarations.
    try:
        for spec in plan:
            binding = collection.execution_binding(config, prepared, spec, authority["source"])
            identity = release.digest_bytes(release.json_bytes(binding))
            authority["run_bindings"][spec["run_id"]] = identity
            selected = [row for row in seed_rows if row["payload"]["run_id"] == spec["run_id"]]
            decoded = [release.decode_row(row["csv"]) for row in selected]
            features = np.array([row[0] for row in decoded], dtype=np.float32).reshape(-1, 11)
            labels = np.array([row[1] for row in decoded], dtype=np.int8)
            rows = np.array([row["payload"]["cell_row"] for row in selected], dtype=np.int64)
            cols = np.array([row["payload"]["cell_col"] for row in selected], dtype=np.int64)
            bits = np.array([row["q32_bits"] for row in selected], dtype=np.uint32)
            family = config["inventory"][spec["family_ordinal"] - 1]
            generic = dict(event_id=family["event_id"], run_id=spec["run_id"], seed=spec["seed"],
                grid_id=authority["source"]["grid_id"], candidate_run_identity_sha256=identity,
                **{name: binding[name] for name in ("teacher_id", "numerical_contract_id",
                    "interpretation_policy", "mask_seed_policy")})
            envelope = dict(schema_version="candidate_comparison_capture.v1",
                run_identity_sha256=identity, provenance_sha256=collection.canonical_metadata_hash(generic),
                generic_provenance=generic, timestep_t=0, timestep_t1=1,
                state_t_sha256="1" * 64, state_t1_sha256="2" * 64,
                eligible_mask_sha256="3" * 64,
                **{name: binding[name] for name in ("teacher_id", "numerical_contract_id",
                    "interpretation_policy", "mask_seed_policy")})
            ids = np.array([release.digest_bytes(release.json_bytes(release.observation_payload(
                binding, envelope, int(row), int(col)))).encode("ascii")
                for row, col in zip(rows, cols, strict=True)], dtype="S64")
            envelope["bindings"] = [dict(observation_id=identifier.decode("ascii"), q32_bits=int(q),
                row_sha256=release.digest_bytes(np.asarray(feature, dtype="<f4").tobytes()
                    + bytes([int(label)]))) for identifier, q, feature, label in
                zip(ids, bits, features, labels, strict=True)]
            arrays = (features, labels, rows, cols, ids, bits)
            for array in arrays:
                array.setflags(write=False)
            batch = collection.CandidateBatch(*arrays, envelope_bytes=release.json_bytes(envelope))
            _, reconciled = release.reconcile_batch(batch, binding)
            expected_rows.extend(reconciled)
            cells, seeds, positives = int(np.prod(config["grid"]["shape"])), len(family["ignition_points"]), int(labels.sum())
            initial = dict(state_timestep=0, committed_rows=0, state_counts=dict(
                non_burnable=0, not_yet_burning=cells - seeds, ignited=0, blazing=seeds, extinguished=0))
            final = dict(state_timestep=1, committed_rows=len(labels), state_counts=dict(
                non_burnable=0, not_yet_burning=cells - seeds - positives, ignited=0,
                blazing=0, extinguished=seeds + positives))
            store.append("runs", dict(kind="initialization", spec=spec, completed=initial))
            store.persist(binding, (batch,))
            store.append("runs", dict(kind="inactive", spec=spec, completed=final))
    finally:
        store.close()
    retained = {name: (store.root / name).read_bytes() for name in ("transitions.jsonl", "runs.jsonl")}
    rows, runs = runner.load_completed_evidence(store.root, config, authority, lambda _: None)
    assert rows == expected_rows and len(rows) == store.rows == 2600
    assert len(runs) == 728 and sum(store.transitions.values()) == len(plan) == 364
    items = release.components(rows, config["complete_event_ids"], authority["custody"]["families"], lambda _: None)
    assert len(items) == 26 and all(item["positives"] == 20 and item["negatives"] == 80 for item in items)
    provenance.update(binding=runner.publication_payload(release.digest_bytes(release.json_bytes(
        runner.execution_payload(config, authority))), authority), run_records=runs,
        evidence_files={name: runner.file_identity(store.root / name, lambda _: None) for name in retained},
        dataset_sha256=release.digest_bytes(release.HEADER + b"".join(row["csv"].encode("ascii") for row in rows)))
    options["expected_binding_sha256"] = release.digest_bytes(release.json_bytes(provenance["binding"]))
    checks = []
    options["check"] = checks.append
    result = release.publish(rows, items, **options)
    package = Path(result["package"])
    manifest_bytes = (package / "custodian/manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert result["committed"] is True and not options["payload"].exists()
    assert json.loads((package / "COMMITTED.json").read_bytes())["manifest_sha256"] == release.digest_bytes(manifest_bytes)
    assert (package / "custodian/master.csv").read_bytes() == release.HEADER + b"".join(row["csv"].encode("ascii") for row in rows)
    assert [json.loads(line) for line in (package / "custodian/observations.jsonl").read_bytes().splitlines()] == rows
    assert manifest["authority"]["run_records"] == runs
    assert manifest["authority"]["evidence_files"] == provenance["evidence_files"]
    assert {name: (store.root / name).read_bytes() for name in retained} == retained
    assert {"allocation_search", "allocation_completed", "publication_completion_decision"}.issubset(checks)
    assert {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file()} == {
        "COMMITTED.json", "custodian/master.csv", "custodian/observations.jsonl",
        "custodian/probabilities.jsonl", "custodian/components.json", "custodian/manifest.json",
        *(folder + "/" + name for folder in ("development/train", "development/validation", "calibration", "custodian/final_test")
          for name in ("data.csv", "membership.jsonl", "probabilities.jsonl", "manifest.json"))}


def test_optimized_and_full_reference_traces_have_identical_durable_evidence(tmp_path, monkeypatch):
    actual, reference, _, config, spec, _ = candidate_reference_pair(monkeypatch,
        scenario="compact", force_zero=True, origin=(0, 0))
    stores = [runner.EvidenceStore(tmp_path / name) for name in ("optimized", "reference")]
    for engine, store in zip((actual, reference), stores, strict=True):
        store.append("runs", dict(kind="initialization", spec=spec, completed=deepcopy(engine.last_completed)))
    try:
        for _ in range(3):
            left, right = actual.step(), reference.step()
            assert_candidate_pair(actual, reference, left, right)
            for engine, store in zip((actual, reference), stores, strict=True):
                snapshot = engine.batches
                runner.persist_and_acknowledge(engine, store)
                assert not engine.batches and snapshot[-1].envelope_bytes == left.envelope_bytes
                assert store.transitions[spec["run_id"]] == engine.timestep
                assert store.run_rows[spec["run_id"]] == engine.row_count
        assert not actual.is_active() and not reference.is_active()
        assert len(actual.rng.calls) == len(reference.rng.calls) == 1
        for engine, store in zip((actual, reference), stores, strict=True):
            store.append("runs", dict(kind="inactive", spec=spec, completed=deepcopy(engine.last_completed)))
    finally:
        for store in stores:
            store.close()
    for name in ("transitions.jsonl", "runs.jsonl"):
        assert (stores[0].root / name).read_bytes() == (stores[1].root / name).read_bytes()
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    authority = {"run_bindings": {spec["run_id"]: actual.run_identity_sha256}}
    contents = [runner.load_completed_evidence(store.root, config, authority, lambda _: None)
                for store in stores]
    assert contents[0] == contents[1]
    rows, runs = contents[0]
    assert len(rows) == actual.row_count == 8 and len(runs) == 2
    eligibility = {spec["scenario_family_id"]: dict(eligible=False,
        prior_use=config["inventory"][0]["prior_use"], assessment_reference="synthetic development fixture")}
    grouped = [release.components(values[0], [], eligibility, lambda _: None) for values in contents]
    assert grouped[0] == grouped[1] and len(grouped[0]) == 1
    # This bounded trace is deliberately inadequate for a release; retain all rows.
    assert grouped[0][0]["positives"] == 8 and grouped[0][0]["negatives"] == 0


def test_runner_reuses_context_but_constructs_fresh_generators_after_all_bindings(tmp_path, monkeypatch):
    real_load, real_build, real_engine = (runner.load_completed_evidence,
        collection.build_candidate_context, collection.CandidateAutomata)
    original_rng = np.random.default_rng
    config, authority, kwargs, first = bounded_runner(tmp_path, monkeypatch)
    prepared = runner.read_inputs()
    plan = collection.planned_runs(config["inventory"])[:2]
    for spec in plan:
        authority["run_bindings"][spec["run_id"]] = release.digest_bytes(release.json_bytes(
            collection.execution_binding(config, prepared, spec, authority["source"])))
    monkeypatch.setattr(runner, "validate_authority", lambda *a, **kw: (plan, {"synthetic": True}))
    monkeypatch.setattr(collection, "planned_runs", lambda _: plan)
    monkeypatch.setattr(runner, "load_completed_evidence", real_load)
    built, engines, calls = [], [], []
    def build(*args):
        calls.append("context")
        result = real_build(*args)
        built.append(result)
        return result
    def construct(preparation, configuration, spec, provenance, **options):
        assert calls.count("binding") == 2
        assert options["static_context"] is built[0] and preparation is built[0].prepared
        engine = real_engine(preparation, configuration, spec, provenance, **options)
        engines.append(engine)
        return engine
    original_binding = collection.execution_binding
    def binding(*args, **options):
        calls.append("binding")
        return original_binding(*args, **options)
    # Record the runner's binding loop separately from constructor-local rechecks.
    def counted_binding(*args, **options):
        if len(engines) == 0 and calls.count("binding") < 2:
            return binding(*args, **options)
        return original_binding(*args, **options)
    monkeypatch.setattr(collection, "build_candidate_context", build)
    monkeypatch.setattr(collection, "execution_binding", counted_binding)
    monkeypatch.setattr(collection, "CandidateAutomata", construct)
    monkeypatch.setattr(np.random, "default_rng", lambda seed:
        TracedCandidateGenerator(original_rng(seed), force_zero=True))
    report = runner.run(config, authority, **kwargs)
    assert report["complete"] and len(built) == 1 and len(engines) == 2
    assert calls[:3] == ["context", "binding", "binding"]
    assert engines[0].rng is not engines[1].rng and engines[0].grid is not engines[1].grid
    assert engines[0].p_base is engines[1].p_base
    assert [e.rng.calls for e in engines] == [[(3, 3)], [(3, 3)]]
    root = Path(authority["destinations"]["evidence"])
    runs = [json.loads(line) for line in (root / "runs.jsonl").read_bytes().splitlines()]
    assert [r["kind"] for r in runs] == ["initialization", "inactive"] * 2


def test_binding_rejection_after_context_precedes_rng_and_evidence_allocation(tmp_path, monkeypatch):
    config, authority, kwargs, spec = bounded_runner(tmp_path, monkeypatch)
    authority["run_bindings"][spec["run_id"]] = "0" * 64
    monkeypatch.setattr(np.random, "default_rng", lambda *_: pytest.fail("rejected binding created RNG"))
    monkeypatch.setattr(runner, "EvidenceStore", lambda *_a, **_k: pytest.fail("rejected binding allocated evidence"))
    with pytest.raises(runner.CandidateFailure) as caught:
        runner.run(config, authority, **kwargs)
    assert caught.value.report["evidence"] is None
    assert "Run binding mismatch" in caught.value.report["original_failure"]["message"]
    assert not Path(authority["destinations"]["evidence"]).exists()


@pytest.mark.parametrize("stage", ["initialization", "completed_transition", "inactive"])
@pytest.mark.parametrize("fault", ["timer", "route", "route_and_beta", "route_and_feature",
                                   "route_record", "failed", "incomplete", "checker_replacement"])
def test_runner_callback_mutation_prevents_continuation_or_publication(tmp_path, monkeypatch, stage, fault):
    config, authority, kwargs, _ = bounded_runner(tmp_path, monkeypatch)
    original_engine, original_rng = collection.CandidateAutomata, np.random.default_rng
    engines, publications = [], []
    def construct(*args, **options):
        result = original_engine(*args, **options)
        engines.append(result)
        return result
    monkeypatch.setattr(collection, "CandidateAutomata", construct)
    monkeypatch.setattr(np.random, "default_rng", lambda seed:
        TracedCandidateGenerator(original_rng(seed), force_zero=True))
    monkeypatch.setattr(release, "publish", lambda *a, **kw: publications.append(True))
    def emit(record):
        if record["kind"] == stage:
            engine = engines[0]
            if fault == "timer":
                engine.ignition_timers[-1, -1] += 1
            elif fault == "route":
                engine._context_bound = False
            elif fault in ("route_and_beta", "route_and_feature", "checker_replacement"):
                object.__setattr__(engine, "_context_bound", False)
                if fault == "route_and_feature":
                    engine.feature_assembler.slope_risk = np.zeros(engine.grid_shape, dtype=np.float32)
                else:
                    engine.p_base = np.zeros(engine.grid_shape, dtype=np.float32)
                if fault == "checker_replacement":
                    engine.validate_handoff = lambda **_: None
            elif fault == "route_record":
                object.__setattr__(engine, "_execution_route", None)
                object.__setattr__(engine, "_context_bound", False)
                object.__setattr__(engine, "_static_context", None)
            elif fault == "failed":
                engine.failed = True
            else:
                engine.incomplete_transition = dict(collection_completed=False, synthetic=True)
    with pytest.raises(runner.CandidateFailure) as caught:
        runner.run(config, authority, emit=emit, **kwargs)
    assert not publications and not caught.value.report["complete"]
    assert caught.value.report["original_failure"] is not None
    engine = engines[0]
    assert engine.failed
    calls, state = list(engine.rng.calls), deepcopy(engine.rng.bit_generator.state)
    assert len(calls) == (0 if stage == "initialization" else 1)
    with pytest.raises(RuntimeError, match="Failed"):
        engine.step()
    assert engine.rng.calls == calls and engine.rng.bit_generator.state == state
    if fault == "incomplete":
        assert engine.incomplete_transition == dict(collection_completed=False, synthetic=True)


def _run_latched_runner_callback(tmp_path, monkeypatch, stage, mutate):
    config, authority, kwargs, _ = bounded_runner(tmp_path, monkeypatch)
    original_engine, original_rng = collection.CandidateAutomata, np.random.default_rng
    original_persist = runner.persist_and_acknowledge
    engines, publications, persisted, observed = [], [], [], []
    names = ("features", "labels", "global_rows", "global_cols", "observation_ids", "q32_bits")
    def batch_bytes(snapshot):
        return tuple((batch.envelope_bytes, tuple(getattr(batch, name).tobytes() for name in names))
                     for batch in snapshot)
    def construct(*args, **options):
        engine = original_engine(*args, **options)
        engines.append(engine)
        return engine
    def persist(engine, store):
        snapshot = engine.batches
        before = batch_bytes(snapshot)
        original_persist(engine, store)
        persisted.append((snapshot, before))
    monkeypatch.setattr(collection, "CandidateAutomata", construct)
    monkeypatch.setattr(np.random, "default_rng", lambda seed:
        TracedCandidateGenerator(original_rng(seed), force_zero=True))
    monkeypatch.setattr(runner, "persist_and_acknowledge", persist)
    monkeypatch.setattr(release, "publish", lambda *a, **kw: publications.append(True))
    root = Path(authority["destinations"]["evidence"])
    def emit(record):
        if record["kind"] != stage:
            return
        engine = engines[0]
        before, batches, completed = engine_snapshot(engine), engine.batches, engine.last_completed
        journals = {name: (root / name).read_bytes() for name in ("runs.jsonl", "transitions.jsonl")}
        mutate(engine)
        assert engine.failed
        assert engine._context_bound is True and engine._static_context is engine._execution_route
        before["failed"] = True
        assert_engine_preserved(engine, before)
        assert engine.batches is batches and engine.last_completed is completed
        observed.append((before, batches, completed, journals))
    with pytest.raises(runner.CandidateFailure) as caught:
        runner.run(config, authority, emit=emit, **kwargs)
    assert len(observed) == 1 and len(engines) == 1
    assert not publications and not caught.value.report["complete"]
    assert caught.value.report["committed"] is None
    assert "Failed or incomplete" in caught.value.report["original_failure"]["message"]
    engine = engines[0]
    before, batches, completed, journals = observed[0]
    assert_engine_preserved(engine, before)
    assert engine.batches is batches and engine.last_completed is completed
    assert caught.value.report["last_completed"] == before["completed"]
    assert caught.value.report["incomplete_transition"] == before["incomplete"]
    assert {name: (root / name).read_bytes() for name in journals} == journals
    assert len(engine.rng.calls) == (1 if stage == "completed_transition" else 0)
    assert len(persisted) == (1 if stage == "completed_transition" else 0)
    if persisted:
        assert before["row_count"] > 0 and not batches
    for snapshot, frozen in persisted:
        assert snapshot and batch_bytes(snapshot) == frozen
    calls = list(engine.rng.calls)
    with pytest.raises(RuntimeError, match="Failed"):
        engine.step()
    assert engine.rng.calls == calls
    assert_engine_preserved(engine, before)
    for snapshot, frozen in persisted:
        assert batch_bytes(snapshot) == frozen


def test_runner_callback_cannot_bless_substituted_probability_with_replacement_members(tmp_path, monkeypatch):
    def mutate(engine):
        members = engine._bound_members
        engine.p_base = np.zeros(engine.grid_shape, dtype=np.float32)
        replacement = dict(members, p_base=engine.p_base)
        with pytest.raises(ValueError, match="validation anchor/helper"):
            engine._bound_members = replacement
        assert engine._bound_members is members and engine.failed
    _run_latched_runner_callback(tmp_path, monkeypatch, "initialization", mutate)


def test_runner_callback_cannot_replace_context_checker_after_committed_evidence(tmp_path, monkeypatch):
    def mutate(engine):
        checker = engine._check_context
        engine.p_base = np.zeros(engine.grid_shape, dtype=np.float32)
        with pytest.raises(ValueError, match="validation anchor/helper"):
            engine._check_context = lambda: None
        assert engine._check_context.__func__ is checker.__func__ and engine.failed
    _run_latched_runner_callback(tmp_path, monkeypatch, "completed_transition", mutate)


def test_runner_callback_cannot_clear_caught_handoff_failure(tmp_path, monkeypatch):
    def mutate(engine):
        original = engine.p_base
        engine.p_base = np.zeros(engine.grid_shape, dtype=np.float32)
        with pytest.raises(ValueError, match="substitution"):
            engine.validate_handoff(expected_context=engine._execution_route)
        assert engine.failed
        engine.p_base = original
        with pytest.raises(ValueError, match="failure latch"):
            engine.failed = False
        assert engine.p_base is original and engine.failed
    _run_latched_runner_callback(tmp_path, monkeypatch, "initialization", mutate)


@pytest.fixture
def retirement_runner(tmp_path, monkeypatch):
    """Already-inactive lifecycle substitutes; real method cycles, no CA steps or RNG."""
    enabled, real_collect = gc.isenabled(), gc.collect
    try:
        gc.disable()
        with monkeypatch.context() as patch:
            config, authority, kwargs, _ = bounded_runner(tmp_path, patch)
            prepared = runner.read_inputs()
            plan = collection.planned_runs(config["inventory"])[:2]
            for spec in plan:
                authority["run_bindings"][spec["run_id"]] = release.digest_bytes(release.json_bytes(
                    collection.execution_binding(config, prepared, spec, authority["source"])))
            patch.setattr(runner, "validate_authority", lambda *a, **kw: (plan, {"synthetic": True}))
            events, refs, snapshots, contexts = [], [], [], []
            state = dict(fault=None, attempts=0, copy_pending=False, copy_rejected=False, last_emit=None)
            failure = OSError("synthetic retirement failure")
            committed = dict(committed=True, package=authority["destinations"]["package"])

            class Engine:
                def __init__(self, ordinal, context):
                    self.ordinal, self.context = ordinal, context
                    self.failed, self.incomplete_transition = False, None
                    self.batches, self.row_count, self.timestep = (), 0, 0
                    self.rng = SimpleNamespace(calls=[])
                    self.last_completed = dict(state_timestep=0, committed_rows=0,
                        state_counts=dict(non_burnable=0, not_yet_burning=9-ordinal,
                                          ignited=0, blazing=0, extinguished=ordinal))
                    self._transition_observer = self.observe
                    self._bound_members = {"_transition_observer": self._transition_observer}
                    assert self._transition_observer.__self__ is self
                    refs.append(weakref.ref(self, lambda _, number=ordinal:
                                           events.append(("collected", number))))
                    snapshots.append(deepcopy(self.last_completed))

                def observe(self, observation):
                    pytest.fail("Lifecycle substitute must not receive a transition")

                def is_active(self):
                    return False

                def validate_handoff(self, *, expected_context):
                    assert expected_context is self.context and not self.failed
                    events.append(("handoff", self.ordinal, state["last_emit"]))

                def latch_handoff_failure(self):
                    self.failed = True
                    events.append(("latch", self.ordinal))

                def step(self):
                    if self.failed:
                        raise RuntimeError("Failed lifecycle engine")
                    pytest.fail("Lifecycle substitute must not execute CA steps")

            def construct(*args, **options):
                state["attempts"] += 1
                ordinal = state["attempts"]
                if refs:
                    assert refs[-1]() is None
                if contexts:
                    assert options["static_context"] is contexts[0]
                contexts.append(options["static_context"])
                events.append(("construct", ordinal))
                if ordinal == 2 and state["fault"] == "successor":
                    raise failure
                return Engine(ordinal, options["static_context"])

            class Guard:
                def __init__(self, *args):
                    self.samples = []

                def check(self, boundary):
                    events.append(("check", boundary, state["attempts"]))
                    self.samples.append(dict(boundary=boundary, elapsed_seconds=len(self.samples)+1,
                        rss_bytes=1, available_bytes=2, free_disk_bytes=3))
                    if boundary == "after_inactive_run":
                        if state["fault"] == "resource":
                            raise failure
                        state["copy_pending"] = True
                    if boundary == "after_engine_retirement":
                        assert refs[-1]() is None

            real_copy, real_append = runner.deepcopy, runner.EvidenceStore.append
            def copy(value):
                engine = refs[-1]() if refs else None
                if state["copy_pending"] and engine is not None:
                    if value is engine.last_completed:
                        events.append(("copy_completed", engine.ordinal))
                        if state["fault"] == "copy" and not state["copy_rejected"]:
                            state["copy_rejected"] = True
                            raise failure
                    elif value is engine.incomplete_transition:
                        events.append(("copy_incomplete", engine.ordinal))
                        state["copy_pending"] = False
                return real_copy(value)

            def collect():
                assert not gc.isenabled() and not state["copy_pending"]
                events.append(("collect", len(refs)))
                assert refs[-1]() is not None  # The self-cycle outlives root removal until collection.
                if state["fault"] == "collection":
                    raise failure
                result = real_collect()
                assert refs[-1]() is None
                return result

            def append(store, name, record):
                result = real_append(store, name, record)
                events.append(("journal", record["kind"], state["attempts"]))
                return result

            def reconstruct(*args):
                assert len(refs) == 2 and all(ref() is None for ref in refs)
                events.append(("reconstruct",))
                return [], []

            def publish(*args, **options):
                assert all(ref() is None for ref in refs)
                events.append(("publish",))
                return deepcopy(committed)  # Publisher substitute; no package files are created.

            def emit(record):
                state["last_emit"] = record["kind"]
                events.append(("emit", record["kind"], state["attempts"]))
                if state["fault"] == "late_reporting" and record["kind"] == "package_committed":
                    raise failure

            patch.setattr(collection, "CandidateAutomata", construct)
            patch.setattr(runner, "Resources", Guard)
            patch.setattr(runner, "deepcopy", copy)
            patch.setattr(runner.gc, "collect", collect)
            patch.setattr(runner.EvidenceStore, "append", append)
            patch.setattr(runner, "load_completed_evidence", reconstruct)
            patch.setattr(release, "publish", publish)
            patch.setattr(np.random, "default_rng", lambda *_: pytest.fail("Unexpected lifecycle RNG"))
            yield SimpleNamespace(config=config, authority=authority, kwargs=kwargs, plan=plan,
                state=state, events=events, refs=refs, snapshots=snapshots, contexts=contexts,
                failure=failure, committed=committed, emit=emit)
    finally:
        if enabled:
            gc.enable()
        else:
            gc.disable()
        assert gc.isenabled() is enabled


def test_runner_retires_cyclic_engines_before_successor_and_reconstruction(retirement_runner):
    fixture = retirement_runner
    report = runner.run(fixture.config, fixture.authority, emit=fixture.emit, **fixture.kwargs)
    assert report["complete"] and fixture.state["attempts"] == 2
    assert all(ref() is None for ref in fixture.refs)
    assert fixture.contexts[0] is fixture.contexts[1]
    for ordinal in (1, 2):
        ordered = [("journal", "inactive", ordinal), ("emit", "inactive", ordinal),
            ("handoff", ordinal, "inactive"), ("check", "after_inactive_run", ordinal),
            ("copy_completed", ordinal), ("copy_incomplete", ordinal), ("collect", ordinal),
            ("collected", ordinal), ("check", "after_engine_retirement", ordinal)]
        positions = [fixture.events.index(event) for event in ordered]
        assert positions == sorted(positions) and len(set(positions)) == len(positions)
    assert fixture.events.index(("check", "after_engine_retirement", 1)) < fixture.events.index(("construct", 2))
    assert fixture.events.index(("check", "after_engine_retirement", 2)) < fixture.events.index(("reconstruct",))
    assert report["last_completed"] == fixture.snapshots[-1] and report["incomplete_transition"] is None
    assert report["diagnostics_source"] == dict(run_id=fixture.plan[-1]["run_id"], boundary="successful_retirement")
    assert not any(event[0] == "latch" for event in fixture.events)


@pytest.mark.parametrize("fault", ["resource", "copy"])
def test_runner_pre_retirement_failure_keeps_live_engine_diagnostics(retirement_runner, fault):
    fixture = retirement_runner
    fixture.state["fault"] = fault
    with pytest.raises(runner.CandidateFailure) as caught:
        runner.run(fixture.config, fixture.authority, emit=fixture.emit, **fixture.kwargs)
    report, engine = caught.value.report, fixture.refs[0]()
    assert caught.value.__cause__ is fixture.failure
    assert report["original_failure"]["type"] == "OSError"
    assert report["original_failure"]["message"] == str(fixture.failure)
    assert engine is not None and engine.failed and fixture.state["attempts"] == 1
    assert report["last_completed"] == fixture.snapshots[0] and report["incomplete_transition"] is None
    assert report["diagnostics_source"] == dict(run_id=fixture.plan[0]["run_id"], boundary="live_engine_finalization")
    assert not any(event[0] in ("collect", "reconstruct", "publish") for event in fixture.events)
    assert ("latch", 1) in fixture.events and not report["complete"] and report["committed"] is None
    with pytest.raises(RuntimeError, match="Failed"):
        engine.step()
    assert engine.rng.calls == []


def test_runner_collection_failure_preserves_retirement_snapshot(retirement_runner):
    fixture = retirement_runner
    fixture.state["fault"] = "collection"
    with pytest.raises(runner.CandidateFailure) as caught:
        runner.run(fixture.config, fixture.authority, emit=fixture.emit, **fixture.kwargs)
    report = caught.value.report
    assert caught.value.__cause__ is fixture.failure and not report["complete"]
    assert report["original_failure"]["message"] == str(fixture.failure) and report["committed"] is None
    assert report["last_completed"] == fixture.snapshots[0] and report["incomplete_transition"] is None
    assert report["diagnostics_source"] == dict(run_id=fixture.plan[0]["run_id"], boundary="successful_retirement")
    assert fixture.state["attempts"] == 1 and ("collect", 1) in fixture.events
    assert not any(event[0] in ("latch", "reconstruct", "publish") for event in fixture.events)
    assert not any(event[:2] == ("check", "after_engine_retirement") for event in fixture.events)


@pytest.mark.parametrize("fault", ["successor", "late_reporting"])
def test_runner_post_retirement_failure_keeps_frozen_diagnostics(retirement_runner, fault):
    fixture = retirement_runner
    fixture.state["fault"] = fault
    with pytest.raises(runner.CandidateFailure) as caught:
        runner.run(fixture.config, fixture.authority, emit=fixture.emit, **fixture.kwargs)
    report = caught.value.report
    assert caught.value.__cause__ is fixture.failure and not report["complete"]
    assert report["original_failure"]["message"] == str(fixture.failure)
    assert all(ref() is None for ref in fixture.refs) and fixture.state["attempts"] == 2
    assert not any(event[0] == "latch" for event in fixture.events)
    assert report["last_completed"] == fixture.snapshots[-1] and report["incomplete_transition"] is None
    last_returned = len(fixture.refs)-1
    assert report["diagnostics_source"] == dict(run_id=fixture.plan[last_returned]["run_id"],
                                               boundary="successful_retirement")
    if fault == "successor":
        assert len(fixture.refs) == 1 and report["diagnostics_source"]["run_id"] != fixture.plan[1]["run_id"]
        assert report["committed"] is None and not any(event[0] in ("reconstruct", "publish") for event in fixture.events)
    else:
        assert len(fixture.refs) == 2 and report["committed"] == fixture.committed
        assert ("publish",) in fixture.events and ("emit", "package_committed", 2) in fixture.events


# Authenticated before editing: runner SHA-256 9a96e437dc7709a0653981916b869d670b34d8bf4cf3dbf523710c9136b4e17b.
# Original 4,979-byte loader text SHA-256: 445ce5fca03d241af49ac564fa603d0db648add067a868b2b3a708c98b86e759.
# Only the function name below differs; strict_json is the unchanged reader helper.
strict_json = runner.strict_json
_chunk_loader_under_test = runner.load_completed_evidence


def _whole_transition_loader_reference(root: Path, config: dict, authority: dict, check) -> tuple:
    rows, transitions, runs, row_counts, positive_counts, prior_states = [], {}, [], {}, {}, {}
    with (root / "transitions.jsonl").open("rb") as stream:
        for line in stream:
            check("transition_reconciliation")
            item = strict_json(line)
            if release.json_bytes(item) + b"\n" != line:
                raise ValueError("Noncanonical or partial transition record")
            binding, envelope, retained = item["binding"], item["envelope"], item["rows"]
            spec = binding["spec"]
            if release.digest_bytes(release.json_bytes(binding)) != authority["run_bindings"][spec["run_id"]]:
                raise ValueError("Retained run binding mismatch")
            columns = [release.decode_row(r["csv"]) for r in retained]
            arrays = [np.array([v[0] for v in columns], dtype=np.float32).reshape(-1, 11),
                np.array([v[1] for v in columns], dtype=np.int8),
                np.array([r["payload"]["cell_row"] for r in retained], dtype=np.int64),
                np.array([r["payload"]["cell_col"] for r in retained], dtype=np.int64),
                np.array([r["observation_id"].encode("ascii") for r in retained], dtype="S64"),
                np.array([r["q32_bits"] for r in retained], dtype=np.uint32)]
            for array in arrays:
                array.setflags(write=False)
            batch = collection.CandidateBatch(*arrays, envelope_bytes=release.json_bytes(envelope))
            _, rebuilt = release.reconcile_batch(batch, binding)
            run_id = spec["run_id"]
            if rebuilt != retained or envelope["timestep_t"] != transitions.get(run_id, 0):
                raise ValueError("Retained transition/row reconciliation failed")
            if run_id in prior_states and envelope["state_t_sha256"] != prior_states[run_id]:
                raise ValueError("Completed transition state-hash chain is discontinuous")
            prior_states[run_id] = envelope["state_t1_sha256"]
            transitions[run_id] = transitions.get(run_id, 0) + 1
            rows.extend(retained)
            row_counts[run_id] = row_counts.get(run_id, 0) + len(retained)
            positive_counts[run_id] = positive_counts.get(run_id, 0) + sum(v[1] for v in columns)
            if len(rows) > collection.MAX_PACKAGE_ROWS:
                raise ValueError("Retained package exceeds cap")
    with (root / "runs.jsonl").open("rb") as stream:
        for line in stream:
            check("run_reconciliation")
            item = strict_json(line)
            if release.json_bytes(item) + b"\n" != line:
                raise ValueError("Noncanonical or partial run record")
            runs.append(item)
    plan = collection.planned_runs(config["inventory"])
    if len(runs) != 2 * len(plan):
        raise ValueError("Every initialization and inactive termination is required")
    for i, spec in enumerate(plan):
        initial, final = runs[2 * i:2 * i + 2]
        total = row_counts.get(spec["run_id"], 0)
        before, after = initial["completed"]["state_counts"], final["completed"]["state_counts"]
        names = {"non_burnable", "not_yet_burning", "ignited", "blazing", "extinguished"}
        cells = config["grid"]["shape"][0] * config["grid"]["shape"][1]
        seeds = len(config["inventory"][spec["family_ordinal"] - 1]["ignition_points"])
        positives = positive_counts.get(spec["run_id"], 0)
        if (set(before) != names or set(after) != names
                or any(type(v) is not int or v < 0 for counts in (before, after) for v in counts.values())
                or sum(before.values()) != cells or sum(after.values()) != cells
                or before["blazing"] != seeds or before["ignited"] != 0 or before["extinguished"] != 0
                or after["non_burnable"] != before["non_burnable"]
                or after["extinguished"] != seeds + positives
                or after["not_yet_burning"] != before["not_yet_burning"] - positives):
            raise ValueError("Inactive state/label conservation mismatch")
        if (initial["kind"] != "initialization" or initial["spec"] != spec
                or initial["completed"]["state_timestep"] != 0
                or initial["completed"]["committed_rows"] != 0
                or final["kind"] != "inactive" or final["spec"] != spec
                or final["completed"]["state_counts"]["ignited"] != 0
                or final["completed"]["state_counts"]["blazing"] != 0
                or final["completed"]["state_timestep"] != transitions.get(spec["run_id"], 0)
                or not 1 <= final["completed"]["state_timestep"] <= collection.MAX_UPDATES
                or final["completed"]["committed_rows"] != total
                or total > collection.MAX_RUN_ROWS):
            raise ValueError("Run completeness/cap/count mismatch")
    release.validate_rows(rows, check)
    return rows, runs


def _constructed_validation_batch(binding, family, count, timestep=0, first_row=0, *, positive_count=None):
    """Fabricated comparisons/state hashes, not an executed stochastic history."""
    ordinal = np.arange(count, dtype=np.int64)
    features = np.tile(np.array([-0.0, .25, 1, .5, 1, 17.7, -.70710677,
                                .70710677, 1, .5, .125], dtype=np.float32), (count, 1))
    features[1::2, 0] = np.float32(1e-5)
    features[:, 4] = ordinal % 5 + 1
    labels = ((ordinal % 5 == 0) if positive_count is None else
              (ordinal < positive_count)).astype(np.int8)
    rows, cols = first_row + ordinal // 100, ordinal % 100
    bits = np.array([0, 0x3E000000, 0x3F800000], dtype=np.uint32)[ordinal % 3]
    identity = release.digest_bytes(release.json_bytes(binding))
    spec = binding["spec"]
    generic = dict(event_id=family["event_id"], run_id=spec["run_id"], seed=spec["seed"],
        grid_id=binding["source"]["grid_id"], candidate_run_identity_sha256=identity,
        **{name: binding[name] for name in ("teacher_id", "numerical_contract_id",
                                          "interpretation_policy", "mask_seed_policy")})
    envelope = dict(schema_version="candidate_comparison_capture.v1",
        run_identity_sha256=identity, provenance_sha256=collection.canonical_metadata_hash(generic),
        generic_provenance=generic, timestep_t=timestep, timestep_t1=timestep + 1,
        state_t_sha256=f"{timestep + 1:064x}", state_t1_sha256=f"{timestep + 2:064x}",
        eligible_mask_sha256="3" * 64,
        row_hash_encoding="eleven_little_endian_float32_then_int8_label.v1",
        **{name: binding[name] for name in ("teacher_id", "numerical_contract_id",
                                          "interpretation_policy", "mask_seed_policy")})
    ids = np.array([release.digest_bytes(release.json_bytes(release.observation_payload(
        binding, envelope, int(row), int(col)))).encode("ascii")
        for row, col in zip(rows, cols, strict=True)], dtype="S64")
    envelope["bindings"] = [dict(observation_id=identifier.decode("ascii"), q32_bits=int(q),
        row_sha256=release.digest_bytes(np.asarray(feature, dtype="<f4").tobytes()
                                       + bytes([int(label)])))
        for identifier, q, feature, label in zip(ids, bits, features, labels, strict=True)]
    arrays = (features, labels, rows, cols, ids, bits)
    for array in arrays:
        array.setflags(write=False)
    return collection.CandidateBatch(*arrays, envelope_bytes=release.json_bytes(envelope))


def _chunk_evidence(tmp_path, sizes):
    from dataclasses import replace
    prepared, config, spec, source, _ = context()
    config["grid"]["shape"] = [128, 128]
    prepared = replace(prepared, full_grid_shape=(128, 128))
    binding = collection.execution_binding(config, prepared, spec, source)
    authority = {"run_bindings": {spec["run_id"]: release.digest_bytes(release.json_bytes(binding))}}
    tmp_path.mkdir(exist_ok=True)
    store = runner.EvidenceStore(tmp_path / "chunk-evidence")
    before = dict(non_burnable=0, not_yet_burning=16383, ignited=0, blazing=1, extinguished=0)
    positives, total = 0, 0
    try:
        store.append("runs", dict(kind="initialization", spec=spec,
            completed=dict(state_timestep=0, committed_rows=0, state_counts=before)))
        for timestep, size in enumerate(sizes):
            batch = _constructed_validation_batch(binding, config["inventory"][0], size, timestep)
            positives += int(batch.labels.sum())
            total += size
            store.persist(binding, (batch,))
        after = dict(before, not_yet_burning=16383-positives, blazing=0, extinguished=1+positives)
        store.append("runs", dict(kind="inactive", spec=spec, completed=dict(
            state_timestep=len(sizes), committed_rows=total, state_counts=after)))
    finally:
        store.close()
    return store.root, config, authority, spec


def _read_constructed_records(root, name="transitions"):
    return [runner.strict_json(line) for line in (root / (name + ".jsonl")).read_bytes().splitlines(keepends=True)]


def _replace_constructed_records(root, records, name="transitions"):
    # Corrupt only future synthetic fixtures; production evidence is never rewritten.
    (root / (name + ".jsonl")).write_bytes(b"".join(release.json_bytes(r) + b"\n" for r in records))


@pytest.mark.parametrize("size", [0, 1, 4095, 4096, 4097, 8192, 8193])
def test_chunk_loader_matches_authenticated_whole_transition_reference(tmp_path, monkeypatch, size):
    root, config, authority, spec = _chunk_evidence(tmp_path, [size])
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    retained = _read_constructed_records(root)[0]
    original_bytes = {name: (root / name).read_bytes() for name in ("transitions.jsonl", "runs.jsonl")}
    expected = _whole_transition_loader_reference(root, config, authority, lambda _: None)
    real, lengths = release.reconcile_batch, []
    def inspect_chunk(batch, binding):
        start = sum(lengths)
        stop = start + len(batch.labels)
        view = json.loads(batch.envelope_bytes)
        assert {k: v for k, v in view.items() if k != "bindings"} == {
            k: v for k, v in retained["envelope"].items() if k != "bindings"}
        assert view["bindings"] == retained["envelope"]["bindings"][start:stop]
        decoded = [release.decode_row(r["csv"]) for r in retained["rows"][start:stop]]
        features = np.array([v[0] for v in decoded], dtype=np.float32).reshape(-1, 11)
        assert np.array_equal(batch.features.view(np.uint32), features.view(np.uint32))
        assert np.array_equal(batch.labels, np.array([v[1] for v in decoded], dtype=np.int8))
        assert batch.global_rows.tolist() == [r["payload"]["cell_row"] for r in retained["rows"][start:stop]]
        assert batch.global_cols.tolist() == [r["payload"]["cell_col"] for r in retained["rows"][start:stop]]
        assert batch.observation_ids.tolist() == [r["observation_id"].encode("ascii") for r in retained["rows"][start:stop]]
        assert batch.q32_bits.tolist() == [r["q32_bits"] for r in retained["rows"][start:stop]]
        assert [a.dtype for a in (batch.features, batch.labels, batch.global_rows,
            batch.global_cols, batch.observation_ids, batch.q32_bits)] == [
                np.dtype("float32"), np.dtype("int8"), np.dtype("int64"), np.dtype("int64"),
                np.dtype("S64"), np.dtype("uint32")]
        lengths.append(len(batch.labels))
        return real(batch, binding)
    monkeypatch.setattr(release, "reconcile_batch", inspect_chunk)
    assert runner.load_completed_evidence(root, config, authority, lambda _: None) == expected
    assert lengths == ([min(4096, size-i) for i in range(0, size, 4096)] if size else [0])
    assert {name: (root / name).read_bytes() for name in original_bytes} == original_bytes
    monkeypatch.setattr(runner, "_reconcile_retained_chunk", lambda *a: pytest.fail("Reference used chunk helper"))
    monkeypatch.setattr(release, "reconcile_batch", real)
    assert _whole_transition_loader_reference(root, config, authority, lambda _: None) == expected


@pytest.mark.parametrize("kind", ["missing", "extra"])
@pytest.mark.parametrize("position", [0, 4096, 8193])
def test_chunk_loader_rejects_complete_record_binding_count_before_slicing(tmp_path, monkeypatch, kind, position):
    root, config, authority, spec = _chunk_evidence(tmp_path, [8193])
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    records = _read_constructed_records(root)
    bindings = records[0]["envelope"]["bindings"]
    if kind == "missing":
        bindings.pop(min(position, len(bindings)-1))
    else:
        bindings.insert(position, dict(bindings[0]))
    _replace_constructed_records(root, records)
    monkeypatch.setattr(runner, "_reconcile_retained_chunk", lambda *a: pytest.fail("Count check was too late"))
    with pytest.raises(ValueError, match="count mismatch"):
        runner.load_completed_evidence(root, config, authority, lambda _: None)


@pytest.mark.parametrize("position", [0, 4096, 8192])
@pytest.mark.parametrize("fault", ["csv", "identity", "row_hash", "q32"])
def test_chunk_loader_rejects_first_middle_and_late_corruption(tmp_path, monkeypatch, position, fault):
    root, config, authority, spec = _chunk_evidence(tmp_path, [8193])
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    records = _read_constructed_records(root)
    row, captured = records[0]["rows"][position], records[0]["envelope"]["bindings"][position]
    if fault == "csv":
        row["csv"] = row["csv"].replace(",0.25,", ",0.5,", 1)
    elif fault == "identity":
        row["observation_id"] = captured["observation_id"] = "f" * 64
    elif fault == "row_hash":
        row["row_sha256"] = captured["row_sha256"] = "f" * 64
    else:
        captured["q32_bits"] ^= 1
    _replace_constructed_records(root, records)
    for loader in (_whole_transition_loader_reference, runner.load_completed_evidence):
        with pytest.raises(ValueError):
            loader(root, config, authority, lambda _: None)


@pytest.mark.parametrize("bits", [0x80000000, 0x7FC00000, 0x3F800001])
def test_chunk_loader_rejects_invalid_q32_in_late_chunk(tmp_path, monkeypatch, bits):
    root, config, authority, spec = _chunk_evidence(tmp_path, [4097])
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    records = _read_constructed_records(root)
    records[0]["rows"][-1]["q32_bits"] = records[0]["envelope"]["bindings"][-1]["q32_bits"] = bits
    _replace_constructed_records(root, records)
    for loader in (_whole_transition_loader_reference, runner.load_completed_evidence):
        with pytest.raises(ValueError, match="q32"):
            loader(root, config, authority, lambda _: None)


@pytest.mark.parametrize("size", [0, 4097])
@pytest.mark.parametrize("fault", ["schema", "run", "provenance", "state", "timestep", "teacher", "seed", "binding"])
def test_chunk_loader_checks_headers_even_for_empty_records(tmp_path, monkeypatch, size, fault):
    root, config, authority, spec = _chunk_evidence(tmp_path, [size])
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    records = _read_constructed_records(root)
    envelope = records[0]["envelope"]
    if fault == "schema":
        envelope["schema_version"] = "unexpected"
    elif fault == "run":
        envelope["run_identity_sha256"] = "f" * 64
    elif fault == "provenance":
        envelope["provenance_sha256"] = "f" * 64
    elif fault == "state":
        envelope["state_t_sha256"] = "invalid"
    elif fault == "timestep":
        envelope["timestep_t1"] += 1
    elif fault == "teacher":
        envelope["teacher_id"] = "unexpected"
    elif fault == "binding":
        records[0]["binding"]["spec"]["seed"] += 1
    else:
        envelope["generic_provenance"]["seed"] += 1
        envelope["provenance_sha256"] = collection.canonical_metadata_hash(envelope["generic_provenance"])
    _replace_constructed_records(root, records)
    for loader in (_whole_transition_loader_reference, runner.load_completed_evidence):
        with pytest.raises(ValueError):
            loader(root, config, authority, lambda _: None)


@pytest.mark.parametrize("fault", ["duplicate_key", "noncanonical", "missing_lf", "partial"])
def test_chunk_loader_preserves_original_record_parsing_checks(tmp_path, monkeypatch, fault):
    root, config, authority, spec = _chunk_evidence(tmp_path, [0])
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    path = root / "transitions.jsonl"
    raw = path.read_bytes()
    if fault == "duplicate_key":
        raw = b'{"binding":null,' + raw[1:]
    elif fault == "noncanonical":
        raw = b" " + raw
    elif fault == "missing_lf":
        raw = raw[:-1]
    else:
        raw = raw[:-10]
    path.write_bytes(raw)
    for loader in (_whole_transition_loader_reference, runner.load_completed_evidence):
        with pytest.raises(ValueError):
            loader(root, config, authority, lambda _: None)


@pytest.mark.parametrize("fault", ["repeat", "gap", "state_chain", "missing_initial", "active_final",
                                  "committed", "conservation", "run_cap", "package_cap"])
def test_chunk_loader_preserves_continuity_totals_caps_and_incomplete_rejection(tmp_path, monkeypatch, fault):
    root, config, authority, spec = _chunk_evidence(tmp_path, [3] if fault.endswith("cap") else [0, 0])
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    if fault in ("repeat", "gap", "state_chain"):
        records = _read_constructed_records(root)
        envelope = records[1]["envelope"]
        if fault == "state_chain":
            envelope["state_t_sha256"] = "f" * 64
        else:
            envelope["timestep_t"] = 0 if fault == "repeat" else 2
            envelope["timestep_t1"] = envelope["timestep_t"] + 1
        _replace_constructed_records(root, records)
    elif fault.endswith("cap"):
        monkeypatch.setattr(collection, "MAX_RUN_ROWS" if fault == "run_cap" else "MAX_PACKAGE_ROWS", 2)
    else:
        records = _read_constructed_records(root, "runs")
        if fault == "missing_initial":
            records.pop(0)
        elif fault == "active_final":
            records[-1]["completed"]["state_counts"]["blazing"] = 1
        elif fault == "committed":
            records[-1]["completed"]["committed_rows"] += 1
        else:
            records[-1]["completed"]["state_counts"]["not_yet_burning"] -= 1
        _replace_constructed_records(root, records, "runs")
    for loader in (_whole_transition_loader_reference, runner.load_completed_evidence):
        with pytest.raises(ValueError):
            loader(root, config, authority, lambda _: None)


@pytest.mark.parametrize("fault", ["order", "duplicate"])
def test_chunk_loader_keeps_cross_chunk_checks_in_global_validation(tmp_path, monkeypatch, fault):
    root, config, authority, spec = _chunk_evidence(tmp_path, [4097])
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    records = _read_constructed_records(root)
    for values in (records[0]["rows"], records[0]["envelope"]["bindings"]):
        if fault == "order":
            values[4095], values[4096] = values[4096], values[4095]
        else:
            values[4096] = deepcopy(values[4094])  # Equal labels preserve conservation.
    _replace_constructed_records(root, records)
    real_validate, visited = release.validate_rows, []
    def validate(rows, check):
        visited.append(len(rows))
        return real_validate(rows, check)
    monkeypatch.setattr(release, "validate_rows", validate)
    with pytest.raises(ValueError, match="order|Repeated"):
        runner.load_completed_evidence(root, config, authority, lambda _: None)
    assert visited == [4097]


@pytest.mark.parametrize("size", [0, 8193])
def test_chunk_loader_retires_secondary_and_record_only_owners(tmp_path, monkeypatch, size):
    root, config, authority, spec = _chunk_evidence(tmp_path, [size])
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    class WeakDict(dict):
        pass
    class WeakList(list):
        pass
    real_json, real_decode = runner.strict_json, release.decode_row
    real_reconcile, real_validate = release.reconcile_batch, release.validate_rows
    records, previous, columns, lengths = [], [], [], []
    def read(data):
        item = real_json(data)
        if "envelope" in item:
            item = WeakDict(item)
            item["envelope"] = WeakDict(item["envelope"])
            item["rows"] = WeakList(item["rows"])
            item["envelope"]["bindings"] = WeakList(item["envelope"]["bindings"])
            records.extend(weakref.ref(v) for v in (item, item["envelope"], item["rows"],
                                                  item["envelope"]["bindings"]))
        return item
    def decode(csv):
        values, label = real_decode(csv)
        columns.append(weakref.ref(values))
        return values, label
    def reconcile(batch, binding):
        prior_columns = columns[:-len(batch.labels)] if len(batch.labels) else columns
        assert all(ref() is None for ref in previous + prior_columns)
        assert records and all(ref() is not None for ref in records)
        envelope, rebuilt = real_reconcile(batch, binding)
        envelope, rebuilt = WeakDict(envelope), WeakList(rebuilt)
        envelope["bindings"] = WeakList(envelope["bindings"])
        owners = (batch, batch.features, batch.labels, batch.global_rows, batch.global_cols,
                  batch.observation_ids, batch.q32_bits, envelope, envelope["bindings"], rebuilt)
        if isinstance(batch.features.base, np.ndarray):
            owners += (batch.features.base,)
        previous[:] = [weakref.ref(v) for v in owners]
        lengths.append(len(batch.labels))
        return envelope, rebuilt
    def validate(rows, check):
        assert all(ref() is None for ref in records + previous + columns)
        return real_validate(rows, check)
    monkeypatch.setattr(runner, "strict_json", read)
    monkeypatch.setattr(release, "decode_row", decode)
    monkeypatch.setattr(release, "reconcile_batch", reconcile)
    monkeypatch.setattr(release, "validate_rows", validate)
    runner.load_completed_evidence(root, config, authority, lambda _: None)
    assert lengths == ([4096, 4096, 1] if size else [0])
    assert all(ref() is None for ref in records + previous + columns)


def test_late_chunk_failure_is_atomic_and_blocks_runner_publication(tmp_path, monkeypatch, request):
    root, config, authority, spec = _chunk_evidence(tmp_path / "late-source", [1, 8193])
    records = _read_constructed_records(root)
    records[-1]["envelope"]["bindings"][-1]["q32_bits"] ^= 1
    _replace_constructed_records(root, records)
    retained = {name: (root / name).read_bytes() for name in ("transitions.jsonl", "runs.jsonl")}
    retirement_runner = request.getfixturevalue("retirement_runner")
    with monkeypatch.context() as runner_patch:
        def load(*args):
            with monkeypatch.context() as patch:
                patch.setattr(collection, "planned_runs", lambda _: [spec])
                return _chunk_loader_under_test(root, config, authority, lambda _: None)
        runner_patch.setattr(runner, "load_completed_evidence", load)
        runner_patch.setattr(release, "components", lambda *a: pytest.fail("Grouped after failed reconstruction"))
        runner_patch.setattr(release, "publish", lambda *a, **kw: pytest.fail("Published after failed reconstruction"))
        with pytest.raises(runner.CandidateFailure) as caught:
            runner.run(retirement_runner.config, retirement_runner.authority,
                       emit=retirement_runner.emit, **retirement_runner.kwargs)
        report = caught.value.report
        assert report["complete"] is False and report["committed"] is None
        assert report["original_failure"]["type"] == "ValueError"
        cause = caught.value.__cause__
        assert isinstance(cause, ValueError)
        trace = cause.__traceback__
        while trace is not None and trace.tb_frame.f_code is not _chunk_loader_under_test.__code__:
            trace = trace.tb_next
        assert trace is not None
        local = trace.tb_frame.f_locals
        run_id = spec["run_id"]
        assert len(local["rows"]) == local["row_counts"][run_id] == 1
        assert local["transitions"][run_id] == local["positive_counts"][run_id] == 1
        assert local["prior_states"][run_id] == f"{2:064x}"
        assert {name: (root / name).read_bytes() for name in retained} == retained
        assert not Path(retirement_runner.authority["destinations"]["package"]).exists()


def test_empty_chunk_record_rejects_extra_binding_before_helper(tmp_path, monkeypatch):
    root, config, authority, spec = _chunk_evidence(tmp_path, [0])
    monkeypatch.setattr(collection, "planned_runs", lambda _: [spec])
    records = _read_constructed_records(root)
    records[0]["envelope"]["bindings"].append(dict(
        observation_id="a"*64, q32_bits=0, row_sha256="b"*64))
    _replace_constructed_records(root, records)
    monkeypatch.setattr(runner, "_reconcile_retained_chunk", lambda *a: pytest.fail("Count check was too late"))
    with pytest.raises(ValueError, match="count mismatch"):
        runner.load_completed_evidence(root, config, authority, lambda _: None)


def test_whole_transition_reference_text_matches_authenticated_before_image():
    import inspect
    original = inspect.getsource(_whole_transition_loader_reference).replace(
        "def _whole_transition_loader_reference(", "def load_completed_evidence(", 1)
    assert len(original.encode("utf-8")) == 4979
    assert release.digest_bytes(original.encode("utf-8")) == (
        "445ce5fca03d241af49ac564fa603d0db648add067a868b2b3a708c98b86e759")


def _reference_package_bytes(rows, items, authority):
    """Bounded test expectations from whole-transition results, not a second publication."""
    files, identities = {}, {}
    def put(name, data):
        files[name] = data
        identities[name] = dict(bytes=len(data), sha256=release.digest_bytes(data))
    master = release.HEADER + b"".join(row["csv"].encode("ascii") for row in rows)
    put("custodian/master.csv", master)
    master_id = identities["custodian/master.csv"]
    assignments = release.allocate(items, master_id["sha256"],
        node_budget=authority["binding"]["allocation_node_budget"], check=lambda _: None)
    roles = {i: assignments[item["component_id"]] for item in items for i in item["indices"]}
    put("custodian/observations.jsonl", b"".join(release.json_bytes(row) + b"\n" for row in rows))
    put("custodian/probabilities.jsonl", b"".join(release.json_bytes(dict(master_row_ordinal=i,
        csv_sha256=master_id["sha256"], observation_id=row["observation_id"],
        row_sha256=row["row_sha256"], q32_bits=row["q32_bits"])) + b"\n"
        for i, row in enumerate(rows)))
    put("custodian/components.json", release.json_bytes(dict(
        schema_version="candidate_component_sorted_observation_ids.v1",
        components=items, assignments=assignments)) + b"\n")
    for role in ("train", "validation", "calibration", "final_test"):
        relative = ("custodian/final_test" if role == "final_test" else
                    "calibration" if role == "calibration" else "development/" + role)
        selected = [row for i, row in enumerate(rows) if roles[i] == role]
        put(relative + "/data.csv", release.HEADER + b"".join(row["csv"].encode("ascii") for row in selected))
        csv_hash = identities[relative + "/data.csv"]["sha256"]
        put(relative + "/membership.jsonl", b"".join(release.json_bytes(dict(row_ordinal=i,
            csv_sha256=csv_hash, observation_id=row["observation_id"], row_sha256=row["row_sha256"],
            payload=row["payload"])) + b"\n" for i, row in enumerate(selected)))
        put(relative + "/probabilities.jsonl", b"".join(release.json_bytes(dict(row_ordinal=i,
            csv_sha256=csv_hash, observation_id=row["observation_id"], row_sha256=row["row_sha256"],
            q32_bits=row["q32_bits"])) + b"\n" for i, row in enumerate(selected)))
        scoped = {name[len(relative)+1:]: value for name, value in identities.items()
                  if name.startswith(relative + "/")}
        put(relative + "/manifest.json", release.json_bytes(dict(
            schema_version="candidate_teacher_shared_dataset_release.v1", role=role,
            files=scoped, rows=len(selected), teacher_id=authority["teacher_id"],
            numerical_contract_id=authority["numerical_contract_id"],
            custody="role-scoped; same-account protection is procedural")) + b"\n")
    put("custodian/manifest.json", release.json_bytes(dict(
        schema_version="candidate_teacher_shared_dataset_release.v1", authority=authority,
        master_path="custodian/master.csv", master=master_id,
        rows=len(rows), files=identities.copy(), complete=True)) + b"\n")
    put("COMMITTED.json", release.json_bytes(dict(schema_version="candidate_teacher_shared_dataset_release.v1",
        manifest_sha256=identities["custodian/manifest.json"]["sha256"])) + b"\n")
    return files


def test_chunked_constructed_evidence_matches_reference_through_real_publication(tmp_path):
    from dataclasses import replace
    from test_candidate_dataset_release_contract import publication_fixture
    options = publication_fixture(tmp_path)[2]
    provenance = options["authority"]
    config, authority = provenance["configuration"], provenance["execution_authority"]
    prepared = replace(context()[0], full_grid_shape=tuple(config["grid"]["shape"]))
    plan = collection.planned_runs(config["inventory"])
    authority["run_bindings"] = {}
    store = runner.EvidenceStore(tmp_path / "chunk-constructed-evidence")
    authority["destinations"]["evidence"] = str(store.root)
    options["payload"] = store.root / "payload"
    # All 364 specifications below are declarations, not executed CA runs.
    try:
        for spec in plan:
            binding = collection.execution_binding(config, prepared, spec, authority["source"])
            authority["run_bindings"][spec["run_id"]] = release.digest_bytes(release.json_bytes(binding))
            family = config["inventory"][spec["family_ordinal"]-1]
            count = (8193 if spec["family_ordinal"] == 1 else 100) if (
                spec["wind_ordinal"] == 1 and spec["replicate_ordinal"] == 1) else 0
            batch = _constructed_validation_batch(binding, family, count,
                first_row=128*spec["family_ordinal"], positive_count=min(20, count))
            cells = config["grid"]["shape"][0] * config["grid"]["shape"][1]
            seeds, positives = len(family["ignition_points"]), int(batch.labels.sum())
            initial = dict(non_burnable=0, not_yet_burning=cells-seeds,
                           ignited=0, blazing=seeds, extinguished=0)
            final = dict(initial, not_yet_burning=cells-seeds-positives,
                         blazing=0, extinguished=seeds+positives)
            store.append("runs", dict(kind="initialization", spec=spec, completed=dict(
                state_timestep=0, committed_rows=0, state_counts=initial)))
            store.persist(binding, (batch,))
            store.append("runs", dict(kind="inactive", spec=spec, completed=dict(
                state_timestep=1, committed_rows=count, state_counts=final)))
    finally:
        store.close()
    retained = {name: (store.root / name).read_bytes() for name in ("transitions.jsonl", "runs.jsonl")}
    expected_rows, expected_runs = _whole_transition_loader_reference(store.root, config, authority, lambda _: None)
    rows, runs = runner.load_completed_evidence(store.root, config, authority, lambda _: None)
    assert rows == expected_rows and runs == expected_runs
    assert len(rows) == store.rows == 10693 and len(runs) == 728
    assert sum(store.transitions.values()) == len(plan) == 364
    families = authority["custody"]["families"]
    items = release.components(rows, config["complete_event_ids"], families, lambda _: None)
    expected_items = release.components(expected_rows, config["complete_event_ids"], families, lambda _: None)
    assert items == expected_items and len(items) == 26
    assert all(item["positives"] == 20 and item["materials"] == [1, 2, 3, 4, 5] for item in items)
    assert sorted(item["negatives"] for item in items) == [80]*25 + [8173]
    provenance.update(binding=runner.publication_payload(release.digest_bytes(release.json_bytes(
        runner.execution_payload(config, authority))), authority), run_records=expected_runs,
        evidence_files={name: runner.file_identity(store.root / name, lambda _: None) for name in retained},
        dataset_sha256=release.digest_bytes(release.HEADER + b"".join(row["csv"].encode("ascii") for row in expected_rows)))
    options["expected_binding_sha256"] = release.digest_bytes(release.json_bytes(provenance["binding"]))
    expected_files = _reference_package_bytes(expected_rows, expected_items, provenance)
    checks = []
    options["check"] = checks.append
    result = release.publish(rows, items, **options)
    package = Path(result["package"])
    assert result["committed"] is True and not options["payload"].exists()
    assert len(expected_files) == 22
    assert {path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file()} == set(expected_files)
    for name, expected in expected_files.items():
        actual = (package / name).read_bytes()
        assert actual == expected
        assert release.digest_bytes(actual) == release.digest_bytes(expected)
    assert {name: (store.root / name).read_bytes() for name in retained} == retained
    assert {"allocation_search", "allocation_completed", "publication_completion_decision"}.issubset(checks)
