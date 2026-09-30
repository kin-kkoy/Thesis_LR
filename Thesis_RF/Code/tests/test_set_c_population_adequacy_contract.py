from __future__ import annotations

import json
from hashlib import sha256
from contextlib import closing
from datetime import datetime, timedelta, timezone
import sqlite3
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import modules.set_c_population_adequacy as adequacy
import modules.set_c_population_replay as replay
import modules.set_c_checkpoint_contract as checkpoint_contract
import run_set_c_population_replay as replay_runner
from modules.set_c_population_adequacy import (
    PopulationAdequacyError,
    PopulationFamily,
    PopulationResourceError,
    PopulationRunResult,
    ResourceSnapshot,
    activate_approved_population_config,
    build_model_free_executor,
    build_run_matrix,
    run_population_adequacy,
)
from orchestrator import load_config
import run_set_c_population_adequacy as runner
from run_set_c_population_adequacy import main


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default_experiment.yaml"


class _Batch:
    def __init__(
        self,
        family_order: int,
        *,
        connect_first_two: bool = False,
        link_cause: str | None = None,
    ) -> None:
        self.labels = np.array([0] * 5 + [1] * 5, dtype=np.int8)
        if link_cause == "bridge":
            duplicate_order = 1 if family_order == 2 else family_order
            block_order = 2 if family_order == 3 else family_order
        else:
            duplicate_order = 1 if family_order in (1, 2) and (
                connect_first_two or link_cause in ("duplicate", "joint")
            ) else family_order
            block_order = 1 if family_order in (1, 2) and (
                connect_first_two or link_cause in ("block", "joint")
            ) else family_order
        self.features = np.asarray(
            [[float(duplicate_order), float(index)] + [0.0] * 9
             for index in range(10)], dtype=np.float32)
        self.duplicate_group_ids = np.array(
            [f"family-{duplicate_order:02d}-cell-{index:02d}" for index in range(10)],
            dtype="S64",
        )
        self.spatial_block_ids = np.array(
            [f"family-{block_order:02d}-block"] * 10,
            dtype="S64",
        )

    @property
    def row_count(self) -> int:
        return 10


def _families() -> tuple[PopulationFamily, ...]:
    return tuple(
        PopulationFamily(
            order=order,
            family_id=f"synthetic_family_{order:02d}",
            material_class=((order - 1) % 5) + 1,
            component_ordinal=order,
            component_size=256 + order,
            ignition_row=order,
            ignition_col=order + 100,
            ignition_set_sha256=f"{order:064x}",
            block_footprint_sha256=f"{order + 100:064x}",
            source_tier="existing" if order <= 6 else "primary",
        )
        for order in range(1, 27)
    )


def _source_context() -> dict[str, object]:
    return {
        "source_revision": "synthetic-revision",
        "source_dirty": False,
        "source_hashes": {"synthetic.py": "a" * 64},
        "input_hashes": {
            "simulation_valid_mask": "b" * 64,
            "mapped_building_mask": "c" * 64,
        },
        "generation_command": "synthetic-test",
        "environment_record": {"environment": "synthetic"},
    }


def _production_source_context() -> dict[str, object]:
    paths = {
        "config/default_experiment.yaml",
        "modules/automata_engine.py",
        "modules/ca_state.py",
        "modules/data_loader.py",
        "modules/feature_pipeline.py",
        "modules/model_free_teacher.py",
        "modules/set_c_collector.py",
        "modules/set_c_pilot.py",
        "modules/set_c_site_inventory.py",
        "modules/set_c_population_adequacy.py",
        "modules/transition_observer.py",
        "modules/wind_convention.py",
        "orchestrator.py",
        "run_set_c_population_adequacy.py",
    }
    return {
        "source_revision": "a" * 40,
        "source_dirty": False,
        "source_hashes": {path: "a" * 64 for path in paths},
        "input_hashes": {
            **adequacy.EXPECTED_RASTER_SHA256,
            "simulation_valid_mask": adequacy.EXPECTED_SIMULATION_VALID_SHA256,
            "mapped_building_mask": adequacy.EXPECTED_MAPPED_BUILDING_SHA256,
        },
        "generation_command": adequacy.PRODUCTION_GENERATION_COMMAND,
        "environment_record": {"environment": "synthetic"},
    }


def _resources(_: Path) -> ResourceSnapshot:
    return ResourceSnapshot(
        available_memory_bytes=8_000_000_000,
        process_rss_bytes=100_000_000,
        free_disk_bytes=10_000_000_000,
        captured_at_utc="2026-09-28T00:00:00+00:00",
    )


def _executor(
    *,
    fail_order: int | None = None,
    unexpected: bool = False,
    connect_first_two: bool = False,
    link_cause: str | None = None,
    result_updates: dict | None = None,
):
    active = 0
    maximum_active = 0

    def execute(spec, consume):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            if spec.order == fail_order:
                raise RuntimeError("synthetic run failure")
            consume(
                _Batch(
                    spec.family.order,
                    connect_first_two=connect_first_two,
                    link_cause=link_cause,
                )
            )
            result = PopulationRunResult(
                run_id=spec.run_id,
                scenario_id=spec.scenario_id,
                family_id=spec.family.family_id,
                seed=spec.seed,
                rows=10,
                batch_count=1,
                observation_count=1,
                released_rows=10,
                pending_rows=0,
                pending_bytes=0,
                termination_reason="inactive",
                completeness_status="complete",
                authoritative=True,
                final_timestep=1,
                final_ignited_count=0,
                final_blazing_count=0,
                expected_transition_count=1,
                captured_transition_count=1,
                configuration_sha256="d" * 64,
                run_identity_sha256="e" * 64,
                provenance_sha256="f" * 64,
                ignition_set_sha256=spec.family.ignition_set_sha256,
                unexpected_outputs=("unexpected.bin",) if unexpected else (),
                max_live_full_grid_observations=maximum_active,
            )
            return replace(result, **(result_updates or {}))
        finally:
            active -= 1

    execute.maximum_active = lambda: maximum_active
    return execute


def _active_config() -> dict:
    return activate_approved_population_config(load_config(str(CONFIG_PATH)))


def test_matrix_has_exact_d015_order_scenarios_and_seeds():
    matrix = build_run_matrix(_families())

    assert len(matrix) == 884
    assert matrix[0].scenario_id == "synthetic_family_01__w000_s00"
    assert matrix[0].seed == 9_010_101
    assert matrix[1].seed == 9_010_102
    assert matrix[2].scenario_id == "synthetic_family_01__w000_s10"
    assert matrix[2].seed == 9_010_201
    assert matrix[18].scenario_id == "synthetic_family_01__w000_s20"
    assert matrix[18].seed == 9_011_001
    assert matrix[33].scenario_id == "synthetic_family_01__w315_s20"
    assert matrix[33].replicate_ordinal == 2
    assert matrix[34].family.order == 2
    assert matrix[-1].family.order == 26
    assert matrix[-1].seed == 9_261_702


def test_success_is_sequential_aggregate_only_atomic_and_non_overwriting(tmp_path):
    destination = tmp_path / "adequacy.aggregate.json"
    executor = _executor()
    result = run_population_adequacy(
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        families=_families(),
        run_executor=executor,
        resource_probe=_resources,
        synthetic_test=True,
        aggregate_store_directory=tmp_path,
    )

    assert result["run_count"] == 884
    assert result["adequacy_status"] == "pass"
    assert executor.maximum_active() == 1
    assert list(tmp_path.iterdir()) == [destination]
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["aggregate_schema_version"] == "phase9_set_c_population_adequacy_aggregate.v2"
    assert "production execution requires separate one-run authorization" in report[
        "authorization_reference"
    ]
    assert report["aggregate"]["totals"]["run_count"] == 884
    assert report["aggregate"]["totals"]["rows"] == 8840
    assert report["aggregate"]["leakage"]["qualifying_component_count"] == 26
    assert report["aggregate"]["leakage"]["role_identities_exposed"] is False
    assert report["integrity"]["aggregate_identity_store"] == {
        "backend": "temporary_sqlite",
        "cache_kib": 8192,
        "per_identity_values": [
            "row_count",
            "class_mask",
            "family_bitmask_26",
        ],
        "committed_per_run": True,
        "cleanup_required_on_success_and_failure": True,
    }
    assert not list(tmp_path.glob("*.sqlite3*"))
    forbidden = {
        "features",
        "predictors",
        "labels",
        "cell_rows",
        "cell_cols",
        "cell_inventory",
        "split_role",
        "split_assignments",
        "model",
        "ground_truth",
    }

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    assert forbidden.isdisjoint(keys(report))
    with pytest.raises(FileExistsError):
        run_population_adequacy(
            _active_config(),
            destination=destination,
            source_context=_source_context(),
            families=_families(),
            run_executor=_executor(),
            resource_probe=_resources,
            synthetic_test=True,
            aggregate_store_directory=tmp_path,
        )


@pytest.mark.parametrize("mode", ["run_failure", "unexpected_output"])
def test_failed_or_unexpected_run_publishes_nothing(tmp_path, mode):
    destination = tmp_path / "adequacy.aggregate.json"
    executor = (
        _executor(fail_order=3)
        if mode == "run_failure"
        else _executor(unexpected=True)
    )

    with pytest.raises((RuntimeError, PopulationAdequacyError)):
        run_population_adequacy(
            _active_config(),
            destination=destination,
            source_context=_source_context(),
            families=_families(),
            run_executor=executor,
            resource_probe=_resources,
            synthetic_test=True,
            aggregate_store_directory=tmp_path,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob("*.sqlite3*"))
    assert list(tmp_path.iterdir()) == []


def test_checked_in_harness_requires_future_execution_flag():
    with pytest.raises(SystemExit) as exc:
        main([])

    assert exc.value.code == 2

    with pytest.raises(SystemExit) as legacy:
        main(["--execute-approved-expansion"])
    assert legacy.value.code == 2


def test_production_executor_checks_resources_during_observation(monkeypatch, tmp_path):
    class FakeCollector:
        pending_rows = 0
        pending_bytes = 0
        live_full_grid_observation_count = 0

        def __init__(self, *_):
            pass

        @staticmethod
        def pending_batches():
            return ()

    def fake_teacher(*_, transition_observer, **__):
        transition_observer(object())
        pytest.fail("resource breach should stop inside the observation")

    low_memory = ResourceSnapshot(
        available_memory_bytes=1,
        process_rss_bytes=100,
        free_disk_bytes=10_000_000_000,
        captured_at_utc="2026-09-28T00:00:00+00:00",
    )
    monkeypatch.setattr(adequacy, "SetCCollector", FakeCollector)
    monkeypatch.setattr(adequacy, "run_model_free_teacher", fake_teacher)
    executor = build_model_free_executor(
        {},
        _active_config(),
        _source_context(),
        destination=tmp_path / "adequacy.aggregate.json",
        resource_probe=lambda _: low_memory,
    )

    with pytest.raises(PopulationResourceError):
        executor(build_run_matrix(_families())[0], lambda _: None)


def test_explicit_entry_point_preflights_before_environment_load(monkeypatch):
    class StopPreflight(RuntimeError):
        pass

    monkeypatch.setattr(runner, "preflight_population_launch", lambda *_: (_ for _ in ()).throw(StopPreflight()))
    monkeypatch.setattr(
        runner,
        "EnvironmentManager",
        lambda *_: pytest.fail("environment loaded before preflight"),
    )

    with pytest.raises(StopPreflight):
        main(["--execute-approved-edge-diagnostic", "--config", str(CONFIG_PATH)])


def test_source_context_uses_slash_normalized_git_safe_directory(monkeypatch):
    repository_root = Path(runner.__file__).resolve().parents[2]
    expected_safe_directory = f"safe.directory={repository_root.as_posix()}"
    calls = []

    def git_with_dubious_ownership(command, **kwargs):
        calls.append(command)
        assert command[:3] == ["git", "-c", expected_safe_directory]
        assert kwargs["cwd"] == repository_root
        return SimpleNamespace(stdout="a" * 40 if "rev-parse" in command else "")

    monkeypatch.setattr(runner.subprocess, "run", git_with_dubious_ownership)
    monkeypatch.setattr(
        runner,
        "_file_sha256",
        lambda path: runner.EXPECTED_RASTER_SHA256.get(path.name, "a" * 64),
    )
    monkeypatch.setattr(runner, "mask_sha256", lambda _: "b" * 64)
    monkeypatch.setattr(runner.platform, "platform", lambda: "synthetic-platform")
    environment = {
        "burnable_mask": np.ones((1, 1), dtype=bool),
        "nodata_mask": np.zeros((1, 1), dtype=bool),
        "building_presence": np.ones((1, 1), dtype=np.int8),
    }

    context = runner._source_context(
        {"environment": {"raster_dir": "synthetic-unused"}},
        CONFIG_PATH,
        environment,
    )

    assert len(calls) == 2
    assert calls[0][3:] == ["rev-parse", "HEAD"]
    assert calls[1][3:] == [
        "status", "--porcelain", "--", ".", ":(exclude)Thesis-try.qgz"
    ]
    assert context["source_revision"] == "a" * 40
    assert context["source_dirty"] is False


def test_production_rejects_injected_resource_probe():
    with pytest.raises(PermissionError, match="exact resource telemetry"):
        run_population_adequacy(
            _active_config(),
            destination=adequacy.approved_destination(),
            source_context=_production_source_context(),
            resource_probe=_resources,
        )


def test_synthetic_paths_outside_system_temp_are_rejected_without_write(tmp_path):
    workspace = CONFIG_PATH.parents[1]
    workspace_destination = workspace / "tests" / "forbidden-adequacy.json"
    workspace_store = workspace / "tests"
    workspace_destination.unlink(missing_ok=True)

    with pytest.raises(PermissionError, match="system temp"):
        run_population_adequacy(
            _active_config(),
            destination=workspace_destination,
            source_context=_source_context(),
            families=_families(),
            run_executor=_executor(),
            resource_probe=_resources,
            synthetic_test=True,
        )
    with pytest.raises(PermissionError, match="system temp"):
        run_population_adequacy(
            _active_config(),
            destination=tmp_path / "adequacy.aggregate.json",
            source_context=_source_context(),
            families=_families(),
            run_executor=_executor(),
            resource_probe=_resources,
            synthetic_test=True,
            aggregate_store_directory=workspace_store,
        )

    assert not workspace_destination.exists()
    assert not list(workspace_store.glob("set-c-adequacy-*.sqlite3*"))
    assert list(tmp_path.iterdir()) == []


def test_alternate_config_is_rejected_before_load_or_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_config", lambda *_: pytest.fail("alternate config loaded"))
    monkeypatch.setattr(
        runner,
        "EnvironmentManager",
        lambda *_: pytest.fail("environment loaded for alternate config"),
    )

    with pytest.raises(PermissionError, match="checked-in default"):
        main(
            [
                "--execute-approved-edge-diagnostic",
                "--config",
                str(tmp_path / "alternate.yaml"),
            ]
        )


def test_fabricated_source_digest_is_rejected(monkeypatch):
    context = _production_source_context()
    context["source_hashes"] = dict(context["source_hashes"])
    context["source_hashes"]["modules/automata_engine.py"] = "b" * 64
    monkeypatch.setattr(
        adequacy,
        "_current_git_state",
        lambda: (context["source_revision"], False),
    )

    with pytest.raises(PopulationAdequacyError, match="digest claims"):
        adequacy._verify_production_evidence(_active_config(), context, {})


def test_fabricated_environment_is_rejected_before_teacher(monkeypatch):
    context = _production_source_context()
    monkeypatch.setattr(
        adequacy, "_current_source_hashes", lambda: dict(context["source_hashes"])
    )
    monkeypatch.setattr(
        adequacy,
        "_current_git_state",
        lambda: (context["source_revision"], False),
    )

    with pytest.raises(Exception, match="Environment grid_shape"):
        adequacy._verify_production_evidence(_active_config(), context, {})


def test_production_input_hash_keys_are_exact_and_forbid_ground_truth():
    context = _production_source_context()
    adequacy._validate_source_context(context, synthetic=False)
    context["input_hashes"]["stack_ground_truth.tif"] = "a" * 64

    with pytest.raises(PermissionError, match="ground_truth"):
        adequacy._validate_source_context(context, synthetic=False)


def test_duplicate_and_block_edges_connect_families(tmp_path):
    destination = tmp_path / "connected.aggregate.json"
    run_population_adequacy(
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        families=_families(),
        run_executor=_executor(connect_first_two=True),
        resource_probe=_resources,
        synthetic_test=True,
        aggregate_store_directory=tmp_path,
    )

    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["aggregate"]["leakage"]["component_count"] == 25
    assert any(
        component["family_count"] == 2
        for component in report["aggregate"]["leakage"]["components"]
    )


@pytest.mark.parametrize(
    ("cause", "duplicate_count", "block_count"),
    [("duplicate", 10, 0), ("block", 0, 1), ("joint", 10, 1)],
)
def test_edge_attribution_keeps_joint_adequacy_and_exposes_no_identity_values(
    tmp_path, cause, duplicate_count, block_count
):
    destination = tmp_path / "diagnostic-v2.aggregate.json"
    run_population_adequacy(
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        families=_families(),
        run_executor=_executor(link_cause=cause),
        resource_probe=_resources,
        synthetic_test=True,
        aggregate_store_directory=tmp_path,
    )
    report = json.loads(destination.read_text(encoding="utf-8"))
    leakage = report["aggregate"]["leakage"]
    attribution = leakage["edge_attribution"]
    assert attribution["family_pair_links"] == [
        {
            "family_ordinals": [1, 2],
            "duplicate_identity_count": duplicate_count,
            "spatial_block_count": block_count,
        }
    ]
    assert attribution["duplicate_only"]["component_count"] == (
        25 if duplicate_count else 26
    )
    assert attribution["block_only"]["component_count"] == (25 if block_count else 26)
    assert attribution["joint"] == {"component_count": 25, "family_counts": [2] + [1] * 24}
    assert leakage["component_count"] == leakage["qualifying_component_count"] == 25
    assert report["adequacy"]["status"] == "pass"
    assert b"family-01-cell" not in destination.read_bytes()
    assert b"family-01-block" not in destination.read_bytes()
    assert list(tmp_path.iterdir()) == [destination]


def test_duplicate_and_block_edges_form_transitive_joint_bridge(tmp_path):
    destination = tmp_path / "bridge-v2.aggregate.json"
    run_population_adequacy(
        _active_config(),
        destination=destination,
        source_context=_source_context(),
        families=_families(),
        run_executor=_executor(link_cause="bridge"),
        resource_probe=_resources,
        synthetic_test=True,
        aggregate_store_directory=tmp_path,
    )
    leakage = json.loads(destination.read_text(encoding="utf-8"))["aggregate"]["leakage"]
    attribution = leakage["edge_attribution"]
    assert attribution["family_pair_links"] == [
        {
            "family_ordinals": [1, 2],
            "duplicate_identity_count": 10,
            "spatial_block_count": 0,
        },
        {
            "family_ordinals": [2, 3],
            "duplicate_identity_count": 0,
            "spatial_block_count": 1,
        },
    ]
    assert attribution["duplicate_only"]["family_counts"] == [2] + [1] * 24
    assert attribution["block_only"]["family_counts"] == [2] + [1] * 24
    assert attribution["joint"]["family_counts"] == [3] + [1] * 23
    assert leakage["component_count"] == 24
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize(
    "updates",
    [
        {"termination_reason": "safety_limit", "authoritative": False},
        {"rows": 11},
        {"run_id": "fabricated-run-id"},
    ],
)
def test_censored_count_or_identity_failure_publishes_nothing(tmp_path, updates):
    destination = tmp_path / "adequacy.aggregate.json"

    with pytest.raises(PopulationAdequacyError):
        run_population_adequacy(
            _active_config(),
            destination=destination,
            source_context=_source_context(),
            families=_families(),
            run_executor=_executor(result_updates=updates),
            resource_probe=_resources,
            synthetic_test=True,
            aggregate_store_directory=tmp_path,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob("*.sqlite3*"))


def test_synthetic_checkpoint_resumes_complete_runs_and_cleans_up(tmp_path, capsys):
    destination = tmp_path / "resumed.aggregate.json"
    checkpoint = tmp_path / "resumed.sqlite3"
    orders = []
    first = _executor(fail_order=203)

    def interrupted(spec, consume):
        orders.append(spec.order)
        return first(spec, consume)

    arguments = dict(
        destination=destination, source_context=_source_context(),
        families=_families(), resource_probe=_resources, synthetic_test=True,
        checkpoint_path=checkpoint,
    )
    with pytest.raises(RuntimeError, match="synthetic run failure"):
        run_population_adequacy(_active_config(), run_executor=interrupted, **arguments)
    assert orders == list(range(1, 204))
    assert capsys.readouterr().out.splitlines() == [
        "Gate 6 progress: 100/884 runs complete",
        "Gate 6 progress: 200/884 runs complete",
    ]
    with closing(sqlite3.connect(checkpoint)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM completed_runs").fetchone() == (202,)
    resumed_orders = []
    second = _executor()

    def continued(spec, consume):
        resumed_orders.append(spec.order)
        return second(spec, consume)

    result = run_population_adequacy(
        _active_config(), run_executor=continued, resume=True, **arguments)
    assert result["run_count"] == 884
    assert resumed_orders == list(range(203, 885))
    assert capsys.readouterr().out.splitlines() == [
        f"Gate 6 progress: {order}/884 runs complete"
        for order in (300, 400, 500, 600, 700, 800, 884)
    ]
    assert list(tmp_path.iterdir()) == [destination]
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["aggregate"]["totals"]["rows"] == 8840
    assert report["resources"]["sample_count"] == 1 + 2 * 884 + 2
    assert "checkpoint" not in report


def test_checkpoint_drift_and_corruption_reject_before_executor(tmp_path):
    destination = tmp_path / "resumed.aggregate.json"
    checkpoint = tmp_path / "resumed.sqlite3"
    arguments = dict(
        destination=destination, source_context=_source_context(),
        families=_families(), resource_probe=_resources, synthetic_test=True,
        checkpoint_path=checkpoint,
    )
    with pytest.raises(RuntimeError):
        run_population_adequacy(
            _active_config(), run_executor=_executor(fail_order=3), **arguments)
    assert checkpoint.exists()

    def forbidden_executor(*_):
        pytest.fail("executor reached before checkpoint validation")

    drifted = dict(arguments)
    drifted["source_context"] = {**_source_context(), "source_revision": "drifted"}
    with pytest.raises(PopulationAdequacyError, match="drifted"):
        run_population_adequacy(
            _active_config(), run_executor=forbidden_executor, resume=True, **drifted)
    with closing(sqlite3.connect(checkpoint)) as connection:
        connection.execute("UPDATE completed_runs SET run_order=4 WHERE run_order=2")
        connection.commit()
    with pytest.raises(PopulationAdequacyError, match="gap"):
        run_population_adequacy(
            _active_config(), run_executor=forbidden_executor, resume=True, **arguments)
    assert not destination.exists()
    assert checkpoint.exists()


def test_checkpoint_rejected_for_production_and_existing_destination(tmp_path):
    with pytest.raises(PermissionError, match="Production checkpoints"):
        run_population_adequacy(
            _active_config(), destination=adequacy.approved_destination(),
            source_context=_production_source_context(),
            checkpoint_path=tmp_path / "forbidden.sqlite3", resume=True,
        )
    destination = tmp_path / "existing.aggregate.json"
    destination.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_population_adequacy(
            _active_config(), destination=destination,
            source_context=_source_context(), families=_families(),
            run_executor=_executor(), resource_probe=_resources,
            synthetic_test=True, checkpoint_path=tmp_path / "new.sqlite3",
        )
    assert not (tmp_path / "new.sqlite3").exists()


def test_failed_attempt_charges_runtime_and_crash_downtime_fails_closed(tmp_path):
    checkpoint = tmp_path / "attempt.sqlite3"
    destination = tmp_path / "attempt.aggregate.json"
    arguments = dict(
        destination=destination, source_context=_source_context(),
        families=_families(), resource_probe=_resources, synthetic_test=True,
        checkpoint_path=checkpoint,
    )
    base = _executor()

    def fails_after_consuming(spec, consume):
        base(spec, consume)
        time.sleep(0.02)
        raise RuntimeError("failed after batch")

    with pytest.raises(RuntimeError, match="failed after batch"):
        run_population_adequacy(
            _active_config(), run_executor=fails_after_consuming, **arguments)
    with closing(sqlite3.connect(checkpoint)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM completed_runs").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM identities").fetchone() == (0,)
        state = json.loads(connection.execute(
            "SELECT value FROM checkpoint_meta WHERE key='state'").fetchone()[0])
        assert state["elapsed_seconds"] >= 0.02
        assert state["attempt_started_at_utc"] is None
        state["attempt_started_at_utc"] = (
            datetime.now(timezone.utc) - timedelta(hours=7)
        ).isoformat()
        connection.execute(
            "UPDATE checkpoint_meta SET value=? WHERE key='state'",
            (json.dumps(state),),
        )
        connection.commit()

    def forbidden_executor(*_):
        pytest.fail("executor reached after crashed attempt exceeded budget")

    with pytest.raises(PopulationAdequacyError, match="runtime limit"):
        run_population_adequacy(
            _active_config(), run_executor=forbidden_executor, resume=True,
            **arguments)
    assert not destination.exists()


def test_policy_replay_is_disabled_and_recomputes_block_only_adequacy(tmp_path):
    source = tmp_path / "source-v2.aggregate.json"
    destination = tmp_path / "replay-v3.aggregate.json"
    run_population_adequacy(
        _active_config(), destination=source, source_context=_source_context(),
        families=_families(), run_executor=_executor(link_cause="duplicate"),
        resource_probe=_resources, synthetic_test=True,
        aggregate_store_directory=tmp_path,
    )
    source_report = json.loads(source.read_text(encoding="utf-8"))
    assert source_report["aggregate"]["leakage"]["component_count"] == 25
    expected = dict(
        expected_file_sha256=sha256(source.read_bytes()).hexdigest(),
        expected_payload_sha256=source_report["payload_sha256"],
        expected_revision="synthetic-revision",
    )
    with pytest.raises(PermissionError, match="separate one-run approval"):
        replay.replay_policy_v3(source, destination, **expected)
    result = replay.replay_policy_v3(
        source, destination, synthetic_test=True, resource_probe=_resources, **expected)
    assert result["adequacy_status"] == "pass"
    published = json.loads(destination.read_text(encoding="utf-8"))
    assert published["feature_collision_family_pair_count"] == 1
    assert published["leakage"]["component_count"] == 26
    assert published["leakage"]["qualifying_component_count"] == 26
    assert published["leakage"]["role_identities_exposed"] is False
    assert "components" not in published["leakage"]
    assert set(tmp_path.iterdir()) == {source, destination}


def test_policy_replay_rejects_hash_and_count_drift(tmp_path):
    source = tmp_path / "source-v2.aggregate.json"
    run_population_adequacy(
        _active_config(), destination=source, source_context=_source_context(),
        families=_families(), run_executor=_executor(), resource_probe=_resources,
        synthetic_test=True, aggregate_store_directory=tmp_path,
    )
    report = json.loads(source.read_text(encoding="utf-8"))
    expected = dict(
        expected_file_sha256=sha256(source.read_bytes()).hexdigest(),
        expected_payload_sha256=report["payload_sha256"],
        expected_revision="synthetic-revision", synthetic_test=True,
        resource_probe=_resources,
    )
    report["aggregate"]["scenario_families"][0]["class_counts"]["positive_1"] += 1
    source.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(PopulationAdequacyError, match="whole-file hash"):
        replay.replay_policy_v3(source, tmp_path / "refused.json", **expected)
    report.pop("payload_sha256")
    payload = adequacy._canonical_json(report)
    report["payload_sha256"] = sha256(payload).hexdigest()
    source.write_bytes(adequacy._canonical_json(report))
    expected["expected_file_sha256"] = sha256(source.read_bytes()).hexdigest()
    expected["expected_payload_sha256"] = report["payload_sha256"]
    with pytest.raises(PopulationAdequacyError, match="family totals mismatch"):
        replay.replay_policy_v3(source, tmp_path / "refused.json", **expected)
    assert not (tmp_path / "refused.json").exists()


def test_policy_replay_runner_requires_gate_and_routes_only_approved_paths(
    tmp_path, monkeypatch,
):
    source = tmp_path / "approved-source.json"
    destination = tmp_path / "approved-destination.json"
    with pytest.raises(SystemExit) as exc:
        replay_runner.main([])
    assert exc.value.code == 2

    captured = {}

    def fake_replay(actual_source, actual_destination, **kwargs):
        captured.update(source=actual_source, destination=actual_destination, **kwargs)
        return {"path": actual_destination, "payload_sha256": "a" * 64,
                "component_count": 26, "adequacy_status": "pass"}

    monkeypatch.setattr(replay_runner, "approved_source", lambda: source)
    monkeypatch.setattr(replay_runner, "approved_destination", lambda: destination)
    monkeypatch.setattr(replay_runner, "replay_policy_v3", fake_replay)
    assert replay_runner.main(["--execute-approved-policy-replay"]) == 0
    assert captured == {
        "source": source,
        "destination": destination,
        "production_authorized": True,
    }


def test_policy_replay_production_paths_and_resources_fail_closed(tmp_path):
    with pytest.raises(PermissionError, match="paths differ"):
        replay.replay_policy_v3(
            tmp_path / "source.json", tmp_path / "destination.json",
            production_authorized=True, resource_probe=_resources,
        )

    source = tmp_path / "source-v2.aggregate.json"
    destination = tmp_path / "replay-v3.aggregate.json"
    run_population_adequacy(
        _active_config(), destination=source, source_context=_source_context(),
        families=_families(), run_executor=_executor(), resource_probe=_resources,
        synthetic_test=True, aggregate_store_directory=tmp_path,
    )
    report = json.loads(source.read_text(encoding="utf-8"))

    def insufficient(_: Path) -> ResourceSnapshot:
        return replace(_resources(destination), available_memory_bytes=1)

    with pytest.raises(adequacy.PopulationResourceError, match="memory"):
        replay.replay_policy_v3(
            source, destination, synthetic_test=True,
            expected_file_sha256=sha256(source.read_bytes()).hexdigest(),
            expected_payload_sha256=report["payload_sha256"],
            expected_revision="synthetic-revision", resource_probe=insufficient,
        )
    assert not destination.exists()


def test_future_checkpoint_storage_telemetry_and_recovery_fail_closed(tmp_path):
    cp = tmp_path / "journal.sqlite3"
    cp.touch()
    telemetry = tmp_path / "resource.jsonl"
    now = datetime.now(timezone.utc)
    digest = "a" * 64
    binding = {
        "schema": checkpoint_contract.CHECKPOINT_SCHEMA,
        "package_id": checkpoint_contract.CHECKPOINT_PACKAGE_ID,
        "expansion_id": checkpoint_contract.CHECKPOINT_PACKAGE_ID,
        "source_revision": "b" * 40, "source_dirty": False,
        "hashes": {key: digest for key in checkpoint_contract.REQUIRED_BINDING_HASHES},
        "source_file_hashes": {
            path: digest for path in adequacy.PRODUCTION_SOURCE_RELATIVE_PATHS},
        "input_hashes": {"synthetic_input": digest},
        "run_count": 884,
    }
    checkpoint_state = {
        "aggregate_counts": {"rows": 1, "positive_1": 1},
        "identity_totals": {"duplicate": 1, "block": 1},
        "cumulative_resource": {"elapsed_seconds": 3.0, "samples": 4},
    }
    record = {
        "binding": binding, "execution_id": "synthetic-execution",
        "created_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(days=30)).isoformat(),
        "completed_run_order": 1,
        "cumulative_resource_sha256": sha256(adequacy._canonical_json(
            checkpoint_state["cumulative_resource"])).hexdigest(),
        "aggregate_counts_sha256": sha256(adequacy._canonical_json(
            checkpoint_state["aggregate_counts"])).hexdigest(),
        "identity_totals_sha256": sha256(adequacy._canonical_json(
            checkpoint_state["identity_totals"])).hexdigest(),
        "checkpoint_state": checkpoint_state,
    }
    def storage(_):
        return {"filesystem": "NTFS", "is_reparse_point": False,
                "inheritance_enabled": False, "owner_sid": "S-1-test",
                "allowed_principals": ["S-1-test", "NT AUTHORITY\\SYSTEM",
                                       "BUILTIN\\Administrators"],
                "denied_principals": []}
    def sample(phase, elapsed, completed, run_order, exit_status=None):
        return {"execution_id": "synthetic-execution", "attempt_id": "attempt-1",
                "process_id": 42, "process_started_at_utc": now.isoformat(),
                "captured_at_utc": (now + timedelta(seconds=elapsed)).isoformat(),
                "monotonic_elapsed_seconds": elapsed, "phase": phase,
                "run_order": run_order, "available_memory_bytes": 8_000_000_000,
                "process_rss_bytes": 100_000_000,
                "process_peak_rss_bytes": 100_000_000,
                "free_disk_bytes": 10_000_000_000,
                "completed_run_order": completed, "process_exit_status": exit_status}
    for item in (sample("launch", 0, 0, None), sample("before_run", 1, 0, 1),
                 sample("after_run", 2, 1, 1), sample("exit", 3, 1, None, 1)):
        checkpoint_contract.append_durable_telemetry(
            telemetry, item, synthetic_test=True)
    args = dict(
        checkpoint_path=cp, telemetry_path=telemetry, checkpoint_record=record,
        expected_binding=binding, storage_verifier=storage,
        executing_user_sid="S-1-test", report_exists=False, report_verifier=None,
        sqlite_integrity_ok=True,
    )
    with pytest.raises(PermissionError):
        checkpoint_contract.inspect_resume_contract(**args)
    assert checkpoint_contract.inspect_resume_contract(
        **args, synthetic_test=True) == "eligible_for_separately_authorized_resume"
    assert checkpoint_contract.inspect_resume_contract(
        **{**args, "report_exists": True, "report_verifier": lambda: {
            "status": "complete", "whole_file_sha256": digest,
            "payload_sha256": digest, "complete_run_count": 884,
            "provenance_verified": True, "integrity_verified": True}},
        synthetic_test=True) == "published_validate_then_owner_approved_cleanup"
    with pytest.raises(PopulationAdequacyError, match="Existing report invalid"):
        checkpoint_contract.inspect_resume_contract(
            **{**args, "report_exists": True,
                "report_verifier": lambda: {"status": "failed"}}, synthetic_test=True)
    with pytest.raises(PopulationAdequacyError, match="ACL"):
        checkpoint_contract.inspect_resume_contract(
            **{**args, "storage_verifier": lambda _: {
                **storage(tmp_path), "inheritance_enabled": True}},
            synthetic_test=True)
    intact_telemetry = telemetry.read_bytes()
    telemetry.write_bytes(b"\n".join(intact_telemetry.splitlines()[:-1]) + b"\n")
    with pytest.raises(PopulationAdequacyError, match="exit record"):
        checkpoint_contract.inspect_resume_contract(**args, synthetic_test=True)
    telemetry.write_bytes(intact_telemetry)
    mutated = [json.loads(line) for line in intact_telemetry.splitlines()]
    mutated[2]["process_peak_rss_bytes"] = adequacy.RSS_CEILING_BYTES + 1
    telemetry.write_bytes(b"\n".join(adequacy._canonical_json(item) for item in mutated) + b"\n")
    with pytest.raises(PopulationAdequacyError, match="breach"):
        checkpoint_contract.inspect_resume_contract(**args, synthetic_test=True)
    mutated[2]["process_peak_rss_bytes"] = 100_000_000
    mutated[2]["monotonic_elapsed_seconds"] = 7
    telemetry.write_bytes(b"\n".join(adequacy._canonical_json(item) for item in mutated) + b"\n")
    with pytest.raises(PopulationAdequacyError, match="sampling gap"):
        checkpoint_contract.inspect_resume_contract(**args, synthetic_test=True)
    telemetry.write_bytes(intact_telemetry)
    expired = {**record, "created_at_utc": (now - timedelta(days=31)).isoformat(),
               "expires_at_utc": (now - timedelta(days=1)).isoformat()}
    with pytest.raises(PopulationAdequacyError, match="retention expired"):
        checkpoint_contract.inspect_resume_contract(
            **{**args, "checkpoint_record": expired}, synthetic_test=True)
