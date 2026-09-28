from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import modules.set_c_population_adequacy as adequacy
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
    def __init__(self, family_order: int, *, connect_first_two: bool = False) -> None:
        self.labels = np.array([0] * 5 + [1] * 5, dtype=np.int8)
        identity_order = 1 if connect_first_two and family_order in (1, 2) else family_order
        self.duplicate_group_ids = np.array(
            [f"family-{identity_order:02d}-cell-{index:02d}" for index in range(10)],
            dtype="S64",
        )
        self.spatial_block_ids = np.array(
            [f"family-{identity_order:02d}-block"] * 10,
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
            consume(_Batch(spec.family.order, connect_first_two=connect_first_two))
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
        main(["--execute-approved-expansion", "--config", str(CONFIG_PATH)])


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
                "--execute-approved-expansion",
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
