import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules import set_c_generation as generation
import run_set_c_generation
from test_phase10_architecture_contract import (
    _batch,
    _package_identity,
    _publisher_config,
    _run,
)
from modules.set_c_publication import _canonical_bytes
from modules.set_c_pilot import ResourceSnapshot
from modules.set_c_population_adequacy import PopulationResourceError


def _passing_replay_report():
    return {
        "replay_id": "phase9_set_c_population_adequacy_policy_replay_v3",
        "policy": "scenario_family_stable_cell_spatial_block_provenance_record; predictor_collisions_audit_only",
        "source_file_sha256": generation.GATE6_SOURCE_FILE_SHA256,
        "source_payload_sha256": generation.GATE6_SOURCE_PAYLOAD_SHA256,
        "source_revision": generation.GATE6_SOURCE_REVISION,
        "population_run_count": generation.EXPECTED_RUN_COUNT,
        "feature_collision_family_pair_count": 97,
        "leakage": {
            "component_count": generation.EXPECTED_FAMILY_COUNT,
            "qualifying_component_count": generation.EXPECTED_FAMILY_COUNT,
            "component_family_counts": [1] * generation.EXPECTED_FAMILY_COUNT,
            "role_identities_exposed": False,
        },
        "adequacy": {
            "status": "pass",
            "gates": {
                "all_884_runs_complete_inactive_authoritative_and_count_matched": True,
                "at_least_23_qualifying_components": True,
                "every_qualifying_component_has_five_of_each_class": True,
                "unexposed_role_minima_material_coverage_and_concentration_feasible": True,
                "one_live_run_and_bounded_complete_batch_draining": True,
                "no_estimator_protected_input_row_split_or_model_output": True,
            },
        },
        "integrity": {
            "row_level_payload_published": False,
            "split_roles_assigned": False,
            "estimator_accessed": False,
            "protected_spatial_reference_accessed": False,
            "sole_persistent_output_is_aggregate_json": True,
        },
    }


def _write_bound_report(path: Path, report: dict) -> tuple[str, str]:
    payload = sha256(_canonical_bytes(report)).hexdigest()
    content = _canonical_bytes({**report, "payload_sha256": payload})
    path.write_bytes(content)
    return sha256(content).hexdigest(), payload


def test_runner_is_disabled_without_approved_execution_flag():
    with pytest.raises(SystemExit) as error:
        run_set_c_generation.main([])
    assert error.value.code == 2


def test_gate6_replay_hash_and_policy_binding_in_temp_path(tmp_path, monkeypatch):
    path = tmp_path / "replay.json"
    file_hash, payload_hash = _write_bound_report(path, _passing_replay_report())
    monkeypatch.setattr(generation, "REPLAY_FILE_SHA256", file_hash)
    monkeypatch.setattr(generation, "REPLAY_PAYLOAD_SHA256", payload_hash)
    assert generation._verify_replay_report(path)["adequacy"]["status"] == "pass"

    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(PermissionError, match="whole-file hash"):
        generation._verify_replay_report(path)


def test_gate6_source_run_order_is_bound_in_temp_path(tmp_path, monkeypatch):
    specs = (SimpleNamespace(run_id="run-1"), SimpleNamespace(run_id="run-2"))
    monkeypatch.setattr(generation, "EXPECTED_FAMILY_COUNT", 2)
    monkeypatch.setattr(generation, "EXPECTED_RUN_COUNT", 2)
    report = {
        "status": "complete",
        "provenance": {"source_revision": generation.GATE6_SOURCE_REVISION},
        "population_contract": {
            "family_count": 2,
            "run_count": 2,
            "run_order_sha256": generation.canonical_metadata_hash(["run-1", "run-2"]),
            "block_size": [128, 128],
            "block_origin": [0, 0],
            "split_roles_assigned": False,
        },
        "aggregate": {"runs": [{"run_id": "run-1"}, {"run_id": "run-2"}]},
    }
    path = tmp_path / "source.json"
    file_hash, payload_hash = _write_bound_report(path, report)
    monkeypatch.setattr(generation, "GATE6_SOURCE_FILE_SHA256", file_hash)
    monkeypatch.setattr(generation, "GATE6_SOURCE_PAYLOAD_SHA256", payload_hash)
    assert generation._verify_gate6_source(path, specs)["status"] == "complete"
    with pytest.raises(PermissionError, match="family or run order"):
        generation._verify_gate6_source(path, tuple(reversed(specs)))


def test_synthetic_generation_publishes_only_under_temp_path(tmp_path):
    package_identity = _package_identity()
    spec = SimpleNamespace(run_id="run-1")

    def fake_executor(_spec, consume_batch):
        consume_batch(_batch())
        return SimpleNamespace(
            authoritative=True,
            completeness_status="complete",
            termination_reason="inactive",
            expected_transition_count=1,
            captured_transition_count=1,
            unexpected_outputs=(),
            rows=2,
            pending_rows=0,
            pending_bytes=0,
            max_live_full_grid_observations=1,
            collected_run=_run(),
        )

    probe = lambda _path: ResourceSnapshot(
        available_memory_bytes=12_000_000_000,
        process_rss_bytes=100_000_000,
        free_disk_bytes=20_000_000_000,
        captured_at_utc="synthetic",
    )
    package = generation.run_synthetic_generation(
        config=_publisher_config(),
        package_identity=package_identity,
        specs=(spec,),
        destination_root=tmp_path,
        executor=fake_executor,
        resource_probe=probe,
    )
    assert package == tmp_path / "synthetic_set_c" / "v1"
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["row_count"] == 2
    assert manifest["split_status"] == "unassigned"
    assert len(manifest["runs"]) == 1


def test_synthetic_resource_breach_discards_staged_package(tmp_path):
    package_identity = _package_identity()
    spec = SimpleNamespace(run_id="run-1")

    def fake_executor(_spec, consume_batch):
        consume_batch(_batch())
        return SimpleNamespace(
            authoritative=True,
            completeness_status="complete",
            termination_reason="inactive",
            expected_transition_count=1,
            captured_transition_count=1,
            unexpected_outputs=(),
            rows=2,
            pending_rows=0,
            pending_bytes=0,
            max_live_full_grid_observations=1,
            collected_run=_run(),
        )

    probe = lambda _path: ResourceSnapshot(1, 1, 1, "synthetic")
    with pytest.raises(PopulationResourceError):
        generation.run_synthetic_generation(
            config=_publisher_config(),
            package_identity=package_identity,
            specs=(spec,),
            destination_root=tmp_path,
            executor=fake_executor,
            resource_probe=probe,
        )
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]
