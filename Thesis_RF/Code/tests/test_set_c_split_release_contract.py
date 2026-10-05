"""Contract checks use synthetic CSVs and temporary destinations only."""

from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from modules.feature_pipeline import (
    AUTHORITATIVE_CSV_COLUMNS,
    CANONICAL_FEATURE_NAMES,
    TARGET_NAME,
)
from modules import set_c_split_release as release


def _synthetic_source(path):
    rows = []
    for component_index in range(26):
        component_id = f"synthetic-family-{component_index:02d}"
        for observation_index in range(100):
            material_class = observation_index // 20 + 1
            label = int(observation_index % 20 < 4)
            material_risk = {1: 0.95, 2: 0.8, 3: 0.48, 4: 0.3, 5: 0.15}[material_class]
            row = {name: 0.0 for name in CANONICAL_FEATURE_NAMES}
            row.update(
                set_id="C", experiment_id="synthetic", event_id="event",
                scenario_family_id=component_id, scenario_id=component_id,
                run_id=f"run-{component_id}", seed=component_index, timestep_t=0,
                timestep_t1=1, cell_row=component_index * 100 + observation_index,
                cell_col=observation_index % 128, grid_id="grid",
                stable_cell_id=f"{component_id}-{observation_index}",
                duplicate_group_id=f"dup-{component_id}-{observation_index}",
                spatial_block_id=f"block-{component_id}", ignition_set_sha256="a" * 64,
                run_identity_sha256="b" * 64, provenance_sha256="c" * 64,
                feature_schema_version="set_c.v1", target_version="ignition_t_plus_1.v1",
                building_presence=1.0, material_class=float(material_class), material_risk=material_risk,
                composite_flammability=material_risk, neighbor_burning_count=1.0,
            )
            row[TARGET_NAME] = label
            rows.append(row)
    pd.DataFrame(rows, columns=AUTHORITATIVE_CSV_COLUMNS).to_csv(path, index=False)
    frame = pd.read_csv(path)
    source_digest = sha256(path.read_bytes()).hexdigest()
    provenance = {
        "source_manifest": {
            "schema_version": release.SOURCE_MANIFEST_VERSION,
            "manifest_id": "synthetic-manifest-001",
            "source_csv_sha256": source_digest,
            "row_count": len(frame),
            "columns_sha256": release._canonical_json_sha256(list(AUTHORITATIVE_CSV_COLUMNS)),
        },
        "configuration": {
            "schema_version": release.ALLOCATION_CONFIG_VERSION,
            "graph_version": release.LEAKAGE_GRAPH_VERSION,
            "policy_version": release.APPROVED_POLICY_VERSION,
            "dataset_sha256": "0" * 64,
            "complete_event_ids": [],
        },
        "source_revision": {
            "schema_version": release.SOURCE_REVISION_VERSION,
            "revision": "synthetic:" + "d" * 40,
        },
        "authorization": {
            "schema_version": release.AUTHORIZATION_VERSION,
            "authorization_id": "synthetic-only",
            "scope": "temporary synthetic allocation/export tests",
            "production_enabled": False,
        },
    }
    return provenance


def test_synthetic_four_role_export_has_bound_schemas_and_no_dev_membership(tmp_path):
    source = tmp_path / "synthetic.csv"
    provenance = _synthetic_source(source)
    protected_parent = tmp_path / "protected-parent"
    development_parent = tmp_path / "development-parent"
    protected_parent.mkdir()
    development_parent.mkdir()
    protected = protected_parent / "set_c_v1"
    development = development_parent / "set_c_v1"

    published_protected, published_development = release.export_set_c_split(
        source,
        protected_destination=protected,
        development_destination=development,
        synthetic=True,
        **provenance,
    )

    expected_protected = {
        "canonical.csv", "split_manifest.csv", "final_test_rows.csv", "attestation.json"
    }
    assert {path.name for path in published_protected.iterdir()} == expected_protected
    assert {path.name for path in published_development.iterdir()} == {
        "train.csv", "validation.csv", "calibration.csv", "attestation.json"
    }
    protected_attestation = json.loads((published_protected / "attestation.json").read_text())
    assert protected_attestation["schema_version"] == release.PROTECTED_ATTESTATION_SCHEMA_VERSION
    assert protected_attestation["package_schema_version"] == release.PROTECTED_SCHEMA_VERSION
    assert protected_attestation["mapping_schema_version"] == release.MAPPING_SCHEMA_VERSION
    assert protected_attestation["final_test_schema_version"] == release.FINAL_TEST_SCHEMA_VERSION
    assert protected_attestation["source_csv_sha256"] == sha256(source.read_bytes()).hexdigest()
    assert protected_attestation["publication_state"] == "protected_only_incomplete_until_paired_release"
    assert protected_attestation["source_provenance"]["hashes"] == {
        "source_manifest_sha256": release._canonical_json_sha256(provenance["source_manifest"]),
        "configuration_sha256": release._canonical_json_sha256(provenance["configuration"]),
        "source_revision_sha256": release._canonical_json_sha256(provenance["source_revision"]),
        "authorization_sha256": release._canonical_json_sha256(provenance["authorization"]),
    }
    mapping = pd.read_csv(published_protected / "split_manifest.csv")
    assert tuple(mapping.columns) == release.SPLIT_MANIFEST_COLUMNS
    assert len(mapping) == 2600
    assert mapping.groupby("leakage_component_id")["split_role"].nunique().max() == 1
    assert mapping.groupby("leakage_component_id")["split_role"].first().value_counts().to_dict() == dict(zip(
        (*release.DEVELOPMENT_SPLIT_ROLES, "final_test"), release.ROLE_COMPONENT_TARGETS
    ))
    final_rows = pd.read_csv(published_protected / "final_test_rows.csv")
    assert len(final_rows) == 500
    assert tuple(final_rows.columns) == (*AUTHORITATIVE_CSV_COLUMNS, "leakage_component_id")
    for role in release.DEVELOPMENT_SPLIT_ROLES:
        rows = pd.read_csv(published_development / f"{role}.csv")
        assert tuple(rows.columns) == release.RELEASE_COLUMNS
        assert not set(rows.columns) & {"scenario_family_id", "scenario_id", "run_id", "stable_cell_id"}
    attestation_hash = sha256((published_development / "attestation.json").read_bytes()).hexdigest()
    combined, _ = release.load_development_release(
        published_development,
        expected_attestation_sha256=attestation_hash,
        allow_synthetic=True,
    )
    assert len(combined) == 2100
    assert "final_test" not in set(combined["split_role"])
    release.verify_publication_pair(
        published_protected, published_development,
        expected_development_attestation_sha256=attestation_hash,
        allow_synthetic=True,
    )


def test_production_export_fails_before_reading_source(tmp_path):
    with pytest.raises(PermissionError, match="operator opt-in"):
        release.export_set_c_split(
            tmp_path / "does-not-exist.csv",
            protected_destination=tmp_path / "protected",
            development_destination=tmp_path / "development",
        )


def test_production_export_requires_nonempty_approval_reference_before_source_access(tmp_path):
    with pytest.raises(PermissionError, match="approval reference"):
        release.export_set_c_split(
            tmp_path / "does-not-exist.csv",
            protected_destination=tmp_path / "protected",
            development_destination=tmp_path / "development",
            operator_opt_in=True,
            authorization_reference="   ",
        )


def test_production_shaped_export_binds_synthetic_package_and_manifest(tmp_path, monkeypatch):
    repo = tmp_path / "fixture-repository"
    source_dir = repo / "Thesis_RF" / "Code" / "output" / "datasets" / "authoritative_set_c" / "v1"
    source_dir.mkdir(parents=True)
    source = source_dir / "set_c.v1.csv"
    _synthetic_source(source)
    source_digest = sha256(source.read_bytes()).hexdigest()
    manifest = {
        "manifest_schema_version": release.PACKAGE_SCHEMA_VERSION,
        "artifact_kind": "authoritative_set_c_observation_package",
        "split_status": "unassigned",
        "dataset_path": source.name,
        "dataset_sha256": source_digest,
        "row_count": 2600,
        "canonical_csv_columns": list(AUTHORITATIVE_CSV_COLUMNS),
        "manifest_payload_sha256": "f" * 64,
    }
    manifest_path = source_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_digest = sha256(manifest_path.read_bytes()).hexdigest()
    fake_module = repo / "Thesis_RF" / "Code" / "modules" / "set_c_split_release.py"
    monkeypatch.setattr(release, "__file__", str(fake_module))
    monkeypatch.setattr(release, "APPROVED_CSV_SHA256", source_digest)
    monkeypatch.setattr(release, "APPROVED_MANIFEST_SHA256", manifest_digest)
    monkeypatch.setattr(release, "validate_published_package", lambda path: manifest)
    monkeypatch.setattr(release, "_verify_clean_source_revision", lambda repo, revision: None)
    protected_parent = repo / release.PROTECTED_DIRECTORY_NAME
    development_parent = repo / release.DEVELOPMENT_DIRECTORY_NAME
    protected_parent.mkdir()
    development_parent.mkdir()

    protected, development = release.export_set_c_split(
        source,
        protected_destination=protected_parent / "set_c_v1",
        development_destination=development_parent / "set_c_v1",
        production_revision="d" * 40,
        authorization_reference="synthetic-production-shape-test",
        operator_opt_in=True,
    )

    protected_attestation = json.loads((protected / "attestation.json").read_text())
    development_attestation = json.loads((development / "attestation.json").read_text())
    assert protected_attestation["source_provenance"]["source_manifest"]["manifest_sha256"] == manifest_digest
    assert protected_attestation["source_provenance"]["configuration"]["dataset_sha256"] == source_digest
    assert protected_attestation["synthetic"] is False
    assert development_attestation["source_csv_sha256"] == source_digest
    assert development_attestation["synthetic"] is False
    assert development_attestation["publication_receipt"]["protected_attestation_sha256"] == sha256(
        (protected / "attestation.json").read_bytes()
    ).hexdigest()
    assert not set(pd.read_csv(development / "train.csv").columns) & {
        "split_role", "leakage_component_id", "scenario_family_id", "stable_cell_id"
    }


def test_pair_verification_rejects_protected_attestation_tampering(tmp_path):
    source = tmp_path / "synthetic.csv"
    provenance = _synthetic_source(source)
    protected_parent = tmp_path / "protected-parent"
    development_parent = tmp_path / "development-parent"
    protected_parent.mkdir()
    development_parent.mkdir()
    protected, development = release.export_set_c_split(
        source,
        protected_destination=protected_parent / "set_c_v1",
        development_destination=development_parent / "set_c_v1",
        synthetic=True,
        **provenance,
    )
    attestation_path = development / "attestation.json"
    expected_hash = sha256(attestation_path.read_bytes()).hexdigest()
    final_rows_path = protected / "final_test_rows.csv"
    original_final_bytes = final_rows_path.read_bytes()
    final_rows_path.write_bytes(original_final_bytes + b"\n")
    with pytest.raises(ValueError, match="attestation hashes"):
        release.verify_publication_pair(
            protected, development,
            expected_development_attestation_sha256=expected_hash,
            allow_synthetic=True,
        )
    final_rows_path.write_bytes(original_final_bytes)
    protected_attestation_path = protected / "attestation.json"
    protected_record = json.loads(protected_attestation_path.read_text(encoding="utf-8"))
    protected_record["publication_state"] = "complete"
    protected_attestation_path.write_text(json.dumps(protected_record), encoding="utf-8")

    with pytest.raises(ValueError, match="do not prove the same completed publication"):
        release.verify_publication_pair(
            protected, development,
            expected_development_attestation_sha256=expected_hash,
            allow_synthetic=True,
        )


def test_production_source_revision_requires_exact_clean_checkout(tmp_path, monkeypatch):
    calls = []

    def clean_git(command, **kwargs):
        calls.append(command)
        output = "a" * 40 if "rev-parse" in command else ""
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(release.subprocess, "run", clean_git)
    release._verify_clean_source_revision(tmp_path, "a" * 40)
    assert len(calls) == 2
    assert all("safe.directory=" + tmp_path.as_posix() in call for call in calls)
    assert all("--porcelain=v1" in call or "rev-parse" in call for call in calls)

    def dirty_git(command, **kwargs):
        return SimpleNamespace(stdout="a" * 40 if "rev-parse" in command else "?? untracked.py")

    monkeypatch.setattr(release.subprocess, "run", dirty_git)
    with pytest.raises(PermissionError, match="checkout is not clean"):
        release._verify_clean_source_revision(tmp_path, "a" * 40)


def test_export_derives_roles_internally_and_rejects_supplied_mapping(tmp_path):
    source = tmp_path / "synthetic.csv"
    provenance = _synthetic_source(source)
    args = {
        "protected_destination": tmp_path / "protected-parent" / "set_c_v1",
        "development_destination": tmp_path / "development-parent" / "set_c_v1",
        "synthetic": True,
        **provenance,
    }
    with pytest.raises(TypeError):
        release.export_set_c_split(source, ["caller-component"], {"caller-component": "train"}, **args)
    (tmp_path / "protected-parent").mkdir()
    (tmp_path / "development-parent").mkdir()
    protected, development = release.export_set_c_split(source, **args)
    assert protected.is_dir() and development.is_dir()


def test_d035_component_allocation_is_deterministic_and_hits_role_cardinalities():
    inventory = pd.DataFrame([
        {
            "component_id": f"synthetic-component-{index:02d}",
            "positives": 200,
            "negatives": 800,
            **{column: 1 for column in release.MATERIAL_COLUMNS},
        }
        for index in range(26)
    ])

    first = release._allocate_components(inventory, dataset_sha256="0" * 64)
    second = release._allocate_components(inventory, dataset_sha256="0" * 64)

    assert first == second
    assert [list(first.values()).count(role) for role in release.FOUR_ROLES] == [11, 5, 5, 5]


def test_export_rejects_source_manifest_hash_mismatch(tmp_path):
    source = tmp_path / "synthetic.csv"
    provenance = _synthetic_source(source)
    (tmp_path / "protected-parent").mkdir()
    (tmp_path / "development-parent").mkdir()
    provenance["source_manifest"]["source_csv_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not bind this CSV"):
        release.export_set_c_split(
            source,
            protected_destination=tmp_path / "protected-parent" / "set_c_v1",
            development_destination=tmp_path / "development-parent" / "set_c_v1",
            synthetic=True,
            **provenance,
        )


def test_existing_publications_are_never_overwritten(tmp_path):
    source = tmp_path / "synthetic.csv"
    provenance = _synthetic_source(source)
    protected_parent = tmp_path / "protected-parent"
    development_parent = tmp_path / "development-parent"
    protected_parent.mkdir()
    development_parent.mkdir()
    protected = protected_parent / "set_c_v1"
    development = development_parent / "set_c_v1"
    args = (source,)
    kwargs = {
        "protected_destination": protected,
        "development_destination": development,
        "synthetic": True,
        **provenance,
    }
    release.export_set_c_split(*args, **kwargs)
    protected_hash = sha256((protected / "attestation.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="overwrite"):
        release.export_set_c_split(*args, **kwargs)
    assert sha256((protected / "attestation.json").read_bytes()).hexdigest() == protected_hash


def test_second_publication_failure_preserves_incomplete_protected_bundle(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.csv"
    provenance = _synthetic_source(source)
    protected_parent = tmp_path / "protected-parent"
    development_parent = tmp_path / "development-parent"
    protected_parent.mkdir()
    development_parent.mkdir()
    protected = protected_parent / "set_c_v1"
    development = development_parent / "set_c_v1"
    real_publish = release._atomic_publish_directory
    calls = 0

    def fail_development(stage, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic injected publication failure")
        real_publish(stage, target)

    monkeypatch.setattr(release, "_atomic_publish_directory", fail_development)
    with pytest.raises(RuntimeError, match="protected publication remains incomplete") as caught:
        release.export_set_c_split(
            source,
            protected_destination=protected,
            development_destination=development,
            synthetic=True,
            **provenance,
        )
    assert caught.value.set_c_diagnostic["stage"] == "development_publication"
    assert caught.value.__cause__.set_c_diagnostic["stage"] == "development_publication"
    assert protected.exists()
    assert not development.exists()
    assert {path.name for path in protected.iterdir()} == {
        "canonical.csv", "split_manifest.csv", "final_test_rows.csv", "attestation.json"
    }
    protected_record = json.loads((protected / "attestation.json").read_text(encoding="utf-8"))
    assert protected_record["publication_state"] == "protected_only_incomplete_until_paired_release"
    with pytest.raises(FileNotFoundError):
        release.verify_publication_pair(
            protected, development,
            expected_development_attestation_sha256="a" * 64,
            allow_synthetic=True,
        )
    with pytest.raises(PermissionError):
        release.load_development_release(
            protected,
            expected_attestation_sha256="a" * 64,
            allow_synthetic=True,
        )
    with pytest.raises(FileExistsError, match="overwrite"):
        release.export_set_c_split(
            source,
            protected_destination=protected,
            development_destination=development,
            synthetic=True,
            **provenance,
        )
    assert not list(protected_parent.glob(".set-c-protected-*"))
    assert not list(development_parent.glob(".set-c-development-*"))


def test_pair_verification_failure_preserves_protected_only_state(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.csv"
    provenance = _synthetic_source(source)
    protected_parent = tmp_path / "protected-parent"
    development_parent = tmp_path / "development-parent"
    protected_parent.mkdir()
    development_parent.mkdir()
    protected = protected_parent / "set_c_v1"
    development = development_parent / "set_c_v1"

    def fail_pair_verification(*_args, **_kwargs):
        raise OSError("synthetic pair verification failure")

    monkeypatch.setattr(release, "_verify_publication_pair", fail_pair_verification)
    with pytest.raises(RuntimeError, match="requires human recovery") as caught:
        release.export_set_c_split(
            source,
            protected_destination=protected,
            development_destination=development,
            synthetic=True,
            **provenance,
        )

    assert caught.value.set_c_diagnostic["stage"] == "pair_validation"
    assert caught.value.__cause__.set_c_diagnostic["stage"] == "pair_validation"
    assert protected.exists() and not development.exists()
    protected_record = json.loads((protected / "attestation.json").read_text(encoding="utf-8"))
    assert protected_record["publication_state"] == "protected_only_incomplete_until_paired_release"
    with pytest.raises(FileExistsError, match="overwrite"):
        release.export_set_c_split(
            source,
            protected_destination=protected,
            development_destination=development,
            synthetic=True,
            **provenance,
        )


def test_unpublished_development_stage_is_rejected_after_receipt_before_rename(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.csv"
    provenance = _synthetic_source(source)
    protected_parent = tmp_path / "protected-parent"
    development_parent = tmp_path / "development-parent"
    protected_parent.mkdir()
    development_parent.mkdir()
    protected = protected_parent / "set_c_v1"
    development = development_parent / "set_c_v1"
    real_publish = release._atomic_publish_directory
    publishes = 0

    def interrupt_before_development_rename(stage, target):
        nonlocal publishes
        publishes += 1
        if publishes == 1:
            return real_publish(stage, target)
        assert target == development
        attestation = stage / "attestation.json"
        digest = sha256(attestation.read_bytes()).hexdigest()
        with pytest.raises(PermissionError, match="Unpublished development staging"):
            release.load_development_release(
                stage, expected_attestation_sha256=digest, allow_synthetic=True,
            )
        with pytest.raises(PermissionError, match="Unpublished development staging"):
            release.verify_publication_pair(
                protected, stage, expected_development_attestation_sha256=digest,
                allow_synthetic=True,
            )
        raise OSError("synthetic interruption after receipt creation")

    monkeypatch.setattr(release, "_atomic_publish_directory", interrupt_before_development_rename)
    with pytest.raises(RuntimeError, match="requires human recovery") as caught:
        release.export_set_c_split(
            source,
            protected_destination=protected,
            development_destination=development,
            synthetic=True,
            **provenance,
        )

    assert caught.value.set_c_diagnostic["stage"] == "development_publication"
    assert caught.value.__cause__.set_c_diagnostic["stage"] == "development_publication"
    assert protected.exists() and not development.exists()
    assert json.loads((protected / "attestation.json").read_text())["publication_state"] == (
        "protected_only_incomplete_until_paired_release"
    )


def test_source_export_requires_provenance_and_temp_paths(tmp_path):
    source = tmp_path / "synthetic.csv"
    provenance = _synthetic_source(source)
    protected_parent = tmp_path / "protected-parent"
    development_parent = tmp_path / "development-parent"
    protected_parent.mkdir()
    development_parent.mkdir()
    with pytest.raises(ValueError, match="Versioned synthetic provenance"):
        release.export_set_c_split(
            source,
            protected_destination=protected_parent / "set_c_v1",
            development_destination=development_parent / "set_c_v1",
            synthetic=True,
        )
    with pytest.raises(PermissionError, match="OS temp"):
        release.export_set_c_split(
            source,
            protected_destination=Path.cwd() / "outside-protected",
            development_destination=development_parent / "set_c_v1",
            synthetic=True,
            **provenance,
        )


def test_synthetic_output_rejects_reparse_parent_before_staging(tmp_path, monkeypatch):
    parent = tmp_path / "protected-parent"
    parent.mkdir()
    real_lstat = os.lstat

    def marked_reparse(path):
        attributes = real_lstat(path)
        if Path(path) == parent:
            return SimpleNamespace(st_file_attributes=0x400)
        return attributes

    monkeypatch.setattr(release.os, "lstat", marked_reparse)
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)

    with pytest.raises(PermissionError, match="reparse point"):
        release._validate_output_path(parent / "set_c_v1", synthetic=True, kind="protected")


def test_abandoned_stage_is_preserved_and_blocks_a_new_publication(tmp_path):
    parent = tmp_path / "protected-parent"
    parent.mkdir()
    abandoned = parent / ".set-c-protected-abandoned"
    abandoned.mkdir()
    marker = abandoned / "synthetic-only-marker"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="manual recovery"):
        release._validate_output_path(parent / "set_c_v1", synthetic=True, kind="protected")

    assert marker.read_text(encoding="utf-8") == "preserve"


def _allocation_inventory():
    return pd.DataFrame([
        {"component_id": f"synthetic-{index:02d}", "positives": 21, "negatives": 81,
         **{column: 1 for column in release.MATERIAL_COLUMNS}}
        for index in range(26)
    ])


@pytest.mark.parametrize("success,status,incumbent", [
    (False, 1, False), (False, 1, True), (False, 2, False),
    (False, 3, False), (False, 4, False), (True, 0, False),
    (False, 0, True), (True, 1, True),
])
def test_allocator_retains_failure_context_without_accepting_incumbent(
    monkeypatch, success, status, incumbent,
):
    calls = []

    def unsuccessful(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            success=success, status=status, x=np.zeros(112) if incumbent else None,
            message="SENSITIVE_SOLVER_TEXT", fun=123, mip_dual_bound=122, mip_gap=1,
        )

    monkeypatch.setattr(release, "milp", unsuccessful)
    with pytest.raises(RuntimeError, match="did not prove an exact optimum") as caught:
        release.allocate_synthetic_components(_allocation_inventory(), dataset_sha256="0" * 64)
    context = caught.value.set_c_diagnostic
    assert len(calls) == 1
    assert calls[0]["options"] == {"time_limit": 60, "mip_rel_gap": 0}
    assert context["stage"] == "allocation"
    assert context["objective_phase"] == "positive_deviation"
    assert context["objective_index"] == context["objective_ordinal"] == 1
    assert context["solver_status"] == status
    assert context["solver_success"] is success
    assert context["has_solution"] is incumbent
    assert context["mip_node_count"] is None  # Optional result field may be absent.
    assert context["failure_reason"] == "exact_optimum_not_proven"
    assert context["time_limit"] == 60
    assert context["elapsed_seconds"] >= 0
    assert not {"x", "message", "fun", "mip_dual_bound", "mip_gap"} & context.keys()


@pytest.mark.parametrize("fractional,reason,message", [
    (True, "fractional_solution", "fractional solution"),
    (False, "hard_constraint_violation", "hard constraint"),
])
def test_allocator_retains_distinct_solution_rejection_reasons(monkeypatch, fractional, reason, message):
    values = np.zeros(112)
    if fractional:
        values[0] = 0.5
    monkeypatch.setattr(release, "milp", lambda *args, **kwargs: SimpleNamespace(
        success=True, status=0, x=values, mip_node_count=7,
    ))
    with pytest.raises(RuntimeError, match=message) as caught:
        release.allocate_synthetic_components(_allocation_inventory(), dataset_sha256="0" * 64)
    assert caught.value.set_c_diagnostic["failure_reason"] == reason
    assert caught.value.set_c_diagnostic["mip_node_count"] == 7


@pytest.mark.parametrize("failure_index,phase,ordinal", [
    (2, "positive_deviation", 2), (6, "row_deviation", 2),
    (9, "tie_break", 1), (34, "tie_break", 26),
])
def test_allocator_identifies_later_failed_objective(monkeypatch, failure_index, phase, ordinal):
    # Uniform counts make these deviation optima and ordered role ties exact.
    values = np.zeros(112)
    component = 0
    for role, count in enumerate(release.ROLE_COMPONENT_TARGETS):
        for _ in range(count):
            values[component * 4 + role] = 1
            component += 1
        values[104 + role] = abs(count * 21 - release.ROLE_POSITIVE_BALANCE[role])
        values[108 + role] = abs(count * 102 - release.ROLE_ROW_BALANCE[role])
    calls = 0

    def fail_later(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["options"] == {"time_limit": 60, "mip_rel_gap": 0}
        return SimpleNamespace(
            success=calls != failure_index, status=0 if calls != failure_index else 1,
            x=values.copy() if calls != failure_index else None,
        )

    monkeypatch.setattr(release, "milp", fail_later)
    with pytest.raises(RuntimeError, match="exact optimum") as caught:
        release.allocate_synthetic_components(_allocation_inventory(), dataset_sha256="0" * 64)
    context = caught.value.set_c_diagnostic
    assert calls == context["objective_index"] == failure_index
    assert (context["objective_phase"], context["objective_ordinal"]) == (phase, ordinal)
    assert context["time_limit"] == 60


def test_solver_exception_keeps_original_type_and_safe_context(monkeypatch):
    error = RuntimeError("SENSITIVE_DEPENDENCY_TEXT")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(release, "milp", fail)
    with pytest.raises(RuntimeError) as caught:
        release.allocate_synthetic_components(_allocation_inventory(), dataset_sha256="0" * 64)
    assert caught.value is error
    assert caught.value.set_c_diagnostic["failure_reason"] == "solver_exception"
    assert "SENSITIVE" not in json.dumps(caught.value.set_c_diagnostic)


def test_allocator_failure_prevents_any_export_staging(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.csv"
    provenance = _synthetic_source(source)
    protected_parent, development_parent = tmp_path / "protected", tmp_path / "development"
    protected_parent.mkdir()
    development_parent.mkdir()
    monkeypatch.setattr(release, "milp", lambda *args, **kwargs: SimpleNamespace(
        success=False, status=1, x=None,
    ))
    touched = []

    def unexpected(*args, **kwargs):
        touched.append(True)
        raise AssertionError("Allocation failure reached staging/publication")

    monkeypatch.setattr(release.tempfile, "mkdtemp", unexpected)
    monkeypatch.setattr(release, "_atomic_publish_directory", unexpected)
    with pytest.raises(RuntimeError, match="exact optimum") as caught:
        release.export_set_c_split(
            source, protected_destination=protected_parent / "set_c_v1",
            development_destination=development_parent / "set_c_v1", synthetic=True, **provenance,
        )
    assert caught.value.set_c_diagnostic["stage"] == "allocation"
    assert not touched
    assert not list(protected_parent.iterdir()) and not list(development_parent.iterdir())


def test_diagnostic_attachment_failure_does_not_mask_original_error():
    class RefusesContext(RuntimeError):
        def __setattr__(self, name, value):
            raise TypeError("cannot attach")

    error = RefusesContext("original")
    release._attach_failure_context(error, {"stage": "allocation"})
    assert str(error) == "original"
