"""Disabled-by-default authoritative Set C generation runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import tempfile
import time
from typing import Any
from .set_c_collector import canonical_metadata_hash
from .set_c_population_adequacy import (
    APPROVED_INVENTORY_FILE_SHA256,
    APPROVED_INVENTORY_PAYLOAD_SHA256,
    APPROVED_PRIMARY_SHA256,
    APPROVED_RESERVE_SHA256,
    BATCH_ROWS,
    BLOCK_ORIGIN,
    BLOCK_SIZE,
    EXPECTED_FAMILY_COUNT,
    EXPECTED_MAPPED_BUILDING_SHA256,
    EXPECTED_RUN_COUNT,
    EXPECTED_SIMULATION_VALID_SHA256,
    EXPANSION_RUNTIME_LIMIT_SECONDS,
    FREE_DISK_FLOOR_BYTES,
    LAUNCH_MEMORY_FLOOR_BYTES,
    MAX_BUFFER_BYTES,
    MAX_BUFFER_ROWS,
    MODEL_FREE_SOURCE,
    PER_RUN_ROW_CEILING,
    RETRY_LIMIT,
    RSS_CEILING_BYTES,
    RUNTIME_MEMORY_FLOOR_BYTES,
    RUN_RUNTIME_LIMIT_SECONDS,
    SAFETY_LIMIT_TIMESTEPS,
    TOTAL_ROW_CEILING,
    _check_resources,
    _current_git_state,
    _run_config,
    build_model_free_executor,
    build_run_matrix,
    load_approved_families,
    validate_approved_environment_config,
)
from .set_c_pilot import ResourceSnapshot, capture_resource_snapshot, _validate_environment
from .set_c_publication import (
    AUTHORITATIVE_VERSIONED_MODE,
    DESTINATION_PATTERN,
    PACKAGE_VERSION,
    SetCPublicationSession,
    _canonical_bytes,
)
from .set_c_population_replay import (
    SOURCE_FILE_SHA256 as GATE6_SOURCE_FILE_SHA256,
    SOURCE_PAYLOAD_SHA256 as GATE6_SOURCE_PAYLOAD_SHA256,
    SOURCE_REVISION as GATE6_SOURCE_REVISION,
)
from .set_c_site_inventory import APPROVED_RELATIVE_DESTINATION as INVENTORY_RELATIVE_PATH
from .set_c_population_adequacy import APPROVED_RELATIVE_DESTINATION as GATE6_SOURCE_RELATIVE_PATH
from .feature_pipeline import CANONICAL_FEATURE_NAMES, SET_C_SCHEMA_VERSION, TARGET_NAME, TARGET_VERSION
from .transition_observer import mask_sha256


GENERATION_ID = "phase9_set_c_authoritative_generation_v1"
GENERATION_AUTHORIZATION = "Phase 9 Gate 7 single-run dataset generation authorization"
PACKAGE_ID = "authoritative_set_c"
PACKAGE_VERSION_FIXED = "1"
PACKAGE_RELATIVE_ROOT = Path("output/datasets")
INVENTORY_FILE_SHA256 = APPROVED_INVENTORY_FILE_SHA256
INVENTORY_PAYLOAD_SHA256 = APPROVED_INVENTORY_PAYLOAD_SHA256
REPLAY_RELATIVE_PATH = Path("output/phase9/phase9_set_c_population_adequacy_policy_replay_v3.aggregate.json")
REPLAY_FILE_SHA256 = "a6cd757ff78da2174625a773ba5a5df34b5dd077da41de4bf17bfe88b7bf8bbc"
REPLAY_PAYLOAD_SHA256 = "3891f54b041f98a0f8cdc0af32dd01485a269b54546c98e5cd5bd7b559d9ca96"
PRIMARY_MANIFEST_SHA256 = "d97e3cb7598dbc9c4b4cc53398c0d0daad0a5236f2d59e357db3cd8cf86a7d33"
RESERVE_MANIFEST_SHA256 = APPROVED_RESERVE_SHA256
FAMILY_ORDER_SHA256 = "6101a4729bfc6c4b4c49f48b6ef19b728620532014fb471adf4bb0e1cefd4046"
RUN_ORDER_SHA256 = "97706c938953e44c08539b758535aa5ebd7731f10abdbfe780f248afaea4c250"
EXPERIMENT_ID = GENERATION_ID
EVENT_ID = "simulated_case_study_no_observed_event"
GRID_ID = "lapu_lapu_epsg32651_3m_5489x6896_v1"
RESOURCE_LIMITS = {
    "safety_limit_timesteps": SAFETY_LIMIT_TIMESTEPS,
    "per_run_row_ceiling": PER_RUN_ROW_CEILING,
    "total_row_ceiling": TOTAL_ROW_CEILING,
    "batch_rows": BATCH_ROWS,
    "max_buffer_rows": MAX_BUFFER_ROWS,
    "max_buffer_bytes": MAX_BUFFER_BYTES,
    "retry_limit": RETRY_LIMIT,
    "launch_memory_floor_bytes": LAUNCH_MEMORY_FLOOR_BYTES,
    "runtime_memory_floor_bytes": RUNTIME_MEMORY_FLOOR_BYTES,
    "rss_ceiling_bytes": RSS_CEILING_BYTES,
    "free_disk_floor_bytes": FREE_DISK_FLOOR_BYTES,
    "run_runtime_limit_seconds": RUN_RUNTIME_LIMIT_SECONDS,
    "total_runtime_limit_seconds": EXPANSION_RUNTIME_LIMIT_SECONDS,
    "maximum_live_runs": 1,
}
GENERATION_SOURCE_FILES = (
    "config/default_experiment.yaml",
    "modules/automata_engine.py",
    "modules/ca_state.py",
    "modules/data_loader.py",
    "modules/feature_pipeline.py",
    "modules/model_free_teacher.py",
    "modules/set_c_collector.py",
    "modules/set_c_pilot.py",
    "modules/set_c_publication.py",
    "modules/set_c_site_inventory.py",
    "modules/set_c_population_adequacy.py",
    "modules/set_c_population_replay.py",
    "modules/set_c_generation.py",
    "modules/transition_observer.py",
    "modules/wind_convention.py",
    "orchestrator.py",
    "run_set_c_generation.py",
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def approved_paths() -> tuple[Path, Path, Path, Path]:
    code_root = Path(__file__).resolve().parents[1]
    return (
        code_root / INVENTORY_RELATIVE_PATH,
        code_root / GATE6_SOURCE_RELATIVE_PATH,
        code_root / REPLAY_RELATIVE_PATH,
        code_root / PACKAGE_RELATIVE_ROOT / PACKAGE_ID / f"v{PACKAGE_VERSION_FIXED}",
    )


def _verify_replay_report(path: Path) -> dict[str, Any]:
    if _file_sha256(path) != REPLAY_FILE_SHA256:
        raise PermissionError("Gate 6 replay whole-file hash mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    payload_hash = report.pop("payload_sha256", None)
    if payload_hash != REPLAY_PAYLOAD_SHA256 or _canonical_hash(report) != payload_hash:
        raise PermissionError("Gate 6 replay payload hash mismatch")
    if (
        report.get("replay_id") != "phase9_set_c_population_adequacy_policy_replay_v3"
        or report.get("policy") != "scenario_family_stable_cell_spatial_block_provenance_record; predictor_collisions_audit_only"
        or report.get("source_file_sha256") != GATE6_SOURCE_FILE_SHA256
        or report.get("source_payload_sha256") != GATE6_SOURCE_PAYLOAD_SHA256
        or report.get("source_revision") != GATE6_SOURCE_REVISION
        or report.get("population_run_count") != EXPECTED_RUN_COUNT
        or report.get("leakage", {}).get("component_count") != EXPECTED_FAMILY_COUNT
        or report.get("leakage", {}).get("qualifying_component_count") != EXPECTED_FAMILY_COUNT
        or report.get("leakage", {}).get("component_family_counts") != [1] * EXPECTED_FAMILY_COUNT
        or report.get("feature_collision_family_pair_count") != 97
        or report.get("adequacy", {}).get("status") != "pass"
        or set(report.get("adequacy", {}).get("gates", {})) != {
            "all_884_runs_complete_inactive_authoritative_and_count_matched",
            "at_least_23_qualifying_components",
            "every_qualifying_component_has_five_of_each_class",
            "unexposed_role_minima_material_coverage_and_concentration_feasible",
            "one_live_run_and_bounded_complete_batch_draining",
            "no_estimator_protected_input_row_split_or_model_output",
        }
        or not all(value is True for value in report.get("adequacy", {}).get("gates", {}).values())
        or report.get("leakage", {}).get("role_identities_exposed") is not False
        or report.get("integrity", {}).get("sole_persistent_output_is_aggregate_json") is not True
    ):
        raise PermissionError("Gate 6 replay is not the approved passing aggregate")
    if any(report.get("integrity", {}).get(key) is not False for key in (
        "row_level_payload_published", "split_roles_assigned", "estimator_accessed",
        "protected_spatial_reference_accessed",
    )):
        raise PermissionError("Gate 6 replay integrity flags are invalid")
    return report


def _verify_gate6_source(path: Path, specs: Sequence[object]) -> dict[str, Any]:
    if _file_sha256(path) != GATE6_SOURCE_FILE_SHA256:
        raise PermissionError("Gate 6 source report whole-file hash mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    payload_hash = report.pop("payload_sha256", None)
    if payload_hash != GATE6_SOURCE_PAYLOAD_SHA256 or _canonical_hash(report) != payload_hash:
        raise PermissionError("Gate 6 source report payload hash mismatch")
    contract = report.get("population_contract", {})
    runs = report.get("aggregate", {}).get("runs", [])
    run_ids = [spec.run_id for spec in specs]
    if (
        report.get("status") != "complete"
        or report.get("provenance", {}).get("source_revision") != GATE6_SOURCE_REVISION
        or contract.get("family_count") != EXPECTED_FAMILY_COUNT
        or contract.get("run_count") != EXPECTED_RUN_COUNT
        or contract.get("run_order_sha256") != canonical_metadata_hash(run_ids)
        or contract.get("block_size") != list(BLOCK_SIZE)
        or contract.get("block_origin") != list(BLOCK_ORIGIN)
        or contract.get("split_roles_assigned") is not False
        or [run.get("run_id") for run in runs] != run_ids
    ):
        raise PermissionError("Gate 6 source family or run order differs from D-015")
    return report


def _source_context(
    config: Mapping[str, object],
    config_path: Path,
    environment: Mapping[str, object],
    generation_command: Sequence[str],
) -> dict[str, Any]:
    code_root = Path(__file__).resolve().parents[1]
    revision, dirty = _current_git_state()
    if dirty:
        raise PermissionError("Authoritative generation requires a clean Git revision")
    source_hashes = {relative: _file_sha256(code_root / relative) for relative in GENERATION_SOURCE_FILES}
    raster_dir = Path(config["environment"]["raster_dir"])
    from .set_c_site_inventory import EXPECTED_RASTER_SHA256
    raster_hashes = {name: _file_sha256(raster_dir / name) for name in EXPECTED_RASTER_SHA256}
    if raster_hashes != EXPECTED_RASTER_SHA256:
        raise PermissionError("Approved environment raster hashes differ")
    input_hashes = {
        **raster_hashes,
        "simulation_valid_mask": mask_sha256(environment["burnable_mask"] & ~environment["nodata_mask"]),
        "mapped_building_mask": mask_sha256(environment["building_presence"] > 0),
    }
    if input_hashes["simulation_valid_mask"] != EXPECTED_SIMULATION_VALID_SHA256 or input_hashes["mapped_building_mask"] != EXPECTED_MAPPED_BUILDING_SHA256:
        raise PermissionError("Approved domain-mask hashes differ")
    packages = {}
    for name in ("numpy", "scipy", "rasterio", "PyYAML"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    return {
        "source_revision": revision,
        "source_dirty": False,
        "source_hashes": source_hashes,
        "configuration_file_sha256": _file_sha256(config_path),
        "configuration_sha256": _canonical_hash(config),
        "generation_command": list(generation_command),
        "input_hashes": input_hashes,
        "environment_sha256": _canonical_hash(input_hashes),
        "environment_record": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
    }


def _package_identity(
    config: Mapping[str, object],
    source_context: Mapping[str, object],
    families: Sequence[object],
    specs: Sequence[object],
    authorization_reference: str,
) -> dict[str, Any]:
    run_configuration_hashes = {
        spec.run_id: canonical_metadata_hash(_run_config(config, spec)) for spec in specs
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "source_revision": source_context["source_revision"],
        "source_dirty": source_context["source_dirty"],
        "source_file_sha256": source_context["source_hashes"],
        "environment_record": source_context["environment_record"],
        "generation_command": source_context["generation_command"],
        "configuration_sha256": source_context["configuration_sha256"],
        "configuration_file_sha256": source_context["configuration_file_sha256"],
        "run_configuration_sha256_by_run_id": run_configuration_hashes,
        "approved_run_order": [spec.run_id for spec in specs],
        "environment_id": GRID_ID,
        "environment_sha256": source_context["environment_sha256"],
        "rng_algorithm": "PCG64",
        "input_hashes": source_context["input_hashes"],
        "simulation_valid_mask_sha256": source_context["input_hashes"]["simulation_valid_mask"],
        "mapped_building_mask_sha256": source_context["input_hashes"]["mapped_building_mask"],
        "teacher_id": MODEL_FREE_SOURCE,
        "teacher_version": "model_free_teacher_run.v1",
        "feature_schema_sha256": _canonical_hash({"version": SET_C_SCHEMA_VERSION, "features": CANONICAL_FEATURE_NAMES}),
        "target_schema_sha256": _canonical_hash({"name": TARGET_NAME, "version": TARGET_VERSION, "positive_label": 1}),
        "authorization_reference": authorization_reference,
        "approved_inventory_file_sha256": INVENTORY_FILE_SHA256,
        "approved_inventory_payload_sha256": INVENTORY_PAYLOAD_SHA256,
        "approved_primary_manifest_sha256": PRIMARY_MANIFEST_SHA256,
        "approved_reserve_manifest_sha256": RESERVE_MANIFEST_SHA256,
        "family_order_sha256": FAMILY_ORDER_SHA256,
        "run_order_sha256": RUN_ORDER_SHA256,
        "gate6_replay_file_sha256": REPLAY_FILE_SHA256,
        "gate6_replay_payload_sha256": REPLAY_PAYLOAD_SHA256,
        "gate6_source_revision": GATE6_SOURCE_REVISION,
        "resource_limits": RESOURCE_LIMITS,
    }


def _publish_run_plan(
    session: SetCPublicationSession,
    specs: Sequence[object],
    executor: Callable[[object, Callable[[object], object]], object],
    *,
    resource_probe: Callable[[Path], ResourceSnapshot],
    total_started: float,
    synthetic_test: bool = False,
) -> Path:
    if not specs:
        raise ValueError("The approved run plan must not be empty")
    rows = 0
    for spec in specs:
        result = executor(spec, session.stage_batch)
        if synthetic_test:
            collected_run = getattr(result, "collected_run", None)
            if collected_run is None:
                raise PermissionError("Synthetic executor must return a sealed run record")
            session.seal_run(collected_run)
        if (
            getattr(result, "authoritative", False) is not True
            or getattr(result, "completeness_status", None) != "complete"
            or getattr(result, "termination_reason", None) != "inactive"
            or getattr(result, "expected_transition_count", -1) != getattr(result, "captured_transition_count", -2)
            or getattr(result, "unexpected_outputs", ())
            or getattr(result, "rows", PER_RUN_ROW_CEILING + 1) > PER_RUN_ROW_CEILING
            or getattr(result, "pending_rows", 1) != 0
            or getattr(result, "pending_bytes", 1) != 0
            or getattr(result, "max_live_full_grid_observations", 2) > 1
        ):
            raise PermissionError("A run failed its completeness, output, or resource contract")
        rows += int(result.rows)
        if rows > TOTAL_ROW_CEILING:
            raise PermissionError("Total package row ceiling exceeded")
        if time.monotonic() - total_started > EXPANSION_RUNTIME_LIMIT_SECONDS:
            raise TimeoutError("Total generation runtime limit exceeded")
        _check_resources(resource_probe(session.final_path), launch=False)
    if len(specs) != EXPECTED_RUN_COUNT and not synthetic_test:
        raise PermissionError("Production generation requires all 884 approved runs")
    _check_resources(resource_probe(session.final_path), launch=False)
    return session.publish(expected_run_count=len(specs))


def preflight_generation(
    *,
    config_path: Path,
    config: Mapping[str, object],
    approved_revision: str,
    resource_probe: Callable[[Path], ResourceSnapshot] = capture_resource_snapshot,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    expected_config = (Path(__file__).resolve().parents[1] / "config" / "default_experiment.yaml").resolve()
    if Path(config_path).resolve() != expected_config:
        raise PermissionError("Only the checked-in default experiment configuration is allowed")
    revision, dirty = _current_git_state()
    if dirty or revision != approved_revision:
        raise PermissionError("Production generation source revision is dirty or differs from its approval")
    inventory_path, source_path, replay_path, final_path = approved_paths()
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable package: {final_path}")
    if _file_sha256(inventory_path) != INVENTORY_FILE_SHA256:
        raise PermissionError("Approved Gate 4 inventory hash mismatch")
    if _file_sha256(source_path) != GATE6_SOURCE_FILE_SHA256:
        raise PermissionError("Approved Gate 6 source report hash mismatch")
    _verify_replay_report(replay_path)
    validate_approved_environment_config(config)
    publisher = config.get("dataset_generation", {}).get("set_c_publisher", {})
    if (
        publisher.get("package_id") != PACKAGE_ID
        or publisher.get("package_version") != PACKAGE_VERSION_FIXED
        or publisher.get("destination_pattern") != DESTINATION_PATTERN
        or publisher.get("enabled") is not False
    ):
        raise PermissionError("Checked-in publisher identity or disabled default changed")
    families = load_approved_families(inventory_path)
    specs = build_run_matrix(families)
    if len(families) != EXPECTED_FAMILY_COUNT or len(specs) != EXPECTED_RUN_COUNT:
        raise PermissionError("Approved Gate 4 family or D-015 run count mismatch")
    if (
        _canonical_hash([asdict(family) for family in families]) != FAMILY_ORDER_SHA256
        or canonical_metadata_hash([spec.run_id for spec in specs]) != RUN_ORDER_SHA256
    ):
        raise PermissionError("Approved family identity or run ordering hash mismatch")
    _verify_gate6_source(source_path, specs)
    _check_resources(resource_probe(final_path), launch=True)
    return families, specs


def run_synthetic_generation(
    *,
    config: Mapping[str, object],
    package_identity: Mapping[str, object],
    specs: Sequence[object],
    destination_root: Path,
    executor: Callable[[object, Callable[[object], object]], object],
    resource_probe: Callable[[Path], ResourceSnapshot],
) -> Path:
    """Exercise staging and promotion only under the system temporary directory."""
    root = Path(destination_root).resolve()
    try:
        root.relative_to(Path(tempfile.gettempdir()).resolve())
    except ValueError as exc:
        raise PermissionError("Synthetic generation destination must be under the system temp directory") from exc
    publisher_config = deepcopy(dict(config))
    publisher_config["dataset_generation"]["set_c_publisher"].update({
        "enabled": True,
        "mode": AUTHORITATIVE_VERSIONED_MODE,
        "authorization_reference": package_identity["authorization_reference"],
        "package_id": "synthetic_set_c",
        "package_version": PACKAGE_VERSION,
        "destination_pattern": DESTINATION_PATTERN,
    })
    with SetCPublicationSession(publisher_config, package_identity, destination_root=root) as session:
        return _publish_run_plan(
            session,
            specs,
            executor,
            resource_probe=resource_probe,
            total_started=time.monotonic(),
            synthetic_test=True,
        )


def run_authorized_generation(
    *,
    config: Mapping[str, object],
    config_path: Path,
    environment: Mapping[str, object],
    authorization_reference: str,
    approved_revision: str,
    generation_command: Sequence[str],
    production_authorized: bool = False,
    resource_probe: Callable[[Path], ResourceSnapshot] = capture_resource_snapshot,
) -> dict[str, object]:
    if production_authorized is not True:
        raise PermissionError("Production Set C publication requires separate Gate 7 authorization")
    if authorization_reference != GENERATION_AUTHORIZATION:
        raise PermissionError("Gate 7 authorization reference differs from the approved contract")
    expected_config = (Path(__file__).resolve().parents[1] / "config" / "default_experiment.yaml").resolve()
    if Path(config_path).resolve() != expected_config:
        raise PermissionError("Only the checked-in default experiment configuration is allowed")
    code_root = Path(__file__).resolve().parents[1]
    config = dict(config)
    families, specs = preflight_generation(
        config_path=config_path,
        config=config,
        approved_revision=approved_revision,
        resource_probe=resource_probe,
    )
    _inventory_path, _source_path, _replay_path, final_path = approved_paths()
    revision, dirty = _current_git_state()
    if dirty or revision != approved_revision:
        raise PermissionError("Production generation source revision changed during preflight")
    source_context = _source_context(config, config_path, environment, generation_command)
    if source_context["source_revision"] != approved_revision:
        raise PermissionError("Source context revision changed during preflight")
    _validate_environment(environment, synthetic=False)
    publisher_config = deepcopy(config)
    publisher_config["dataset_generation"]["set_c_publisher"].update({
        "enabled": True,
        "mode": AUTHORITATIVE_VERSIONED_MODE,
        "authorization_reference": authorization_reference,
        "package_id": PACKAGE_ID,
        "package_version": PACKAGE_VERSION_FIXED,
        "destination_pattern": DESTINATION_PATTERN,
    })
    identity = _package_identity(config, source_context, families, specs, authorization_reference)
    if set(identity["run_configuration_sha256_by_run_id"]) != {spec.run_id for spec in specs}:
        raise PermissionError("Per-run configuration hash map is not complete")
    session_root = final_path.parent.parent
    with SetCPublicationSession(publisher_config, identity, destination_root=session_root) as session:
        executor = build_model_free_executor(
            environment,
            config,
            source_context,
            destination=final_path,
            resource_probe=resource_probe,
            on_run_sealed=lambda _spec, sealed: session.seal_run(sealed),
            expected_output_path=session.staging_path,
            experiment_id=EXPERIMENT_ID,
            authorization_reference=authorization_reference,
        )
        package = _publish_run_plan(
            session,
            specs,
            executor,
            resource_probe=resource_probe,
            total_started=time.monotonic(),
        )
    return {
        "path": package,
        "package_id": PACKAGE_ID,
        "package_version": PACKAGE_VERSION_FIXED,
        "family_manifest_sha256": identity["family_order_sha256"],
        "run_order_sha256": identity["run_order_sha256"],
        "source_revision": revision,
        "rows": sum(
            int(entry["content"]["row_count"])
            for entry in json.loads((package / "manifest.json").read_text(encoding="utf-8"))["runs"]
        ),
    }
