"""Disabled aggregate-only Phase 9 site-inventory preflight."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
from scipy import ndimage

from .set_c_pilot import (
    FAMILIES,
    ResourceSnapshot,
    _validate_environment as validate_phase8_environment,
    capture_resource_snapshot,
    publish_aggregate_report,
)
from .transition_observer import canonical_metadata_hash


INVENTORY_ID = "phase9_set_c_site_inventory_v1"
AGGREGATE_SCHEMA_VERSION = "phase9_set_c_site_inventory_aggregate.v1"
AUTHORIZATION_REFERENCE = "Phase 9 Task 9.2 source and synthetic-test approval only"
APPROVED_RELATIVE_DESTINATION = Path(
    "output/phase9/phase9_set_c_site_inventory_v1.aggregate.json"
)
EXPECTED_GRID_SHAPE = (5489, 6896)
EXPECTED_GRID_CRS = "EPSG:32651"
EXPECTED_GRID_RESOLUTION_M = 3
EXPECTED_GRID_IDENTITY_SHA256 = (
    "26e9da1104da2caf15039b5f9fbcf719fea506adcee95f9c2da247e8a9e2bc46"
)
EXPECTED_AUTHORITATIVE_DOMAIN_SHA256 = (
    "f9bb9dbfe00c61a5b08d228bcbec4b4313853a47f4ca2ea82b7a4359d81814a4"
)
EXPECTED_RASTER_SHA256 = {
    "stack_buildings.tif": "da53c3e63b661471b3366fc68249793c6a5a84e0d63bcc96492f60baaf4b83c7",
    "stack_materials.tif": "21f5c98c7936b4be6e368e7c8c0472a3c045b9c85b1abcb0a0e3fc8c1be0336a",
    "stack_proximity.tif": "1b12936cd8bc78d1e638fd31ddf1241bd753cb040cc9db556c623b8193de14b5",
    "stack_slope_final.tif": "dd61a0da016cd9c0d21d957ffedebd7ce871ef94f9084d5d4feb90e7e32b8482",
}
EXPECTED_ENVIRONMENT_FILES = {
    "buildings_file": "stack_buildings.tif",
    "materials_file": "stack_materials.tif",
    "proximity_file": "stack_proximity.tif",
    "slope_file": "stack_slope_final.tif",
}
BLOCK_SIZE = (128, 128)
BLOCK_ORIGIN = (0, 0)
COMPONENT_SIZE_MIN = 256
COMPONENT_SIZE_MAX = 6_000
MATERIAL_CLASSES = (1, 2, 3, 4, 5)
PRIMARY_PER_MATERIAL = 4
RESERVE_PER_MATERIAL = 4
TOTAL_PER_MATERIAL = PRIMARY_PER_MATERIAL + RESERVE_PER_MATERIAL
LAUNCH_MEMORY_FLOOR_BYTES = 6_442_450_944
DISK_FLOOR_BYTES = 2_147_483_648
SOURCE_RELATIVE_PATHS = (
    "modules/set_c_site_inventory.py",
    "modules/set_c_pilot.py",
    "modules/transition_observer.py",
    "modules/data_loader.py",
    "run_set_c_site_inventory.py",
    "config/default_experiment.yaml",
)
_FORBIDDEN_PATH_TOKENS = (
    "ground_truth",
    "teacher",
    "model",
    "split",
    "training",
    "calibration",
    "manuscript",
)


class SiteInventoryContractError(RuntimeError):
    """Raised when the frozen site-inventory contract is violated."""


def ignition_set_sha256(canonical_coordinates: tuple[tuple[int, int], ...]) -> str:
    """Hash the already-canonical single-site coordinates without teacher access."""
    return canonical_metadata_hash(canonical_coordinates)


@dataclass(frozen=True, slots=True)
class ExistingSite:
    family_id: str
    row: int
    col: int
    material_class: int
    component_ordinal: int
    component_size: int
    ignition_set_sha256: str


_EXISTING_MATERIALS = (1, 3, 2, 5, 3, 5)
_EXISTING_COMPONENT_ORDINALS = (322, 1052, 4974, 160, 4146, 6201)
EXISTING_SITES = tuple(
    ExistingSite(
        family_id=family.family_id,
        row=family.ignition_coordinates[0][0],
        col=family.ignition_coordinates[0][1],
        material_class=material,
        component_ordinal=ordinal,
        component_size=family.component_size,
        ignition_set_sha256=family.ignition_set_sha256,
    )
    for family, material, ordinal in zip(
        FAMILIES,
        _EXISTING_MATERIALS,
        _EXISTING_COMPONENT_ORDINALS,
        strict=True,
    )
)


def approved_destination() -> Path:
    return Path(__file__).resolve().parents[1] / APPROVED_RELATIVE_DESTINATION


def activate_approved_inventory_config(config: Mapping[str, object]) -> dict[str, Any]:
    """Enable only an in-memory copy of the checked-in disabled contract."""
    active = deepcopy(dict(config))
    inventory = active.get("phase9_set_c_site_inventory")
    if not isinstance(inventory, dict):
        raise SiteInventoryContractError("Phase 9 site-inventory configuration is required")
    if inventory.get("enabled") is not False:
        raise SiteInventoryContractError("The checked-in site inventory must remain disabled")
    _require_exact_contract(active, enabled=False)
    inventory["enabled"] = True
    return active


def _require_exact_contract(config: Mapping[str, object], *, enabled: bool = True) -> None:
    inventory = config.get("phase9_set_c_site_inventory")
    if not isinstance(inventory, Mapping) or inventory.get("enabled") is not enabled:
        if enabled:
            raise PermissionError("Phase 9 site inventory is disabled by default")
        raise SiteInventoryContractError("Checked-in Phase 9 inventory gate changed")
    exact = {
        "inventory_id": INVENTORY_ID,
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "authorization_reference": AUTHORIZATION_REFERENCE,
        "aggregate_destination": str(APPROVED_RELATIVE_DESTINATION).replace("\\", "/"),
        "grid_shape": list(EXPECTED_GRID_SHAPE),
        "grid_crs": EXPECTED_GRID_CRS,
        "grid_resolution_m": EXPECTED_GRID_RESOLUTION_M,
        "grid_identity_sha256": EXPECTED_GRID_IDENTITY_SHA256,
        "authoritative_domain_sha256": EXPECTED_AUTHORITATIVE_DOMAIN_SHA256,
        "raster_sha256": EXPECTED_RASTER_SHA256,
        "component_connectivity": 8,
        "component_size_min": COMPONENT_SIZE_MIN,
        "component_size_max": COMPONENT_SIZE_MAX,
        "block_size": list(BLOCK_SIZE),
        "block_origin": list(BLOCK_ORIGIN),
        "primary_candidates_per_material": PRIMARY_PER_MATERIAL,
        "reserve_candidates_per_material": RESERVE_PER_MATERIAL,
        "material_classes": list(MATERIAL_CLASSES),
        "launch_memory_floor_bytes": LAUNCH_MEMORY_FLOOR_BYTES,
        "disk_floor_bytes": DISK_FLOOR_BYTES,
    }
    for name, expected in exact.items():
        if inventory.get(name) != expected:
            raise SiteInventoryContractError(
                f"phase9_set_c_site_inventory.{name} must equal {expected!r}"
            )
    environment = config.get("environment")
    if not isinstance(environment, Mapping):
        raise SiteInventoryContractError("Environment configuration is required")
    for key, expected in EXPECTED_ENVIRONMENT_FILES.items():
        if environment.get(key) != expected:
            raise SiteInventoryContractError(f"environment.{key} changed")
    if config.get("dataset_generation", {}).get("artifact_write_enabled") is not False:
        raise PermissionError("Dataset artifact writes must remain disabled")
    if config.get("dataset_generation", {}).get("set_c_publisher", {}).get("enabled") is not False:
        raise PermissionError("Set C row publication must remain disabled")
    if config.get("ml_model", {}).get("artifact_write_enabled") is not False:
        raise PermissionError("Model artifact writes must remain disabled")


def label_authoritative_components(domain: np.ndarray) -> tuple[np.ndarray, int]:
    """Label a 2D domain with constant-boundary 8-connectivity."""
    mask = np.asarray(domain)
    if mask.ndim != 2 or mask.dtype != np.dtype(bool):
        raise SiteInventoryContractError("Authoritative domain must be a 2D boolean mask")
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return labels, int(count)


def _component_cells(
    labels: np.ndarray,
    component_slices: list[tuple[slice, ...] | None],
    ordinal: int,
) -> tuple[np.ndarray, np.ndarray]:
    component_slice = component_slices[ordinal - 1]
    if component_slice is None:
        raise SiteInventoryContractError(f"Missing component slice for ordinal {ordinal}")
    local = np.argwhere(labels[component_slice] == ordinal)
    rows = local[:, 0] + int(component_slice[0].start)
    cols = local[:, 1] + int(component_slice[1].start)
    return rows, cols


def _block_footprint(rows: np.ndarray, cols: np.ndarray) -> tuple[set[tuple[int, int]], str]:
    block_pairs = np.unique(
        np.column_stack((rows // BLOCK_SIZE[0], cols // BLOCK_SIZE[1])), axis=0
    )
    blocks = {(int(row), int(col)) for row, col in block_pairs}
    digest = canonical_metadata_hash(
        {
            "block_size": list(BLOCK_SIZE),
            "block_origin": list(BLOCK_ORIGIN),
            "blocks": sorted([list(block) for block in blocks]),
        }
    )
    return blocks, digest


def _choose_ignition(
    domain: np.ndarray,
    materials: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    material_class: int,
) -> tuple[int, int]:
    requested = materials[rows, cols] == material_class
    candidate_rows = rows[requested]
    candidate_cols = cols[requested]
    if candidate_rows.size == 0:
        raise SiteInventoryContractError("Component does not contain the requested material")
    neighbour_counts = np.zeros(candidate_rows.size, dtype=np.int8)
    height, width = domain.shape
    for row_delta in (-1, 0, 1):
        for col_delta in (-1, 0, 1):
            if row_delta == 0 and col_delta == 0:
                continue
            adjacent_rows = candidate_rows + row_delta
            adjacent_cols = candidate_cols + col_delta
            inside = (
                (adjacent_rows >= 0)
                & (adjacent_rows < height)
                & (adjacent_cols >= 0)
                & (adjacent_cols < width)
            )
            neighbour_counts[inside] += domain[
                adjacent_rows[inside], adjacent_cols[inside]
            ]
    centroid_row = float(rows.mean())
    centroid_col = float(cols.mean())
    distance_sq = (candidate_rows - centroid_row) ** 2 + (
        candidate_cols - centroid_col
    ) ** 2
    order = np.lexsort(
        (candidate_cols, candidate_rows, distance_sq, -neighbour_counts)
    )
    selected = int(order[0])
    return int(candidate_rows[selected]), int(candidate_cols[selected])


def _revalidate_existing_sites(
    labels: np.ndarray,
    component_sizes: np.ndarray,
    component_slices: list[tuple[slice, ...] | None],
    domain: np.ndarray,
    valid: np.ndarray,
    mapped: np.ndarray,
    materials: np.ndarray,
    sites: tuple[ExistingSite, ...],
) -> tuple[list[dict[str, object]], set[int], set[tuple[int, int]]]:
    records: list[dict[str, object]] = []
    excluded_components: set[int] = set()
    occupied_blocks: set[tuple[int, int]] = set()
    for site in sites:
        if not (0 <= site.row < labels.shape[0] and 0 <= site.col < labels.shape[1]):
            raise SiteInventoryContractError(f"Existing site {site.family_id} is outside the grid")
        ordinal = int(labels[site.row, site.col])
        if (
            not domain[site.row, site.col]
            or not valid[site.row, site.col]
            or not mapped[site.row, site.col]
            or int(materials[site.row, site.col]) != site.material_class
            or ordinal != site.component_ordinal
            or int(component_sizes[ordinal]) != site.component_size
            or ignition_set_sha256(((site.row, site.col),)) != site.ignition_set_sha256
        ):
            raise SiteInventoryContractError(
                f"Existing site {site.family_id} failed exact revalidation"
            )
        rows, cols = _component_cells(labels, component_slices, ordinal)
        blocks, block_hash = _block_footprint(rows, cols)
        excluded_components.add(ordinal)
        occupied_blocks.update(blocks)
        records.append(
            {
                "existing_family_id": site.family_id,
                "component_ordinal": ordinal,
                "component_size": site.component_size,
                "ignition_row": site.row,
                "ignition_col": site.col,
                "ignition_set_sha256": site.ignition_set_sha256,
                "material_class": site.material_class,
                "mapped_building": True,
                "simulation_valid": True,
                "block_footprint_sha256": block_hash,
                "occupied_block_count": len(blocks),
                "contract_status": "pass",
            }
        )
    return records, excluded_components, occupied_blocks


def _select_candidates(
    labels: np.ndarray,
    component_count: int,
    domain: np.ndarray,
    valid: np.ndarray,
    mapped: np.ndarray,
    materials: np.ndarray,
    existing_sites: tuple[ExistingSite, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    slices = list(ndimage.find_objects(labels, max_label=component_count))
    existing, excluded, occupied_blocks = _revalidate_existing_sites(
        labels, sizes, slices, domain, valid, mapped, materials, existing_sites
    )
    ordered = sorted(
        (
            ordinal
            for ordinal in range(1, component_count + 1)
            if COMPONENT_SIZE_MIN <= int(sizes[ordinal]) <= COMPONENT_SIZE_MAX
            and ordinal not in excluded
        ),
        key=lambda ordinal: (-int(sizes[ordinal]), ordinal),
    )
    candidates: list[dict[str, object]] = []
    for material_class in MATERIAL_CLASSES:
        selected = 0
        for ordinal in ordered:
            rows, cols = _component_cells(labels, slices, ordinal)
            blocks, block_hash = _block_footprint(rows, cols)
            if blocks & occupied_blocks or not np.any(
                materials[rows, cols] == material_class
            ):
                continue
            ignition_row, ignition_col = _choose_ignition(
                domain, materials, rows, cols, material_class
            )
            if not (
                valid[ignition_row, ignition_col]
                and mapped[ignition_row, ignition_col]
                and int(materials[ignition_row, ignition_col]) == material_class
            ):
                raise SiteInventoryContractError("Selected ignition violates domain contract")
            selected += 1
            candidates.append(
                {
                    "proposed_family_id": (
                        f"p9f_m{material_class}_r{selected}_c{ordinal}"
                    ),
                    "candidate_tier": (
                        "primary" if selected <= PRIMARY_PER_MATERIAL else "reserve"
                    ),
                    "material_class": material_class,
                    "rank": selected,
                    "component_ordinal": ordinal,
                    "component_size": int(sizes[ordinal]),
                    "ignition_row": ignition_row,
                    "ignition_col": ignition_col,
                    "ignition_set_sha256": ignition_set_sha256(
                        ((ignition_row, ignition_col),)
                    ),
                    "mapped_building": True,
                    "simulation_valid": True,
                    "block_footprint_sha256": block_hash,
                    "occupied_block_count": len(blocks),
                    "contract_status": "pass",
                }
            )
            occupied_blocks.update(blocks)
            if selected == TOTAL_PER_MATERIAL:
                break
        if selected != TOTAL_PER_MATERIAL:
            raise SiteInventoryContractError(
                f"Material class {material_class} has only {selected} valid candidates"
            )
    return existing, candidates


def _validate_source_context(
    source_context: Mapping[str, object], *, synthetic_test: bool
) -> None:
    input_hashes = source_context.get("input_hashes")
    if not isinstance(input_hashes, Mapping) or dict(input_hashes) != EXPECTED_RASTER_SHA256:
        raise SiteInventoryContractError("Environmental raster identities changed")
    source_hashes = source_context.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise SiteInventoryContractError("Source hashes are required")
    allowed_sources = set(SOURCE_RELATIVE_PATHS)
    if not synthetic_test and set(source_hashes) != allowed_sources:
        raise SiteInventoryContractError("Production source provenance is incomplete")
    for name, digest in source_hashes.items():
        normalized = str(name).replace("\\", "/")
        if normalized not in allowed_sources or not _is_sha256(digest):
            raise SiteInventoryContractError("Source provenance contains an unapproved path")
    command = source_context.get("generation_command")
    inspected_paths = [*source_hashes.keys(), *input_hashes.keys(), command]
    if any(
        token in str(value).lower()
        for value in inspected_paths
        for token in _FORBIDDEN_PATH_TOKENS
    ):
        raise PermissionError(
            "Ground-truth, teacher, model, split, or training paths are forbidden"
        )
    if not str(source_context.get("source_revision", "")).strip():
        raise SiteInventoryContractError("Source revision is required")
    if not isinstance(source_context.get("source_dirty"), bool):
        raise SiteInventoryContractError("Source dirty-state is required")
    if not synthetic_test and source_context["source_dirty"] is not False:
        raise SiteInventoryContractError("Production source must be clean")
    if not isinstance(source_context.get("environment_record"), Mapping):
        raise SiteInventoryContractError("Environment record is required")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _validate_resource(snapshot: ResourceSnapshot) -> None:
    if snapshot.available_memory_bytes < LAUNCH_MEMORY_FLOOR_BYTES:
        raise SiteInventoryContractError("Available memory is below the launch floor")
    if snapshot.free_disk_bytes < DISK_FLOOR_BYTES:
        raise SiteInventoryContractError("Free disk is below the launch floor")
    if snapshot.process_rss_bytes < 0:
        raise SiteInventoryContractError("Process RSS telemetry is invalid")


def _validate_aggregate_only(value: object) -> None:
    forbidden_keys = {
        "features",
        "predictors",
        "labels",
        "cell_inventory",
        "cell_inventories",
        "masks",
        "grids",
        "rasters",
        "teacher",
        "model",
        "split",
        "training",
        "calibration",
        "manuscript",
        "ignition_coordinates",
        "component_cells",
        "block_coordinates",
    }
    if isinstance(value, np.ndarray):
        raise SiteInventoryContractError("Aggregate report cannot contain arrays")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in forbidden_keys:
                raise SiteInventoryContractError(f"Aggregate report contains forbidden key: {key}")
            _validate_aggregate_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_aggregate_only(item)


def run_site_inventory(
    config: Mapping[str, object],
    *,
    environment_loader: Callable[[], Mapping[str, object]],
    destination: Path,
    source_context: Mapping[str, object],
    resource_probe: Callable[[Path], ResourceSnapshot] = capture_resource_snapshot,
    synthetic_test: bool = False,
    existing_sites: tuple[ExistingSite, ...] | None = None,
) -> dict[str, object]:
    """Validate, inventory, and atomically publish one aggregate-only report."""
    _require_exact_contract(config)
    destination = destination.resolve()
    if not synthetic_test and destination != approved_destination().resolve():
        raise PermissionError("Production inventory destination is not approved")
    if not synthetic_test and resource_probe is not capture_resource_snapshot:
        raise PermissionError("Production inventory requires approved resource telemetry")
    if not synthetic_test and existing_sites is not None:
        raise PermissionError("Production existing-site contract cannot be substituted")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite inventory report: {destination}")
    _validate_source_context(source_context, synthetic_test=synthetic_test)
    resource = resource_probe(destination)
    _validate_resource(resource)
    environment = environment_loader()
    materials = np.asarray(environment["material_class"])
    if not np.issubdtype(materials.dtype, np.integer) or not np.all(
        np.isin(materials, (0, *MATERIAL_CLASSES))
    ):
        raise SiteInventoryContractError(
            "Material classes must be integer values in the approved 0..5 domain"
        )
    identity = validate_phase8_environment(environment, synthetic=synthetic_test)
    if not synthetic_test and (
        identity["grid_identity_sha256"] != EXPECTED_GRID_IDENTITY_SHA256
        or identity["authoritative_domain_sha256"]
        != EXPECTED_AUTHORITATIVE_DOMAIN_SHA256
    ):
        raise SiteInventoryContractError("Production grid/domain identity changed")
    burnable = np.asarray(environment["burnable_mask"])
    nodata = np.asarray(environment["nodata_mask"])
    mapped = np.asarray(environment["building_presence"]) > 0
    valid = burnable & (~nodata)
    domain = valid & mapped
    labels, component_count = label_authoritative_components(domain)
    checked_existing, candidates = _select_candidates(
        labels,
        component_count,
        domain,
        valid,
        mapped,
        materials,
        EXISTING_SITES if existing_sites is None else existing_sites,
    )
    report = {
        "inventory_id": INVENTORY_ID,
        "aggregate_schema_version": AGGREGATE_SCHEMA_VERSION,
        "status": "complete",
        "completeness": "complete",
        "contract_status": "pass",
        "execution_mode": "synthetic_contract_test" if synthetic_test else "production",
        "provenance": {
            "authorization_reference": AUTHORIZATION_REFERENCE,
            "configuration_sha256": canonical_metadata_hash(config),
            "source_revision": source_context["source_revision"],
            "source_dirty": source_context["source_dirty"],
            "source_hashes": dict(source_context["source_hashes"]),
            "generation_command": source_context["generation_command"],
            "environment_record": dict(source_context["environment_record"]),
            "raster_sha256": dict(source_context["input_hashes"]),
            "grid_and_domain": identity,
            "resource_snapshot": {
                "available_memory_bytes": resource.available_memory_bytes,
                "process_rss_bytes": resource.process_rss_bytes,
                "free_disk_bytes": resource.free_disk_bytes,
                "captured_at_utc": resource.captured_at_utc,
            },
        },
        "selection_contract": {
            "component_connectivity": 8,
            "constant_non_wrapping_boundary": True,
            "component_size_min": COMPONENT_SIZE_MIN,
            "component_size_max": COMPONENT_SIZE_MAX,
            "block_size": list(BLOCK_SIZE),
            "block_origin": list(BLOCK_ORIGIN),
            "component_order": "descending_size_then_ascending_ordinal",
            "ignition_order": (
                "maximum_authoritative_moore_neighbours_then_minimum_centroid_"
                "distance_then_row_then_col"
            ),
            "primary_candidates_per_material": PRIMARY_PER_MATERIAL,
            "reserve_candidates_per_material": RESERVE_PER_MATERIAL,
        },
        "component_count": component_count,
        "existing_site_count": len(checked_existing),
        "existing_sites": checked_existing,
        "candidate_count": len(candidates),
        "candidate_counts_by_material": {
            str(material): sum(
                candidate["material_class"] == material for candidate in candidates
            )
            for material in MATERIAL_CLASSES
        },
        "candidates": candidates,
    }
    _validate_aggregate_only(report)
    path, payload_hash = publish_aggregate_report(report, destination)
    return {
        "status": "complete",
        "contract_status": "pass",
        "candidate_count": len(candidates),
        "path": str(path),
        "payload_sha256": payload_hash,
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_is_dirty(repo_root: Path) -> bool:
    """Ignore only the preserved visualization project, never other source changes."""
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root.as_posix()}",
            "-C",
            str(repo_root),
            "status",
            "--porcelain",
            "--",
            ".",
            ":(exclude)Thesis-try.qgz",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def build_source_context(
    config: Mapping[str, object], config_path: Path
) -> dict[str, object]:
    """Bind the inventory to its source and four approved environmental rasters."""
    code_dir = Path(__file__).resolve().parents[1]
    repo_root = code_dir.parents[1]
    approved_config = code_dir / "config" / "default_experiment.yaml"
    if config_path.resolve() != approved_config.resolve():
        raise PermissionError("Only the approved default experiment configuration is allowed")
    source_paths = {
        relative: code_dir / relative for relative in SOURCE_RELATIVE_PATHS[:-1]
    }
    source_paths[SOURCE_RELATIVE_PATHS[-1]] = config_path.resolve()
    if any(not path.is_file() for path in source_paths.values()):
        raise FileNotFoundError("Every site-inventory source/configuration file must exist")
    environment = config["environment"]
    raster_dir = Path(str(environment["raster_dir"]))
    raster_paths = {
        filename: raster_dir / filename for filename in EXPECTED_RASTER_SHA256
    }
    if any(
        token in path.name.lower()
        for path in raster_paths.values()
        for token in _FORBIDDEN_PATH_TOKENS
    ):
        raise PermissionError("Unapproved environmental input path")
    if any(not path.is_file() for path in raster_paths.values()):
        raise FileNotFoundError("Every approved environmental raster must exist")
    git_base = [
        "git",
        "-c",
        f"safe.directory={repo_root.as_posix()}",
        "-C",
        str(repo_root),
    ]
    revision = subprocess.run(
        [*git_base, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = _source_is_dirty(repo_root)
    return {
        "source_revision": revision,
        "source_dirty": dirty,
        "source_hashes": {
            relative: _sha256_file(path) for relative, path in source_paths.items()
        },
        "input_hashes": {
            filename: _sha256_file(path) for filename, path in raster_paths.items()
        },
        "generation_command": (
            ".venv\\Scripts\\python.exe -B Thesis_RF\\Code\\run_set_c_site_inventory.py "
            "--config Thesis_RF\\Code\\config\\default_experiment.yaml "
            "--execute-approved-inventory"
        ),
        "environment_record": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("numpy", "scipy", "rasterio", "PyYAML")
            },
        },
    }


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "APPROVED_RELATIVE_DESTINATION",
    "AUTHORIZATION_REFERENCE",
    "EXISTING_SITES",
    "ExistingSite",
    "INVENTORY_ID",
    "SiteInventoryContractError",
    "activate_approved_inventory_config",
    "approved_destination",
    "build_source_context",
    "label_authoritative_components",
    "run_site_inventory",
]
