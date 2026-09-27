from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml

import modules.set_c_site_inventory as inventory
import run_set_c_site_inventory as inventory_cli


def test_source_dirty_check_ignores_only_preserved_visualization(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "synthetic@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Synthetic Test"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "fixture"], check=True)

    (tmp_path / "Thesis-try.qgz").write_text("preserved", encoding="utf-8")
    assert inventory._source_is_dirty(tmp_path) is False

    (tmp_path / "unexpected.py").write_text("unexpected", encoding="utf-8")
    assert inventory._source_is_dirty(tmp_path) is True


def _raw_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "config" / "default_experiment.yaml"
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def _active_config() -> dict:
    return inventory.activate_approved_inventory_config(_raw_config())


def _source_context() -> dict:
    return {
        "source_revision": "synthetic-revision",
        "source_dirty": False,
        "source_hashes": {"modules/set_c_site_inventory.py": "a" * 64},
        "input_hashes": dict(inventory.EXPECTED_RASTER_SHA256),
        "generation_command": "synthetic-contract-test",
        "environment_record": {"python": "synthetic"},
    }


def _resource_probe(_destination: Path) -> inventory.ResourceSnapshot:
    return inventory.ResourceSnapshot(
        available_memory_bytes=inventory.LAUNCH_MEMORY_FLOOR_BYTES + 1,
        process_rss_bytes=1,
        free_disk_bytes=inventory.DISK_FLOOR_BYTES + 1,
        captured_at_utc="2026-09-27T00:00:00+08:00",
    )


def _synthetic_case(
    *, larger_late_component: bool = False
) -> tuple[dict, tuple[inventory.ExistingSite, ...]]:
    shape = (6 * 128, 8 * 128)
    mapped = np.zeros(shape, dtype=np.int8)
    materials = np.zeros(shape, dtype=np.int8)
    existing_materials = (1, 3, 2, 5, 3, 5)
    existing_points: list[tuple[int, int, int, str]] = []

    for block_index, material in enumerate(existing_materials):
        block_row, block_col = divmod(block_index, 8)
        row, col = block_row * 128 + 2, block_col * 128 + 2
        mapped[row : row + 16, col : col + 16] = 1
        materials[row : row + 16, col : col + 16] = material
        existing_points.append((row + 5, col + 5, material, f"existing-{block_index + 1}"))

    # Valid-size decoy in an existing block: component-disjoint but block-overlapping.
    mapped[64:80, 64:80] = 1
    materials[64:80, 64:80] = 1

    for candidate_index in range(40):
        block_index = candidate_index + 8
        block_row, block_col = divmod(block_index, 8)
        row, col = block_row * 128 + 2, block_col * 128 + 2
        material = candidate_index // 8 + 1
        height = 17 if larger_late_component and candidate_index == 7 else 16
        mapped[row : row + height, col : col + 16] = 1
        materials[row : row + height, col : col + 16] = material

    environment = {
        "slope_risk": np.zeros(shape, dtype=np.float32),
        "proximity_risk": np.zeros(shape, dtype=np.float32),
        "building_presence": mapped,
        "material_class": materials,
        "burnable_mask": np.ones(shape, dtype=bool),
        "nodata_mask": np.zeros(shape, dtype=bool),
        "grid_shape": shape,
        "transform": (3.0, 0.0, 100.0, 0.0, -3.0, 200.0, 0.0, 0.0, 1.0),
        "crs": "EPSG:32651",
    }
    labels, count = inventory.label_authoritative_components(mapped.astype(bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sites = tuple(
        inventory.ExistingSite(
            family_id=family_id,
            row=row,
            col=col,
            material_class=material,
            component_ordinal=int(labels[row, col]),
            component_size=int(sizes[labels[row, col]]),
            ignition_set_sha256=inventory.ignition_set_sha256(((row, col),)),
        )
        for row, col, material, family_id in existing_points
    )
    return environment, sites


def _run(tmp_path: Path) -> tuple[dict, dict, dict, tuple[inventory.ExistingSite, ...]]:
    environment, sites = _synthetic_case()
    destination = tmp_path / "inventory.aggregate.json"
    result = inventory.run_site_inventory(
        _active_config(),
        environment_loader=lambda: environment,
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        synthetic_test=True,
        existing_sites=sites,
    )
    report = json.loads(destination.read_text(encoding="utf-8"))
    return result, report, environment, sites


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value), set())
    return set()


def test_checked_in_inventory_is_disabled_before_environment_or_resource_access(tmp_path):
    accessed = False

    def forbidden() -> dict:
        nonlocal accessed
        accessed = True
        raise AssertionError("environment accessed")

    with pytest.raises(PermissionError, match="disabled"):
        inventory.run_site_inventory(
            _raw_config(),
            environment_loader=forbidden,
            destination=tmp_path / "inventory.aggregate.json",
            source_context=_source_context(),
            resource_probe=lambda _path: pytest.fail("resource probe reached"),
            synthetic_test=True,
            existing_sites=(),
        )
    assert not accessed
    assert list(tmp_path.iterdir()) == []


def test_cli_requires_flag_before_configuration_access(monkeypatch):
    monkeypatch.setattr(
        inventory_cli,
        "load_config",
        lambda *_args: pytest.fail("disabled CLI accessed configuration"),
    )
    with pytest.raises(SystemExit):
        inventory_cli.main([])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grid_shape", [1, 1]),
        ("grid_crs", "EPSG:4326"),
        ("grid_resolution_m", 30),
        ("grid_identity_sha256", "0" * 64),
        ("authoritative_domain_sha256", "0" * 64),
        ("raster_sha256", {}),
    ],
)
def test_exact_grid_domain_and_raster_contract_is_bound(field, value):
    config = _active_config()
    config["phase9_set_c_site_inventory"][field] = value
    with pytest.raises(inventory.SiteInventoryContractError):
        inventory._require_exact_contract(config)


def test_raster_hash_drift_fails_before_environment_access(tmp_path):
    context = _source_context()
    context["input_hashes"]["stack_buildings.tif"] = "0" * 64
    with pytest.raises(inventory.SiteInventoryContractError, match="identities"):
        inventory.run_site_inventory(
            _active_config(),
            environment_loader=lambda: pytest.fail("environment accessed"),
            destination=tmp_path / "inventory.aggregate.json",
            source_context=context,
            resource_probe=_resource_probe,
            synthetic_test=True,
            existing_sites=(),
        )
    assert list(tmp_path.iterdir()) == []


def test_eight_connectivity_is_diagonal_and_non_wrapping():
    domain = np.zeros((4, 4), dtype=bool)
    domain[0, 0] = True
    domain[1, 1] = True
    domain[3, 3] = True
    labels, count = inventory.label_authoritative_components(domain)
    assert count == 2
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 0] != labels[3, 3]


def test_deterministic_selection_revalidates_sites_and_produces_four_plus_four(tmp_path):
    result, report, _environment, _sites = _run(tmp_path)
    assert result["candidate_count"] == 40
    assert report["candidate_counts_by_material"] == {str(i): 8 for i in range(1, 6)}
    for material in range(1, 6):
        candidates = [
            candidate
            for candidate in report["candidates"]
            if candidate["material_class"] == material
        ]
        assert [candidate["rank"] for candidate in candidates] == list(range(1, 9))
        assert [candidate["candidate_tier"] for candidate in candidates] == [
            "primary"
        ] * 4 + ["reserve"] * 4
        assert [candidate["component_size"] for candidate in candidates] == [256] * 8
        assert [candidate["component_ordinal"] for candidate in candidates] == sorted(
            candidate["component_ordinal"] for candidate in candidates
        )
    assert report["existing_site_count"] == 6
    assert all(site["contract_status"] == "pass" for site in report["existing_sites"])
    assert all(candidate["component_ordinal"] != 7 for candidate in report["candidates"])
    assert len(
        {candidate["block_footprint_sha256"] for candidate in report["candidates"]}
    ) == 40


def test_ignition_tie_breaks_by_neighbours_centroid_row_and_col(tmp_path):
    _result, report, _environment, _sites = _run(tmp_path)
    first = report["candidates"][0]
    assert (first["ignition_row"], first["ignition_col"]) == (137, 9)
    assert first["proposed_family_id"] == (
        f"p9f_m1_r1_c{first['component_ordinal']}"
    )


def test_component_ranking_prefers_size_before_ordinal(tmp_path):
    environment, sites = _synthetic_case(larger_late_component=True)
    destination = tmp_path / "inventory.aggregate.json"
    inventory.run_site_inventory(
        _active_config(),
        environment_loader=lambda: environment,
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        synthetic_test=True,
        existing_sites=sites,
    )
    candidates = json.loads(destination.read_text(encoding="utf-8"))["candidates"]
    material_one = [item for item in candidates if item["material_class"] == 1]
    assert material_one[0]["component_size"] == 272
    assert material_one[1]["component_size"] == 256
    assert material_one[0]["component_ordinal"] > material_one[1]["component_ordinal"]


def test_mixed_material_component_is_reserved_by_first_material_class(tmp_path):
    environment, sites = _synthetic_case()
    environment["material_class"][130:146, 10:18] = 2
    labels, _count = inventory.label_authoritative_components(
        environment["building_presence"].astype(bool)
    )
    mixed_ordinal = int(labels[130, 2])
    destination = tmp_path / "inventory.aggregate.json"
    inventory.run_site_inventory(
        _active_config(),
        environment_loader=lambda: environment,
        destination=destination,
        source_context=_source_context(),
        resource_probe=_resource_probe,
        synthetic_test=True,
        existing_sites=sites,
    )
    candidates = json.loads(destination.read_text(encoding="utf-8"))["candidates"]
    selected = [
        candidate
        for candidate in candidates
        if candidate["component_ordinal"] == mixed_ordinal
    ]
    assert len(selected) == 1
    assert selected[0]["material_class"] == 1


def test_component_size_filter_fails_without_report_when_one_class_is_short(tmp_path):
    environment, sites = _synthetic_case()
    # The last material-5 component falls from 256 to 255 cells.
    environment["building_presence"][657, 913] = 0
    environment["material_class"][657, 913] = 0
    destination = tmp_path / "inventory.aggregate.json"
    with pytest.raises(inventory.SiteInventoryContractError, match="Material class 5"):
        inventory.run_site_inventory(
            _active_config(),
            environment_loader=lambda: environment,
            destination=destination,
            source_context=_source_context(),
            resource_probe=_resource_probe,
            synthetic_test=True,
            existing_sites=sites,
        )
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_site_context_failure_writes_nothing(tmp_path):
    environment, sites = _synthetic_case()
    row, col = sites[0].row, sites[0].col
    environment["material_class"][row, col] = 2
    destination = tmp_path / "inventory.aggregate.json"
    with pytest.raises(inventory.SiteInventoryContractError, match="revalidation"):
        inventory.run_site_inventory(
            _active_config(),
            environment_loader=lambda: environment,
            destination=destination,
            source_context=_source_context(),
            resource_probe=_resource_probe,
            synthetic_test=True,
            existing_sites=sites,
        )
    assert not destination.exists()


@pytest.mark.parametrize("invalid_material", [-1, 6])
def test_invalid_material_domain_fails_without_report(tmp_path, invalid_material):
    environment, sites = _synthetic_case()
    environment["material_class"][500, 500] = invalid_material
    destination = tmp_path / "inventory.aggregate.json"
    with pytest.raises(inventory.SiteInventoryContractError, match="0..5"):
        inventory.run_site_inventory(
            _active_config(),
            environment_loader=lambda: environment,
            destination=destination,
            source_context=_source_context(),
            resource_probe=_resource_probe,
            synthetic_test=True,
            existing_sites=sites,
        )
    assert not destination.exists()


def test_report_is_self_hashed_atomic_non_overwriting_and_aggregate_only(tmp_path):
    result, report, environment, sites = _run(tmp_path)
    destination = Path(result["path"])
    original = destination.read_bytes()
    payload_hash = report.pop("payload_sha256")
    canonical = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    assert payload_hash == sha256(canonical).hexdigest()
    assert list(tmp_path.iterdir()) == [destination]
    with pytest.raises(FileExistsError, match="overwrite"):
        inventory.run_site_inventory(
            _active_config(),
            environment_loader=lambda: environment,
            destination=destination,
            source_context=_source_context(),
            resource_probe=_resource_probe,
            synthetic_test=True,
            existing_sites=sites,
        )
    assert destination.read_bytes() == original
    assert not {
        "features",
        "predictors",
        "labels",
        "cell_inventory",
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
    }.intersection(_keys(report))


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "stack_ground_truth.tif",
        "modules/model_free_teacher.py",
        "models/fire_rf_model.joblib",
        "output/split_manifest.json",
        "data/training_rows.csv",
    ],
)
def test_forbidden_ground_truth_teacher_model_split_and_training_paths_are_rejected(
    tmp_path, forbidden_path
):
    context = _source_context()
    context["source_hashes"] = {forbidden_path: "a" * 64}
    destination = tmp_path / "inventory.aggregate.json"
    with pytest.raises((PermissionError, inventory.SiteInventoryContractError)):
        inventory.run_site_inventory(
            _active_config(),
            environment_loader=lambda: pytest.fail("environment accessed"),
            destination=destination,
            source_context=context,
            resource_probe=_resource_probe,
            synthetic_test=True,
            existing_sites=(),
        )
    assert not destination.exists()
