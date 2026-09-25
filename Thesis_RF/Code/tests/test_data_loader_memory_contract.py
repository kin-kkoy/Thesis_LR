"""Value, domain, and ownership tests for bounded raster preparation."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from modules.data_loader import EnvironmentManager
from modules.feature_pipeline import FeatureAssembler


def _write_raster(path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs="EPSG:32651",
        transform=from_origin(100.0, 200.0, 3.0, 3.0),
    ) as dataset:
        dataset.write(values, 1)


def _manager(tmp_path) -> tuple[EnvironmentManager, dict[str, np.ndarray]]:
    source = {
        "slope": np.array(
            [[-9999.0, 0.0, 2.5], [-9999.0, 10.0, 4.25]], dtype=np.float32
        ),
        "proximity": np.array(
            [[-9999.0, 0.0, 15.0], [20.0, 8.0, 2.5]], dtype=np.float32
        ),
        "buildings": np.array(
            [[-9999.0, 0.0, 10.0], [10.0, 10.0, 0.0]], dtype=np.float32
        ),
        "materials": np.array(
            [[-9999, 0, 1], [2, 5, 0]], dtype=np.int16
        ),
    }
    names = {
        "slope": "slope.tif",
        "proximity": "proximity.tif",
        "buildings": "buildings.tif",
        "materials": "materials.tif",
    }
    for name, filename in names.items():
        _write_raster(tmp_path / filename, source[name])
    manager = EnvironmentManager(
        {
            "raster_dir": str(tmp_path),
            "slope_file": names["slope"],
            "proximity_file": names["proximity"],
            "buildings_file": names["buildings"],
            "materials_file": names["materials"],
            "nodata_value": -9999,
        }
    )
    return manager, source


def _expected_environment(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    slope_raw = source["slope"].astype(np.float32)
    proximity_raw = source["proximity"].astype(np.float32)
    buildings_raw = source["buildings"].astype(np.float32)
    materials_raw = source["materials"].astype(np.float32)
    nodata = (
        (slope_raw == -9999)
        | (proximity_raw == -9999)
        | (buildings_raw == -9999)
        | (materials_raw == -9999)
        | (slope_raw == -9999)
    )
    ocean = (
        (slope_raw == 0)
        & (proximity_raw == 0)
        & (buildings_raw == 0)
        & (materials_raw == 0)
    )
    nodata = nodata | ocean
    slope = slope_raw.copy()
    slope[(slope < 0) | (slope > 10)] = 0
    slope_risk = slope / 10.0
    slope_risk[nodata] = 0
    proximity_risk = proximity_raw / 10.0
    proximity_risk[nodata] = 0
    building_presence = np.where(buildings_raw == 10, 1, 0).astype(np.int8)
    material_invalid = (
        ~np.isfinite(materials_raw)
        | (materials_raw == -9999)
    )
    materials_raw[material_invalid] = 0
    material_class = materials_raw.astype(np.int8)
    return {
        "slope_risk": slope_risk,
        "proximity_risk": proximity_risk,
        "building_presence": building_presence,
        "material_class": material_class,
        "nodata_mask": nodata,
        "burnable_mask": ~nodata,
    }


def test_direct_dtype_read_and_corrected_normalization_match_contract(tmp_path):
    manager, source = _manager(tmp_path)
    expected = _expected_environment(source)

    manager.load_rasters()
    assert manager.materials_raw.dtype == np.dtype(np.float32)
    assert np.array_equal(manager.materials_raw, source["materials"].astype(np.float32))
    slope_buffer = manager.slope_raw
    proximity_buffer = manager.proximity_raw
    manager.build_masks()
    manager.normalize_layers()
    environment = manager.get_environment()

    assert manager.slope_risk is slope_buffer
    assert manager.proximity_risk is proximity_buffer
    for name, expected_values in expected.items():
        assert np.array_equal(environment[name], expected_values)

    # The combined mask excludes this slope-invalid mapped building, but its
    # independently valid material class remains intact for full-grid identity.
    assert environment["nodata_mask"][1, 0]
    assert not environment["burnable_mask"][1, 0]
    assert environment["building_presence"][1, 0] == 1
    assert environment["material_class"][1, 0] == 2
    assert np.array_equal(
        environment["building_presence"] > 0,
        environment["material_class"] > 0,
    )
    authoritative = (
        environment["burnable_mask"]
        & (~environment["nodata_mask"])
        & (environment["building_presence"] > 0)
    )
    assert not authoritative[1, 0]


def test_raw_grids_and_redundant_material_risk_are_not_retained(tmp_path, capsys):
    manager, _source = _manager(tmp_path)
    manager.load_rasters()
    manager.build_masks()
    manager.normalize_layers()

    assert manager.slope_raw is None
    assert manager.proximity_raw is None
    assert manager.buildings_raw is None
    assert manager.materials_raw is None
    assert not hasattr(manager, "material_risk")
    assert "material_risk" not in manager.get_environment()

    assembler = FeatureAssembler(
        manager.get_environment(),
        {
            "speed_kmh": 10.0,
            "direction_deg": 315.0,
            "direction_convention": "meteorological_from",
            "direction_units": "degrees_clockwise",
            "north_reference": "projected_grid_north",
            "schema_version": "simulated_wind.v1",
            "calm_representation": "speed_zero_direction_zero",
        },
    )
    expected_risk = FeatureAssembler.MATERIAL_CLASS_TO_RISK[
        manager.material_class
    ]
    expected_risk = np.where(
        manager.building_presence > 0, expected_risk, 0.0
    ).astype(np.float32)
    assert np.array_equal(assembler.material_risk, expected_risk)

    manager.summary()
    assert "Material risk range:" in capsys.readouterr().out
