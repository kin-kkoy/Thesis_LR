"""Synthetic array/reader substitutes only; no real TIFF fixture files."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import Affine

from modules import candidate_dataset_inputs as inputs


def arrays(size=3):
    raw = {name: np.full((size, size), value, dtype=np.float64)
           for name, value in (("slope", 0), ("proximity", 0), ("buildings", 10), ("materials", 1))}
    masks = {name: np.full((size, size), 255, dtype=np.uint8) for name in raw}
    return raw, masks


def prepared_arrays(raw=None, masks=None, origin=(0, 0), shape=None):
    if raw is None:
        raw, masks = arrays()
    return inputs.prepare_arrays(raw, masks, dict.fromkeys(inputs.LAYERS),
        configured_nodata=None, crs="EPSG:32651", transform=Affine(3, 0, 0, 0, -3, 9),
        input_hashes=dict.fromkeys(inputs.LAYERS, "a" * 64),
        interpretation={"policy_id": inputs.POLICY_ID, "fixture": "explicit synthetic arrays"},
        global_origin=origin, full_grid_shape=shape)


def test_zero_background_and_candidate_masks_do_not_use_ocean_heuristic():
    raw, masks = arrays()
    raw["buildings"][0, 0] = raw["materials"][0, 0] = 0
    result = prepared_arrays(raw, masks)
    e = result.environment
    assert e["burnable_mask"][0, 0] and not e["nodata_mask"][0, 0]
    assert e["building_presence"][0, 0] == e["material_class"][0, 0] == 0
    assert not result.receiving_mask[0, 0]
    assert e["slope_risk"].dtype == e["proximity_risk"].dtype == np.float32
    assert e["building_presence"].dtype == e["material_class"].dtype == np.int8
    assert not result.receiving_mask.flags.writeable


@pytest.mark.parametrize("name,value", [
    ("materials", 1.00000000001), ("materials", 5.99999999999),
    ("materials", 256.0), ("buildings", 10.00000000001), ("buildings", 256.0)])
def test_raw_categorical_precision_is_checked_outside_joint_validity(name, value):
    raw, masks = arrays()
    raw["slope"][0, 0] = -9999
    raw[name][0, 0] = value
    with pytest.raises(ValueError, match="raw .* code"):
        prepared_arrays(raw, masks)


@pytest.mark.parametrize("invalid", ["native", "nan", "configured", "legacy", "declared"])
def test_layer_validity_is_native_finite_and_all_declared_sentinels(invalid):
    raw, masks = arrays()
    declared = dict.fromkeys(inputs.LAYERS)
    configured = -777
    if invalid == "native":
        masks["slope"][1, 1] = 0
    else:
        raw["slope"][1, 1] = {"nan": np.nan, "configured": -777,
                              "legacy": -9999, "declared": -888}[invalid]
        declared["slope"] = -888
    result = inputs.prepare_arrays(raw, masks, declared, configured_nodata=configured,
        crs="EPSG:32651", transform=Affine(3, 0, 0, 0, -3, 9),
        input_hashes=dict.fromkeys(inputs.LAYERS, "a" * 64),
        interpretation={"policy_id": inputs.POLICY_ID})
    assert not result.environment["burnable_mask"][1, 1]
    assert not result.receiving_mask[1, 1]
    assert result.environment["material_class"][1, 1] == 1


@pytest.mark.parametrize("layer,value", [("materials", 0), ("materials", np.nan),
                                       ("buildings", 0), ("buildings", np.nan)])
def test_full_grid_categorical_disagreement_is_not_waived_by_bad_slope(layer, value):
    raw, masks = arrays()
    raw["slope"][0, 0] = -9999
    raw[layer][0, 0] = value
    with pytest.raises(ValueError, match="Full-grid"):
        prepared_arrays(raw, masks)


def test_independently_invalid_categorical_layers_normalize_to_background():
    raw, masks = arrays()
    raw["buildings"][0, 0] = raw["materials"][0, 0] = -9999
    prepared = prepared_arrays(raw, masks)
    assert prepared.environment["material_class"][0, 0] == 0
    assert not prepared.receiving_mask[0, 0]


@pytest.mark.parametrize("bad", [[(0, 0), (3, 0)], [(0, 0), (0, 0)],
                               [(True, 0)], [(0.1, 0)], []])
def test_whole_seed_list_rejects_atomically_without_mutation(bad):
    prepared = prepared_arrays()
    before = {k: v.copy() for k, v in prepared.environment.items() if isinstance(v, np.ndarray)}
    with pytest.raises(ValueError):
        inputs.validate_seeds(bad, prepared, "grid_row_col")
    for key, value in before.items():
        np.testing.assert_array_equal(value, prepared.environment[key])


def test_global_and_projected_seed_conventions_agree_and_reject_background():
    raw, masks = arrays()
    raw["buildings"][0, 0] = raw["materials"][0, 0] = 0
    prepared = prepared_arrays(raw, masks, origin=(128, 256), shape=(140, 270))
    assert inputs.validate_seeds([(129, 257)], prepared, "grid_row_col") == ((1, 1),)
    assert inputs.validate_seeds([(4.5, 4.5)], prepared, "projected_xy") == ((1, 1),)
    with pytest.raises(ValueError):
        inputs.validate_seeds([(128, 256)], prepared, "grid_row_col")


@pytest.fixture
def reader(monkeypatch):
    state = SimpleNamespace(active=False, calls=[], hashes=0)
    transform = Affine(3, 0, 0, 0, -3, 9)
    crs = inputs.CRS.from_epsg(32651)
    paths = {name: str(Path(__file__).resolve().parent / (name + ".tif")) for name in inputs.LAYERS}
    pins = dict.fromkeys(inputs.LAYERS, "a" * 64)
    capability = dict(policy_id=inputs.POLICY_ID,
        runtime={"rasterio": inputs.rasterio.__version__, "gdal": inputs.rasterio.__gdal_version__},
        controls_verified=True, environment_options=dict(inputs.ENV_OPTIONS),
        open_options=dict(inputs.OPEN_OPTIONS), verification_reference="synthetic reader substitute")
    @contextmanager
    def environment(**kwargs):
        assert kwargs == inputs.ENV_OPTIONS
        state.active = True
        try:
            yield
        finally:
            state.active = False
    class Dataset:
        driver, mode, count = "GTiff", "r", 1
        subdatasets, shape, nodatavals = [], (3, 3), (None,)
        dtypes = ("float64",)
        mask_flag_enums = ((SimpleNamespace(name="all_valid"),),)
        def __init__(self, path, kwargs):
            self.files, self.options = [path], dict(kwargs)
            self.crs, self.transform = crs, transform
        def __enter__(self):
            return self
        def __exit__(self, *args):
            assert state.active
        def read(self, band):
            assert state.active and band == 1
            name = Path(self.files[0]).stem
            return np.full((3, 3), {"slope": 0, "proximity": 0,
                                    "buildings": 10, "materials": 1}[name], dtype=np.float64)
        def read_masks(self, band):
            assert state.active and band == 1
            return np.full((3, 3), 255, dtype=np.uint8)
    def opening(path, **kwargs):
        assert state.active
        assert kwargs.pop("mode") == "r"
        assert kwargs.pop("driver") == "GTiff"
        assert kwargs.pop("sharing") is False
        assert kwargs == inputs.OPEN_OPTIONS
        dataset = Dataset(path, kwargs)
        state.calls.append(dataset)
        if hasattr(state, "change"):
            state.change(dataset)
        return dataset
    monkeypatch.setattr(inputs.rasterio, "Env", environment)
    monkeypatch.setattr(inputs.rasterio, "open", opening)
    monkeypatch.setattr(inputs, "get_gdal_config", lambda key, **kwargs: inputs.ENV_OPTIONS[key])
    monkeypatch.setattr(inputs, "_file_hash", lambda path: "a" * 64)
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    monkeypatch.setattr(Path, "is_symlink", lambda path: False)
    monkeypatch.setattr(Path, "is_junction", lambda path: False)
    state.invoke = lambda: inputs.read_inputs(paths, pins,
        expected_grid=dict(shape=[3, 3], crs=str(crs), transform=list(transform)),
        configured_nodata=None, capability=capability)
    state.capability = capability
    state.paths = paths
    return state


def test_controlled_native_reads_and_only_associated_tiff(reader):
    prepared = reader.invoke()
    assert len(reader.calls) == 4 and not reader.active
    assert prepared.interpretation["policy_id"] == "embedded_tiff_only.v1"
    assert np.all(prepared.receiving_mask)


@pytest.mark.parametrize("layer", inputs.LAYERS)
@pytest.mark.parametrize("network_path", [
    r"\\server\share\layer.tif",
    "//server/share/layer.tif",
    r"\\?\UNC\server\share\layer.tif",
    "//?/UNC/server/share/layer.tif",
    r"\\?\unc\server\share\layer.tif",
    r"\\server/share\layer.tif",
    r"\\.\UNC\server\share\layer.tif",
])
def test_network_path_guard_precedes_all_filesystem_hash_and_gdal_access(
        reader, monkeypatch, layer, network_path):
    reader.paths[layer] = network_path
    def unexpected(*args, **kwargs):
        pytest.fail("Filesystem/hash/GDAL access before network-path rejection")
    with monkeypatch.context() as guard:
        for name in ("stat", "lstat", "exists", "is_file", "is_symlink", "is_junction"):
            guard.setattr(Path, name, unexpected)
        guard.setattr(inputs, "_file_hash", unexpected)
        guard.setattr(inputs.rasterio, "Env", unexpected)
        guard.setattr(inputs.rasterio, "open", unexpected)
        with pytest.raises(ValueError, match="UNC/network TIFF paths are not permitted"):
            reader.invoke()
    assert not reader.calls


@pytest.mark.parametrize("change", [
    lambda d: d.options.clear(),
    lambda d: setattr(d, "driver", "VRT"),
    lambda d: setattr(d, "count", 2),
    lambda d: d.files.append(d.files[0] + ".aux.xml"),
    lambda d: setattr(d, "mask_flag_enums", ((SimpleNamespace(name="unknown"),),)),
    lambda d: setattr(d, "transform", Affine(3, 0, 1, 0, -3, 9)),
])
def test_unknown_ignored_or_contradictory_interpretation_rejects(reader, change):
    reader.change = change
    with pytest.raises(ValueError):
        reader.invoke()


def test_effective_gdal_control_failure_has_no_normal_open_fallback(reader, monkeypatch):
    monkeypatch.setattr(inputs, "get_gdal_config", lambda *a, **k: "YES")
    with pytest.raises(RuntimeError):
        reader.invoke()
    assert not reader.calls


def test_missing_capability_rejects_before_artifact_access(reader, monkeypatch):
    reader.capability["controls_verified"] = False
    monkeypatch.setattr(inputs, "_file_hash", lambda *_: pytest.fail("artifact access"))
    with pytest.raises(PermissionError):
        reader.invoke()


def test_identity_change_rejects_after_reads(reader, monkeypatch):
    hashes = iter(["a" * 64] * 4 + ["b" * 64])
    monkeypatch.setattr(inputs, "_file_hash", lambda *_: next(hashes))
    with pytest.raises(ValueError, match="changed"):
        reader.invoke()
