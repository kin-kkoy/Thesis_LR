"""Source-prepared installed-reader checks; execution needs separate approval.

No opener is mocked. All TIFF/companion creation is deferred to these tests.
The adapter capability argument below is an explicit bootstrap assumption,
not a capability receipt. Only independent review of complete future evidence
can establish a version-bound operational capability record.
"""
from contextlib import contextmanager
import hashlib
from html import escape
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.env import get_gdal_config
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from modules import candidate_dataset_inputs as inputs


CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parents[1]
SOURCE_PINS = {
    "modules/candidate_dataset_inputs.py":
        "a3c03d082b32742020f0ab9fac5c4b5c523ddc9556d26410dda236ae503f226d",
    "config/candidate_teacher_shared_dataset_v1.yaml":
        "c5d9443dda133bdc112ccb45f4734220be7e7bea30bb8f6e7d565d5a37f292d0",
}
GRID = dict(shape=(3, 3), crs="EPSG:32651",
            transform=tuple(Affine(3, 0, 600000, 0, -3, 1140000)))
OTHER_TRANSFORM = Affine(3, 0, 600300, 0, -3, 1140300)
KEYS = ("GDAL_PAM_ENABLED", "GDAL_DISABLE_READDIR_ON_OPEN",
        "GDAL_GEOREF_SOURCES", "GDAL_TIFF_INTERNAL_MASK",
        "GDAL_TIFF_INTERNAL_MASK_TO_8BIT")
ORDINARY = dict(GDAL_PAM_ENABLED="YES", GDAL_DISABLE_READDIR_ON_OPEN="FALSE",
                GDAL_GEOREF_SOURCES="PAM,INTERNAL,WORLDFILE",
                GDAL_TIFF_INTERNAL_MASK_TO_8BIT="YES")


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings():
    return {key: get_gdal_config(key, normalize=False) for key in KEYS}


@contextmanager
def _environment(evidence, label, **options):
    before = _settings()
    entry = dict(label=label, requested=options, before=before)
    evidence["environments"].append(entry)
    try:
        with rasterio.Env(**options):
            entry["effective"] = _settings()
            yield
    finally:
        entry["after"] = _settings()
        entry["restored"] = entry["after"] == before
        assert entry["restored"], "GDAL setting restoration failed: " + label


def _inventory(folder):
    result = {}
    for path in sorted(folder.iterdir()):
        assert not path.is_symlink() and not path.is_junction() and path.is_file()
        result[path.name] = dict(bytes=path.stat().st_size, sha256=_hash(path))
    return result


@pytest.fixture
def case(request, tmp_path_factory):
    # Validate the separately approved root before requesting any tmp_path.
    text = os.environ.get("CANDIDATE_READER_VERIFICATION_ROOT")
    assert text, "Explicit separately approved synthetic verification root required"
    assert not text.replace("/", "\\").startswith("\\\\")
    root = Path(text)
    base_text = request.config.getoption("basetemp")
    assert root.is_absolute() and base_text, "Explicit local root and basetemp required"
    base = Path(base_text)
    assert base.is_absolute() and base != root and base.is_relative_to(root)
    assert not root.is_relative_to(REPO) and not REPO.is_relative_to(root)
    for path in (root, *root.parents):
        assert not path.is_symlink() and not path.is_junction()
    assert root.is_dir()
    for path in (base, *base.parents):
        assert not path.is_symlink() and not path.is_junction()
    assert base.resolve() != root.resolve() and base.resolve().is_relative_to(root.resolve())
    assert not root.resolve().is_relative_to(REPO) and not REPO.is_relative_to(root.resolve())
    folder = request.getfixturevalue("tmp_path")
    assert folder.resolve().is_relative_to(root.resolve())
    folder = folder / "reader-fixtures"
    folder.mkdir(exist_ok=False)
    before_settings = _settings()
    evidence = dict(schema="candidate_reader_capability.synthetic.v1",
        nodeid=request.node.nodeid, fixture_directory=str(folder),
        operational_capability_established=False,
        capability_argument_status="UNVERIFIED_TEST_BOOTSTRAP_ONLY",
        runtime=dict(rasterio=rasterio.__version__, gdal=rasterio.__gdal_version__,
                     rasterio_origin=str(Path(rasterio.__file__).resolve())),
        sources={}, environments=[], views=[], adapter=None, expected={})
    for relative, expected in SOURCE_PINS.items():
        actual = _hash(CODE / relative)
        evidence["sources"][relative] = dict(expected=expected, observed=actual)
        assert actual == expected, "Source/configuration binding changed"
    evidence["test_source_sha256"] = _hash(Path(__file__))
    assert Path(inputs.__file__).resolve() == CODE / "modules/candidate_dataset_inputs.py"
    assert rasterio.__version__ == "1.5.0" and rasterio.__gdal_version__ == "3.12.1"
    try:
        yield folder, evidence
    finally:
        evidence["final_inventory"] = _inventory(folder)
        evidence["settings_after"] = _settings()
        evidence["settings_restored"] = evidence["settings_after"] == before_settings
        evidence["sources_after"] = {name: _hash(CODE / name) for name in SOURCE_PINS}
        print("READER_CAPABILITY_CASE " + json.dumps(evidence, sort_keys=True, allow_nan=False),
              flush=True)
        assert evidence["settings_restored"]
        assert evidence["sources_after"] == SOURCE_PINS
        if "identity_before_reads" in evidence:
            assert evidence["final_inventory"] == evidence["identity_before_reads"]


def _arrays(*, mixed=False):
    raw = {
        "slope": np.arange(9, dtype=np.int16).reshape(3, 3),
        "proximity": (np.arange(9, dtype=np.int16) * 2).reshape(3, 3),
        "buildings": np.array([[0, 10, 10], [10, 10, 10], [10, 10, 10]], dtype=np.int16),
        "materials": np.array([[0, 1, 2], [3, 4, 5], [1, 2, 3]], dtype=np.int16),
    }
    if mixed:
        raw["slope"] = np.array([[0, 0.5, 1.25], [2.5, 4.25, 5.5],
                                 [8.75, 10.5, -0.5]], dtype=np.float32)
        raw["proximity"] = np.array([[0, 0.25, 2.5], [3.75, 5.5, 12.25],
                                     [16.5, 20.75, 30.125]], dtype=np.float32)
        raw["buildings"] = raw["buildings"].astype(np.float32)
    return raw


def _create(case, *, kind="all_valid", missing_grid=None, mixed=False,
            fractional_building=False):
    folder, evidence = case
    raw = _arrays(mixed=mixed)
    if fractional_building:
        assert mixed
        raw["buildings"][1, 1] = np.float32(10.5)
    if kind == "embedded_nodata":
        raw["slope"][1, 1] = -888
    paths = {name: folder / (name + ".tif") for name in inputs.LAYERS}
    internal = kind != "external_mask"
    with _environment(evidence, "fixture_writer", GDAL_PAM_ENABLED="NO",
                      GDAL_TIFF_INTERNAL_MASK=internal,
                      GDAL_TIFF_INTERNAL_MASK_TO_8BIT="YES"):
        for name, path in paths.items():
            with path.open("xb"):
                pass  # Reserve a new owned fixture before the writer initializes it.
            profile = dict(driver="GTiff", height=3, width=3, count=1, dtype=str(raw[name].dtype))
            if name != "slope" or missing_grid != "pam":
                profile["crs"] = GRID["crs"]
            if name != "slope" or missing_grid is None:
                profile["transform"] = Affine(*GRID["transform"][:6])
            if name == "slope" and kind == "embedded_nodata":
                profile["nodata"] = -888
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("error")
                warnings.simplefilter("always", NotGeoreferencedWarning)
                with rasterio.open(path, "w", **profile) as dataset:
                    dataset.write(raw[name], 1)
                    if name == "slope" and kind in ("internal_mask", "external_mask"):
                        mask = np.full((3, 3), 255, dtype=np.uint8)
                        mask[1, 1] = 0
                        dataset.write_mask(mask)
                evidence.setdefault("construction_warnings", []).extend(str(w.message) for w in caught)
    evidence["expected"]["raw"] = {name: values.tolist() for name, values in raw.items()}
    evidence["expected"]["raw_dtypes"] = {name: str(values.dtype) for name, values in raw.items()}
    return paths, raw


def _text(path, text):
    data = text.encode("utf-8")
    with path.open("xb") as stream:
        assert stream.write(data) == len(data)
    assert path.read_bytes() == data


def _pam(path, transform, crs, nodata=None):
    gt = (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)
    band = "" if nodata is None else (
        '<PAMRasterBand band="1"><NoDataValue>' + str(nodata)
        + '</NoDataValue></PAMRasterBand>')
    _text(Path(str(path) + ".aux.xml"), "<PAMDataset><SRS>"
          + escape(CRS.from_user_input(crs).to_wkt()) + "</SRS><GeoTransform>"
          + ",".join(str(n) for n in gt) + "</GeoTransform>" + band + "</PAMDataset>\n")


def _world(path, transform):
    values = (transform.a, transform.d, transform.b, transform.e,
              transform.c + (transform.a + transform.b) / 2,
              transform.f + (transform.d + transform.e) / 2)
    _text(path.with_suffix(".tfw"), "".join(str(n) + "\n" for n in values))


def _view(path, evidence, *, controlled, georef_sources=None):
    options = dict(ORDINARY)
    if georef_sources is not None:
        assert not controlled
        options["GDAL_GEOREF_SOURCES"] = georef_sources
    if controlled:
        options.update(inputs.ENV_OPTIONS)
    with _environment(evidence, "controlled" if controlled else "ordinary_control", **options):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            warnings.simplefilter("always", NotGeoreferencedWarning)
            with rasterio.open(path, mode="r", driver="GTiff", sharing=False,
                               **(inputs.OPEN_OPTIONS if controlled else {})) as dataset:
                values, mask = dataset.read(1), dataset.read_masks(1)
                view = dict(path=str(path), controlled=controlled,
                    driver=dataset.driver, count=dataset.count, shape=list(dataset.shape),
                    dtype=dataset.dtypes[0], crs=None if dataset.crs is None else dataset.crs.to_string(),
                    transform=list(dataset.transform), nodata=dataset.nodatavals[0],
                    pixels=values.tolist(), masks=mask.tolist(),
                    mask_dtype=str(mask.dtype), mask_flags=sorted(f.name for f in dataset.mask_flag_enums[0]),
                    associated_files=list(dataset.files), open_options=dict(dataset.options),
                    effective_controls=_settings())
            view["warnings"] = [str(w.message) for w in caught]
    evidence["views"].append(view)
    return view


def _assert_view(view, raw, mask, *, crs=GRID["crs"], transform=GRID["transform"],
                 nodata=None, flags=("all_valid",)):
    view["expected"] = dict(pixels=raw.tolist(), masks=mask.tolist(), crs=crs,
        transform=list(transform), nodata=nodata, flags=list(flags), dtype=str(raw.dtype),
        mask_dtype="uint8", shape=[3, 3], count=1, driver="GTiff")
    assert view["driver"] == "GTiff" and view["count"] == 1
    assert view["shape"] == [3, 3] and view["dtype"] == str(raw.dtype)
    assert view["crs"] == crs and tuple(view["transform"]) == tuple(transform)
    assert view["nodata"] == nodata and view["mask_dtype"] == "uint8"
    assert view["mask_flags"] == list(flags)
    np.testing.assert_array_equal(view["pixels"], raw)
    np.testing.assert_array_equal(view["masks"], mask)
    if view["controlled"]:
        assert view["associated_files"] == [view["path"]]
        assert view["open_options"].get("GEOREF_SOURCES") == "INTERNAL"
        assert all(view["effective_controls"][k] == v for k, v in inputs.ENV_OPTIONS.items())


def _adapt(paths, evidence):
    pins = {name: _hash(path) for name, path in paths.items()}
    # Necessary to enter the implementation under test; deliberately unverified.
    capability = dict(policy_id=inputs.POLICY_ID,
        runtime={"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__},
        controls_verified=True, environment_options=dict(inputs.ENV_OPTIONS),
        open_options=dict(inputs.OPEN_OPTIONS), verification_reference="UNVERIFIED_TEST_BOOTSTRAP_ONLY")
    evidence["adapter"] = dict(input_pins=pins, capability_argument=capability)
    with _environment(evidence, "production_adapter_outer_control", **ORDINARY):
        try:
            prepared = inputs.read_inputs(paths, pins, expected_grid=GRID,
                configured_nodata=None, capability=capability)
        except BaseException as error:
            evidence["adapter"]["exception"] = dict(type=type(error).__name__, message=str(error))
            raise
    evidence["adapter"]["interpretation"] = prepared.interpretation
    evidence["adapter"]["masks"] = {
        "environment": prepared.environment["burnable_mask"].tolist(),
        "receiving": prepared.receiving_mask.tolist()}
    return prepared


def _exercise(case, paths, raw, *, invalid=False, kind="all_valid"):
    folder, evidence = case
    expected_environment = np.ones((3, 3), dtype=bool)
    if invalid:
        expected_environment[1, 1] = False
    evidence["expected"].update(environment=expected_environment.tolist(),
        receiving=(expected_environment & (raw["buildings"] == 10)).tolist())
    evidence.setdefault("identity_before_reads", _inventory(folder))
    assert _inventory(folder) == evidence["identity_before_reads"]
    for name, path in paths.items():
        mask = np.full((3, 3), 255, dtype=np.uint8)
        nodata, flags = None, ("all_valid",)
        if name == "slope" and invalid:
            mask[1, 1] = 0
            nodata = -888 if kind == "embedded_nodata" else None
            flags = ("nodata",) if nodata is not None else ("per_dataset",)
        evidence["expected"].setdefault("native", {})[name] = dict(
            masks=mask.tolist(), nodata=nodata, flags=list(flags), crs=GRID["crs"],
            transform=list(GRID["transform"]), dtype=str(raw[name].dtype))
        _assert_view(_view(path, evidence, controlled=True), raw[name], mask,
                     nodata=nodata, flags=flags)
    prepared = _adapt(paths, evidence)
    e = prepared.environment
    assert e["grid_shape"] == GRID["shape"] and e["crs"] == CRS.from_user_input(GRID["crs"])
    assert tuple(e["transform"]) == GRID["transform"]
    np.testing.assert_array_equal(e["burnable_mask"], expected_environment)
    np.testing.assert_array_equal(e["nodata_mask"], ~expected_environment)
    np.testing.assert_array_equal(prepared.receiving_mask,
                                  expected_environment & (raw["buildings"] == 10))
    np.testing.assert_array_equal(e["building_presence"], (raw["buildings"] == 10).astype(np.int8))
    np.testing.assert_array_equal(e["material_class"], raw["materials"].astype(np.int8))
    for name, field in (("slope", "slope_risk"), ("proximity", "proximity_risk")):
        expected = np.where(expected_environment, raw[name], 0).astype(np.float32)
        if name == "slope":
            expected[(expected < 0) | (expected > 10)] = 0
        expected = expected / np.float32(10)
        evidence["expected"].setdefault("canonical_predictors", {})[field] = expected.tolist()
        evidence["adapter"].setdefault("canonical_predictors", {})[field] = dict(
            dtype=str(e[field].dtype), values=e[field].tolist())
        assert e[field].dtype == np.float32
        np.testing.assert_array_equal(e[field], expected)
    assert e["burnable_mask"][0, 0] and e["slope_risk"][0, 0] == 0
    assert not prepared.receiving_mask[0, 0]
    assert prepared.input_hashes == evidence["adapter"]["input_pins"]
    assert {name: layer["dtype"] for name, layer in
            prepared.interpretation["layers"].items()} == evidence["expected"]["raw_dtypes"]


@pytest.mark.parametrize("kind", ["all_valid", "embedded_nodata", "internal_mask"])
def test_embedded_pixels_masks_and_production_adapter(case, kind):
    paths, raw = _create(case, kind=kind)
    _exercise(case, paths, raw, invalid=kind != "all_valid", kind=kind)


@pytest.mark.parametrize("companion", ["mask", "pam", "world"])
def test_companion_exclusion_with_interpretable_positive_control(case, companion):
    _exercise_companion(case, companion)


def _exercise_companion(case, companion, *, mixed=False):
    folder, evidence = case
    paths, raw = _create(case, kind="external_mask" if companion == "mask" else "all_valid",
                         mixed=mixed)
    slope = paths["slope"]
    external_mask = np.full((3, 3), 255, dtype=np.uint8)
    external_mask[1, 1] = 0
    pam_nodata = 4.25 if mixed else 4
    if companion == "pam":
        assert raw["slope"][1, 1] == pam_nodata
        external_mask = np.where(raw["slope"] == pam_nodata, 0, 255).astype(np.uint8)
        _pam(slope, OTHER_TRANSFORM, "EPSG:32650", nodata=pam_nodata)
    elif companion == "world":
        _world(slope, OTHER_TRANSFORM)
    companion_name = {"mask": "slope.tif.msk", "pam": "slope.tif.aux.xml",
                      "world": "slope.tfw"}[companion]
    expected_names = {path.name for path in paths.values()} | {companion_name}
    assert set(_inventory(folder)) == expected_names
    evidence["identity_before_reads"] = _inventory(folder)
    if companion == "world":
        # INTERNAL precedes WORLDFILE by default; declare the positive-control priority.
        ordinary = _view(slope, evidence, controlled=False,
                         georef_sources="WORLDFILE,INTERNAL")
        _assert_view(ordinary, raw["slope"], np.full((3, 3), 255, dtype=np.uint8),
                     transform=OTHER_TRANSFORM)
    else:
        ordinary = _view(slope, evidence, controlled=False)
        _assert_view(ordinary, raw["slope"], external_mask,
            crs="EPSG:32650" if companion == "pam" else GRID["crs"],
            transform=OTHER_TRANSFORM if companion == "pam" else GRID["transform"],
            nodata=pam_nodata if companion == "pam" else None,
            flags=("nodata",) if companion == "pam" else ("per_dataset",))
    _exercise(case, paths, raw)


@pytest.mark.parametrize("source", ["pam", "world"])
def test_external_only_georeferencing_is_not_an_internal_grid(case, source):
    _exercise_external_grid(case, source)


def _exercise_external_grid(case, source, *, mixed=False):
    folder, evidence = case
    paths, raw = _create(case, missing_grid=source, mixed=mixed)
    if source == "pam":
        _pam(paths["slope"], Affine(*GRID["transform"][:6]), GRID["crs"])
    else:
        _world(paths["slope"], Affine(*GRID["transform"][:6]))
    evidence["identity_before_reads"] = _inventory(folder)
    mask = np.full((3, 3), 255, dtype=np.uint8)
    _assert_view(_view(paths["slope"], evidence, controlled=False), raw["slope"], mask)
    controlled = _view(paths["slope"], evidence, controlled=True)
    assert controlled["pixels"] == raw["slope"].tolist() and controlled["masks"] == mask.tolist()
    assert (controlled["crs"] != GRID["crs"]
            or tuple(controlled["transform"]) != GRID["transform"])
    with pytest.raises((ValueError, NotGeoreferencedWarning)) as failure:
        _adapt(paths, evidence)
    if isinstance(failure.value, ValueError):
        assert "Raster alignment or approved grid mismatch" in str(failure.value)


@pytest.mark.parametrize("kind", ["all_valid", "embedded_nodata", "internal_mask"])
def test_mixed_embedded_pixels_masks_and_production_adapter(case, kind):
    paths, raw = _create(case, kind=kind, mixed=True)
    _exercise(case, paths, raw, invalid=kind != "all_valid", kind=kind)


@pytest.mark.parametrize("companion", ["mask", "pam", "world"])
def test_mixed_companion_exclusion_with_interpretable_positive_control(case, companion):
    _exercise_companion(case, companion, mixed=True)


@pytest.mark.parametrize("source", ["pam", "world"])
def test_mixed_external_only_georeferencing_is_not_an_internal_grid(case, source):
    _exercise_external_grid(case, source, mixed=True)


@pytest.mark.parametrize("slope_invalid", [False, True])
def test_mixed_fractional_building_rejected_before_narrowing(case, slope_invalid):
    folder, evidence = case
    paths, raw = _create(case, kind="internal_mask" if slope_invalid else "all_valid",
                         mixed=True, fractional_building=True)
    assert raw["buildings"].dtype == np.float32 and raw["buildings"][1, 1] == np.float32(10.5)
    evidence["expected"].update(rejected_building=dict(row=1, col=1, value=10.5,
        dtype="float32", slope_native_invalid=slope_invalid),
        exception=dict(type="ValueError", message="Unsupported raw building code before conversion"))
    evidence["identity_before_reads"] = _inventory(folder)
    for name, path in paths.items():
        mask = np.full((3, 3), 255, dtype=np.uint8)
        flags = ("all_valid",)
        if name == "slope" and slope_invalid:
            mask[1, 1] = 0
            flags = ("per_dataset",)
        _assert_view(_view(path, evidence, controlled=True), raw[name], mask, flags=flags)
    with pytest.raises(ValueError, match="^Unsupported raw building code before conversion$"):
        _adapt(paths, evidence)
