"""104 bounded comparisons; passing characterizations are not release acceptance.

Future execution/imports and the two synthetic CSVs need separate approval.
No learned inference, real RNG, fitting, CA step or publication is performed.
"""
import builtins
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine

from modules.automata_engine import FireAutomata as RFAutomata
from modules.candidate_dataset_inputs import prepare_arrays, validate_seeds
from modules import candidate_dataset_release as release
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES, FeatureAssembler, compute_moore_neighbor_count
from modules.wind_convention import WindContract, compute_wind_weighted_score


CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parents[1]
LR = REPO / "Thesis_LR/Code/sandbox_kent"
PREFIX = "_candidate_lr_compat_"
NAMES = ("slope_risk", "proximity_risk", "building_presence", "material_risk",
         "material_class", "wind_speed", "wind_sin", "wind_cos", "neighbor_burning_count",
         "composite_flammability", "wind_weighted_score")
RISKS = np.array([0, .95, .8, .48, .30, .15], dtype=np.float32)
TRANSFORM = Affine(3, 0, 600000, 0, -3, 1140000)
WIND = WindContract(17.7, 315.0).to_config()
OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
SOURCE_PINS = {
    "Thesis_RF/Code/modules/feature_pipeline.py": "b404e82cac952f3111defedf857ecf0b7891214afe4f0876a3bf7617680daeb1",
    "Thesis_RF/Code/modules/wind_convention.py": "99f0d24b1e79561cdf21004f467685ebfeaf17c14e7ac2e15301121542e09044",
    "Thesis_RF/Code/modules/candidate_dataset_inputs.py": "a3c03d082b32742020f0ab9fac5c4b5c523ddc9556d26410dda236ae503f226d",
    "Thesis_RF/Code/modules/candidate_dataset_release.py": "80c1c9ff23225b0bda5ce908b1c72bec5a40dd572b4a8d9158ae284f76a5eef3",
    "Thesis_RF/Code/modules/automata_engine.py": "a2e4518876e64b67a831e81ce89c345d577a340e19ecbafaa0e8562d227286d8",
    "Thesis_LR/Code/sandbox_kent/modules/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "Thesis_LR/Code/sandbox_kent/modules/feature_pipeline.py": "4c01b5669772f384f5a5c7771109d341a2afa8e2daf451c73be871f0623b075f",
    "Thesis_LR/Code/sandbox_kent/modules/automata_engine.py": "02698abf2b4e4a13ec38d199b2a829698d5890f2e338f17b48615a27686ba4d5",
    "Thesis_LR/Code/sandbox_kent/modules/data_loader.py": "6415ec09f154ef1675ec8c4695bbff9a8b5b410d21a74049d277714878cf4f97",
    "Thesis_LR/Code/sandbox_kent/orchestrator.py": "6ccc09ef7b76f1ee86700d24968d6f7c0676ae209551c92c8ef61f5971b1e6d0",
    "Thesis_LR/Code/sandbox_kent/dataset_generator.py": "ef1099fbb4d1d372ebbea7558eba8e540df534aedb789ba41593ca7a841aaefc",
    "Thesis_LR/Code/sandbox_kent/train_lr_stage_c.py": "924f3c9eb224d7413247d3be66099196b9fa1a457f81a6f69182c4790c340c3b",
    "Thesis_LR/Code/sandbox_kent/config/default_experiment.yaml": "590b05100686db1c97a2b72887e70aea84585ee29d2375280d9aa6bf4bb93145",
    "Thesis_RF/Code/config/candidate_teacher_shared_dataset_v1.yaml": "c5d9443dda133bdc112ccb45f4734220be7e7bea30bb8f6e7d565d5a37f292d0",
    "Thesis_RF/Code/config/phase15_additive_exposure_r2.yaml": "02389d7e1bbee4568f271bd270f5e21785ad5761ae3e291df0ec4815c8a50389",
    "Thesis_RF/Code/config/phase15_setc_rf_ca_v1.yaml": "88922d9b3366c480330580c6f22926987a3ec0b252fcdf9f4d2afacc051d3dfb",
}


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pins():
    return {name: _hash(REPO / name) for name in SOURCE_PINS}


def _affected(name):
    return (name == "modules" or name.startswith("modules.") or name == "Code"
            or name.startswith("Code.") or name.startswith(PREFIX))


def _namespace():
    return {name: value for name, value in list(sys.modules.items()) if _affected(name)}


def _same(left, right):
    return left.keys() == right.keys() and all(left[name] is right[name] for name in left)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path
    return module


@contextmanager
def _lr_namespace(evidence):
    saved = _namespace()
    path_object, paths = sys.path, list(sys.path)
    cache_object, cache = sys.path_importer_cache, dict(sys.path_importer_cache)
    record = dict(namespace_restored=False, paths_restored=False, cache_restored=False,
                  original_failure=None, restoration_failures=[])
    evidence["imports"].append(record)
    try:
        for name in tuple(sys.modules):
            if _affected(name):
                del sys.modules[name]
        sys.modules["Code"] = None  # Reject the orchestrator's fallback namespace.
        sys.path = [str(LR), *[p for p in paths if p != str(LR)]]
        modules = {}
        for name in ("modules", "modules.feature_pipeline", "modules.data_loader", "modules.automata_engine"):
            modules[name] = importlib.import_module(name)
            relative = "modules/__init__.py" if name == "modules" else name.replace(".", "/") + ".py"
            assert Path(modules[name].__file__).resolve() == LR / relative
        orchestrator = _load(PREFIX + "orchestrator", LR / "orchestrator.py")
        generator = _load(PREFIX + "generator", LR / "dataset_generator.py")
        stage = _load(PREFIX + "stage_c", LR / "train_lr_stage_c.py")
        assert stage.__name__ != "__main__"
        engine = modules["modules.automata_engine"]
        assert orchestrator.FireAutomata is generator.FireAutomata is engine.FireAutomata
        assert engine.FeatureAssembler is modules["modules.feature_pipeline"].FeatureAssembler
        record["origins"] = {name: str(Path(module.__file__).resolve()) for name, module in _namespace().items()
                             if module is not None}
        yield SimpleNamespace(feature=modules["modules.feature_pipeline"],
            loader=modules["modules.data_loader"], engine=engine, orchestrator=orchestrator,
            generator=generator, stage=stage)
    except BaseException as error:
        record["original_failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        for label, restore in (("namespace", lambda: _restore_namespace(saved)),
                               ("paths", lambda: _restore_list(path_object, paths)),
                               ("cache", lambda: _restore_cache(cache_object, cache))):
            try:
                restore()
                record[label + "_restored"] = True
            except BaseException as error:
                record["restoration_failures"].append(dict(target=label,
                    type=type(error).__name__, message=str(error)))
        if record["restoration_failures"] and record["original_failure"] is None:
            raise AssertionError("Import-state restoration failed; retain the case evidence")


def _restore_namespace(saved):
    for name in tuple(sys.modules):
        if _affected(name):
            del sys.modules[name]
    sys.modules.update(saved)
    assert _same(_namespace(), saved)


def _restore_list(original, values):
    sys.path = original
    original[:] = values
    assert sys.path is original and sys.path == values


def _restore_cache(original, values):
    sys.path_importer_cache = original
    original.clear()
    original.update(values)
    assert sys.path_importer_cache is original and _same(original, values)


class _NoRNG:
    def __getattr__(self, name):
        raise AssertionError("Randomness is forbidden in compatibility checks: " + name)


@pytest.fixture
def context(request, monkeypatch):
    assert sys.dont_write_bytecode, "Future verification must disable repository bytecode writes"
    evidence = dict(schema="candidate_rf_lr_compatibility.synthetic.v1", nodeid=request.node.nodeid,
        overall_compatibility_accepted=False, source_pins=_pins(), imports=[], comparisons=[],
        characterizations=[], rng_factory_calls=[])
    assert evidence["source_pins"] == SOURCE_PINS, "Reviewed source/configuration identity changed"
    factory = np.random.default_rng
    def no_rng(*args, **kwargs):
        evidence["rng_factory_calls"].append(dict(args=list(args), kwargs=kwargs))
        return _NoRNG()
    try:
        with monkeypatch.context() as guard:
            guard.setattr(np.random, "default_rng", no_rng)
            with _lr_namespace(evidence) as lr:
                yield SimpleNamespace(lr=lr, evidence=evidence, request=request, monkeypatch=monkeypatch)
    finally:
        evidence["rng_factory_restored"] = np.random.default_rng is factory
        try:
            evidence["source_pins_after"] = _pins()
        except BaseException as error:
            evidence["source_recheck_failure"] = dict(type=type(error).__name__, message=str(error))
        print("RF_LR_COMPATIBILITY_CASE " + json.dumps(evidence, sort_keys=True, allow_nan=False), flush=True)
        assert evidence["rng_factory_restored"]
        assert evidence.get("source_pins_after") == SOURCE_PINS
        assert all(not entry["restoration_failures"] for entry in evidence["imports"])


def _array(value):
    a = np.asarray(value)
    values = [float(v) if np.isfinite(v) else str(float(v)) for v in a.ravel()]
    return dict(dtype=str(a.dtype), shape=list(a.shape), bytes_hex=a.tobytes().hex(), values=values)


def _float_record(context, name, left, right):
    a, b = np.asarray(left), np.asarray(right)
    record = dict(name=name, left=_array(a), right=_array(b), bit_equal=False)
    context.evidence["comparisons"].append(record)
    if a.dtype == b.dtype == np.float32 and a.shape == b.shape:
        delta = a.astype(np.float64) - b.astype(np.float64)
        record.update(left_bits=a.view(np.uint32).ravel().tolist(),
            right_bits=b.view(np.uint32).ravel().tolist(), signed_difference=_array(delta),
            max_absolute_difference=(float(np.max(np.abs(delta))) if delta.size and np.all(np.isfinite(delta)) else None),
            bit_equal=np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    return record


def _exact(records):
    for record in records:
        assert record["bit_equal"], "Exact float32 compatibility unresolved: " + record["name"]


def _characterize(context, name, **observed):
    context.evidence["characterizations"].append(dict(name=name,
        matching_characterization_is_not_compatibility_acceptance=True, observed=observed))


def _raw(shape=(3, 3)):
    n = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return dict(slope=n % 11, proximity=n * 2, buildings=np.full(shape, 10.0), materials=n % 5 + 1)


def _prepare(raw=None, masks=None, origin=(0, 0), transform=TRANSFORM, full_shape=None):
    raw = _raw() if raw is None else raw
    masks = {name: np.full(value.shape, 255, dtype=np.uint8) for name, value in raw.items()} if masks is None else masks
    return prepare_arrays(raw, masks, dict.fromkeys(raw), configured_nodata=None,
        crs="EPSG:32651", transform=transform, input_hashes=dict.fromkeys(raw, "a" * 64),
        interpretation={"fixture": "synthetic arrays, no reader capability claim"},
        global_origin=origin, full_grid_shape=full_shape)


class _Recorder:
    def predict_proba(self, features):
        self.features = np.array(features, copy=True)
        return np.full((len(features), 2), .5, dtype=np.float64)


def _paired(context, environment, state):
    lr = context.lr
    kernel = lr.engine.FireAutomata._compute_wind_kernel(SimpleNamespace(
        wind_cfg=dict(WIND), transition_cfg={"wind_weight": .20}))
    assembler = lr.feature.FeatureAssembler(environment, dict(WIND))
    recorder = _Recorder()
    harness = SimpleNamespace(grid=state, wind_kernel=kernel, grid_shape=state.shape,
        feature_assembler=assembler, model=recorder, ml_cfg={"proba_threshold": 0.0})
    lr.engine.FireAutomata._predict_with_model(harness)
    count = compute_moore_neighbor_count((state == 4).astype(np.int8))
    score, rf_kernel = compute_wind_weighted_score((state == 4).astype(np.int8), WindContract(17.7, 315), .20)
    rf_features = FeatureAssembler(environment, dict(WIND)).assemble_grid_features(count, score)
    lr_features = recorder.features
    records = [_float_record(context, "kernel", rf_kernel, kernel),
               _float_record(context, "score", score, lr_features[:, 10].reshape(state.shape)),
               _float_record(context, "eleven_features", rf_features, lr_features)]
    context.evidence["recording_estimator_only"] = True
    context.evidence["counts"] = dict(rf=count.tolist(), lr=lr_features[:, 8].reshape(state.shape).tolist())
    assert tuple(CANONICAL_FEATURE_NAMES) == tuple(assembler.feature_names) == NAMES
    assert rf_features.shape == lr_features.shape == (state.size, 11)
    assert rf_features.dtype == lr_features.dtype == np.float32
    np.testing.assert_array_equal(count.ravel(), lr_features[:, 8])
    for column in (2, 4, 8):
        np.testing.assert_array_equal(rf_features[:, column], lr_features[:, column])
    _exact(records)
    return count, score, rf_kernel, rf_features


@pytest.mark.parametrize("target", [(r, c) for r in range(3) for c in range(3)])
@pytest.mark.parametrize("pattern", [*range(8), "zero", "all"])
def test_neighbourhoods(context, target, pattern):
    state = np.full((3, 3), 2, dtype=np.int8)
    offsets = OFFSETS if pattern == "all" else () if pattern == "zero" else (OFFSETS[pattern],)
    sources = [(target[0] + dr, target[1] + dc) for dr, dc in offsets
               if 0 <= target[0] + dr < 3 and 0 <= target[1] + dc < 3]
    for cell in sources:
        state[cell] = 4
    context.evidence["pattern"] = dict(target=list(target), requested=str(pattern),
        in_bounds_sources=[list(cell) for cell in sources], out_of_grid_sources_absent=True)
    count, score, kernel, _ = _paired(context, _prepare().environment, state)
    expected = sum(kernel[r - target[0] + 1, c - target[1] + 1] for r, c in sources) / kernel.sum(dtype=np.float32)
    assert count[target] == len(sources)
    if not sources:
        assert score[target] == 0
    elif len(sources) == 1:
        assert score[target].view(np.uint32) == np.float32(expected).view(np.uint32)
    assert kernel[0, 0] > kernel[2, 2] and kernel[1, 1] == 0


def _legacy(context, raw, transform=TRANSFORM):
    manager = context.lr.loader.EnvironmentManager(dict(raster_dir="synthetic-only",
        slope_file="slope.tif", proximity_file="proximity.tif", buildings_file="buildings.tif",
        materials_file="materials.tif", nodata_value=-9999))
    for name, value in raw.items():
        setattr(manager, name + "_raw", value.astype(np.float32))
    manager.grid_shape, manager.crs, manager.transform = raw["slope"].shape, CRS.from_epsg(32651), transform
    manager.build_masks()
    manager.normalize_layers()
    return manager.get_environment()


def test_material_and_static_representations(context):
    raw = _raw((2, 3))
    raw["materials"][:] = np.arange(6).reshape(2, 3)
    raw["buildings"][0, 0] = 0
    raw["slope"][:] = [[0, -1, 11], [10, 5, 1]]
    environment = _prepare(raw).environment
    legacy = _legacy(context, raw)
    state = np.full((2, 3), 2, dtype=np.int8)
    _, _, _, features = _paired(context, environment, state)
    records = [_float_record(context, "material_mapping", FeatureAssembler.MATERIAL_CLASS_TO_RISK,
                             context.lr.feature.FeatureAssembler.MATERIAL_CLASS_TO_RISK),
               _float_record(context, "declared_mapping", FeatureAssembler.MATERIAL_CLASS_TO_RISK, RISKS),
               _float_record(context, "static_slope", environment["slope_risk"], legacy["slope_risk"]),
               _float_record(context, "static_proximity", environment["proximity_risk"], legacy["proximity_risk"])]
    _exact(records)
    np.testing.assert_array_equal(features[:, 4], np.arange(6))
    np.testing.assert_array_equal(features[:, 3], RISKS)
    np.testing.assert_array_equal(features[:, 9], features[:, 2] * features[:, 3])


def _engines(context, rf_environment, lr_environment):
    rf = dict(rf_environment)
    rf["material_risk"] = RISKS[rf["material_class"]]
    config = dict(simulation={"seed": 0, "emergency_checkpoint_enabled": False}, wind=dict(WIND),
        placeholder_transition={"wind_weight": .20, "T_3_to_4": 1, "T_4_to_5": 1},
        ml_model={"inference_mode": "stochastic_probability", "threshold": None})
    return RFAutomata(rf, config, model_free=True), context.lr.engine.FireAutomata(lr_environment, deepcopy(config))


@pytest.mark.parametrize("difference", ["zero_background", "native_mask", "nonfinite", "raw_categorical"])
def test_population_and_initialization_characterizations(context, difference):
    raw = _raw()
    masks = {name: np.full((3, 3), 255, dtype=np.uint8) for name in raw}
    if difference == "zero_background":
        for value in raw.values():
            value[0, 0] = 0
        raw["buildings"][0, 1] = raw["materials"][0, 1] = 0
    elif difference == "native_mask":
        masks["slope"][1, 1] = 0
    elif difference == "nonfinite":
        raw["slope"][1, 1] = np.nan
    else:
        raw["materials"][1, 1] = 1.00000000001
        legacy = _legacy(context, raw)
        _characterize(context, difference, original_value=repr(raw["materials"][1, 1]),
                      lr_class=int(legacy["material_class"][1, 1]), rf_requires_raw_rejection=True)
        with pytest.raises(ValueError, match="raw material code"):
            _prepare(raw, masks)
        assert legacy["material_class"][1, 1] == 1
        return
    prepared = _prepare(raw, masks)
    legacy = _legacy(context, raw)
    rf, lr = _engines(context, prepared.environment, legacy)
    _characterize(context, difference, rf_environment=prepared.environment["burnable_mask"].tolist(),
        lr_environment=legacy["burnable_mask"].tolist(), rf_initial=rf.grid.tolist(), lr_initial=lr.grid.tolist(),
        rf_receiving=prepared.receiving_mask.tolist(), lr_initial_receiving=(lr.grid == 2).tolist())
    assert rf.grid.dtype == lr.grid.dtype == np.int8
    assert rf.ignition_timers.dtype == lr.ignition_timers.dtype == np.int16
    if difference == "zero_background":
        assert prepared.environment["burnable_mask"][0, 0] and not legacy["burnable_mask"][0, 0]
        assert rf.grid[0, 1] == 1 and lr.grid[0, 1] == 2
    else:
        assert not prepared.receiving_mask[1, 1] and legacy["burnable_mask"][1, 1]
        assert rf.grid[1, 1] == 1 and lr.grid[1, 1] == 2


def test_masked_helper_characterization(context):
    raw = _raw()
    raw["buildings"][0, 0] = raw["materials"][0, 0] = 0
    environment = _prepare(raw).environment
    state = np.full((3, 3), 2, dtype=np.int8)
    state[1, 1] = 4
    count, score, _, _ = _paired(context, environment, state)
    rf_values, rf_indices = FeatureAssembler(environment, dict(WIND)).assemble_masked_features(count, score)
    lr_values, lr_indices = context.lr.feature.FeatureAssembler(environment, dict(WIND)).assemble_masked_features(count)
    context.evidence["comparisons"].append(dict(name="masked_helper", rf=_array(rf_values), lr=_array(lr_values)))
    _characterize(context, "masked_helper_score_and_population", rf_indices=rf_indices.tolist(),
                  lr_indices=lr_indices.tolist(), lr_defaults_directional_score_to_zero=True)
    assert 0 not in rf_indices and 0 in lr_indices
    assert np.all(lr_values[:, 10] == 0) and np.any(rf_values[:, 10] > 0)


@pytest.mark.parametrize("state_code", [3, 5])
def test_nonblazing_sources(context, state_code):
    state = np.full((3, 3), 2, dtype=np.int8)
    state[0, 0] = state_code
    count, score, _, _ = _paired(context, _prepare().environment, state)
    assert not np.any(count) and not np.any(score)


@pytest.mark.parametrize("case", ["origin_and_crop", "boundary", "atomic_seeds"])
def test_coordinate_characterizations(context, case):
    lr_convert = context.lr.orchestrator._convert_geographic_ignition_points
    if case == "origin_and_crop":
        origin = (1, 2)
        local = TRANSFORM * Affine.translation(origin[1], origin[0])
        prepared = _prepare(origin=origin, transform=local, full_shape=(5, 6))
        harness = SimpleNamespace(grid_shape=(3, 3), transform=local)
        xy = list(local * (1.5, 1.5))
        assert validate_seeds([xy], prepared, "projected_xy") == ((1, 1),)
        assert validate_seeds([[2, 3]], prepared, "grid_row_col") == ((1, 1),)
        assert lr_convert(harness, [xy]) == [(1, 1)]
        full = dict(_prepare(_raw((5, 6))).environment)
        full["material_risk"] = RISKS[full["material_class"]]
        cropped = context.lr.generator.SyntheticDatasetGenerator._crop_environment(None, full,
            dict(row_start=1, row_end=4, col_start=2, col_end=5))
        _characterize(context, "crop_transform", expected_local=list(local),
                      lr_crop_transform=list(cropped["transform"]), global_origin=list(origin))
        assert cropped["transform"] == TRANSFORM and cropped["transform"] != local
    elif case == "boundary":
        prepared = _prepare()
        harness = SimpleNamespace(grid_shape=(3, 3), transform=TRANSFORM)
        for pixel, expected in (((.5, .5), (0, 0)), ((1., 1.), (1, 1)), ((2.5, 2.5), (2, 2))):
            xy = list(TRANSFORM * pixel)
            assert validate_seeds([xy], prepared, "projected_xy") == (expected,)
            assert lr_convert(harness, [xy]) == [expected]
        outside = list(TRANSFORM * (3., 1.))
        with pytest.raises(ValueError):
            validate_seeds([outside], prepared, "projected_xy")
        with pytest.raises(ValueError):
            lr_convert(harness, [outside])
    else:
        raw = _raw()
        raw["buildings"][0, 1] = raw["materials"][0, 1] = 0
        prepared = _prepare(raw)
        rf, lr = _engines(context, prepared.environment, _legacy(context, raw))
        xy = list(TRANSFORM * (1.5, 1.5))
        before = rf.grid.copy()
        with pytest.raises(ValueError, match="Duplicate"):
            validate_seeds([xy, xy], prepared, "projected_xy")
        converted = lr_convert(lr, [xy, xy])
        assert converted == [(1, 1), (1, 1)]
        lr.set_ignition(converted)
        background = list(TRANSFORM * (1.5, .5))
        with pytest.raises(ValueError, match="receiving"):
            validate_seeds([background], prepared, "projected_xy")
        lr.set_ignition(lr_convert(lr, [background]))
        _characterize(context, "seed_duplicates_and_background", lr_duplicates=converted,
                      lr_background_state=int(lr.grid[0, 1]), rf_unchanged=np.array_equal(rf.grid, before))
        assert lr.grid[0, 1] == 4
        np.testing.assert_array_equal(rf.grid, before)


class _BeforeSplit(Exception):
    pass


def _csv_path(context, name):
    text = os.environ.get("CANDIDATE_RF_LR_VERIFICATION_ROOT")
    assert text and not text.replace("/", "\\").startswith("\\\\")
    root = Path(text)
    base_text = context.request.config.getoption("basetemp")
    assert root.is_absolute() and root.is_dir() and base_text
    base = Path(base_text)
    assert base.is_absolute() and base.resolve() != root.resolve() and base.resolve().is_relative_to(root.resolve())
    assert not root.resolve().is_relative_to(REPO) and not REPO.is_relative_to(root.resolve())
    for path in (base, *base.parents):
        assert not path.is_symlink() and not path.is_junction()
    folder = context.request.getfixturevalue("tmp_path")
    assert folder.resolve().is_relative_to(root.resolve())
    return folder / name


def _native_frame(frame):
    return dict(columns=list(frame.columns), index=frame.index.tolist(),
        dtypes={name: str(dtype) for name, dtype in frame.dtypes.items()},
        columns_native={name: _array(frame[name].to_numpy(copy=True)) for name in frame.columns})


def _stage_prefix(context, path):
    stage = context.lr.stage
    holder, forbidden = {}, []
    read_csv, builtin_open, path_open = stage.pd.read_csv, builtins.open, Path.open
    tracked = [(stage, name) for name in ("CSV_PATH", "train_test_split", "Pipeline", "LogisticRegression", "StandardScaler")]
    tracked += [(cls, "fit") for cls in (stage.Pipeline, stage.LogisticRegression, stage.StandardScaler)]
    tracked += [(stage.pd, "read_csv"), (stage.joblib, "dump"), (Path, "mkdir"),
                (Path, "open"), (builtins, "open")]
    originals = [(owner, name, getattr(owner, name)) for owner, name in tracked]
    def reject(*args, **kwargs):
        forbidden.append("training/construction/output")
        raise AssertionError("Stage C crossed the pre-split compatibility boundary")
    def read_only_csv(value, *args, **kwargs):
        assert Path(value).resolve() == path.resolve(), "Unexpected CSV access"
        return read_csv(value, *args, **kwargs)
    def capture(X, y, *args, **kwargs):
        assert not holder, "More than one split attempt"
        holder.update(X_native=X.copy(deep=True), y_native=y.copy(deep=True),
                      native=_native_frame(X), native_y=_array(y.to_numpy(copy=True)))
        context.evidence["csv_native"] = holder["native"]
        context.evidence["csv_native_y"] = holder["native_y"]
        raise _BeforeSplit("Captured native X/y; no split or estimator construction")
    def no_write_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            reject()
        return builtin_open(file, mode, *args, **kwargs)
    def no_write_path(self, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            reject()
        return path_open(self, mode, *args, **kwargs)
    try:
        with context.monkeypatch.context() as guard:
            guard.setattr(stage, "CSV_PATH", path)
            guard.setattr(stage.pd, "read_csv", read_only_csv)
            guard.setattr(stage, "train_test_split", capture)
            for cls in (stage.Pipeline, stage.LogisticRegression, stage.StandardScaler):
                guard.setattr(cls, "fit", reject)
            for name in ("Pipeline", "LogisticRegression", "StandardScaler"):
                guard.setattr(stage, name, reject)
            guard.setattr(stage.joblib, "dump", reject)
            guard.setattr(Path, "mkdir", reject)
            guard.setattr(Path, "open", no_write_path)
            guard.setattr(builtins, "open", no_write_open)
            with pytest.raises(_BeforeSplit):
                stage.main()
    finally:
        context.evidence["stage_guard_restoration"] = [dict(attribute=name,
            restored=getattr(owner, name) is original) for owner, name, original in originals]
        context.evidence["stage_forbidden_calls"] = forbidden
        if sys.exc_info()[0] is None:
            assert all(row["restored"] for row in context.evidence["stage_guard_restoration"])
    context.evidence["csv_native"] = holder.get("native")
    context.evidence["csv_native_y"] = holder.get("native_y")
    context.evidence["stage_forbidden_calls"] = forbidden
    assert holder and not forbidden
    assert stage.pd.read_csv is read_csv and builtins.open is builtin_open and Path.open is path_open
    return holder


@pytest.mark.parametrize("kind", ["compliant", "zero_neighbour_sentinel"])
def test_csv_stage_c_loading_without_training(context, kind):
    state = np.full((3, 3), 2, dtype=np.int8)
    state[1, 1] = 4
    _, _, _, grid = _paired(context, _prepare().environment, state)
    features = grid[[0, 1, 2, 3]].copy()
    features[0, 1] = np.float32(-0.0)
    labels = [0, 1, 0, 1]
    if kind == "zero_neighbour_sentinel":
        sentinel = features[:1].copy()
        sentinel[0, 8] = sentinel[0, 10] = 0
        features = np.vstack([features, sentinel])
        labels.append(0)
    payload = release.HEADER + b"".join(release.row_bytes(row, label) for row, label in zip(features, labels, strict=True))
    path = _csv_path(context, "shared-" + kind + ".csv")
    with path.open("xb") as stream:
        assert stream.write(payload) == len(payload)
    assert path.read_bytes() == payload
    context.evidence["csv_fixture"] = dict(path=str(path), bytes=len(payload), sha256=_hash(path),
        synthetic_labels_only=True, negative_zero_serialization_stress=True,
        zero_neighbour_row_is_not_release_population=kind != "compliant")
    decoded = [release.decode_row(line.decode("ascii"))[0] for line in payload.splitlines(keepends=True)[1:]]
    serialization = _float_record(context, "canonical_decode", features, np.stack(decoded))
    holder = _stage_prefix(context, path)
    X, y = holder["X_native"], holder["y_native"]
    recovered = X.to_numpy(dtype=np.float32, copy=True)
    recovery = _float_record(context, "stage_float32_recovery", features[:4], recovered)
    _characterize(context, "stage_csv_dtypes_and_filter", native_dtypes=holder["native"]["dtypes"],
                  supplied_rows=len(features), retained_rows=len(X), indices=X.index.tolist(),
                  native_float32_conversion_absent=True, frozen_role_accommodation_pending=True)
    assert tuple(X.columns) == NAMES and str(y.dtype) == "int8" and X.index.tolist() == [0, 1, 2, 3]
    np.testing.assert_array_equal(y.to_numpy(), labels[:4])
    assert _native_frame(X) == holder["native"] and _array(y.to_numpy()) == holder["native_y"]
    assert _hash(path) == context.evidence["csv_fixture"]["sha256"]
    _exact([serialization, recovery])


def test_namespace_origins_and_failure_restoration(context):
    saved, path_object, paths = _namespace(), sys.path, list(sys.path)
    importer = importlib.import_module
    class InjectedImportFailure(Exception):
        pass
    def fail_engine(name, *args, **kwargs):
        if name == "modules.automata_engine":
            raise InjectedImportFailure("bounded import failure")
        return importer(name, *args, **kwargs)
    with context.monkeypatch.context() as guard:
        guard.setattr(importlib, "import_module", fail_engine)
        with pytest.raises(InjectedImportFailure, match="bounded import failure"):
            with _lr_namespace(context.evidence):
                pytest.fail("Injected import failure was not reached")
    assert _same(saved, _namespace()) and sys.path is path_object and sys.path == paths
    assert context.lr.stage.__name__ != "__main__"
    assert Path(context.lr.engine.__file__).resolve() == LR / "modules/automata_engine.py"
    assert context.lr.engine.FeatureAssembler is not FeatureAssembler
