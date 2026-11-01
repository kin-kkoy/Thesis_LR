"""Tiny-array viewer contracts; no production raster access or display."""
from copy import deepcopy
from contextlib import nullcontext

import numpy as np
import pytest
from matplotlib.figure import Figure
from rasterio.crs import CRS
from rasterio.transform import Affine

import final_results as viewer


def tiny_arrays():
    return {
        "prediction": np.array([[5, 5, 2], [2, 2, 2]], dtype=np.int8),
        "reference": np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int16),
        "buildings": np.array([[10, 10, 10], [10, 10, 0]], dtype=np.float32),
        "environment": np.array([[1, 1, 1], [1, 0, 1]], dtype=np.uint8),
        **{name + "_valid": np.ones((2, 3), dtype=bool)
           for name in ("prediction", "reference", "buildings", "environment")},
    }


def test_confusion_categories_and_disjoint_exclusions():
    arrays = tiny_arrays()
    before = {name: value.copy() for name, value in arrays.items()}
    result = viewer.evaluate_arrays(arrays)
    m = result["metrics"]
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["accuracy"] == 0.5
    assert m["evaluation_pixel_count"] == 4
    assert m["predicted_burned_cells"] == 2
    assert m["excluded_percentage"] == 50
    assert result["difference"].tolist() == [[2, 3, 4], [1, 0, 0]]
    c = m["reference_coverage"]
    assert c["reference_valid_burned_total"] == 4
    assert c["reference_burned_included_in_primary_domain"] == 2
    assert c["reference_burned_excluded_from_primary_domain"] == 2
    assert c["exclusion_priority_partition"] == {
        "outside_environmental_validity": 1, "outside_prediction_validity": 0,
        "outside_building_raster_validity": 0, "outside_mapped_buildings": 1,
        "included_in_primary_domain": 2,
    }
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], before[name])


def test_validity_intersection_and_unknown_reference():
    arrays = tiny_arrays()
    arrays["prediction_valid"][0, 0] = False
    arrays["buildings_valid"][0, 2] = False
    arrays["reference_valid"][1, 2] = False
    result = viewer.evaluate_arrays(arrays)
    assert result["domain"].tolist() == [[False, True, False], [True, False, False]]
    assert result["metrics"]["confusion_matrix"] == [[1, 1], [0, 0]]
    partition = result["metrics"]["reference_coverage"]["exclusion_priority_partition"]
    assert list(partition.values()) == [1, 1, 1, 0, 0]
    assert result["difference"][0, 0] == 0  # Excluded TP is not TN.
    assert result["difference"][1, 0] == 1


@pytest.mark.parametrize("name,value", [
    ("prediction", 3), ("prediction", 4), ("prediction", 0),
    ("reference", 0.5), ("buildings", 10.5), ("environment", 2),
    ("reference", np.inf),
])
def test_bad_values_rejected_without_narrowing(name, value):
    arrays = tiny_arrays()
    arrays[name] = arrays[name].astype(np.float64)
    arrays[name][0, 0] = value
    with pytest.raises(ValueError):
        viewer.evaluate_arrays(arrays)


@pytest.mark.parametrize("fault", ["shape", "mask_dtype", "environment_invalid", "empty_domain"])
def test_mask_and_population_rejection(fault):
    arrays = tiny_arrays()
    if fault == "shape":
        arrays["reference"] = arrays["reference"][:, :2]
    elif fault == "mask_dtype":
        arrays["reference_valid"] = arrays["reference_valid"].astype(np.uint8)
    elif fault == "environment_invalid":
        arrays["environment_valid"][0, 0] = False
    else:
        arrays["buildings"][:] = 0
    with pytest.raises(ValueError):
        viewer.evaluate_arrays(arrays)


@pytest.mark.parametrize("fault", ["shape", "crs", "transform"])
def test_geometry_mismatch_rejected(fault):
    shape, crs, transform = viewer.CROP_SHAPE, CRS.from_epsg(32651), viewer.CROP_TRANSFORM
    if fault == "shape":
        shape = viewer.PARENT_SHAPE
    elif fault == "crs":
        crs = CRS.from_epsg(4326)
    else:
        transform = Affine.translation(1, 0) * transform
    with pytest.raises(ValueError, match="Geometry"):
        viewer.validate_geometry(shape, crs, transform, parent=False)


def test_exact_parent_and_crop_geometry():
    viewer.validate_geometry(viewer.PARENT_SHAPE, CRS.from_epsg(32651),
                             viewer.PARENT_TRANSFORM, parent=True)
    viewer.validate_geometry(viewer.CROP_SHAPE, CRS.from_epsg(32651),
                             viewer.CROP_TRANSFORM, parent=False)
    assert viewer.PARENT_TRANSFORM * Affine.translation(2176, 256) == viewer.CROP_TRANSFORM


def test_display_extent_and_zoom_do_not_change_metrics():
    result = viewer.evaluate_arrays(tiny_arrays())
    before = deepcopy(result["metrics"])
    figure = Figure()
    extent = (100, 106, 194, 200)
    viewer.draw_figure(figure, result, extent)
    assert len(figure.axes) == 3
    for ax in figure.axes:
        assert tuple(ax.images[0].get_extent()) == extent
        assert ax.images[0].get_interpolation() == "nearest"
        assert ax.images[0].origin == "upper"
        assert ax.get_aspect() == 1
        ax.set_xlim(101, 102)
        ax.set_ylim(195, 196)
    np.testing.assert_array_equal(figure.axes[2].images[0].get_array(), result["difference"])
    assert result["metrics"] == before
    assert "50.00%" in figure.texts[0].get_text()
    figure.clear()


def test_metrics_are_printed_before_plot_and_disagreement_blocks_display(monkeypatch, capsys):
    arrays = tiny_arrays()
    monkeypatch.setattr(
        viewer, "load_inputs",
        lambda *args, **kwargs: (arrays, synthetic_identities("original315")))
    def forbidden(*args, **kwargs):
        raise AssertionError("Disagreeing case must not display")
    monkeypatch.setattr(viewer, "draw_figure", forbidden)
    with pytest.raises(ValueError, match="DISAGREEMENT"):
        viewer.main([])
    output = capsys.readouterr().out
    assert "[[1, 1], [1, 1]]" in output
    assert "Accuracy: 0.500000" in output
    assert "Spatial ROC/AUC is not reported" in output


def test_identity_rejection_precedes_any_raster_open(monkeypatch):
    class FakePath:
        def is_symlink(self): return False
        def is_junction(self): return False
        def is_file(self): return True
    monkeypatch.setattr(viewer.rasterio, "__gdal_version__", "3.12.1")
    monkeypatch.setattr(viewer, "Path", lambda value: FakePath())
    # Constructing the environment path also uses Path; supply its parent operator.
    FakePath.parent = property(lambda self: self)
    FakePath.__truediv__ = lambda self, value: self
    monkeypatch.setattr(viewer, "file_sha256", lambda path: "0" * 64)
    def forbidden(*args, **kwargs):
        raise AssertionError("Identity failure must precede native interpretation")
    monkeypatch.setattr(viewer.rasterio, "open", forbidden)
    with pytest.raises(ValueError, match="identity mismatch"):
        viewer.load_inputs()


def test_bad_building_category_is_not_hidden_by_other_layer_invalidity():
    arrays = tiny_arrays()
    arrays["buildings"][1, 1] = 10.5
    assert arrays["environment"][1, 1] == 0
    with pytest.raises(ValueError, match="buildings category"):
        viewer.evaluate_arrays(arrays)


def synthetic_identities(case_id):
    identities = {
        name: {"dtype": dtype, "nodata": None, "mask_flags": ["all_valid"]}
        for name, dtype in (
            ("prediction", "int8"), ("reference", "int16"),
            ("buildings", "float32"), ("environment", "uint8"))}
    digest, case = next(
        (digest, case) for digest, case in viewer.PREDICTION_CASES.items()
        if case["case_id"] == case_id)
    identities["prediction"].update(case)
    identities["prediction"]["sha256"] = digest
    return identities


def comparison_pair():
    original = tiny_arrays()
    original["prediction"][1, 1:] = 1
    selected = deepcopy(original)
    selected["prediction"][0, 0] = 2
    selected["prediction"][1, 0] = 5
    return original, selected


def test_explicit_path_defaults_and_overrides():
    defaults = viewer.parse_args([])
    assert defaults.final_state == viewer.FINAL_STATE_PATH
    assert defaults.ground_truth == viewer.GROUND_TRUTH_PATH
    assert defaults.environment is None
    assert defaults.baseline_final_state is None
    args = viewer.parse_args([
        "--final-state", "new.tif", "--ground-truth", "reference.tif",
        "--environment", "mask.tif", "--buildings", "buildings.tif",
        "--baseline-final-state", "original.tif"])
    assert str(args.final_state) == "new.tif"
    assert str(args.ground_truth) == "reference.tif"
    assert str(args.environment) == "mask.tif"
    assert str(args.buildings) == "buildings.tif"
    assert str(args.baseline_final_state) == "original.tif"


def test_new_pin_and_selected_sibling_are_checked_before_native_read(monkeypatch):
    prediction = viewer.CODE_DIR / "synthetic-new/final_state.tif"
    reference = viewer.CODE_DIR / "synthetic-reference.tif"
    sibling = prediction.parent / "simulation_environment_valid.tif"
    new_pin = synthetic_identities("nasa49")["prediction"]["sha256"]
    expected = {
        reference: viewer.INPUT_SHA256["reference"],
        prediction: new_pin,
        viewer.BUILDINGS_PATH: viewer.INPUT_SHA256["buildings"],
        sibling: viewer.INPUT_SHA256["environment"],
    }
    observed_paths = []
    monkeypatch.setattr(viewer.Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(viewer.Path, "is_junction", lambda self: False)
    monkeypatch.setattr(viewer.Path, "is_file", lambda self: True)
    monkeypatch.setattr(viewer.Path, "resolve", lambda self: self)
    monkeypatch.setattr(viewer.rasterio, "__gdal_version__", "3.12.1")
    monkeypatch.setattr(viewer.rasterio, "Env", lambda **kwargs: nullcontext())
    monkeypatch.setattr(viewer, "check_controls", lambda: None)
    def fake_hash(path):
        observed_paths.append(path)
        return expected[path]
    def stop_at_open(*args, **kwargs):
        raise RuntimeError("synthetic native-read boundary")
    monkeypatch.setattr(viewer, "file_sha256", fake_hash)
    monkeypatch.setattr(viewer.rasterio, "open", stop_at_open)
    with pytest.raises(RuntimeError, match="synthetic native-read boundary"):
        viewer.load_inputs(prediction, reference)
    assert set(observed_paths) == set(expected)


def test_new_selection_does_not_implicitly_open_original(monkeypatch):
    calls = []
    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        return tiny_arrays(), synthetic_identities("nasa49")
    def forbidden(*args, **kwargs):
        raise AssertionError("No scoring before explicit baseline")
    monkeypatch.setattr(viewer, "load_inputs", fake_load)
    monkeypatch.setattr(viewer, "evaluate_arrays", forbidden)
    with pytest.raises(ValueError, match="explicit --baseline-final-state"):
        viewer.evaluate_selected(final_state="new.tif")
    assert len(calls) == 1


@pytest.mark.parametrize("baseline_ok", [False, True])
def test_original_gate_precedes_new_scoring_without_new_expectations(
        monkeypatch, baseline_ok):
    original, selected = comparison_pair()
    evaluated, checked = [], []
    def fake_load(final_state, *args, **kwargs):
        return ((original, synthetic_identities("original315"))
                if final_state == "original.tif"
                else (selected, synthetic_identities("nasa49")))
    real_evaluate = viewer.evaluate_arrays
    def evaluate(arrays):
        evaluated.append(arrays)
        return real_evaluate(arrays)
    def check(result):
        checked.append(result)
        return [] if baseline_ok else ["synthetic baseline mismatch"]
    monkeypatch.setattr(viewer, "load_inputs", fake_load)
    monkeypatch.setattr(viewer, "evaluate_arrays", evaluate)
    monkeypatch.setattr(viewer, "accepted_disagreements", check)
    if not baseline_ok:
        with pytest.raises(ValueError, match="DISAGREEMENT"):
            viewer.evaluate_selected(
                "new.tif", baseline_final_state="original.tif")
        assert len(evaluated) == 1 and evaluated[0] is original
    else:
        result, comparison, report = viewer.evaluate_selected(
            "new.tif", baseline_final_state="original.tif")
        assert len(evaluated) == 2
        assert evaluated[0] is original and evaluated[1] is selected
        assert result["metrics"] != checked[0]["metrics"]
        assert report["new"]["metrics"] == result["metrics"]
        assert "new_minus_original_metrics" in comparison["summary"]
    assert len(checked) == 1


@pytest.mark.parametrize("fault", [
    "environment", "eligibility", "reference_validity", "nodata"])
def test_comparison_rejects_unexplained_domain_or_metadata_changes(fault):
    original, selected = comparison_pair()
    original_ids = synthetic_identities("original315")
    selected_ids = synthetic_identities("nasa49")
    if fault == "environment":
        selected["environment"][1, 1] = 1
    elif fault == "eligibility":
        selected["prediction"][1, 1] = 2
    elif fault == "reference_validity":
        selected["reference_valid"][0, 0] = False
    else:
        selected_ids["reference"]["nodata"] = -9999
    with pytest.raises(ValueError):
        viewer.compare_arrays(original, selected, original_ids, selected_ids)


def test_symmetric_comparison_and_empty_burn_union():
    original, selected = comparison_pair()
    ids = (synthetic_identities("original315"), synthetic_identities("nasa49"))
    comparison = viewer.compare_arrays(original, selected, *ids)
    summary = comparison["summary"]
    assert summary["original_only"] == summary["new_only"] == 1
    assert summary["both_burned"] == 1
    assert summary["symmetric_difference"] == 2
    assert sum(map(sum, summary[
        "state_cross_table_full_crop_rows_original_columns_new_codes_1_to_5"])) == 6
    assert summary["state_cross_table_scope"] == "full_crop"
    assert summary["state_counts_scope"] == "full_crop"
    assert summary["eligible_building_cells"] == summary["reference_evaluation_cells"] == 4
    assert summary["state_cross_table_full_crop_rows_original_columns_new_codes_1_to_5"][0][0] == 2
    assert all(sum(counts.values()) == 6 for counts in summary["state_counts"].values())
    for arrays in (original, selected):
        arrays["prediction"][arrays["prediction"] != 1] = 2
    empty = viewer.compare_arrays(original, selected, *ids)["summary"]
    assert empty["empty_union"] is True
    assert empty["symmetric_iou"] is None
    assert empty["symmetric_dice"] is None


def test_comparison_plot_uses_descriptive_labels_without_changing_counts():
    original, selected = comparison_pair()
    comparison = viewer.compare_arrays(
        original, selected, synthetic_identities("original315"),
        synthetic_identities("nasa49"))
    before = deepcopy(comparison["summary"])
    figure = Figure()
    viewer.draw_figure(figure, comparison, (100, 106, 194, 200), comparison=True)
    assert len(figure.axes) == 3
    assert "315" in figure.axes[0].get_title()
    assert "48.825208" in figure.axes[1].get_title()
    labels = [text.get_text()
              for text in figure.axes[2].get_legend().get_texts()]
    assert labels == ["Excluded", "Neither", "Both", "New only", "Original only"]
    assert comparison["summary"] == before
    figure.clear()


def test_wrong_case_baseline_stops_before_any_scoring(monkeypatch):
    monkeypatch.setattr(
        viewer, "load_inputs",
        lambda *args, **kwargs: (tiny_arrays(), synthetic_identities("nasa49")))
    def forbidden(*args, **kwargs):
        raise AssertionError("Wrong-case baseline must not be scored")
    monkeypatch.setattr(viewer, "evaluate_arrays", forbidden)
    with pytest.raises(ValueError, match="Baseline must be the identified original-315"):
        viewer.evaluate_selected("new.tif", baseline_final_state="wrong.tif")


def test_cli_ancillary_overrides_reach_selected_loader(monkeypatch):
    args = viewer.parse_args([
        "--final-state", "new.tif", "--ground-truth", "reference.tif",
        "--environment", "mask.tif", "--buildings", "buildings.tif"])
    calls = []
    def stop_at_loader(*positional, **keywords):
        calls.append((positional, keywords))
        raise RuntimeError("synthetic selected-loader boundary")
    monkeypatch.setattr(viewer, "load_inputs", stop_at_loader)
    with pytest.raises(RuntimeError, match="synthetic selected-loader boundary"):
        viewer.evaluate_selected(**vars(args))
    assert calls == [((args.final_state, args.ground_truth), {
        "environment": args.environment, "buildings": args.buildings})]


def synthetic_native_loader(monkeypatch, *, changed=None, ancillary_overrides=False):
    root = viewer.CODE_DIR / "synthetic-loader"
    paths = dict(reference=root / "reference.tif", prediction=root / "final_state.tif",
                 buildings=root / "buildings.tif",
                 environment=root / "simulation_environment_valid.tif")
    if ancillary_overrides:
        paths["environment"] = root / "explicit-mask.tif"
        paths["buildings"] = root / "explicit-buildings.tif"
    raw = tiny_arrays()
    names = {path: name for name, path in paths.items()}
    hash_calls = {name: 0 for name in paths}
    opened = []
    transform = Affine(3, 0, 100, 0, -3, 200)
    monkeypatch.setattr(viewer, "CROP_SHAPE", (2, 3))
    monkeypatch.setattr(viewer, "PARENT_SHAPE", (2, 3))
    monkeypatch.setattr(viewer, "CROP_TRANSFORM", transform)
    monkeypatch.setattr(viewer, "PARENT_TRANSFORM", transform)
    monkeypatch.setattr(viewer, "CROP_WINDOW", viewer.Window(0, 0, 3, 2))
    monkeypatch.setattr(viewer, "BUILDINGS_PATH", paths["buildings"])
    monkeypatch.setattr(viewer.Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(viewer.Path, "is_junction", lambda self: False)
    monkeypatch.setattr(viewer.Path, "is_file", lambda self: True)
    monkeypatch.setattr(viewer.Path, "resolve", lambda self: self)
    monkeypatch.setattr(viewer.rasterio, "__gdal_version__", "3.12.1")
    monkeypatch.setattr(viewer.rasterio, "Env", lambda **kwargs: nullcontext())
    monkeypatch.setattr(viewer, "check_controls", lambda: None)
    def fake_hash(path):
        name = names[path]
        hash_calls[name] += 1
        return ("0" * 64 if name == changed and hash_calls[name] > 1
                else viewer.INPUT_SHA256[name])
    class Flag:
        name = "all_valid"
    class NativeRaster:
        driver, mode, count, subdatasets, nodata = "GTiff", "r", 1, (), None
        shape, crs = (2, 3), CRS.from_epsg(32651)
        options = {"GEOREF_SOURCES": "INTERNAL"}
        mask_flag_enums = ((Flag(),),)
        def __init__(self, path):
            self.name = names[path]
            self.files = [str(path)]
            self.transform = transform
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def window_transform(self, window):
            assert window == viewer.CROP_WINDOW
            return transform
        def read(self, band, **options):
            assert band == 1
            assert options == ({"window": viewer.CROP_WINDOW}
                               if self.name in ("reference", "buildings") else {})
            return raw[self.name].copy()
        def read_masks(self, band, **options):
            assert band == 1
            assert options == ({"window": viewer.CROP_WINDOW}
                               if self.name in ("reference", "buildings") else {})
            return np.full((2, 3), 255, dtype=np.uint8)
    def fake_open(path, **options):
        assert options == {"mode": "r", "driver": "GTiff", "sharing": False,
                           "GEOREF_SOURCES": "INTERNAL"}
        opened.append(names[path])
        return NativeRaster(path)
    monkeypatch.setattr(viewer, "file_sha256", fake_hash)
    monkeypatch.setattr(viewer.rasterio, "open", fake_open)
    return paths, hash_calls, opened


def test_ancillary_overrides_select_exact_loader_inputs(monkeypatch):
    paths, hash_calls, opened = synthetic_native_loader(
        monkeypatch, ancillary_overrides=True)
    arrays, identities = viewer.load_inputs(
        paths["prediction"], paths["reference"],
        environment=paths["environment"], buildings=paths["buildings"])
    assert set(opened) == set(paths)
    assert all(calls == 2 for calls in hash_calls.values())
    for name in paths:
        assert identities[name]["path"] == str(paths[name])
        assert arrays[name].shape == (2, 3)


@pytest.mark.parametrize("changed", ["reference", "prediction", "buildings", "environment"])
def test_changed_during_read_is_rejected(monkeypatch, changed):
    paths, hash_calls, opened = synthetic_native_loader(monkeypatch, changed=changed)
    with pytest.raises(ValueError, match="Input changed during reading: " + changed):
        viewer.load_inputs(paths["prediction"], paths["reference"])
    assert set(opened) == set(paths)
    assert hash_calls[changed] == 2


def test_unknown_prediction_is_rejected_at_prediction_branch(monkeypatch):
    paths, _, opened = synthetic_native_loader(monkeypatch)
    known_hash = viewer.file_sha256
    monkeypatch.setattr(
        viewer, "file_sha256",
        lambda path: "0" * 64 if path == paths["prediction"] else known_hash(path))
    with pytest.raises(ValueError, match="F01 prediction identity mismatch"):
        viewer.load_inputs(paths["prediction"], paths["reference"])
    assert opened == []


def accepted_original_result():
    return {"metrics": {
        "confusion_matrix": [[133219, 863], [115, 1795]],
        "evaluation_pixel_count": 135992,
        "predicted_burned_cells": 2658, "predicted_burned_full_crop": 2658,
        "accuracy": 0.992808, "precision": 0.675320, "recall": 0.939791,
        "f1_score": 0.785902, "jaccard_index": 0.647313,
        "excluded_percentage": 42.33,
        "reference_coverage": {
            "reference_valid_burned_total": 3312,
            "reference_burned_included_in_primary_domain": 1910,
            "reference_burned_excluded_from_primary_domain": 1402,
            "exclusion_priority_partition": {
                "outside_environmental_validity": 20,
                "outside_prediction_validity": 0,
                "outside_building_raster_validity": 0,
                "outside_mapped_buildings": 1382}}}}


@pytest.mark.parametrize("path", [
    ("confusion_matrix",), ("evaluation_pixel_count",),
    ("predicted_burned_cells",), ("predicted_burned_full_crop",),
    ("accuracy",), ("precision",), ("recall",), ("f1_score",), ("jaccard_index",),
    ("excluded_percentage",),
    *[("reference_coverage", key) for key in (
        "reference_valid_burned_total", "reference_burned_included_in_primary_domain",
        "reference_burned_excluded_from_primary_domain")],
    *[("reference_coverage", "exclusion_priority_partition", key) for key in (
        "outside_environmental_validity", "outside_prediction_validity",
        "outside_building_raster_validity", "outside_mapped_buildings")],
])
def test_every_original_acceptance_expectation_is_retained(path):
    result = accepted_original_result()
    assert viewer.accepted_disagreements(result) == []
    field = result["metrics"]
    for key in path[:-1]:
        field = field[key]
    field[path[-1]] = ([[0, 0], [0, 0]] if path[-1] == "confusion_matrix"
                      else field[path[-1]] + 1)
    assert viewer.accepted_disagreements(result)


def test_undefined_original_exclusion_percentage_is_a_disagreement():
    result = accepted_original_result()
    result["metrics"]["excluded_percentage"] = None
    assert viewer.accepted_disagreements(result)
