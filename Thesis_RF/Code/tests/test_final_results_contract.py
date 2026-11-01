"""Tiny-array viewer contracts; no production raster access or display."""
from copy import deepcopy

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
    monkeypatch.setattr(viewer, "load_inputs", lambda: (arrays, {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Disagreeing case must not display")
    monkeypatch.setattr(viewer, "draw_figure", forbidden)
    with pytest.raises(ValueError, match="DISAGREEMENT"):
        viewer.main()
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
