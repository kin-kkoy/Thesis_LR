import numpy as np
import pytest

import airbase_results as a
from airbase_results import evaluate_airbase_arrays


def _arrays(state=None):
    state = np.array([[5, 1, 5], [1, 1, 1]], dtype=np.uint8) if state is None else state
    return {
        "prediction": state,
        "prediction_valid": np.ones((2, 3), dtype=bool),
        "reference": np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8),
        "reference_valid": np.ones((2, 3), dtype=bool),
        "buildings": np.full((2, 3), 10, dtype=np.uint8),
        "buildings_valid": np.ones((2, 3), dtype=bool),
        "environment": np.ones((2, 3), dtype=np.uint8),
        "environment_valid": np.ones((2, 3), dtype=bool),
    }


def test_excluded_reference_cells_do_not_enter_primary_metrics():
    arrays = _arrays()
    arrays["environment"][0, 1] = 0
    result = evaluate_airbase_arrays(arrays, termination_reason="inactive")
    assert result["domain_cells"] == 5
    assert result["metrics"]["confusion_matrix"] == [[2, 1], [1, 1]]
    assert result["metrics"]["reference_coverage"]["reference_burned_excluded_from_primary_domain"] == 1


def test_invalid_reference_category_is_rejected():
    arrays = _arrays()
    arrays["reference"][0, 0] = 2
    with pytest.raises(ValueError, match="0/1"):
        evaluate_airbase_arrays(arrays, termination_reason="inactive")


def test_timestep_limit_counts_active_cells_as_censored_predicted_footprint():
    state = np.array([[3, 4, 1], [5, 1, 1]], dtype=np.uint8)
    result = evaluate_airbase_arrays(_arrays(state), termination_reason="timestep_limit")
    assert result["metrics"]["termination"]["status"] == "censored_incomplete_footprint_through_stopping_time"
    assert result["metrics"]["predicted_burned_cells"] == 3
    with pytest.raises(ValueError, match="inactive termination"):
        evaluate_airbase_arrays(_arrays(state), termination_reason="inactive")


def test_reader_enforces_hash_and_geometry_without_resampling(tmp_path, monkeypatch):
    import rasterio
    from rasterio.transform import from_origin
    from hashlib import sha256
    path = tmp_path / "reference.tif"
    transform = from_origin(100, 200, 3, 3)
    with rasterio.open(path, "w", driver="GTiff", height=2, width=3, count=1,
                       dtype="uint8", crs="EPSG:32651", transform=transform) as dst:
        dst.write(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8), 1)
    monkeypatch.setattr(a, "PARENT_SHAPE", (2, 3))
    monkeypatch.setattr(a, "PARENT_TRANSFORM", transform)
    monkeypatch.setattr(a.orchestrator, "PHASE16_AIRBASE_WINDOW", {"global_origin": [0, 1], "shape": [2, 2]})
    digest = sha256(path.read_bytes()).hexdigest()
    raw, valid, _ = a._read_raster(path, parent=True, expected_sha256=digest)
    np.testing.assert_array_equal(raw, [[0, 0], [1, 0]])
    assert valid.all()
    with pytest.raises(ValueError, match="hash mismatch"):
        a._read_raster(path, parent=True, expected_sha256="0" * 64)
    monkeypatch.setattr(a, "PARENT_SHAPE", (3, 3))
    with pytest.raises(ValueError, match="geometry"):
        a._read_raster(path, parent=True, expected_sha256=digest)
