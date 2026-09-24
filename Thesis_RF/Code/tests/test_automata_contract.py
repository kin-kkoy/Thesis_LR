import numpy as np
import pytest

import modules.automata_engine as automata_engine_module
from modules.automata_engine import (
    STATE_BLAZING,
    STATE_NOT_YET_BURNING,
    FireAutomata,
)
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES
from modules.wind_convention import (
    CALM_REPRESENTATION,
    DIRECTION_CONVENTION,
    DIRECTION_UNITS,
    NORTH_REFERENCE,
    WIND_SCHEMA_VERSION,
)


def _environment() -> dict:
    return {
        "slope_risk": np.zeros((1, 2), dtype=np.float32),
        "proximity_risk": np.zeros((1, 2), dtype=np.float32),
        "building_presence": np.ones((1, 2), dtype=np.float32),
        "material_class": np.ones((1, 2), dtype=np.int8),
        "burnable_mask": np.ones((1, 2), dtype=bool),
        "nodata_mask": np.zeros((1, 2), dtype=bool),
        "grid_shape": (1, 2),
        "transform": None,
        "crs": None,
    }


def _config() -> dict:
    return {
        "simulation": {"seed": 7},
        "wind": {
            "speed_kmh": 10.0,
            "direction_deg": 0.0,
            "direction_convention": DIRECTION_CONVENTION,
            "direction_units": DIRECTION_UNITS,
            "north_reference": NORTH_REFERENCE,
            "schema_version": WIND_SCHEMA_VERSION,
            "calm_representation": CALM_REPRESENTATION,
        },
        "placeholder_transition": {"wind_weight": 3.0},
        "ml_model": {"inference_mode": "stochastic_probability", "threshold": None},
    }


class _ReorderedEstimator:
    feature_names_in_ = np.asarray(CANONICAL_FEATURE_NAMES)
    classes_ = np.array([1, 0], dtype=np.int64)

    def predict_proba(self, features):
        return np.tile(np.array([[0.8, 0.2]]), (len(features), 1))


class _FixedRng:
    def random(self, shape):
        return np.array([[0.99, 0.30]], dtype=np.float64)


def test_model_probability_uses_positive_class_even_when_classes_are_reordered():
    automata = FireAutomata(_environment(), _config())
    automata.grid[0, 0] = STATE_BLAZING
    automata.model = _ReorderedEstimator()

    probabilities = automata._predict_with_model(
        np.array([[0, 1]], dtype=np.int8),
        np.array([[False, True]]),
    )

    assert probabilities[0, 1] == pytest.approx(0.8)


def test_ml_probability_is_not_multiplied_by_placeholder_wind_weight(monkeypatch):
    automata = FireAutomata(_environment(), _config())
    automata.grid[0, 0] = STATE_BLAZING
    automata.model = object()
    automata.rng = _FixedRng()
    monkeypatch.setattr(
        automata,
        "_predict_with_model",
        lambda neighbor_count, susceptible, wind_score: np.array(
            [[0.0, 0.25]], dtype=np.float32
        ),
    )

    automata.step()

    assert automata.grid[0, 1] == STATE_NOT_YET_BURNING
    assert automata.grid.dtype == np.int8


def test_stochastic_ml_mode_rejects_a_ca_threshold():
    config = _config()
    config["ml_model"]["threshold"] = 0.5
    with pytest.raises(ValueError, match="must be absent or null"):
        FireAutomata(_environment(), config)


def test_model_free_probability_does_not_reapply_wind(monkeypatch):
    config = _config()
    config["placeholder_transition"]["base_ignition_prob"] = 0.2
    automata = FireAutomata(_environment(), config, model_free=True)
    automata.grid[0, 0] = STATE_BLAZING

    class FixedDraw:
        def random(self, shape):
            assert shape == (1, 2)
            return np.array([[0.99, 0.03]], dtype=np.float64)

    automata.rng = FixedDraw()
    monkeypatch.setattr(
        automata_engine_module,
        "compute_wind_weighted_score",
        lambda *_args, **_kwargs: (
            np.array([[0.0, 0.1]], dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
        ),
    )

    automata.step()

    # D-014 gives p_effective = 0.2 * 0.1 = 0.02. Reapplying the
    # configured wind_weight of 3.0 would incorrectly ignite this cell.
    assert automata.grid[0, 1] == STATE_NOT_YET_BURNING
    assert automata.grid.dtype == np.int8


def test_model_free_engine_rejects_all_estimator_boundaries(monkeypatch, tmp_path):
    automata = FireAutomata(_environment(), _config(), model_free=True)
    joblib_calls = []
    monkeypatch.setattr(
        automata_engine_module.joblib,
        "load",
        lambda path: joblib_calls.append(path),
    )

    with pytest.raises(RuntimeError, match="cannot attach"):
        automata.model = object()
    with pytest.raises(RuntimeError, match="cannot load"):
        automata.load_model("forbidden.joblib")
    with pytest.raises(RuntimeError, match="cannot invoke"):
        automata._predict_with_model(
            np.zeros((1, 2), dtype=np.int8),
            np.zeros((1, 2), dtype=bool),
        )
    with pytest.raises(RuntimeError, match="cannot write checkpoints"):
        automata.save_checkpoint(str(tmp_path / "forbidden.npz"))
    with pytest.raises(RuntimeError, match="cannot load checkpoints"):
        automata.load_checkpoint(str(tmp_path / "forbidden.npz"))

    assert joblib_calls == []
    assert not (tmp_path / "forbidden.npz").exists()
    assert automata.model is None
    assert automata.model_free is True
