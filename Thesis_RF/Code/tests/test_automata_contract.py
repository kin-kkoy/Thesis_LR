import numpy as np
import pytest

import modules.automata_engine as automata_engine_module
from modules.automata_engine import (
    STATE_BLAZING,
    STATE_IGNITED,
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


def test_model_free_sparse_probability_preserves_full_grid_seeded_outcome(monkeypatch):
    shape = (3, 3)
    environment = {
        "slope_risk": np.zeros(shape, dtype=np.float32),
        "proximity_risk": np.zeros(shape, dtype=np.float32),
        "building_presence": np.ones(shape, dtype=np.float32),
        "material_class": np.ones(shape, dtype=np.int8),
        "burnable_mask": np.ones(shape, dtype=bool),
        "nodata_mask": np.zeros(shape, dtype=bool),
        "grid_shape": shape,
        "transform": None,
        "crs": None,
    }
    config = _config()
    config["simulation"]["seed"] = 31415
    config["placeholder_transition"]["base_ignition_prob"] = 0.45
    automata = FireAutomata(environment, config, model_free=True)
    automata.grid[1, 1] = STATE_BLAZING
    wind_score = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.0, 0.6], [0.7, 0.8, 0.9]],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        automata_engine_module,
        "compute_wind_weighted_score",
        lambda *_args, **_kwargs: (
            wind_score.copy(),
            np.zeros((3, 3), dtype=np.float32),
        ),
    )
    reference_draw = np.random.default_rng(31415).random(shape)
    reference_susceptible = np.ones(shape, dtype=bool)
    reference_susceptible[1, 1] = False
    expected_ignited = reference_susceptible & (
        reference_draw < np.clip(automata.p_base * wind_score, 0.0, 1.0)
    )

    automata.step()

    assert np.array_equal(automata.grid == STATE_IGNITED, expected_ignited)
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


@pytest.mark.parametrize("enabled", [False, True])
def test_emergency_switch_suppresses_pilot_writes_and_preserves_legacy_default(monkeypatch, enabled):
    config = _config()
    if not enabled:
        config["simulation"]["emergency_checkpoint_enabled"] = False
    automata = FireAutomata(_environment(), config)
    attempts = []
    monkeypatch.setattr(automata, "_save_emergency_checkpoint", lambda: attempts.append("write") or "synthetic.npz")

    def fail(_state):
        raise ArithmeticError("synthetic step failure")

    monkeypatch.setattr(automata_engine_module, "compute_moore_neighbor_count", fail)
    with pytest.raises(RuntimeError):
        automata.step()
    assert attempts == (["write"] if enabled else [])


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_emergency_switch_requires_a_boolean(value):
    config = _config()
    config["simulation"]["emergency_checkpoint_enabled"] = value
    with pytest.raises(TypeError, match="must be boolean"):
        FireAutomata(_environment(), config)


def test_pilot_switch_does_not_change_model_state_timers_or_rng():
    engines = []
    for enabled in (True, False):
        config = _config()
        config["simulation"]["emergency_checkpoint_enabled"] = enabled
        engine = FireAutomata(_environment(), config)
        engine.model = _ReorderedEstimator()
        engine.set_ignition([(0, 0)])
        engines.append(engine)
    for _ in range(3):
        for engine in engines:
            engine.step()
    left, right = engines
    for name in ("grid", "ignition_timers", "blazing_timers"):
        assert np.array_equal(getattr(left, name), getattr(right, name))
    assert left.timestep == right.timestep
    assert left.rng.bit_generator.state == right.rng.bit_generator.state


def test_compensating_malformed_chunks_fail_at_the_first_chunk():
    environment = _environment()
    for name in ("slope_risk", "proximity_risk", "building_presence", "material_class", "burnable_mask", "nodata_mask"):
        environment[name] = np.repeat(environment[name], 3, axis=0)
    environment["grid_shape"] = (3, 2)
    config = _config()
    config["ml_model"]["inference_chunk_size"] = 1

    class Malformed(_ReorderedEstimator):
        calls = 0

        def predict_proba(self, features):
            self.calls += 1
            return np.tile([0.8, 0.2], (0 if self.calls == 1 else 2, 1))

    engine = FireAutomata(environment, config)
    engine.model = Malformed()
    with pytest.raises(ValueError, match="row count"):
        engine._predict_with_model(np.ones((3, 2), dtype=np.int8), np.ones((3, 2), dtype=bool))
    assert engine.model.calls == 1



@pytest.mark.parametrize("bearing", [315.0, 48.825208])
@pytest.mark.parametrize("probability,draw,ignites", [
    (0.0, 0.0, False),
    (1.0, 0.99, True),
    (0.25, 0.30, False),
    (0.50000001, 0.500000005, False),
    (0.49999999, 0.499999995, True),
])
def test_candidate_calibrated_probability_is_compared_as_float32_without_extra_factors(
        monkeypatch, probability, draw, ignites, bearing):
    config = _config()
    config["wind"].update(speed_kmh=17.7, direction_deg=bearing)
    config["placeholder_transition"].update(wind_weight=0.20, T_3_to_4=1, T_4_to_5=1)
    config["simulation"].update(seed=42, emergency_checkpoint_enabled=False, checkpoint_interval=0)
    engine = FireAutomata(_environment(), config)
    engine.set_ignition([(0, 0)])
    engine.p_base.fill(0.0)  # An accidental teacher multiplier would destroy p=1.
    calls = []
    class Calibrated:
        feature_names_in_ = np.asarray(CANONICAL_FEATURE_NAMES)
        classes_ = np.array([1, 0], dtype=np.int64)
        def predict_proba(self, features):
            assert list(features.columns) == list(CANONICAL_FEATURE_NAMES)
            assert all(dtype == np.dtype("float32") for dtype in features.dtypes)
            calls.append(features.to_numpy(copy=True))
            return np.tile([probability, 1.0 - probability], (len(features), 1))
    class Draw:
        def random(self, shape):
            assert shape == (1, 2)
            return np.array([[0.0, draw]], dtype=np.float64)
    engine.model, engine.rng = Calibrated(), Draw()
    monkeypatch.setattr(engine, "_save_emergency_checkpoint", lambda: pytest.fail("checkpoint"))
    engine.step()
    assert len(calls) == 1 and calls[0].shape == (1, 11)
    assert (engine.grid[0, 1] == STATE_IGNITED) == ignites
    assert ignites == (draw < float(np.float32(probability)))


@pytest.mark.parametrize("bearing", [315.0, 48.825208])
def test_candidate_multistep_matches_independent_scalar_state_features_timers_and_rng(bearing):
    # Independent scalar neighbourhood/feature/transition expectation; no engine,
    # feature assembler, wind helper, teacher or observer is used for the reference.
    shape = (3, 4)
    rr, cc = np.indices(shape)
    material = ((rr + cc) % 5 + 1).astype(np.int8)
    receiving = np.ones(shape, dtype=bool)
    receiving[0, 3] = False
    environment = dict(slope_risk=(rr / 10).astype(np.float32),
        proximity_risk=(cc / 10).astype(np.float32),
        building_presence=receiving.astype(np.int8), material_class=material,
        burnable_mask=np.ones(shape, dtype=bool), nodata_mask=np.zeros(shape, dtype=bool),
        grid_shape=shape, transform=None, crs=None)
    config = _config()
    config["simulation"].update(seed=42, emergency_checkpoint_enabled=False, checkpoint_interval=0)
    config["wind"].update(speed_kmh=17.7, direction_deg=bearing)
    config["placeholder_transition"].update(wind_weight=0.20, T_3_to_4=1, T_4_to_5=1)
    engine = FireAutomata(environment, config)
    engine.set_ignition([(1, 1)])
    reference_rng = np.random.default_rng(42)
    state = np.where(receiving, 2, 1).astype(np.int8)
    state[1, 1] = 4
    ignition = np.zeros(shape, dtype=np.int16)
    blazing = np.zeros(shape, dtype=np.int16)
    sine, cosine = np.sin(np.deg2rad(bearing)), np.cos(np.deg2rad(bearing))
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0)]
    kernel = np.zeros((3, 3), dtype=np.float32)
    for dr, dc in offsets:
        distance = float(np.hypot(dr, dc))
        alignment = (-dr / distance) * cosine + (-dc / distance) * (-sine)
        kernel[dr + 1, dc + 1] = np.float32(max(1.0 + 0.20 * (17.7 / 10.0) * alignment, 0.05))
    risk_table = np.array([0.0, 0.95, 0.8, 0.48, 0.30, 0.15], dtype=np.float32)
    expected_features, call_counts = [], []
    class Calibrated:
        feature_names_in_ = np.asarray(CANONICAL_FEATURE_NAMES)
        classes_ = np.array([1, 0], dtype=np.int64)
        def predict_proba(self, features):
            actual = features.to_numpy()
            assert actual.dtype == np.float32
            np.testing.assert_array_equal(actual.view(np.uint32), expected_features[0].view(np.uint32))
            call_counts.append(len(features))
            return np.tile([0.675, 0.325], (len(features), 1))
    engine.model = Calibrated()
    expected_draws = 0
    for update in range(1, 7):
        counts = np.zeros(shape, dtype=np.int8)
        score = np.zeros(shape, dtype=np.float32)
        for r in range(shape[0]):
            for c in range(shape[1]):
                sources = [(dr, dc) for dr, dc in offsets
                           if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]
                           and state[r + dr, c + dc] == 4]
                counts[r, c] = len(sources)
                score[r, c] = np.float32(sum(float(kernel[dr + 1, dc + 1]) for dr, dc in sources))
        score = score / np.float32(kernel.sum())
        eligible = receiving & (state == 2) & (counts > 0)
        rows = []
        for r, c in zip(*np.nonzero(eligible)):
            presence = np.float32(environment["building_presence"][r, c])
            risk = risk_table[material[r, c]] if presence > 0 else np.float32(0)
            rows.append([environment["slope_risk"][r, c], environment["proximity_risk"][r, c],
                presence, risk, material[r, c], 17.7, sine, cosine, counts[r, c],
                np.float32(presence * risk), score[r, c]])
        expected_features[:] = [np.asarray(rows, dtype=np.float32).reshape(-1, 11)]
        next_state = state.copy()
        for r in range(shape[0]):
            for c in range(shape[1]):
                if state[r, c] == 3:
                    ignition[r, c] += 1
                    if ignition[r, c] >= 1:
                        next_state[r, c], ignition[r, c], blazing[r, c] = 4, 0, 0
                if state[r, c] == 4:
                    blazing[r, c] += 1
                    if blazing[r, c] >= 1:
                        next_state[r, c], blazing[r, c] = 5, 0
        if np.any(eligible):
            draw = reference_rng.random(shape)
            expected_draws += 1
            new = eligible & (draw < np.float32(0.675))
            next_state[new], ignition[new], blazing[new] = 3, 0, 0
        engine.step()
        state = next_state
        for actual, expected in ((engine.grid, state), (engine.ignition_timers, ignition),
                                 (engine.blazing_timers, blazing)):
            np.testing.assert_array_equal(actual, expected)
            assert actual.dtype == expected.dtype
        assert engine.timestep == update
        assert engine.rng.bit_generator.state == reference_rng.bit_generator.state
    assert len(call_counts) == expected_draws
