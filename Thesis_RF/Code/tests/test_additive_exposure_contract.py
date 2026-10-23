"""Deterministic R2 checks only; no campaign or file fixtures."""
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import struct

import numpy as np
import pytest
import yaml

import _additive_exposure_oracle as oracle
from modules.additive_exposure_teacher import (
    compute_susceptibility32 as susceptibility,
    compute_additive_exposure_probability as probability,
)
from modules.additive_exposure_assessment import (
    WINDS, OFFSETS, LAYOUTS, BASELINE_ID, TEACHER_ID, _AssessmentAutomata,
    disabled_configuration, validate_configuration, prepare_environment,
    uniform_raw, dynamic_fixture, pattern_offsets, engine_configuration,
)
from modules.wind_convention import WindContract, compute_wind_weighted_score
from modules.feature_pipeline import compute_moore_neighbor_count


def bits(value):
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def assert_probability_layout(result, shape):
    for array, dtype in ((result.exposure64, np.float64),
                         (result.q64, np.float64), (result.q32, np.float32)):
        assert isinstance(array, np.ndarray)
        assert array.shape == shape
        assert array.dtype == np.dtype(dtype)


def judge(beta, score, result):
    assert_probability_layout(result, (1,))
    if beta == 0 or score == 0:
        assert result.q64.view(np.uint64)[0] == 0
        assert result.q32.view(np.uint32)[0] == 0
    elif beta == 1:
        assert result.q64.view(np.uint64)[0] == 0x3FF0000000000000
        assert result.q32.view(np.uint32)[0] == 0x3F800000
    try:
        exposure, q64, q32 = oracle.probability(beta, score)
    except oracle.OracleInconclusive as error:
        pytest.fail("ORACLE-INCONCLUSIVE: " + str(error))
    assert result.exposure64.view(np.uint64)[0] == exposure
    reference64 = struct.unpack(">d", struct.pack(">Q", q64))[0]
    reference32 = struct.unpack(">f", struct.pack(">I", q32))[0]
    endpoint = beta in (0, 1) or score == 0
    assert oracle.distance(result.q64[0], reference64, 64) <= (0 if endpoint else 8)
    assert result.q32.view(np.uint32)[0] == oracle.round_bits(Fraction.from_float(float(result.q64[0])), 32)
    assert oracle.distance(result.q32[0], reference32, 32) <= (0 if endpoint else 1)


def test_disabled_yaml_matches_fixed_contract():
    path = Path(__file__).resolve().parents[1] / "config/phase15_additive_exposure_r2.yaml"
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded == disabled_configuration()
    validate_configuration(loaded)
    with pytest.raises(PermissionError):
        validate_configuration(loaded, active=True)


def test_oracle_rounding_and_encoding_self_checks():
    assert oracle.round_bits(Fraction(1) + Fraction(1, 2**24), 32) == 0x3F800000
    assert oracle.round_bits(Fraction(1) + Fraction(3, 2**24), 32) == 0x3F800002
    assert oracle.round_bits(Fraction(1, 2**150), 32) == 0
    assert oracle.round_bits(Fraction(3, 2**150), 32) == 2
    assert oracle.round_bits(-Fraction(1, 2**150), 32) == 0x80000000
    assert oracle.distance(-0.0, 0.0, 32) == oracle.distance(-0.0, 0.0, 64) == 0
    assert oracle.distance(0.0, 1.0, 32) == 0x3F800000
    assert oracle.distance(0.0, 1.0, 64) == 0x3FF0000000000000
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            oracle.distance(value, 0, 64)
    for width in (32, 64):
        one = oracle.round_bits(1, width)
        assert oracle.decode(one, width) == 1
        assert oracle.round_bits(oracle.decode(one + 1, width), width) == one + 1
    assert oracle.probability(0.5, 0.125)[1:] == (
        oracle.round_bits(Fraction(1, 2), 64), oracle.round_bits(Fraction(1, 2), 32))


def test_oracle_inconclusive_is_not_a_pass(monkeypatch):
    monkeypatch.setattr(oracle, "_interval", lambda *_: (Fraction(0), Fraction(1)))
    with pytest.raises(oracle.OracleInconclusive):
        oracle.probability(0.5, np.float32(0.1))


@pytest.mark.parametrize("beta", [0, 2**-149, 2**-24, 0.25, 0.5, 1 - 2**-24, 1])
@pytest.mark.parametrize("score", [0, 2**-149, 0.125, 0.5, 1])
def test_direct_probability_cases(beta, score):
    beta, score = np.float32(beta), np.float32(score)
    result = probability(beta32=np.array([beta]), wind_weighted_score32=np.array([score]))
    judge(beta, score, result)


def test_endpoints_do_not_evaluate_logarithms(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Endpoint logarithm evaluated")
    monkeypatch.setattr(np, "log1p", forbidden)
    result = probability(beta32=np.array([1, 1, 0], dtype=np.float32),
                         wind_weighted_score32=np.array([0, 1, 1], dtype=np.float32))
    assert_probability_layout(result, (3,))
    assert np.array_equal(result.q64.view(np.uint64),
                          np.array([0, 0x3FF0000000000000, 0], dtype=np.uint64))
    assert np.array_equal(result.q32.view(np.uint32),
                          np.array([0, 0x3F800000, 0], dtype=np.uint32))


@pytest.mark.parametrize("which,value", [
    ("beta", np.array([0.5], dtype=np.float64)),
    ("score", np.array([0.5], dtype=np.float64)),
    ("beta", np.array([np.nan], dtype=np.float32)),
    ("score", np.array([np.inf], dtype=np.float32)),
    ("beta", np.array([-1], dtype=np.float32)),
    ("score", np.array([1.01], dtype=np.float32)),
    ("score", np.array([0, 1], dtype=np.float32)),
])
def test_probability_rejects_invalid_inputs(which, value):
    arguments = {"beta32": np.array([0.5], dtype=np.float32),
                 "wind_weighted_score32": np.array([0.5], dtype=np.float32)}
    arguments["beta32" if which == "beta" else "wind_weighted_score32"] = value
    with pytest.raises((ValueError, TypeError)):
        probability(**arguments)


def test_interior_range_failure_is_not_clipped(monkeypatch):
    monkeypatch.setattr(np, "expm1", lambda a, **kw: np.full(a.shape, -1.01))
    with pytest.raises(ValueError, match="Interior"):
        probability(beta32=np.array([0.5], dtype=np.float32),
                    wind_weighted_score32=np.array([0.5], dtype=np.float32))


def test_susceptibility_clips_and_rejects_before_clip(monkeypatch):
    inputs = dict(slope32=np.array([0, 1], dtype=np.float32),
                  building32=np.ones(2, dtype=np.float32),
                  proximity32=np.array([-2, 2], dtype=np.float32),
                  material_risk32=np.array([0.15, 0.95], dtype=np.float32))
    result = susceptibility(**inputs)
    assert np.array_equal(result.beta32, [0, 1])
    for index in range(2):
        expected = oracle.susceptibility(*(a[index] for a in inputs.values()))
        assert [bits(a[index]) for a in result.products] == expected[0]
        assert [bits(a[index]) for a in result.additions] == expected[1]
        assert bits(result.beta32[index]) == expected[2]
    inputs["slope32"][0] = np.nan
    with pytest.raises(ValueError):
        susceptibility(**inputs)
    monkeypatch.setattr(np, "add", lambda a, b, **kw: np.full(a.shape, np.inf, dtype=np.float32))
    one = np.ones(1, dtype=np.float32)
    with pytest.raises(ValueError, match="Nonfinite"):
        susceptibility(slope32=one, building32=one, proximity32=one, material_risk32=one)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("material", [1, 5])
def test_dynamic_fixture_geometry_masks_and_initialization(layout, material):
    environment, ignition = dynamic_fixture(layout, material)
    assert environment["transform"] == (3, 0, 0, 0, -3, 27)
    assert environment["crs"] == "EPSG:32651"
    expected = np.ones((9, 9), dtype=bool)
    if layout == "corridor":
        expected[:] = False
        expected[4] = True
    elif layout == "separated_clusters":
        expected[:] = False
        expected[np.ix_([3, 4, 5], [0, 1, 2, 6, 7, 8])] = True
    assert np.array_equal(environment["burnable_mask"], expected)
    assert np.array_equal(environment["nodata_mask"], ~expected)
    assert np.array_equal(environment["building_presence"] > 0, expected)
    assert np.all(environment["material_class"][expected] == material)
    assert ignition == ((0, 4) if layout == "boundary" else
                        (4, 1) if layout == "separated_clusters" else (4, 4))


@pytest.mark.parametrize("mutation", ["slope_nodata", "nonbuilding", "state3", "state4", "state5", "mismatch"])
def test_eligibility_negative_fixtures(mutation):
    raw = uniform_raw(3)
    if mutation == "slope_nodata":
        raw["slope"][1, 1] = -9999
    if mutation == "nonbuilding":
        raw["buildings"][1, 1] = raw["materials"][1, 1] = 0
        raw["proximity"][1, 1] = 10
    if mutation == "mismatch":
        raw["materials"][1, 1] = 0
        with pytest.raises(ValueError, match="inconsistency"):
            prepare_environment(raw)
        return
    environment = prepare_environment(raw)
    engine = _AssessmentAutomata(environment, engine_configuration(0, (0, 0)), teacher_id=TEACHER_ID)
    engine.set_ignition([(1, 0)])
    if mutation.startswith("state"):
        engine.grid[1, 1] = int(mutation[-1])
    else:
        assert engine.grid[1, 1] == 1
    count = compute_moore_neighbor_count((engine.grid == 4).astype(np.int8))
    eligible = (environment["burnable_mask"] & ~environment["nodata_mask"] &
                (environment["building_presence"] > 0) & (engine.grid == 2) & (count > 0))
    assert not eligible[1, 1]
    record = engine.step()
    capture = engine._transition_capture
    if capture is None:
        assert record["eligible_opportunities"] == 0
        assert record["probability_sum"] == 0
        assert record["rng_draw_calls"] == 0
    else:
        indices, probabilities = capture
        assert len(indices) > 0
        assert record["eligible_opportunities"] == len(indices) == len(probabilities)
        assert 4 not in indices  # Target (1, 1), in the pre-update 3x3 grid.


@pytest.mark.parametrize("wind", WINDS)
@pytest.mark.parametrize("case", list(product((0, 10), (0, 10), (1, 5))) + [(10, 20, 1)])
def test_all_valid_patterns_oracle_and_source_addition_monotonicity(wind, case):
    slope, proximity, material = case
    environment = prepare_environment(uniform_raw(3, material, slope, proximity))
    trace = susceptibility(slope32=environment["slope_risk"],
                           building32=environment["building_presence"].astype(np.float32),
                           proximity32=environment["proximity_risk"],
                           material_risk32=np.full((3, 3), np.float32(0.95 if material == 1 else 0.15)))
    expected = oracle.susceptibility(np.float32(slope) / np.float32(10), 1,
                                    np.float32(proximity) / np.float32(10),
                                    np.float32(0.95 if material == 1 else 0.15))
    assert [bits(a[0, 0]) for a in trace.products] == expected[0]
    assert [bits(a[0, 0]) for a in trace.additions] == expected[1]
    assert bits(trace.beta32[0, 0]) == expected[2]
    beta = trace.beta32[0, 0]
    for target in product(range(3), repeat=2):
        offsets = pattern_offsets(target)
        outcomes = {}
        for pattern in range(1 << len(offsets)):
            state = np.zeros((3, 3), dtype=np.int8)
            active = [offset for bit, offset in enumerate(offsets) if pattern & (1 << bit)]
            for dr, dc in active:
                state[target[0] + dr, target[1] + dc] = 1
            score, _ = compute_wind_weighted_score(state, WindContract(*wind), 0.20)
            result = probability(beta32=np.array([beta]),
                                 wind_weighted_score32=np.array([score[target]], dtype=np.float32))
            judge(beta, score[target], result)
            outcomes[pattern] = (result.q64[0], result.q32[0])
            print(json.dumps({"target": target, "pattern": pattern, "case": case, "wind": wind,
                              "q64": float(result.q64[0]), "q32": float(result.q32[0]),
                              **oracle.ideal_comparison(beta, score[target], active, wind)}))
        for pattern, values in outcomes.items():
            for bit in range(len(offsets)):
                added = outcomes[pattern | (1 << bit)]
                assert added[0] >= values[0] and added[1] >= values[1]


@pytest.mark.parametrize("defect", ["mask_dtype", "material_dtype", "nonfinite", "state_dtype"])
def test_canonical_environment_and_state_rejections(defect):
    environment = prepare_environment(uniform_raw(3))
    if defect == "mask_dtype":
        environment["burnable_mask"] = environment["burnable_mask"].astype(np.int8)
    elif defect == "material_dtype":
        environment["material_class"] = environment["material_class"].astype(np.float32)
    elif defect == "nonfinite":
        environment["slope_risk"][0, 0] = np.inf
    config = engine_configuration(0, (0, 0))
    if defect == "state_dtype":
        engine = _AssessmentAutomata(environment, config, teacher_id=TEACHER_ID)
        engine.grid = engine.grid.astype(np.float32)
        with pytest.raises(TypeError):
            engine.step()
        assert engine.failed_transition["transition_completed"] is False
    else:
        with pytest.raises((TypeError, ValueError)):
            _AssessmentAutomata(environment, config, teacher_id=TEACHER_ID)


def test_component_monotonicity_in_both_precisions():
    betas = np.array([0, 2**-149, 2**-24, 0.25, 0.5, 1 - 2**-24, 1], dtype=np.float32)
    scores = np.array([0, 2**-149, 0.125, 0.5, 1], dtype=np.float32)
    for beta in betas:
        result = probability(beta32=np.full(scores.shape, beta), wind_weighted_score32=scores)
        assert_probability_layout(result, scores.shape)
        assert np.all(np.diff(result.q64) >= 0) and np.all(np.diff(result.q32) >= 0)
    for score in scores:
        result = probability(beta32=betas, wind_weighted_score32=np.full(betas.shape, score))
        assert_probability_layout(result, betas.shape)
        assert np.all(np.diff(result.q64) >= 0) and np.all(np.diff(result.q32) >= 0)
