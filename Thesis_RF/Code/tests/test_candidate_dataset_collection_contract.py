"""Source-prepared bounded synthetic checks; no campaign or fixture artifacts."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import _additive_exposure_oracle as oracle
from test_candidate_dataset_inputs_contract import arrays, prepared_arrays
from modules import candidate_dataset_collection as collection
from modules.additive_exposure_teacher import compute_additive_exposure_probability
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES, FeatureAssembler
from modules.transition_observer import canonical_metadata_hash
from modules.wind_convention import WindContract, compute_wind_weighted_score


def context(*, origin=(0, 0), observer=None, raw=None, masks=None):
    local_shape = (3, 3) if raw is None else raw["slope"].shape
    prepared = prepared_arrays(raw, masks, origin=origin,
                               shape=tuple(o + n for o, n in zip(origin, local_shape, strict=True)))
    capability = dict(policy_id="embedded_tiff_only.v1", controls_verified=True,
        runtime={"backend": "synthetic array fixture; not an installed-reader assertion"},
        verification_reference="synthetic verification substitute")
    prepared.interpretation["capability"] = capability
    config = collection.disabled_configuration()
    config.update(enabled=True, authorization_reference="synthetic unit-test execution substitute",
        inventory=[dict(ordinal=f, scenario_family_id=f"synthetic-family-{f}",
            event_id=f"synthetic-event-{f}", complete_event=False, prior_use="synthetic_fixture",
            coordinate_system="grid_row_col",
            ignition_points=[[origin[0] + 1, origin[1] + 1]]) for f in range(1, 27)],
        complete_event_ids=[], inputs={name: {"path": str(Path(__file__).resolve().parent / f"synthetic-{name}.tif"), "sha256": digest}
                                      for name, digest in prepared.input_hashes.items()},
        grid=dict(shape=list(prepared.full_grid_shape), transform=list(prepared.full_grid_transform),
                  crs=str(prepared.environment["crs"])),
        reader_capability=capability,
        resources=dict.fromkeys(config["resources"], 1))
    spec = collection.planned_runs(config["inventory"])[0]
    provenance = dict(source_revision="synthetic source fixture", source_dirty=False,
        source_sha256={"synthetic-source.py": "b" * 64}, grid_id="synthetic-global-grid",
        execution_id="synthetic-unit-execution")
    def construct(**overrides):
        binding = collection.execution_binding(config, prepared, spec, provenance)
        kwargs = dict(execution_authorized=True, approval_reference=config["authorization_reference"],
            expected_binding_sha256=hashlib.sha256(collection._bytes(binding)).hexdigest(), observer=observer)
        kwargs.update(overrides)
        return collection.CandidateAutomata(prepared, config, spec, provenance, **kwargs)
    return prepared, config, spec, provenance, construct


def test_disabled_config_exact_fixed_design_and_pending_activation():
    path = Path(__file__).resolve().parents[1] / "config/candidate_teacher_shared_dataset_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config == collection.disabled_configuration()
    collection.validate_configuration(config)
    assert not config["enabled"] and not config["publication"]["enabled"]
    assert config["inventory"] is config["authorization_reference"] is None
    assert all(value is None for value in config["resources"].values())
    with pytest.raises(PermissionError):
        collection.validate_configuration(config, active=True)


def test_plan_seven_winds_364_runs_order_seed_formula_and_no_campaign():
    _, config, _, _, _ = context()
    plan = collection.planned_runs(config["inventory"])
    assert len(plan) == 364
    assert len({r["run_id"] for r in plan}) == len({r["seed"] for r in plan}) == 364
    assert [tuple(r[k] for k in ("family_ordinal", "wind_ordinal", "replicate_ordinal"))
            for r in plan] == [(f, w, r) for f in range(1, 27) for w in range(1, 8) for r in (1, 2)]
    assert plan[0]["seed"] == 9010101 and plan[-1]["seed"] == 9260702
    assert plan[12]["wind"] == [17.7, 315.0]
    assert (collection.MAX_UPDATES, collection.MAX_RUN_ROWS, collection.MAX_PACKAGE_ROWS) == (
        300, 1800000, 2000000)
    assert not hasattr(collection, "run_campaign") and not hasattr(collection, "publish")


@pytest.mark.parametrize("field", ["inventory", "inputs", "grid", "reader_capability"])
def test_missing_activation_values_reject(field):
    _, config, _, _, _ = context()
    config[field] = None
    with pytest.raises((PermissionError, ValueError)):
        collection.validate_configuration(config, active=True)


def test_capture_marker_alone_and_bad_binding_cannot_authorize_rng(monkeypatch):
    _, _, _, _, construct = context()
    monkeypatch.setattr(np.random, "default_rng", lambda *_: pytest.fail("RNG construction"))
    with pytest.raises(PermissionError):
        construct(execution_authorized=False)
    with pytest.raises(PermissionError):
        construct(expected_binding_sha256="0" * 64)
    with pytest.raises(PermissionError):
        construct(approval_reference="d013_source_only_in_memory")


def test_constructor_order_original_generator_and_atomic_bad_seed(monkeypatch):
    _, config, _, _, construct = context()
    created = []
    original = np.random.default_rng
    def track(seed):
        generator = original(seed)
        created.append(generator)
        return generator
    monkeypatch.setattr(np.random, "default_rng", track)
    original_precompute = collection.CandidateAutomata._precompute_base_probability
    dispatched = []
    def precompute(instance):
        assert instance.teacher_id == collection.TEACHER_ID
        assert instance.prepared is not None and instance._capture is None
        assert not hasattr(instance, "feature_assembler")
        dispatched.append(True)
        return original_precompute(instance)
    monkeypatch.setattr(collection.CandidateAutomata, "_precompute_base_probability", precompute)
    engine = construct()
    assert dispatched == [True]
    assert engine.rng is created[0] and len(created) == 1
    assert engine.p_base.dtype == np.float32 and engine.feature_assembler is not None
    assert engine.teacher_id == collection.TEACHER_ID
    assert engine.grid[1, 1] == 4
    assert not np.any(engine.ignition_timers) and not np.any(engine.blazing_timers)
    config["inventory"][0]["ignition_points"].append([99, 99])
    with pytest.raises(ValueError):
        construct()
    assert len(created) == 1


def test_actual_q32_vector_bits_rng_shape_order_and_global_observation_identity():
    _, _, spec, _, construct = context(origin=(128, 256))
    engine = construct()
    calls, compared = [], []
    class Draws:
        def random(self, shape):
            calls.append(shape)
            indices, actual, bits, _ = engine._capture
            assert actual.dtype == np.float32 and not actual.flags.writeable
            compared.append((indices.copy(), actual, bits.copy()))
            draws = np.ones(shape, dtype=np.float64)
            draws.ravel()[indices[0]] = float(actual[0])  # Equality must not ignite.
            draws.ravel()[indices[1]] = 0.0
            return draws
    engine.rng = Draws()
    batch = engine.step()
    assert calls == [(3, 3)] and len(compared) == 1
    assert batch.labels[0] == 0 and batch.labels[1] == 1
    np.testing.assert_array_equal(batch.q32_bits, compared[0][2])
    np.testing.assert_array_equal(compared[0][1].view(np.uint32), batch.q32_bits)
    assert batch.features.dtype == np.float32 and batch.features.shape == (8, 11)
    assert len(CANONICAL_FEATURE_NAMES) == 11
    rows, cols = np.unravel_index(compared[0][0], (3, 3))
    np.testing.assert_array_equal(batch.global_rows, rows + 128)
    np.testing.assert_array_equal(batch.global_cols, cols + 256)
    envelope = json.loads(batch.envelope_bytes)
    payload = dict(identity_version="sha256_observation_record_provenance.v2",
        scenario_family_id=spec["scenario_family_id"], scenario_id=spec["scenario_id"],
        run_id=spec["run_id"], seed=spec["seed"], timestep_t=0, timestep_t1=1,
        grid_id="synthetic-global-grid", feature_schema_version="set_c.v1",
        target_version="ignition_t_plus_1.v1", run_identity_sha256=engine.run_identity_sha256,
        provenance_sha256=envelope["provenance_sha256"],
        cell_row=int(batch.global_rows[0]), cell_col=int(batch.global_cols[0]))
    golden = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    assert batch.observation_ids[0].decode() == hashlib.sha256(golden).hexdigest()
    assert envelope["teacher_id"] == "exploratory_additive_exposure_teacher.r2"
    assert len(envelope["bindings"]) == len(batch.labels)
    assert [r["q32_bits"] for r in envelope["bindings"]] == batch.q32_bits.tolist()


def test_fresh_seeded_generator_state_matches_one_full_grid_draw():
    _, _, spec, _, construct = context()
    engine = construct()
    expected = np.random.default_rng(spec["seed"])
    expected.random((3, 3))
    engine.step()
    assert engine.rng.bit_generator.state == expected.bit_generator.state


def test_capture_clears_when_no_susceptible_hook_exists():
    _, _, _, _, construct = context()
    engine = construct()
    engine.step()
    engine.grid[:] = 5
    engine.ignition_timers[:] = engine.blazing_timers[:] = 0
    before = deepcopy(engine.rng.bit_generator.state)
    batch = engine.step()
    assert batch.features.shape == (0, 11) and not len(batch.q32_bits)
    assert engine._capture is None and engine._pending is None
    assert engine.rng.bit_generator.state == before


@pytest.mark.parametrize("excluded", ["environment", "background"])
def test_actual_probability_capture_excludes_ineligible_target(excluded):
    raw, masks = arrays()
    if excluded == "environment":
        masks["slope"][0, 0] = 0
    else:
        raw["buildings"][0, 0] = raw["materials"][0, 0] = 0
    _, _, _, _, construct = context(raw=raw, masks=masks)
    engine = construct()
    batch = engine.step()
    assert not np.any((batch.global_rows == 0) & (batch.global_cols == 0))
    assert len(batch.q32_bits) == 7


@pytest.mark.parametrize("failure_stage", ["observer", "post_step", "captured_bits"])
def test_failure_never_commits_partial_rows_and_preserves_last_success(monkeypatch, failure_stage):
    _, _, _, _, construct = context()
    engine = construct()
    first = engine.step()
    before = deepcopy(engine.last_completed)
    old_batches, old_rows = engine.batches, engine.row_count
    def fail(*_):
        raise RuntimeError("injected collection failure")
    if failure_stage == "observer":
        engine._observer = fail
    elif failure_stage == "post_step":
        monkeypatch.setattr(engine, "_make_batch", fail)
    else:
        # Ensure exposure exists regardless of the preceding Bernoulli outcome.
        engine.grid[:] = 2
        engine.grid[1, 1] = 4
        original = engine._stage_observation
        def corrupt(observation):
            original(observation)
            indices, q, bits, score = engine._capture
            bits[0] ^= np.uint32(1)
        engine._transition_observer = corrupt
    with pytest.raises((RuntimeError, ValueError)):
        engine.step()
    assert engine.failed and engine.batches is old_batches and engine.batches[0] is first
    assert engine.row_count == old_rows and engine.last_completed == before
    assert engine.incomplete_transition["collection_completed"] is False
    assert engine._capture is None and engine._pending is None
    with pytest.raises(RuntimeError):
        engine.step()


def test_caps_fail_without_truncation_or_more_draws(monkeypatch):
    _, _, _, _, construct = context()
    engine = construct()
    other = construct()
    monkeypatch.setattr(collection, "MAX_RUN_ROWS", 0)
    with pytest.raises(RuntimeError, match="row cap"):
        engine.step()
    assert not engine.batches and engine.row_count == 0
    other.timestep = collection.MAX_UPDATES
    before = deepcopy(other.rng.bit_generator.state)
    with pytest.raises(RuntimeError):
        other.step()
    assert other.rng.bit_generator.state == before


@pytest.mark.parametrize("wind", collection.WINDS)
@pytest.mark.parametrize("target", [(r, c) for r in range(3) for c in range(3)])
def test_all_patterns_interior_edges_corners_quantized_oracle(wind, target):
    row, col = target
    offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
               if (dr or dc) and 0 <= row + dr < 3 and 0 <= col + dc < 3]
    for pattern in range(1 << len(offsets)):
        blazing = np.zeros((3, 3), dtype=np.int8)
        for bit, (dr, dc) in enumerate(offsets):
            blazing[row + dr, col + dc] = bool(pattern & (1 << bit))
        score, _ = compute_wind_weighted_score(blazing, WindContract(*wind), 0.20)
        beta = np.array([np.float32(0.5525)])
        result = compute_additive_exposure_probability(beta32=beta,
            wind_weighted_score32=np.array([score[target]], dtype=np.float32))
        assert result.exposure64.dtype == result.q64.dtype == np.float64
        assert result.q32.dtype == np.float32 and result.q32.shape == (1,)
        try:
            exposure, q64, q32 = oracle.probability(beta[0], score[target])
        except oracle.OracleInconclusive as error:
            pytest.fail("ORACLE-INCONCLUSIVE: " + str(error))
        assert result.exposure64.view(np.uint64)[0] == exposure
        reference64 = struct.unpack(">d", struct.pack(">Q", q64))[0]
        reference32 = struct.unpack(">f", struct.pack(">I", q32))[0]
        assert oracle.distance(result.q64[0], reference64, 64) <= (0 if not pattern else 8)
        assert oracle.distance(result.q32[0], reference32, 32) <= (0 if not pattern else 1)
        assert result.q32.view(np.uint32)[0] == oracle.round_bits(
            Fraction.from_float(float(result.q64[0])), 32)
        if not pattern:
            assert result.q64.view(np.uint64)[0] == result.q32.view(np.uint32)[0] == 0


@pytest.mark.parametrize("wind", collection.WINDS[1:])
def test_directional_single_source_semantics_separate_from_roundoff(wind):
    contract = WindContract(*wind)
    kernel = contract.directional_kernel(0.20)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not (dr or dc):
                continue
            blazing = np.zeros((3, 3), dtype=np.int8)
            blazing[1 + dr, 1 + dc] = 1
            score, actual_kernel = compute_wind_weighted_score(blazing, contract, 0.20)
            np.testing.assert_array_equal(actual_kernel.view(np.uint32), kernel.view(np.uint32))
            expected = np.float32(kernel[dr + 1, dc + 1] / kernel.sum(dtype=np.float32))
            assert score[1, 1].view(np.uint32) == expected.view(np.uint32)
    if wind[1] == 315:
        assert kernel[0, 0] > kernel[2, 2]  # NW source favours SE travel.


def test_preserved_engine_r2_and_historical_config_source_anchors():
    root = Path(__file__).resolve().parents[1]
    pins = {
        "modules/automata_engine.py": "a2e4518876e64b67a831e81ce89c345d577a340e19ecbafaa0e8562d227286d8",
        "modules/additive_exposure_teacher.py": "128d44cc1096a90fe7b9c2e6da733312ff70e89d1651e99251747b30c389465f",
        "modules/additive_exposure_assessment.py": "6b04cc9a74e1534797c418ae8295c9e043a40bea2014f47e567588a225db1e3c",
        "config/phase15_additive_exposure_r2.yaml": "02389d7e1bbee4568f271bd270f5e21785ad5761ae3e291df0ec4815c8a50389",
        "config/phase15_setc_rf_ca_v1.yaml": "88922d9b3366c480330580c6f22926987a3ec0b252fcdf9f4d2afacc051d3dfb",
    }
    for name, pin in pins.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == pin


def test_zero_probability_opportunities_remain_rows_and_consume_full_grid_draw():
    raw, masks = arrays()
    raw["proximity"][:] = -100  # Synthetic arithmetic case; no urban-domain claim.
    _, _, spec, _, construct = context(raw=raw, masks=masks)
    engine = construct()
    expected = np.random.default_rng(spec["seed"])
    expected.random((3, 3))
    batch = engine.step()
    assert len(batch.labels) == 8 and np.all(batch.labels == 0)
    assert np.all(batch.q32_bits == 0)
    assert engine.rng.bit_generator.state == expected.bit_generator.state


@pytest.mark.parametrize("missing", ["authorization_reference", "max_process_rss_bytes",
    "min_available_physical_memory_bytes", "min_free_disk_bytes", "max_wall_seconds",
    "observation_interval_seconds"])
def test_unresolved_authorization_or_resources_block_activation(missing):
    _, config, _, _, _ = context()
    if missing == "authorization_reference":
        config[missing] = None
    else:
        config["resources"][missing] = None
    with pytest.raises((PermissionError, ValueError)):
        collection.validate_configuration(config, active=True)


def test_fixed_wind_order_and_complete_event_designations_are_not_relaxed():
    _, config, _, _, _ = context()
    config["winds"].reverse()
    with pytest.raises(ValueError):
        collection.validate_configuration(config, active=True)
    _, config, _, _, _ = context()
    config["inventory"][1].update(event_id=config["inventory"][0]["event_id"],
                                  complete_event=True)
    with pytest.raises(ValueError):
        collection.validate_configuration(config, active=True)


def _feature_sources_case(wind=(0.0, 0.0), layout="c"):
    # Three validation strips; component-only holder, not an authorized CA engine.
    shape = (67, 4)
    rows, cols = np.indices(shape)
    classes = ((rows + cols) % 6).astype(np.int8)
    environment = dict(grid_shape=shape, slope_risk=((rows + cols) % 11 / 10).astype(np.float32),
        proximity_risk=((rows * 4 + cols - 9) / 7).astype(np.float32),
        building_presence=(classes > 0).astype(np.float32), material_class=classes,
        burnable_mask=np.ones(shape, dtype=bool))
    assembler = FeatureAssembler(environment, WindContract(*wind).to_manifest())
    assembler.slope_risk[0, 0] = assembler.proximity_risk[0, 0] = np.float32(-0.0)
    if wind == (0.0, 0.0):
        assembler.wind_sin = -0.0
    counts = ((rows + cols) % 9).astype(np.int8)
    scores = (counts / 8).astype(np.float32)
    scores[0, 0] = np.float32(-0.0)
    def arrange(values):
        if layout == "f":
            return np.asfortranarray(values)
        if layout == "strided":
            backing = np.empty((shape[0], shape[1] * 2), dtype=values.dtype)
            backing[:, ::2] = values
            return backing[:, ::2]
        if layout == "reversed":
            return values[::-1, ::-1].copy()[::-1, ::-1]
        return values.copy(order="C")
    for name in ("slope_risk", "proximity_risk", "building_presence", "material_risk",
                 "material_class", "composite_flammability"):
        setattr(assembler, name, arrange(getattr(assembler, name)))
    observation = SimpleNamespace(blazing_neighbor_count_t=arrange(counts),
                                  wind_weighted_score_t=arrange(scores))
    return SimpleNamespace(grid_shape=shape, feature_assembler=assembler), observation


def _reference_features(holder, observation, indices):
    return holder.feature_assembler.assemble_grid_features(
        observation.blazing_neighbor_count_t, observation.wind_weighted_score_t)[indices]


def _set_feature_source(holder, observation, name, value):
    if name in ("wind_speed", "wind_sin", "wind_cos"):
        setattr(holder.feature_assembler, name, value)
        return
    target = observation if name in ("neighbor_burning_count", "wind_weighted_score") else holder.feature_assembler
    attribute = {"neighbor_burning_count": "blazing_neighbor_count_t",
                 "wind_weighted_score": "wind_weighted_score_t"}.get(name, name)
    values = getattr(target, attribute).astype(np.float64)
    values[-1, -1] = value
    setattr(target, attribute, values)


@pytest.mark.parametrize("wind", collection.WINDS)
@pytest.mark.parametrize("selection", ["empty", "sparse", "dense"])
@pytest.mark.parametrize("layout", ["c", "f", "strided", "reversed"])
def test_eligible_features_match_reference_bits_and_leave_sources_unchanged(wind, selection, layout):
    holder, observation = _feature_sources_case(wind, layout)
    indices = {"empty": np.empty(0, dtype=np.int64),
               "sparse": np.array([0, 3, 128, 267]), "dense": np.arange(268)}[selection]
    sources = [getattr(holder.feature_assembler, name) for name in
        ("slope_risk", "proximity_risk", "building_presence", "material_risk",
         "material_class", "composite_flammability")]
    sources += [observation.blazing_neighbor_count_t, observation.wind_weighted_score_t]
    before = [(value.dtype.str, value.shape, value.strides, value.tobytes()) for value in sources]
    expected = _reference_features(holder, observation, indices)
    actual = collection.CandidateAutomata._assemble_eligible_features(holder, observation, indices)
    assert expected.dtype == actual.dtype == np.float32
    assert actual.shape == expected.shape == (len(indices), 11)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    if selection != "empty":
        assert actual[0, 0].view(np.uint32) == actual[0, 1].view(np.uint32) == 0x80000000
        assert actual[0, 10].view(np.uint32) == 0x80000000
    assert before == [(value.dtype.str, value.shape, value.strides, value.tobytes()) for value in sources]


@pytest.mark.parametrize("conversion", ["dynamic_float64", "integer_common_promotion"])
def test_eligible_features_preserve_reference_conversion_sequence(conversion):
    holder, observation = _feature_sources_case()
    if conversion == "dynamic_float64":
        observation.blazing_neighbor_count_t = observation.blazing_neighbor_count_t.astype(np.float64)
        observation.blazing_neighbor_count_t[1, 1] = 2.0 + 2.0 ** -25
        observation.wind_weighted_score_t = observation.wind_weighted_score_t.astype(np.float64)
        observation.wind_weighted_score_t[1, 1] = 0.25 + 2.0 ** -28
    else:
        holder.feature_assembler.proximity_risk = np.full(holder.grid_shape,
            (1 << 60) + (1 << 36) + 1, dtype=np.int64)
    indices = np.arange(268)
    expected = _reference_features(holder, observation, indices)
    actual = collection.CandidateAutomata._assemble_eligible_features(holder, observation, indices)
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("name", CANONICAL_FEATURE_NAMES)
@pytest.mark.parametrize("empty", [False, True])
def test_all_feature_columns_reject_nonfinite_values_outside_selection(name, empty):
    holder, observation = _feature_sources_case()
    _set_feature_source(holder, observation, name, np.nan)
    indices = np.empty(0, dtype=np.int64) if empty else np.array([0])
    for route in (_reference_features, collection.CandidateAutomata._assemble_eligible_features):
        with pytest.raises(ValueError, match="non-finite"):
            route(holder, observation, indices)


@pytest.mark.parametrize("name,value", [
    ("slope_risk", -1), ("slope_risk", 2), ("building_presence", 0.5),
    ("material_class", 1.5), ("material_class", -1), ("material_class", 6),
    ("material_risk", 0.1), ("wind_speed", -1), ("wind_sin", 2), ("wind_cos", -2),
    ("neighbor_burning_count", 0.5), ("neighbor_burning_count", -1),
    ("neighbor_burning_count", 9), ("composite_flammability", 0.1),
    ("wind_weighted_score", -0.1), ("wind_weighted_score", 1.1),
    ("proximity_risk", 1e100), ("wind_speed", np.inf)])
@pytest.mark.parametrize("empty", [False, True])
def test_full_grid_feature_predicates_reject_excluded_values(name, value, empty):
    holder, observation = _feature_sources_case()
    _set_feature_source(holder, observation, name, value)
    indices = np.empty(0, dtype=np.int64) if empty else np.array([0])
    with np.errstate(over="ignore", invalid="ignore"):
        for route in (_reference_features, collection.CandidateAutomata._assemble_eligible_features):
            with pytest.raises(ValueError):
                route(holder, observation, indices)


@pytest.mark.parametrize("field", ["count", "score", "static", "assembler"])
@pytest.mark.parametrize("empty", [False, True])
def test_eligible_feature_shapes_reject_before_empty_return(field, empty):
    holder, observation = _feature_sources_case()
    if field == "count":
        observation.blazing_neighbor_count_t = np.zeros((1, 1), dtype=np.int8)
    elif field == "score":
        observation.wind_weighted_score_t = np.zeros((1, 1), dtype=np.float32)
    elif field == "static":
        holder.feature_assembler.slope_risk = np.zeros((1, 1), dtype=np.float32)
    else:
        holder.feature_assembler.grid_shape = (1, 1)
    indices = np.empty(0, dtype=np.int64) if empty else np.array([0])
    for route in (_reference_features, collection.CandidateAutomata._assemble_eligible_features):
        with pytest.raises(ValueError):
            route(holder, observation, indices)


@pytest.mark.parametrize("name,value", [
    ("material_class", 3.0 + 2e-5), ("neighbor_burning_count", 6.0 + 5e-5),
    ("material_risk", 0.48 + 5e-6), ("composite_flammability", 0.48 + 5e-6)])
def test_eligible_features_preserve_existing_relative_and_absolute_tolerances(name, value):
    holder, observation = _feature_sources_case()
    _set_feature_source(holder, observation, name, value)
    indices = np.array([267])
    expected = _reference_features(holder, observation, indices)
    actual = collection.CandidateAutomata._assemble_eligible_features(holder, observation, indices)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("wind_ordinal", range(1, 8))
@pytest.mark.parametrize("empty", [False, True])
def test_complete_batches_match_reference_without_full_grid_assembly(monkeypatch, wind_ordinal, empty):
    raw, masks = arrays()
    if empty:
        raw["buildings"][:] = raw["materials"][:] = 0
        raw["buildings"][1, 1], raw["materials"][1, 1] = 10, 1
    _, config, spec, _, construct = context(origin=(128, 256), raw=raw, masks=masks)
    spec.update(collection.planned_runs(config["inventory"])[(wind_ordinal - 1) * 2])
    actual, reference = construct(), construct()
    monkeypatch.setattr(reference, "_assemble_eligible_features",
        lambda observation, indices: _reference_features(reference, observation, indices))
    monkeypatch.setattr(actual.feature_assembler, "assemble_grid_features",
        lambda *_: pytest.fail("candidate invoked full-grid feature assembly"))
    class TracedDraws:
        def __init__(self, engine):
            self.engine, self.generator, self.calls = engine, engine.rng, []
        def random(self, shape):
            indices, q, bits, score = self.engine._capture
            self.calls.append((shape, indices.tobytes(), q.dtype.str, q.tobytes(),
                               bits.tobytes(), score.tobytes()))
            return self.generator.random(shape)
    actual.rng, reference.rng = TracedDraws(actual), TracedDraws(reference)
    first_snapshot = None
    for _ in range(3):
        left, right = actual.step(), reference.step()
        assert left.envelope_bytes == right.envelope_bytes
        for name in ("features", "labels", "global_rows", "global_cols", "observation_ids", "q32_bits"):
            a, b = getattr(left, name), getattr(right, name)
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            assert not a.flags.writeable and not b.flags.writeable
        assert actual.grid.tobytes() == reference.grid.tobytes()
        for name in ("ignition_timers", "blazing_timers", "p_base"):
            assert getattr(actual, name).tobytes() == getattr(reference, name).tobytes()
        assert actual.last_completed == reference.last_completed and actual.row_count == reference.row_count
        assert actual.rng.calls == reference.rng.calls
        assert all(call[0] == (3, 3) for call in actual.rng.calls)
        assert actual.rng.generator.bit_generator.state == reference.rng.generator.bit_generator.state
        assert actual._capture is reference._capture is actual._pending is reference._pending is None
        if first_snapshot is None:
            first_snapshot = actual.batches
    assert bool(actual.rng.calls) is (not empty)
    current, count, completed = actual.batches, actual.row_count, deepcopy(actual.last_completed)
    unchanged = (actual.grid.tobytes(), actual.ignition_timers.tobytes(), actual.blazing_timers.tobytes(),
                 deepcopy(actual.rng.generator.bit_generator.state), actual.failed, actual.incomplete_transition)
    for bad in ((), tuple(list(current)), first_snapshot):
        with pytest.raises(ValueError):
            actual.acknowledge_batches(bad)
        assert actual.batches is current
    actual.acknowledge_batches(current)  # Identity/release check only; not a claim of persistence.
    assert actual.batches == () and actual.row_count == count and actual.last_completed == completed
    assert unchanged == (actual.grid.tobytes(), actual.ignition_timers.tobytes(), actual.blazing_timers.tobytes(),
        actual.rng.generator.bit_generator.state, actual.failed, actual.incomplete_transition)


@pytest.mark.parametrize("empty", [False, True])
def test_feature_rejection_retains_prior_committed_batch_at_existing_failure_boundary(empty):
    _, _, spec, _, construct = context()
    engine = construct()
    engine.step()
    old_batches, old_count, old_record = engine.batches, engine.row_count, deepcopy(engine.last_completed)
    old_evidence = [(batch.envelope_bytes, tuple(getattr(batch, name).tobytes() for name in
        ("features", "labels", "global_rows", "global_cols", "observation_ids", "q32_bits"))) for batch in old_batches]
    engine.grid[:] = 5 if empty else 2
    if not empty:
        engine.grid[1, 1] = 4
    engine.ignition_timers[:] = engine.blazing_timers[:] = 0
    engine.feature_assembler.slope_risk = engine.feature_assembler.slope_risk.copy()
    engine.feature_assembler.slope_risk[1, 1] = np.nan  # Not an eligible receiving cell.
    expected = np.random.default_rng(spec["seed"])
    expected.bit_generator.state = deepcopy(engine.rng.bit_generator.state)
    if not empty:
        expected.random((3, 3))
    timestep = engine.timestep
    with pytest.raises(ValueError, match="non-finite"):
        engine.step()
    assert engine.timestep == timestep + 1 and engine.failed
    assert engine.batches is old_batches and engine.row_count == old_count and engine.last_completed == old_record
    assert old_evidence == [(batch.envelope_bytes, tuple(getattr(batch, name).tobytes() for name in
        ("features", "labels", "global_rows", "global_cols", "observation_ids", "q32_bits"))) for batch in engine.batches]
    assert engine.incomplete_transition == dict(from_timestep=timestep,
        to_timestep=timestep + 1, collection_completed=False)
    assert engine.rng.bit_generator.state == expected.bit_generator.state
    assert engine._capture is engine._pending is None


class FullGridReferenceCandidate(collection.CandidateAutomata):
    """Unchanged engine/observer, R2 probability override and full reference assembler."""
    def __init__(self, *args, **kwargs):
        assert kwargs.get("static_context") is None
        super().__init__(*args, **kwargs)

    def _precompute_base_probability(self):
        from modules.additive_exposure_teacher import compute_susceptibility32
        self.p_base = compute_susceptibility32(slope32=self.slope_risk,
            building32=self.building_presence, proximity32=self.proximity_risk,
            material_risk32=self.material_risk).beta32

    def _model_free_probability(self, indices, score):
        q = compute_additive_exposure_probability(beta32=self.p_base.ravel()[indices],
            wind_weighted_score32=score.ravel()[indices]).q32
        q.setflags(write=False)
        self._capture = indices.copy(), q, q.view(np.uint32).copy(), score.ravel()[indices].copy()
        return q

    def _stage_observation(self, observation):
        from modules.transition_observer import validate_transition_observation
        assert self._pending is None
        self._pending = validate_transition_observation(observation)
        if self._observer is not None:
            self._observer(self._pending)

    def _assemble_eligible_features(self, observation, indices):
        return self.feature_assembler.assemble_grid_features(observation.blazing_neighbor_count_t,
            observation.wind_weighted_score_t)[indices]

    def step(self):
        from modules.automata_engine import FireAutomata
        if self.failed or self.timestep >= collection.MAX_UPDATES:
            raise RuntimeError("Failed or capped reference run")
        self._capture = self._pending = None
        timestep = self.timestep
        try:
            FireAutomata.step(self)
            for timer in (self.ignition_timers, self.blazing_timers):
                if timer.dtype != np.int16 or timer.shape != self.grid_shape or np.any(timer < 0):
                    raise ValueError("Reference timer invariant")
            batch = self._make_batch()
            count = self.row_count + len(batch.labels)
            if count > collection.MAX_RUN_ROWS:
                raise RuntimeError("Reference row cap")
            batches = self._batches + (batch,)
            record = dict(state_timestep=self.timestep, state_counts=self.get_state_counts(), committed_rows=count)
        except BaseException:
            self.failed = True
            self.incomplete_transition = dict(from_timestep=timestep, to_timestep=self.timestep,
                                              collection_completed=False)
            self._capture = self._pending = None
            raise
        self._batches, self.row_count, self.last_completed = batches, count, record
        self._capture = self._pending = None
        return batch


class TracedCandidateGenerator:
    def __init__(self, generator, *, force_zero=False):
        self.generator, self.force_zero, self.calls = generator, force_zero, []

    @property
    def bit_generator(self):
        return self.generator.bit_generator

    def random(self, shape):
        self.calls.append(tuple(shape))
        result = self.generator.random(shape)
        if self.force_zero:
            result.fill(0.0)  # Explicit synthetic draw substitute, after the counted real draw.
        return result


def candidate_reference_pair(monkeypatch, *, scenario="mixed", wind_ordinal=7, layout="c", force_zero=False, origin=(128, 256)):
    size = 129 if scenario == "seams" else 3 if scenario == "compact" else 9
    raw, masks = arrays(size)
    rr, cc = np.indices((size, size))
    raw["materials"][:] = (rr // 2 + cc // 2) % 5 + 1
    raw["slope"][:] = (rr + cc) % 11
    raw["proximity"][:] = (rr - cc) / 7
    raw["slope"][0, 0] = raw["proximity"][0, 0] = -0.0
    if scenario == "zero_q":
        raw["proximity"][:] = -100
    points = [[1, 1], [size - 2, size - 2]]
    if scenario == "compact":
        points = [[1, 1]]
    if scenario == "seams":
        points = [[0, 0], [63, 63], [64, 65], [127, 127], [128, 0]]
    if scenario == "background":
        raw["materials"][0, :] = raw["buildings"][0, :] = 0
        masks["slope"][3, 3] = 0
    if scenario == "isolated":
        raw["materials"][:] = raw["buildings"][:] = 0
        raw["materials"][1, 1], raw["buildings"][1, 1] = 1, 10
        points = [[1, 1]]
    prepared, config, spec, provenance, _ = context(raw=raw, masks=masks, origin=origin)
    for family in config["inventory"]:
        family["ignition_points"] = [[r + origin[0], c + origin[1]] for r, c in points]
    spec.update(collection.planned_runs(config["inventory"])[2 * (wind_ordinal - 1)])
    if layout == "strided":
        for name, value in tuple(prepared.environment.items()):
            if isinstance(value, np.ndarray):
                backing = np.empty((size, size * 2), dtype=value.dtype)
                backing[:, ::2] = value
                prepared.environment[name] = backing[:, ::2]
    config_before = deepcopy(config)
    binding = collection.execution_binding(config, prepared, spec, provenance)
    static = collection.build_candidate_context(prepared, config)
    assert config == config_before
    assert collection.execution_binding(config, static.prepared, spec, provenance,
                                        static_context=static) == binding
    original_factory = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda seed:
        TracedCandidateGenerator(original_factory(seed), force_zero=force_zero))
    observations = ([], [])
    options = dict(execution_authorized=True, approval_reference=config["authorization_reference"],
        expected_binding_sha256=hashlib.sha256(collection._bytes(binding)).hexdigest())
    actual = collection.CandidateAutomata(static.prepared, config, spec, provenance,
        static_context=static, observer=observations[0].append, **options)
    reference = FullGridReferenceCandidate(prepared, config, spec, provenance,
        observer=observations[1].append, **options)
    return actual, reference, observations, config, spec, provenance


def assert_candidate_pair(actual, reference, left, right):
    assert left.envelope_bytes == right.envelope_bytes
    for name in ("features", "labels", "global_rows", "global_cols", "observation_ids", "q32_bits"):
        a, b = getattr(left, name), getattr(right, name)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    for name in ("grid", "ignition_timers", "blazing_timers", "p_base"):
        a, b = getattr(actual, name), getattr(reference, name)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert actual.timestep == reference.timestep
    assert actual.is_active() == reference.is_active()
    assert actual.last_completed == reference.last_completed
    assert actual.rng.calls == reference.rng.calls
    assert actual.rng.bit_generator.state == reference.rng.bit_generator.state
    assert not actual.failed and not reference.failed
    assert actual.incomplete_transition is reference.incomplete_transition is None


@pytest.mark.parametrize("wind", collection.WINDS)
@pytest.mark.parametrize("pattern", ["none", "corner", "overlap", "separate", "seams", "dense"])
@pytest.mark.parametrize("tile_size", [3, 64, 127])
def test_active_fields_match_unchanged_full_grid_helpers_bitwise(wind, pattern, tile_size):
    from modules.feature_pipeline import compute_moore_neighbor_count
    shape = (129, 130)
    state = np.ones(shape, dtype=np.int8)
    points = dict(none=[], corner=[(0, 0), (128, 129)], overlap=[(5, 5), (5, 6), (6, 5)],
        separate=[(4, 4), (80, 90)], seams=[(63, 63), (64, 65), (126, 127), (127, 128)],
        dense=[(r, c) for r in range(60, 69) for c in range(60, 69)])[pattern]
    for point in points:
        state[point] = 4
    blazing = (state == 4).astype(np.int8)
    expected_count = compute_moore_neighbor_count(blazing)
    expected_score, kernel = compute_wind_weighted_score(blazing, WindContract(*wind), 0.20)
    before = state.tobytes()
    count, score = collection._active_neighbour_fields(state, kernel, tile_size)
    assert state.tobytes() == before
    assert count.dtype == np.int8 and score.dtype == np.float32
    assert count.tobytes() == expected_count.tobytes()
    assert score.view(np.uint32).tobytes() == expected_score.view(np.uint32).tobytes()
    assert np.all(score[count == 0] == 0)


@pytest.mark.parametrize("wind_ordinal", range(1, 8))
@pytest.mark.parametrize("scenario", ["mixed", "background", "isolated", "zero_q", "seams"])
@pytest.mark.parametrize("layout", ["c", "strided"])
def test_owned_candidate_multistep_matches_independent_r2_reference(monkeypatch, wind_ordinal, scenario, layout):
    from dataclasses import fields
    actual, reference, observations, _, _, _ = candidate_reference_pair(monkeypatch,
        scenario=scenario, wind_ordinal=wind_ordinal, layout=layout)
    retained = None
    for _ in range(5):
        left, right = actual.step(), reference.step()
        assert_candidate_pair(actual, reference, left, right)
        a, b = observations[0][-1], observations[1][-1]
        for field in fields(a):
            x, y = getattr(a, field.name), getattr(b, field.name)
            if isinstance(x, np.ndarray):
                assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
                with pytest.raises(ValueError):
                    x.setflags(write=True)
            else:
                assert canonical_metadata_hash({"value": x}) == canonical_metadata_hash({"value": y})
        if actual.draw_evidence is not None:
            evidence = actual.draw_evidence
            assert evidence.generator is actual.rng and evidence.bit_generator is actual.rng.bit_generator
            assert evidence.state_after == collection._bytes(actual.rng.bit_generator.state)
            assert len(evidence.indices.array) == len(left.labels)
            with pytest.raises(ValueError):
                evidence.draw.array.setflags(write=True)
        if retained is None:
            retained = a, tuple(getattr(a, f.name).tobytes() for f in fields(a)
                                if isinstance(getattr(a, f.name), np.ndarray))
    assert retained[1] == tuple(getattr(retained[0], f.name).tobytes() for f in fields(retained[0])
        if isinstance(getattr(retained[0], f.name), np.ndarray))
    if scenario == "zero_q":
        assert actual.rng.calls == [(9, 9)]
        assert np.all(actual.batches[0].q32_bits == 0) and not np.any(actual.batches[0].labels)
    if scenario == "isolated":
        assert actual.rng.calls == [] and all(len(b.labels) == 0 for b in actual.batches)


def test_context_creation_has_no_rng_and_reuses_owned_storage_with_fresh_runs(monkeypatch):
    prepared, config, spec, provenance, _ = context()
    original_factory = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda *_: pytest.fail("context created RNG"))
    static = collection.build_candidate_context(prepared, config)
    # The old mutable buffers are retired, not authoritative aliases of the owned context.
    prepared.environment["slope_risk"].setflags(write=True)
    prepared.environment["slope_risk"][:] = np.nan
    static.check(static.prepared, config)
    with pytest.raises(ValueError, match="does not bind"):
        static.check(prepared, config)
    monkeypatch.setattr(np.random, "default_rng", original_factory)
    engines = []
    for spec in collection.planned_runs(config["inventory"])[:2]:
        binding = collection.execution_binding(config, static.prepared, spec, provenance, static_context=static)
        engines.append(collection.CandidateAutomata(static.prepared, config, spec, provenance,
            static_context=static, execution_authorized=True, approval_reference=config["authorization_reference"],
            expected_binding_sha256=hashlib.sha256(collection._bytes(binding)).hexdigest()))
    a, b = engines
    assert a.rng is not b.rng and a.rng.bit_generator is not b.rng.bit_generator
    assert a.p_base is b.p_base and a.feature_assembler.slope_risk is b.feature_assembler.slope_risk
    for name in ("grid", "ignition_timers", "blazing_timers"):
        assert not np.shares_memory(getattr(a, name), getattr(b, name))
    before = b.grid.tobytes(), b.ignition_timers.tobytes(), deepcopy(b.rng.bit_generator.state)
    a.step()
    assert before == (b.grid.tobytes(), b.ignition_timers.tobytes(), b.rng.bit_generator.state)


@pytest.mark.parametrize("field", ["slope_risk", "proximity_risk", "building_presence", "material_class"])
def test_context_rejects_invalid_excluded_static_cells_before_rng(monkeypatch, field):
    raw, masks = arrays()
    masks["slope"][0, 0] = 0
    prepared, config, _, _, _ = context(raw=raw, masks=masks)
    values = prepared.environment[field].astype(np.float32)
    values[0, 0] = np.nan if field != "material_class" else 6
    prepared.environment[field] = values
    monkeypatch.setattr(np.random, "default_rng", lambda *_: pytest.fail("invalid context created RNG"))
    with pytest.raises((ValueError, TypeError)):
        collection.build_candidate_context(prepared, config)


def test_candidate_checker_uses_actual_draw_without_replaying_generator(monkeypatch):
    actual, reference, _, _, _, _ = candidate_reference_pair(monkeypatch, force_zero=True)
    left, right = actual.step(), reference.step()
    assert_candidate_pair(actual, reference, left, right)
    assert actual.rng.calls == reference.rng.calls == [(9, 9)]
    assert np.all(actual.draw_evidence.draw.array == 0) and np.all(left.labels == 1)


@pytest.mark.parametrize("fault", ["different_preparation", "metadata", "feature_shape",
                                   "receiving_shape", "copied_context", "dependencies", "configuration"])
def test_supplied_context_never_falls_back_or_creates_rng_on_mismatch(monkeypatch, fault):
    from dataclasses import replace
    prepared, config, spec, provenance, _ = context()
    static = collection.build_candidate_context(prepared, config)
    binding = collection.execution_binding(config, static.prepared, spec, provenance, static_context=static)
    supplied = static.prepared
    if fault == "different_preparation":
        supplied = prepared
    elif fault == "metadata":
        static.prepared.interpretation["changed"] = True
    elif fault == "feature_shape":
        static.features["slope_risk"].shape = (9,)
    elif fault == "receiving_shape":
        static.prepared.receiving_mask.shape = (9,)
    elif fault == "copied_context":
        static = replace(static)
    elif fault == "dependencies":
        changed = FeatureAssembler.MATERIAL_CLASS_TO_RISK.copy()
        changed[1] = 0.5
        monkeypatch.setattr(FeatureAssembler, "MATERIAL_CLASS_TO_RISK", changed)
    else:
        config["inputs"]["slope"]["path"] += "-changed"
    monkeypatch.setattr(np.random, "default_rng", lambda *_: pytest.fail("mismatched context created RNG"))
    with pytest.raises((ValueError, PermissionError)):
        collection.CandidateAutomata(supplied, config, spec, provenance, static_context=static,
            execution_authorized=True, approval_reference=config["authorization_reference"],
            expected_binding_sha256=hashlib.sha256(collection._bytes(binding)).hexdigest())


@pytest.mark.parametrize("fault", ["state", "ignition_timer", "blazing_timer", "feature_view",
    "context", "rng_state", "grid_view", "observation_view", "metadata", "capture",
    "accounting", "record", "retained_batch_shape", "observer_exception", "route_flag"])
def test_owned_candidate_callback_mutations_latch_without_new_commit(fault):
    prepared, config, spec, provenance, _ = context()
    static = collection.build_candidate_context(prepared, config)
    binding = collection.execution_binding(config, static.prepared, spec, provenance, static_context=static)
    holder = {}
    def callback(observation):
        if holder.get("corrupt") is not True:
            return
        engine = holder["engine"]
        if fault == "state":
            engine.grid[-1, -1] = 1
        elif fault == "ignition_timer":
            engine.ignition_timers[-1, -1] += 1
        elif fault == "blazing_timer":
            engine.blazing_timers[-1, -1] += 1
        elif fault == "feature_view":
            engine.feature_assembler.slope_risk = engine.feature_assembler.slope_risk.view()
        elif fault == "context":
            engine._static_context = None
        elif fault == "rng_state":
            state = deepcopy(engine.rng.bit_generator.state)
            state["state"]["state"] ^= 1
            engine.rng.bit_generator.state = state
        elif fault == "grid_view":
            engine.grid = engine.grid.view()
        elif fault == "observation_view":
            object.__setattr__(observation, "state_t1", observation.state_t1.view())
        elif fault == "metadata":
            object.__setattr__(observation, "seed", 0)
        elif fault == "capture":
            if engine._capture is None:
                engine._capture = (np.array([0]),) * 4
            else:
                engine._capture[2][0] ^= np.uint32(1)
        elif fault == "accounting":
            engine.row_count += 1
        elif fault == "record":
            engine.last_completed["state_counts"]["blazing"] = 0
        elif fault == "retained_batch_shape":
            engine.batches[0].features.shape = (engine.batches[0].features.size,)
        elif fault == "observer_exception":
            raise RuntimeError("injected optimized observer failure")
        else:
            engine._context_bound = False
    engine = collection.CandidateAutomata(static.prepared, config, spec, provenance,
        static_context=static, observer=callback, execution_authorized=True,
        approval_reference=config["authorization_reference"],
        expected_binding_sha256=hashlib.sha256(collection._bytes(binding)).hexdigest())
    holder["engine"] = engine
    first = engine.step()
    old, count, record = engine.batches, engine.row_count, deepcopy(engine.last_completed)
    holder["corrupt"] = True
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        engine.step()
    assert engine.failed and engine.batches is old and old[0] is first
    assert engine.last_completed == record
    assert engine.row_count == count
    assert engine._capture is engine._pending is None
    with pytest.raises(RuntimeError):
        engine.step()


@pytest.mark.parametrize("fault", ["count", "score"])
def test_independent_candidate_check_rejects_fields_outside_producer_footprint(monkeypatch, fault):
    actual, _, _, _, _, _ = candidate_reference_pair(monkeypatch, scenario="isolated")
    original = collection._active_neighbour_fields
    def corrupted(state, kernel, tile_size):
        count, score = original(state, kernel, tile_size)
        if tile_size == 64:
            (count if fault == "count" else score)[-1, -1] = 1
        return count, score
    monkeypatch.setattr(collection, "_active_neighbour_fields", corrupted)
    with pytest.raises(ValueError, match="neighbourhood"):
        actual.step()
    assert actual.failed and not actual.batches and actual.row_count == 0


@pytest.mark.parametrize("fault", ["indices", "q32", "draw_dtype", "missing_draw", "state", "timer"])
def test_candidate_check_rejects_producer_decision_and_complement_faults(monkeypatch, fault):
    actual, _, _, _, _, _ = candidate_reference_pair(monkeypatch, scenario="compact", force_zero=True)
    original = actual._verify_candidate_transition
    def corrupt(before, timestep, rng_before, count, score, evidence):
        if fault == "indices":
            values = evidence.indices.array.copy()
            values[0] = values[-1]
            object.__setattr__(evidence, "indices", collection._OwnedArray.make(values))
        elif fault == "q32":
            actual._capture[2][0] ^= np.uint32(1)
        elif fault == "draw_dtype":
            object.__setattr__(evidence, "draw", collection._OwnedArray.make(evidence.draw.array, np.float32))
        elif fault == "missing_draw":
            actual._draw_evidence = None
        elif fault == "state":
            actual.grid[1, 1] = 4  # Original blazing source must have extinguished.
        else:
            actual.ignition_timers[1, 1] = 1  # Outside the ignition-update footprint.
        return original(before, timestep, rng_before, count, score, evidence)
    monkeypatch.setattr(actual, "_verify_candidate_transition", corrupt)
    with pytest.raises(ValueError):
        actual.step()
    assert actual.failed and actual.row_count == 0 and not actual.batches
    assert actual._capture is actual._pending is None


@pytest.mark.parametrize("cap", ["rows", "updates"])
def test_owned_candidate_caps_preserve_accounting_and_prevent_extra_updates(monkeypatch, cap):
    actual, _, _, _, _, _ = candidate_reference_pair(monkeypatch, scenario="compact")
    before = deepcopy(actual.last_completed), actual.row_count, len(actual.rng.calls)
    if cap == "rows":
        monkeypatch.setattr(collection, "MAX_RUN_ROWS", 0)
        with pytest.raises(RuntimeError, match="row cap"):
            actual.step()
        assert actual.failed and not actual.batches and len(actual.rng.calls) == before[2] + 1
    else:
        actual.timestep = collection.MAX_UPDATES
        with pytest.raises(RuntimeError, match="capped"):
            actual.step()
        assert len(actual.rng.calls) == before[2]
    assert actual.last_completed == before[0] and actual.row_count == before[1]


@pytest.mark.parametrize("name", ["grid", "ignition_timers", "blazing_timers"])
def test_owned_candidate_rejects_equal_mutable_state_views_before_rng(monkeypatch, name):
    actual, _, _, _, _, _ = candidate_reference_pair(monkeypatch, scenario="compact")
    previous = deepcopy(actual.last_completed), deepcopy(actual.rng.bit_generator.state)
    setattr(actual, name, getattr(actual, name).view())
    with pytest.raises(ValueError, match="state/timer mutation"):
        actual.step()
    assert actual.failed and not actual.rng.calls and actual.timestep == 0
    assert (actual.last_completed, actual.rng.bit_generator.state) == previous


def test_batch_envelope_preserves_preoptimization_keys_and_canonical_bytes(monkeypatch):
    # Before-image SHA-256: 2751fe94cdc3196aed11661a5e8aa17a24ae201fc1ee355b91a0bea1025e5339.
    actual, reference, _, _, _, _ = candidate_reference_pair(monkeypatch,
        scenario="compact", force_zero=True)
    left, right = actual.step(), reference.step()
    before = np.full((3, 3), 2, dtype=np.int8)
    after = np.full((3, 3), 3, dtype=np.int8)
    before[1, 1], after[1, 1] = 4, 5
    eligible = np.ones((3, 3), dtype=bool)
    eligible[1, 1] = False
    provenance = json.loads(actual._capture_provenance_bytes)
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False).encode("utf-8")
    bindings = [dict(observation_id=identity.decode("ascii"), q32_bits=int(bits),
        row_sha256=hashlib.sha256(struct.pack("<11f", *map(float, row)) +
                                  struct.pack("b", int(label))).hexdigest())
        for identity, bits, row, label in zip(left.observation_ids, left.q32_bits,
                                             left.features, left.labels, strict=True)]
    expected = dict(schema_version="candidate_comparison_capture.v1",
        run_identity_sha256=actual.run_identity_sha256,
        provenance_sha256=hashlib.sha256(encoded(provenance)).hexdigest(),
        teacher_id="exploratory_additive_exposure_teacher.r2",
        numerical_contract_id="additive_exposure_quantized_f32_f64.r2",
        interpretation_policy="embedded_tiff_only.v1",
        mask_seed_policy="candidate_common_environment_receiving_atomic_seeds.v1",
        timestep_t=0, timestep_t1=1, generic_provenance=provenance,
        state_t_sha256=hashlib.sha256(before.tobytes()).hexdigest(),
        state_t1_sha256=hashlib.sha256(after.tobytes()).hexdigest(),
        eligible_mask_sha256=hashlib.sha256(str(eligible.shape).encode("ascii") +
                                           b"|bool|" + eligible.tobytes()).hexdigest(),
        row_hash_encoding="eleven_little_endian_float32_then_int8_label.v1", bindings=bindings)
    keys = {"schema_version", "run_identity_sha256", "provenance_sha256", "teacher_id",
        "numerical_contract_id", "interpretation_policy", "mask_seed_policy", "timestep_t",
        "timestep_t1", "generic_provenance", "state_t_sha256", "state_t1_sha256",
        "eligible_mask_sha256", "row_hash_encoding", "bindings"}
    assert set(json.loads(left.envelope_bytes)) == set(expected) == keys
    assert left.envelope_bytes == right.envelope_bytes == encoded(expected)
    assert all(set(item) == {"observation_id", "q32_bits", "row_sha256"} for item in bindings)
