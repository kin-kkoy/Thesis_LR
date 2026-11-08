"""Synthetic only: source/config text, tiny estimators and fresh temporary native exports."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import builtins
import io
import json
import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import orchestrator as o
import dswd_snapshot_results as scoring
import dswd_inference_guard as inference_guard
from modules.automata_engine import FireAutomata
from modules.feature_pipeline import CANONICAL_FEATURE_NAMES


@pytest.fixture(autouse=True)
def forbid_production_io(monkeypatch, tmp_path):
    code = Path(o.__file__).parent.as_posix().lower()
    temporary = tmp_path.as_posix().lower()
    def guard(path):
        if isinstance(path, int):
            return
        value = str(path).replace("\\", "/").lower()
        if value.startswith(temporary + "/"):
            return
        if ("/thesis_lr/" in value or any(value.startswith(code + "/" + part)
                for part in ("output/", "processeddata/", "datafiles/", "models/"))):
            raise AssertionError("Synthetic test attempted production I/O or resolution: " + value)
    real_open, real_io_open = builtins.open, io.open
    real_raster, real_resolve = rasterio.open, Path.resolve
    def opened(delegate, path, *args, **kwargs):
        guard(path)
        return delegate(path, *args, **kwargs)
    def resolve(path, *args, **kwargs):
        guard(path)
        return real_resolve(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", lambda path, *a, **k: opened(real_open, path, *a, **k))
    monkeypatch.setattr(io, "open", lambda path, *a, **k: opened(real_io_open, path, *a, **k))
    monkeypatch.setattr(rasterio, "open", lambda path, *a, **k: opened(real_raster, path, *a, **k))
    monkeypatch.setattr(Path, "resolve", resolve)


def base_config(incident):
    name = ("phase16_airbase_rf_nocutoff_s42_exec_r1.yaml" if incident == "airbase"
            else "phase16_candidate_r2_f01_nasa_hr11_exec_r1.yaml")
    return o._phase15_yaml((Path(o.__file__).parent / "config" / name).read_bytes())


def snapshot_config(incident, seed=42):
    config = base_config(incident)
    scenario = (o.PHASE16_AIRBASE_SCENARIO_ID if incident == "airbase" else o.PHASE16_SCENARIO_ID)
    config["phase15"].update(
        schema_version=o.PHASE16_DSWD_SCHEMA, enabled=False,
        authorization_reference="synthetic-owner-only",
        run_id=f"p16-dswd-{incident.replace('_', '-')}-rf-s{seed}-r1",
        scenario_id=scenario.replace("-s42-", f"-s{seed}-") + "-dswd-snap-v1")
    config["simulation"].update(seed=seed, max_timesteps=172 if incident == "airbase" else 424)
    config["snapshot_basis"] = o._phase16_snapshot_basis(incident)
    return config


def tiny_basis(horizons=(1, 2, 3)):
    return {"horizons": [{"requested_update": h, "assumed_minutes_per_update": 1,
                          "clock_role": "synthetic", "clock_status": "ASSUMPTION"}
                         for h in horizons], "duration_basis": {"source": "synthetic"}}


class Scripted:
    def __init__(self, states):
        self.states = [np.array([s], dtype=np.int8) for s in states]
        self.grid, self.timestep = self.states[0].copy(), 0
    def get_grid(self):
        return self.grid
    def step(self):
        self.timestep += 1
        self.grid[:] = self.states[min(self.timestep, len(self.states) - 1)]


def collect(series, engine, check=lambda boundary: None):
    captures = []
    def capture(row, state, reached):
        captures.append((row, state, reached))
        return []
    result = series.run(engine, check, capture)
    return result, captures


def test_disabled_manifest_is_exact_and_not_dispatchable(monkeypatch):
    manifest = o._phase15_yaml((Path(o.__file__).parent / o.PHASE16_DSWD_MANIFEST).read_bytes())
    assert manifest == {"dswd_snapshots": o.PHASE16_DSWD_PROTOCOL}
    assert manifest["dswd_snapshots"]["enabled"] is False
    assert manifest["dswd_snapshots"]["execution_authorized"] is False
    assert (manifest["dswd_snapshots"]["trajectory_count"], manifest["dswd_snapshots"]["planned_outcome_count"]) == (10, 30)
    monkeypatch.setattr(o, "_run_legacy_simulation", lambda *a, **k: pytest.fail("manifest dispatched"))
    with pytest.raises(PermissionError, match="preparation only"):
        o.run_simulation(manifest)


@pytest.mark.parametrize("incident,maximum", [("airbase", 172), ("santa_maria", 424)])
@pytest.mark.parametrize("seed", range(42, 47))
def test_frozen_snapshot_profile(incident, maximum, seed):
    config = snapshot_config(incident, seed)
    before = deepcopy(config)
    active = o._phase15_normalize(config)
    assert config == before and config["phase15"]["enabled"] is False
    assert active["simulation"]["max_timesteps"] == maximum
    assert active["simulation"]["seed"] == seed
    assert active["ml_model"]["inference_mode"] == "stochastic_probability"
    assert active["ml_model"]["threshold"] is None
    assert active["wind"]["direction_deg"] == (60.826537 if incident == "airbase" else 48.825208)
    assert active["placeholder_transition"] == {"T_3_to_4": 1, "T_4_to_5": 1, "wind_weight": 0.2}


@pytest.mark.parametrize("field,value", [("seed", 41), ("seed", True), ("max_timesteps", 300),
                                         ("max_timesteps", 424.0), ("ignition_points", [[0, 0]])])
def test_invalid_snapshot_simulation_controls(field, value):
    config = snapshot_config("santa_maria")
    config["simulation"][field] = value
    with pytest.raises((ValueError, TypeError)):
        o._phase15_normalize(config)


@pytest.mark.parametrize("mutation", ["clock", "horizon", "basis", "lr", "gate", "input", "wind", "enabled"])
def test_invalid_snapshot_bindings(mutation):
    config = snapshot_config("airbase")
    if mutation == "clock": config["snapshot_basis"]["horizons"][0]["assumed_minutes_per_update"] = True
    elif mutation == "horizon": config["snapshot_basis"]["horizons"][0]["requested_update"] = 171
    elif mutation == "basis": config["snapshot_basis"]["duration_basis"]["minutes"] = 90
    elif mutation == "lr": config["ml_model"]["model_sha256"] = o.PHASE16_LR_MODEL_SHA256
    elif mutation == "gate": config["ml_model"]["inference_mode"] = "stochastic_probability_gate"
    elif mutation == "input": config["environment"]["input_sha256"]["slope"] = "0" * 64
    elif mutation == "wind": config["wind"]["direction_deg"] = 61.0
    else: config["phase15"]["enabled"] = True
    with pytest.raises((ValueError, TypeError)):
        o._phase15_normalize(config)


@pytest.mark.parametrize("horizons", [(0,), (True,), (1.0,), (1, 1), ()])
def test_invalid_observer_horizons(horizons):
    with pytest.raises(ValueError):
        o._Phase16Snapshots(tiny_basis(horizons))


@pytest.mark.parametrize("stop", [0, 1, 2])
def test_initial_early_and_exact_horizon_inactivity(stop):
    engine = Scripted([[4]] * stop + [[5]])
    series = o._Phase16Snapshots(tiny_basis((2, 3, 4)))
    result, captures = collect(series, engine)
    assert result[0] == "inactive" and result[1] == stop
    assert len(series.curve) == stop + 1 and len(captures) == 3
    assert all(row["actual_capture_update"] == stop for row in series.rows)
    assert all(row["actual_stop_update"] == stop for row in series.rows)
    assert all(row["absorbing_carry_forward"] == (stop < row["requested_update"]) for row in series.rows)
    assert all(row["status"] == "complete" and row["output_valid"] for row in series.rows)


def test_post_update_boundaries_active_censor_and_immutable_copies():
    engine = Scripted([[4, 2, 2], [4, 3, 2], [5, 4, 3], [5, 5, 4]])
    series = o._Phase16Snapshots(tiny_basis())
    result, captures = collect(series, engine)
    assert result[:2] == ("timestep_limit", 3)
    for update, (row, native, reached) in enumerate(captures, 1):
        np.testing.assert_array_equal(native, engine.states[update])
        np.testing.assert_array_equal(reached, np.isin(native, (3, 4, 5)))
        assert row["actual_capture_update"] == row["requested_update"] == update
        assert not native.flags.writeable and not reached.flags.writeable
    saved = captures[0][1].copy()
    engine.grid[:] = 5
    np.testing.assert_array_equal(captures[0][1], saved)
    assert series.initial["initial_ignition_count"] == 1
    assert [r["new_ignition_count"] for r in series.curve] == [0, 1, 1, 0]
    assert all(row["status"] == "censored" for row in series.rows)


@pytest.mark.parametrize("failure", ["resource", "output", "step"])
def test_failure_never_becomes_successful_carry_forward(failure):
    engine = Scripted([[4], [4], [4], [5]])
    series = o._Phase16Snapshots(tiny_basis())
    def check(boundary):
        if failure == "resource" and boundary == "after_step" and series.completed == 2:
            raise o._Phase15Stop("resource_limit")
    def capture(row, state, reached):
        if failure == "output": raise OSError("synthetic export failure")
        return []
    if failure == "step":
        engine.step = lambda: (_ for _ in ()).throw(RuntimeError("synthetic step failure"))
    with pytest.raises((RuntimeError, OSError)):
        series.run(engine, check, capture)
    assert all(row["status"] == "incomplete" and not row["output_valid"] for row in series.rows)


class StubEstimator:
    feature_names_in_ = np.asarray(CANONICAL_FEATURE_NAMES)
    classes_ = np.array([0, 1])
    def __init__(self): self.calls = []
    def predict_proba(self, features):
        self.calls.append(np.asarray(features).copy())
        return np.tile([0.45, 0.55], (len(features), 1))


def tiny_engine(seed=42):
    shape = (5, 5)
    environment = dict(slope_risk=np.ones(shape, dtype=np.float32) * 0.1,
        proximity_risk=np.zeros(shape, dtype=np.float32), building_presence=np.ones(shape, dtype=np.int8),
        material_class=np.ones(shape, dtype=np.int8), burnable_mask=np.ones(shape, dtype=bool),
        nodata_mask=np.zeros(shape, dtype=bool), grid_shape=shape,
        crs=rasterio.crs.CRS.from_epsg(32651), transform=from_origin(0, 15, 3, 3))
    config = dict(simulation={"seed": seed, "emergency_checkpoint_enabled": False},
        wind={"speed_kmh": 0, "direction_deg": 0,
              "direction_convention": "meteorological_from", "direction_units": "degrees_clockwise",
              "north_reference": "projected_grid_north", "schema_version": "simulated_wind.v1",
              "calm_representation": "speed_zero_direction_zero"},
        placeholder_transition={"T_3_to_4": 1, "T_4_to_5": 1, "wind_weight": 0.2},
        ml_model={"inference_mode": "stochastic_probability", "positive_label": 1},
        flammability_weights={"base_weight": 0.5, "slope_weight": 0.3, "proximity_weight": 0.2})
    engine = FireAutomata(environment, config)
    engine.model = StubEstimator()
    engine.set_ignition([(2, 2)])
    return engine


def equal_trajectory(a, b):
    np.testing.assert_array_equal(a.grid, b.grid)
    np.testing.assert_array_equal(a.ignition_timers, b.ignition_timers)
    np.testing.assert_array_equal(a.blazing_timers, b.blazing_timers)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert len(a.model.calls) == len(b.model.calls)
    for left, right in zip(a.model.calls, b.model.calls):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("seed", range(42, 47))
def test_observation_and_separately_stopped_prefixes_preserve_state_timers_rng_inference(seed):
    observed, disabled = tiny_engine(seed), tiny_engine(seed)
    series = o._Phase16Snapshots(tiny_basis((2, 4, 6)))
    def capture(row, native, reached):
        independent = tiny_engine(seed)
        for _ in range(row["actual_capture_update"]): independent.step()
        equal_trajectory(observed, independent)
        np.testing.assert_array_equal(native, independent.get_grid())
        return []
    _, completed, _ = series.run(observed, lambda boundary: None, capture)
    for _ in range(completed): disabled.step()
    equal_trajectory(observed, disabled)


def test_native_export_counts_and_profile_only_allowlist(tmp_path):
    basis = tiny_basis((1, 2, 3))
    allowed = o._phase16_snapshot_outputs(basis)
    native = np.array([[1, 2, 3], [4, 5, 2]], dtype=np.int8)
    reached = np.isin(native, (3, 4, 5)).astype(np.uint8)
    engine = SimpleNamespace(crs=rasterio.crs.CRS.from_epsg(32651), transform=from_origin(0, 6, 3, 3))
    monitor = SimpleNamespace(check=lambda *args, **kwargs: None)
    for kind, array in (("state", native), ("reached", reached)):
        filename = f"snapshot_h0001_{kind}.tif"
        with pytest.raises(ValueError, match="output policy"):
            o._phase15_raster(tmp_path, filename, array, engine, monitor)
        identity = o._phase15_raster(tmp_path, filename, array, engine, monitor, allowed_outputs=allowed)
        with rasterio.open(tmp_path / filename) as src:
            np.testing.assert_array_equal(src.read(1), array)
            assert src.nodata is None
        assert identity["sha256"] == sha256((tmp_path / filename).read_bytes()).hexdigest()
        assert (tmp_path / (filename + ".partial")).exists()
        with pytest.raises(FileExistsError):
            o._phase15_raster(tmp_path, filename, array, engine, monitor, allowed_outputs=allowed)
    assert int(reached.sum()) == 3


@pytest.mark.parametrize("profile,cap", [("original315", 300), ("rf49", 300), ("airbase", 300), ("h90", 90)])
def test_historical_caps_and_output_policy_remain_frozen(profile, cap):
    config = base_config("airbase" if profile in ("airbase", "h90") else "santa_maria")
    if profile == "h90":
        config["phase15"].update(schema_version=o.PHASE16_AIRBASE_DURATION90_SCHEMA,
                                scenario_id=o.PHASE16_AIRBASE_SCENARIO_ID + "-h90-dt1m")
        config["duration_basis"] = deepcopy(o.PHASE16_AIRBASE_DURATION90_BASIS)
        config["simulation"]["max_timesteps"] = 90
    if profile == "original315":
        config["phase15"].update(schema_version=o.PHASE15_CANDIDATE_SCHEMA,
                                scenario_id="candidate-r2-f01-1024-w17p7-from315-s42-t1-1")
        del config["wind_input_basis"]
        config["wind"]["direction_deg"] = 315.0
    assert o._phase15_normalize(config)["simulation"]["max_timesteps"] == cap
    config["simulation"]["max_timesteps"] = 424
    with pytest.raises(ValueError): o._phase15_normalize(config)
    assert o.PHASE15_OUTPUTS == frozenset(("start.json", "termination.json", "final_state.tif",
        "simulation_environment_valid.tif", "final_state.tif.partial", "simulation_environment_valid.tif.partial",
        "termination.json.partial"))


@pytest.mark.parametrize("section,key,value", [
    ("phase15", "enabled", True), ("simulation", "resume_checkpoint", "not-permitted"),
    ("simulation", "checkpoint_interval", 1), ("output", "save_every_n_steps", 1),
    ("ml_model", "threshold", 0.4), ("simulation", "early_stop", False)])
def test_historical_rf49_guards_remain_strict(section, key, value):
    config = base_config("santa_maria")
    config[section][key] = value
    with pytest.raises(ValueError):
        o._phase15_normalize(config)


def test_binding_includes_protocol_source_and_authority_stays_required(monkeypatch, tmp_path):
    active = o._phase15_normalize(snapshot_config("santa_maria"))
    monkeypatch.setattr(o, "_phase15_source", lambda: {"files": {}})
    monkeypatch.setattr(o, "_phase15_runtime", lambda: {"runtime": "synthetic"})
    binding = o._phase15_binding(active, tmp_path / "execution.yaml", "0" * 64)
    assert binding["snapshot_basis"] == active["snapshot_basis"]
    assert "dswd_snapshot_results.py" in " ".join(binding["source"]["files"])
    assert binding["protocol_manifest"]["sha256"] == sha256(
        (Path(o.__file__).parent / o.PHASE16_DSWD_MANIFEST).read_bytes()).hexdigest()
    with pytest.raises(PermissionError):
        o._run_phase15({}, explicitly_authorized=False, approval_reference=None,
                      expected_binding_sha256=None, config_path=None, clock=(0, "synthetic"), invocation={})


def test_alignment_missing_lr_evidence_and_failed_groups_never_accepted():
    table = scoring.alignment_table()
    assert table["airbase"]["domain_cells"] == 215553
    assert table["santa_maria"]["reference_positive_included"] == 1910
    assert all(row["domain_mask_sha256"] is None for row in table.values())
    assert not scoring.verify_parity(table["airbase"], None)
    assert all(row["status"] == "incomplete" and row["mean_sample_sd"] is None
               for row in scoring.summarize_five_seed_groups([]))


@pytest.fixture
def guarded_trajectory(monkeypatch, tmp_path):
    config = snapshot_config("santa_maria")
    code = tmp_path / "Code"
    root = code / "output/phase15"
    root.mkdir(parents=True)
    (code / "dswd_snapshot_results.py").write_text("# synthetic binding source\n", encoding="utf-8")
    monkeypatch.setattr(o, "PHASE15_CODE_DIR", code)
    monkeypatch.setattr(o, "PHASE15_CANDIDATE_MODEL",
                        (code / config["ml_model"]["model_path"]).as_posix())
    monkeypatch.setattr(o, "_phase15_source", lambda: {"files": {}, "head": "synthetic", "dirty": True})
    monkeypatch.setattr(o, "_phase15_runtime", lambda: {"runtime": "synthetic"})
    monkeypatch.setattr(o, "_phase16_snapshot_manifest", lambda: {"path": "synthetic-protocol", "sha256": "a" * 64})
    sample = SimpleNamespace(process_rss_bytes=64, available_memory_bytes=10**10,
        free_disk_bytes=10**10, captured_at_utc="2026-11-03T00:00:00+00:00")
    monkeypatch.setattr(o, "capture_resource_snapshot", lambda path: sample)
    template = tiny_engine()
    environment = dict(slope_risk=template.slope_risk, proximity_risk=template.proximity_risk,
        building_presence=template.building_presence, material_class=template.material_class,
        burnable_mask=template.simulation_valid_mask, nodata_mask=template.nodata_mask,
        grid_shape=template.grid_shape, crs=template.crs, transform=template.transform)
    accesses = []
    def synthetic_environment(*args):
        accesses.append("environment")
        return environment, {"converted_ignitions": [[2, 2]]}
    def synthetic_model(engine, *args):
        accesses.append("model")
        engine.model = StubEstimator()
        return {"interface_validated": True, "synthetic": True}
    monkeypatch.setattr(o, "_phase15_candidate_environment", synthetic_environment)
    monkeypatch.setattr(o, "_phase15_load_model", synthetic_model)
    path = code / "execution.yaml"
    path.write_text(json.dumps(config), encoding="utf-8")
    active = o._phase15_normalize(config)
    binding = o._phase15_binding(active, path, sha256(path.read_bytes()).hexdigest())
    def invoke(**overrides):
        options = dict(explicitly_authorized=True, approval_reference=config["phase15"]["authorization_reference"],
                       expected_binding_sha256=o.canonical_metadata_hash(binding), config_path=path)
        options.update(overrides)
        return o.run_simulation(config, **options)
    return SimpleNamespace(invoke=invoke, destination=root / config["phase15"]["run_id"],
                           accesses=accesses, binding=binding)


@pytest.mark.parametrize("overrides", [
    {"explicitly_authorized": False}, {"approval_reference": "not-owner"},
    {"expected_binding_sha256": "0" * 64}])
def test_snapshot_authority_and_binding_precede_synthetic_startup(guarded_trajectory, overrides):
    r = guarded_trajectory
    with pytest.raises(PermissionError): r.invoke(**overrides)
    assert r.accesses == [] and not r.destination.exists()


def test_guarded_native_snapshots_collision_and_final_binding(guarded_trajectory):
    r = guarded_trajectory
    r.invoke()
    end = json.loads((r.destination / "termination.json").read_text())
    assert end["output_valid"] is True
    rows = end["snapshot_series"]["outcomes"]
    assert len(rows) == 3 and all(row["output_valid"] for row in rows)
    assert all(row["actual_stop_update"] == end["last_completed_timestep"] for row in rows)
    assert all(row["execution_binding_sha256"] == end["binding_sha256"] for row in rows)
    for row in rows:
        assert len(row["outputs"]) == 2
        assert row["configuration_sha256"] == r.binding["configuration"]["activated_effective_sha256"]
        assert row["actual_capture_update"] <= row["requested_update"]
    before = {p.name: p.read_bytes() for p in r.destination.iterdir()}
    with pytest.raises(FileExistsError): r.invoke()
    assert before == {p.name: p.read_bytes() for p in r.destination.iterdir()}


@pytest.mark.parametrize("failure", ["resource", "output"])
def test_guarded_failure_retains_evidence_and_incomplete_outcomes(guarded_trajectory, monkeypatch, failure):
    r = guarded_trajectory
    if failure == "resource":
        real = o._Phase15Resources.check
        def check(self, boundary, **kwargs):
            if boundary == "after_step": raise o._Phase15Stop("resource_limit")
            return real(self, boundary, **kwargs)
        monkeypatch.setattr(o._Phase15Resources, "check", check)
    else:
        real = o._phase15_raster
        def export(destination, name, *args, **kwargs):
            if name.endswith("_reached.tif"):
                (destination / (name + ".partial")).write_bytes(b"synthetic incomplete export")
                raise OSError("synthetic output failure")
            return real(destination, name, *args, **kwargs)
        monkeypatch.setattr(o, "_phase15_raster", export)
    with pytest.raises((RuntimeError, OSError)): r.invoke()
    end = json.loads((r.destination / "termination.json").read_text())
    assert end["termination_reason"] == ("resource_limit" if failure == "resource" else "output_failure")
    assert end["output_valid"] is False
    assert all(row["status"] == "incomplete" and not row["output_valid"]
               for row in end["snapshot_series"]["outcomes"])
    assert (r.destination / "start.json").exists()
    if failure == "output": assert list(r.destination.glob("snapshot_*_reached.tif.partial"))


def test_scoring_reuses_reached_semantics_and_requires_spatial_parity(monkeypatch):
    protocol = deepcopy(o.PHASE16_DSWD_PROTOCOL)
    protocol["incidents"]["santa_maria"].update(
        shape=[2, 3], expected_domain_cells=4,
        expected_reference_positive_included=2, expected_reference_positive_total=3)
    monkeypatch.setattr(o, "PHASE16_DSWD_PROTOCOL", protocol)
    state = np.array([[4, 3, 1], [5, 2, 1]], dtype=np.int8)
    arrays = dict(prediction=state, reference=np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8),
        buildings=np.array([[10, 10, 0], [10, 10, 0]], dtype=np.float32),
        environment=np.ones((2, 3), dtype=np.uint8), reached=np.isin(state, (3, 4, 5)).astype(np.uint8))
    arrays.update({name + "_valid": np.ones((2, 3), dtype=bool)
                   for name in ("prediction", "reference", "buildings", "environment")})
    outcome = dict(status="censored", output_valid=True, termination_reason="timestep_limit",
        seed=42, assumed_minutes_per_update=1, cumulative_reached_count=3, active_cell_count=2,
        state_counts={"non_burnable": 2, "not_yet_burning": 1, "ignited": 1, "blazing": 1, "extinguished": 1})
    metadata = {name: {"dtype": str(arrays[name].dtype), "nodata": None, "mask_flags": ["all_valid"]}
                for name in ("prediction", "reference", "buildings", "environment")}
    scored = scoring.evaluate_snapshot_arrays(arrays, outcome, "santa_maria", native_metadata=metadata)
    assert scored["metrics"]["termination"]["predicted_footprint_state_values"] == [3, 4, 5]
    assert scored["metrics"]["evaluation_pixel_count"] == 4
    assert scored["accepted_for_cross_simulator_comparison"] is False
    lr = deepcopy(scored["alignment"])
    lr["confirmation_reference"] = "synthetic LR-side alignment contract"
    assert scoring.evaluate_snapshot_arrays(arrays, outcome, "santa_maria",
        native_metadata=metadata, lr_alignment=lr)["accepted_for_cross_simulator_comparison"] is True
    lr["domain_mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="spatial domain parity"):
        scoring.evaluate_snapshot_arrays(arrays, outcome, "santa_maria", native_metadata=metadata, lr_alignment=lr)
    arrays["reached"][0, 0] = 0
    with pytest.raises(ValueError, match="reached export"):
        scoring.evaluate_snapshot_arrays(arrays, outcome, "santa_maria")
    outcome["output_valid"] = False
    with pytest.raises(ValueError, match="Incomplete/failed"):
        scoring.evaluate_snapshot_arrays(arrays, outcome, "santa_maria")


def test_five_seed_reporting_does_not_pool_or_drop_failed_seeds():
    rows = []
    for incident in ("airbase", "santa_maria"):
        for clock in (0.5, 1, 2):
            for seed in range(42, 47):
                rows.append({"incident": incident, "assumed_minutes_per_update": clock, "seed": seed,
                    "metrics": {key: (seed - 42) * 0.01 for key in scoring.METRICS},
                    "outcome": {"output_valid": True, "status": "censored", "cumulative_reached_count": seed,
                                "active_cell_count": 1, "actual_capture_update": 10}})
    summary = scoring.summarize_five_seed_groups(rows)
    assert len(summary) == 6 and all(group["n"] == 5 for group in summary)
    assert all(group["mean_sample_sd"]["cumulative_reached_count"]["sample_sd"] == 2.5**0.5
               for group in summary)
    rows[0]["outcome"]["output_valid"] = False
    broken = scoring.summarize_five_seed_groups(rows)
    assert broken[0]["status"] == "incomplete" and broken[0]["mean_sample_sd"] is None
    assert all(group["trajectories_independent_across_clocks"] is False for group in broken)


# D-184 additions: all native reads/exports below stay inside tmp_path.
import dswd_snapshot_readiness as readiness


def synthetic_reader_capability():
    return {"policy_id": readiness.native.POLICY_ID, "controls_verified": True,
            "windowed_reads_verified": True,
            "runtime": {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__},
            "environment_options": dict(readiness.native.ENV_OPTIONS),
            "open_options": dict(readiness.native.OPEN_OPTIONS),
            "verification_reference": "synthetic fixture controls only",
            "window_verification_reference": "synthetic fixture window only"}


def noncalm_chunk_engine(seed):
    template = tiny_engine(seed)
    config = deepcopy(template.config)
    config["wind"].update(speed_kmh=10.6, direction_deg=60.826537)
    config["ml_model"]["inference_chunk_size"] = 2
    engine = FireAutomata(template.environment, config)
    engine.model = StubEstimator()
    engine.set_ignition([(2, 2)])
    return engine


@pytest.mark.parametrize("seed", range(42, 47))
def test_export_enabled_noncalm_chunk_prefixes_are_observational(seed, tmp_path):
    observed, disabled = noncalm_chunk_engine(seed), noncalm_chunk_engine(seed)
    series = o._Phase16Snapshots(tiny_basis((1, 2, 3)))
    allowed = o._phase16_snapshot_outputs(series.basis)
    monitor = SimpleNamespace(check=lambda *a, **k: None)
    grid = {"shape": [5, 5], "crs": "EPSG:32651", "transform": list(observed.transform)}
    captures = []
    def capture(row, native, reached):
        checkpoint = (observed.grid.copy(), observed.ignition_timers.copy(),
                      observed.blazing_timers.copy(), deepcopy(observed.rng.bit_generator.state),
                      [call.copy() for call in observed.model.calls])
        identities = []
        for kind, values in (("state", native), ("reached", reached)):
            name = f"snapshot_h{row['requested_update']:04d}_{kind}.tif"
            identity = o._phase15_raster(tmp_path, name, values, observed, monitor, allowed_outputs=allowed)
            actual, mask, nodata, metadata = readiness._read_native(
                tmp_path / name, identity["sha256"], grid, capability=synthetic_reader_capability())
            np.testing.assert_array_equal(actual, values)
            assert np.all(mask) and nodata is None and metadata["mask_flags"] == ["all_valid"]
            identities.append(identity)
        np.testing.assert_array_equal(observed.grid, checkpoint[0])
        np.testing.assert_array_equal(observed.ignition_timers, checkpoint[1])
        np.testing.assert_array_equal(observed.blazing_timers, checkpoint[2])
        assert observed.rng.bit_generator.state == checkpoint[3]
        assert len(observed.model.calls) == len(checkpoint[4])
        for actual, prior in zip(observed.model.calls, checkpoint[4]):
            np.testing.assert_array_equal(actual, prior)
        independent = noncalm_chunk_engine(seed)
        for _ in range(row["actual_capture_update"]):
            independent.step()
        equal_trajectory(observed, independent)
        np.testing.assert_array_equal(native, independent.grid)
        captures.append(row["actual_capture_update"])
        return identities
    _, completed, _ = series.run(observed, lambda boundary: None, capture)
    for _ in range(completed):
        disabled.step()
    equal_trajectory(observed, disabled)
    assert len(captures) == 3
    assert observed.wind_cfg["speed_kmh"] == 10.6
    assert observed.t_3_to_4 == observed.t_4_to_5 == 1
    assert len(observed.model.calls) >= 4
    assert all(0 < len(call) <= 2 for call in observed.model.calls)
    assert any(len(call) == 2 for call in observed.model.calls)


def test_late_export_failure_retains_first_capture_and_invalidates_every_outcome(tmp_path):
    engine = Scripted([[4], [4], [4], [4]])
    engine.crs = rasterio.crs.CRS.from_epsg(32651)
    engine.transform = from_origin(0, 3, 3, 3)
    series = o._Phase16Snapshots(tiny_basis())
    allowed = o._phase16_snapshot_outputs(series.basis)
    monitor = SimpleNamespace(check=lambda *a, **k: None)
    attempts = []
    def capture(row, native, reached):
        result = []
        for kind, values in (("state", native), ("reached", reached)):
            name = f"snapshot_h{row['requested_update']:04d}_{kind}.tif"
            attempts.append(name)
            if row["requested_update"] == 2 and kind == "reached":
                (tmp_path / (name + ".partial")).write_bytes(b"synthetic partial")
                raise OSError("late synthetic export failure")
            result.append(o._phase15_raster(tmp_path, name, values, engine, monitor, allowed_outputs=allowed))
        return result
    with pytest.raises(OSError, match="late synthetic"):
        series.run(engine, lambda boundary: None, capture)
    assert engine.timestep == 2
    assert attempts == ["snapshot_h0001_state.tif", "snapshot_h0001_reached.tif",
                        "snapshot_h0002_state.tif", "snapshot_h0002_reached.tif"]
    assert (tmp_path / "snapshot_h0001_state.tif").exists()
    assert (tmp_path / "snapshot_h0001_reached.tif").exists()
    assert (tmp_path / "snapshot_h0002_reached.tif.partial").read_bytes() == b"synthetic partial"
    assert not (tmp_path / "snapshot_h0003_state.tif").exists()
    assert all(row["status"] == "incomplete" and not row["output_valid"] for row in series.rows)
    before = (tmp_path / "snapshot_h0001_state.tif").read_bytes()
    with pytest.raises(FileExistsError):
        o._phase15_raster(tmp_path, "snapshot_h0001_state.tif",
                         np.array([[4]], dtype=np.int8), engine, monitor, allowed_outputs=allowed)
    assert (tmp_path / "snapshot_h0001_state.tif").read_bytes() == before


def test_final_publication_failure_retains_orphans_and_reader_rejects(guarded_trajectory, monkeypatch):
    r = guarded_trajectory
    real = o._phase15_finalize
    publications = []
    def finalize(partial, final):
        if final.name == "termination.json":
            publications.append(final.name)
            assert partial.exists()
            raise OSError("synthetic final publication failure")
        return real(partial, final)
    monkeypatch.setattr(o, "_phase15_finalize", finalize)
    with pytest.raises(OSError, match="final publication"):
        r.invoke()
    assert publications == ["termination.json"]
    assert not (r.destination / "termination.json").exists()
    partial = json.loads((r.destination / "termination.json.partial").read_text())
    assert partial["output_valid"] is True
    assert all(row["output_valid"] for row in partial["snapshot_series"]["outcomes"])
    assert list(r.destination.glob("snapshot_*_state.tif"))
    before = {p.name: p.read_bytes() for p in r.destination.iterdir()}
    expected = {"review_reference": "synthetic orphan rejection",
                "start_sha256": sha256((r.destination / "start.json").read_bytes()).hexdigest(),
                "termination_sha256": "0" * 64}
    with pytest.raises(FileNotFoundError):
        readiness.read_snapshot_run(r.destination, expected=expected, alignment={},
            capability=synthetic_reader_capability(), authorized=True,
            authorization_reference="synthetic explicit read only")
    assert before == {p.name: p.read_bytes() for p in r.destination.iterdir()}


@pytest.fixture
def native_alignment_fixture(monkeypatch, tmp_path):
    code = tmp_path / "Code"
    monkeypatch.setattr(o, "PHASE15_CODE_DIR", code)
    protocol = deepcopy(o.PHASE16_DSWD_PROTOCOL)
    transform = from_origin(0, 15, 3, 3)
    protocol["alignment_policy"].update(parent_shape=[5, 5], parent_transform=list(transform)[:6])
    for item in protocol["incidents"].values():
        item.update(crop_origin=[0, 0], shape=[5, 5], ignition=[[2, 2]],
                    crop_transform=list(transform)[:6], horizons=[8, 4, 2], max_updates=8,
                    expected_domain_cells=21, expected_reference_positive_included=2,
                    expected_reference_positive_total=3, qualification="synthetic native fixture only")
        item["duration"] = {"minutes": 4, "source": "synthetic", "approximate": True}
    monkeypatch.setattr(o, "PHASE16_DSWD_PROTOCOL", protocol)
    paths = readiness.alignment_read_whitelist()
    raw = {name: np.ones((5, 5), dtype=np.float32) for name in readiness.native.LAYERS}
    raw["buildings"][:] = 10
    raw["slope"][4, 0] = -9999
    raw["proximity"][4, 1] = np.nan
    raw["materials"][4, 2] = -9999
    raw["buildings"][4, 2] = 0
    raw["buildings"][4, 3] = -9999
    raw["materials"][4, 3] = 0
    reference = np.zeros((5, 5), dtype=np.uint8)
    reference[0, 0] = reference[0, 1] = reference[4, 0] = 1
    nodata = {name: (-9999 if name in ("buildings", "materials") else None)
              for name in readiness.native.LAYERS}
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        values = raw[name] if name in raw else reference
        with rasterio.open(path, "w", driver="GTiff", height=5, width=5, count=1,
                           dtype=values.dtype.name, crs="EPSG:32651", transform=transform,
                           nodata=nodata.get(name)) as dst:
            dst.write(values, 1)
        digest = sha256(path.read_bytes()).hexdigest()
        if name in raw:
            protocol["input_sha256"][name] = digest
        else:
            protocol["incidents"][name.removesuffix("_reference")]["reference_sha256"] = digest
    capability = synthetic_reader_capability()
    alignment = readiness.collect_alignment(paths, capability=capability, authorized=True,
                                            authorization_reference="synthetic six-file read")
    masks = {name: np.where(values == nodata[name], 0, 255).astype(np.uint8)
             if nodata[name] is not None else np.full((5, 5), 255, dtype=np.uint8)
             for name, values in raw.items()}
    prepared = readiness.native.prepare_arrays(raw, masks, nodata, configured_nodata=-9999,
        crs="EPSG:32651", transform=transform, input_hashes=protocol["input_sha256"],
        interpretation={}, global_origin=(0, 0), full_grid_shape=(5, 5))
    return SimpleNamespace(code=code, paths=paths, protocol=protocol, alignment=alignment,
                           capability=capability, prepared=prepared, transform=transform)


def test_native_alignment_all_four_layers_masks_and_missing_lr(native_alignment_fixture):
    f = native_alignment_fixture
    for asset in f.alignment.values():
        report = asset["report"]
        assert report["masks"]["environmental_valid"]["count"] == 21
        assert len(report["parent_grid"]["transform"]) == 9
        assert report["masks"]["mapped_building"]["count"] == 23
        assert report["masks"]["evaluation_domain"]["count"] == 21
        assert report["reference_positive"] == {"total_in_crop": 3, "included": 2, "excluded": 1}
        assert not np.any(asset["arrays"]["environment"][4, :4])
        assert asset["arrays"]["environment"][4, 4] == 1
        assert asset["parity"]["status"] == "NEEDS EVIDENCE" and not asset["parity"]["accepted"]
        assert not asset["arrays"]["reference"].flags.writeable
        assert set(report["native_metadata"]) == {"slope", "proximity", "buildings", "materials", "reference"}
        incomplete = deepcopy(report)
        incomplete["confirmation_reference"] = None
        checked = readiness.compare_lr_packet(report, incomplete, o.canonical_metadata_hash(incomplete))
        assert checked["status"] == "NEEDS EVIDENCE" and not checked["accepted"]


@pytest.mark.parametrize("mismatch", ["hash", "geometry", "mask", "authorization", "whitelist"])
def test_native_alignment_rejects_identity_metadata_mask_and_scope(native_alignment_fixture, mismatch):
    f = native_alignment_fixture
    paths = dict(f.paths)
    if mismatch == "authorization":
        with pytest.raises(PermissionError):
            readiness.collect_alignment(paths, capability=f.capability)
        return
    if mismatch == "whitelist":
        paths["slope"] = f.code / "arbitrary.tif"
    elif mismatch == "hash":
        f.protocol["input_sha256"]["slope"] = "0" * 64
    else:
        path = paths["slope"]
        values = np.ones((5, 5), dtype=np.float32)
        # This overwrites only a fresh synthetic fixture, never production data.
        with rasterio.open(path, "w", driver="GTiff", height=5, width=5, count=1,
                dtype="float32", crs="EPSG:32651",
                transform=from_origin(3 if mismatch == "geometry" else 0, 15, 3, 3),
                nodata=0 if mismatch == "mask" else None) as dst:
            if mismatch == "mask": values[0, 2] = 0
            dst.write(values, 1)
        f.protocol["input_sha256"]["slope"] = sha256(path.read_bytes()).hexdigest()
    with pytest.raises((ValueError, PermissionError)):
        readiness.collect_alignment(paths, capability=f.capability, authorized=True,
                                    authorization_reference="synthetic mismatch read")


def synthetic_receipt(f, monkeypatch, *, arm=None, incident="santa_maria", revision=1,
                      source_revision="synthetic"):
    window = {"family_ordinal": 1, "scenario_family_id": "p8f01_cluster_322_m1",
              "global_origin": [0, 0], "shape": [5, 5]}
    monkeypatch.setattr(o, "PHASE15_CANDIDATE_WINDOW", window)
    monkeypatch.setattr(o, "PHASE15_CANDIDATE_IGNITIONS", [[2, 2]])
    if incident == "airbase":
        monkeypatch.setattr(o, "PHASE16_AIRBASE_WINDOW", window)
    monkeypatch.setattr(o.candidate_crops, "PARENT_SHAPE", (5, 5))
    config = shared_config(incident, arm) if arm else snapshot_config(incident)
    if revision != 1:
        run = config["phase15"]["run_id"][:-2] + f"r{revision}"
        config["phase15"].update(run_id=run, scenario_id=run + "-snapshots")
    config["candidate_inputs"]["window"] = deepcopy(window)
    config["candidate_inputs"]["expected_grid"].update(shape=[5, 5], transform=list(f.transform))
    config["simulation"]["ignition_points"] = [[2, 2]]
    config["simulation"]["max_timesteps"] = 8
    config["environment"]["input_sha256"] = deepcopy(f.protocol["input_sha256"])
    if arm == "lr":
        monkeypatch.setattr(o, "PHASE16_LR_MODEL",
                            (f.code / config["ml_model"]["model_path"]).as_posix())
    else:
        monkeypatch.setattr(o, "PHASE15_CANDIDATE_MODEL",
                            (f.code / config["ml_model"]["model_path"]).as_posix())
    monkeypatch.setattr(o, "_phase15_source", lambda: {"files": {}, "head": source_revision, "dirty": True})
    monkeypatch.setattr(o, "_phase15_runtime", lambda: {"runtime": "synthetic"})
    monkeypatch.setattr(o, "_phase16_snapshot_manifest", lambda *args: {"path": "synthetic-protocol", "sha256": "a" * 64})
    if not (f.code / "dswd_snapshot_results.py").exists():
        (f.code / "dswd_snapshot_results.py").write_text("# synthetic source\n", encoding="utf-8")
    if arm is not None and not (f.code / "dswd_inference_guard.py").exists():
        (f.code / "dswd_inference_guard.py").write_text("# synthetic guard binding source\n", encoding="utf-8")
    if arm is not None and not (f.code / "run_dswd_shared_ca_v1.ps1").exists():
        (f.code / "run_dswd_shared_ca_v1.ps1").write_bytes(
            (Path(o.__file__).parent / "run_dswd_shared_ca_v1.ps1").read_bytes())
    config_path = f.code / (f"execution-{arm}.yaml" if arm else "execution.yaml")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    active = o._phase15_normalize(config)
    binding = o._phase15_binding(active, config_path, sha256(config_path.read_bytes()).hexdigest())
    digest = o.canonical_metadata_hash(binding)
    root = Path(binding["run"]["destination"])
    root.mkdir(parents=True)
    start = {"schema_version": "phase15_run_start.v1", **binding["run"],
             "started_at_utc": "2026-11-03T00:00:00+00:00", "clock_boundary": "phase15_entry",
             "invocation": {"kind": "synthetic"}, "initial_resource_observations": {"synthetic": True},
             "authorization_reference": binding["approval_reference"],
             "binding": {"payload": binding, "expected_sha256": digest, "calculated_sha256": digest},
             "configuration": {**binding["configuration"], "effective": active},
             "source": binding["source"], "runtime": binding["runtime"],
             "model": {**binding["model"], "observed_sha256": None, "interface_validated": None},
             "inputs": {k: {**v, "observed_sha256": None} for k, v in binding["inputs"].items()},
             "scenario": {"seed": 42, "wind": active["wind"], "ignition_points": [[2, 2]],
                          "converted_ignitions": None, "grid": None, "timing": active["placeholder_transition"],
                          "max_timesteps": 8}, "resource_policy": active["phase15"]["resources"],
             "output_policy": {"allowlist": sorted(o._phase16_snapshot_outputs(active["snapshot_basis"])),
                               "resume": False}}
    start_hash = o._phase15_write_json(root / "start.json", start)
    series = o._Phase16Snapshots(active["snapshot_basis"])
    for row in series.rows:
        row.update(execution_binding_sha256=digest,
                   configuration_sha256=binding["configuration"]["activated_effective_sha256"],
                   start_record_sha256=start_hash, run_id=root.name, seed=42, incident=incident)
    engine = FireAutomata(f.prepared.environment, active)
    engine.model = StubEstimator()
    engine.set_ignition([(2, 2)])
    monitor = SimpleNamespace(check=lambda *a, **k: None)
    allowed = o._phase16_snapshot_outputs(active["snapshot_basis"])
    outputs = []
    def capture(row, native, reached):
        result = []
        for kind, values in (("state", native), ("reached", reached)):
            result.append(o._phase15_raster(root, f"snapshot_h{row['requested_update']:04d}_{kind}.tif",
                          values, engine, monitor, allowed_outputs=allowed))
        outputs.extend(result)
        return result
    reason, stop, counts = series.run(engine, lambda boundary: None, capture)
    for name, values in (("final_state.tif", engine.grid),
                         ("simulation_environment_valid.tif", engine.simulation_valid_mask.astype(np.uint8))):
        outputs.append(o._phase15_raster(root, name, values, engine, monitor, allowed_outputs=allowed))
    end = {"schema_version": "phase15_run_termination.v1", "run_id": root.name,
           "ended_at_utc": "2026-11-03T00:00:01+00:00", "decision_elapsed_seconds": 1.0,
           "clock_policy": {"boundary": "phase15_entry", "termination_publication_excluded": True,
                            "limits_are_cooperative": True}, "resource_observations": {"synthetic": True},
           "start_record_sha256": start_hash, "binding_sha256": digest,
           "status": "complete" if reason == "inactive" else "censored", "termination_reason": reason,
           "exit_code": 0, "output_valid": True, "error": None, "violations": [], "failed_step": None,
           "last_completed_timestep": stop, "state_counts": counts,
           "active_cell_count": counts["ignited"] + counts["blazing"], "snapshot_series": series.record(),
           "outputs": outputs, "observed_bindings": {
               "inputs": {name: {"observed_sha256": pin} for name, pin in f.protocol["input_sha256"].items()},
               "model": {"observed_sha256": o.PHASE15_CANDIDATE_MODEL_SHA256, "interface_validated": True,
                         "manifest": {"observed_sha256": o.PHASE15_CANDIDATE_MANIFEST_SHA256}},
               "converted_ignitions": [[2, 2]], "mask_identities": f.prepared.mask_identities(),
               "grid": {"shape": [5, 5], "crs": engine.crs.to_wkt(), "transform": list(engine.transform)[:6]}}}
    if arm == "lr":
        end["observed_bindings"]["model"] = {
            "observed_sha256": o.PHASE16_LR_MODEL_SHA256, "interface_validated": True}
    o._phase15_publish(root, end)
    expected = {"review_reference": "synthetic independently supplied receipt pins",
                "start_sha256": start_hash, "termination_sha256": sha256((root / "termination.json").read_bytes()).hexdigest(),
                "binding": binding, "effective_configuration": active,
                "alignment_sha256": f.alignment[incident]["report_sha256"]}
    return SimpleNamespace(root=root, start=start, end=end, expected=expected)


def test_verified_snapshot_reader_and_lr_packet_gate(native_alignment_fixture, monkeypatch):
    f = native_alignment_fixture
    receipt = synthetic_receipt(f, monkeypatch)
    asset = f.alignment["santa_maria"]
    result = readiness.read_snapshot_run(receipt.root, expected=receipt.expected, alignment=asset,
        capability=f.capability, authorized=True, authorization_reference="synthetic root read")
    assert len(result["outcomes"]) == 3
    assert all(not row["accepted_for_cross_simulator_comparison"] for row in result["outcomes"])
    packet = deepcopy(asset["report"])
    packet.update(confirmation_reference="synthetic LR custodian packet", reader_reference="synthetic reader",
                  reader_source_sha256="1" * 64, simulation_configuration_reference="synthetic planned fixture")
    asset["lr_packet"] = packet
    receipt.expected["lr_packet_sha256"] = o.canonical_metadata_hash(packet)
    matched = readiness.read_snapshot_run(receipt.root, expected=receipt.expected, alignment=asset,
        capability=f.capability, authorized=True, authorization_reference="synthetic root read")
    assert all(row["accepted_for_cross_simulator_comparison"] for row in matched["outcomes"])
    packet["ignition_global_row_col"] = [[3, 2]]
    with pytest.raises(ValueError, match="spatial masks"):
        readiness.compare_lr_packet(asset["report"], packet, o.canonical_metadata_hash(packet))
    packet["ignition_global_row_col"] = [[2, 2]]
    packet["masks"]["evaluation_domain"]["sha256"] = "0" * 64
    receipt.expected["lr_packet_sha256"] = o.canonical_metadata_hash(packet)
    with pytest.raises(ValueError, match="spatial masks"):
        readiness.read_snapshot_run(receipt.root, expected=receipt.expected, alignment=asset,
            capability=f.capability, authorized=True, authorization_reference="synthetic root read")


@pytest.mark.parametrize("mismatch", ["record_pin", "outcome_clock", "outcome_horizon", "outcome_seed",
                                     "outcome_stop", "output_link", "effective", "failed"])
def test_reader_rejects_receipt_and_outcome_tampering(native_alignment_fixture, monkeypatch, mismatch):
    f = native_alignment_fixture
    receipt = synthetic_receipt(f, monkeypatch)
    expected = deepcopy(receipt.expected)
    end = deepcopy(receipt.end)
    row = end["snapshot_series"]["outcomes"][0]
    if mismatch == "record_pin": expected["start_sha256"] = "0" * 64
    elif mismatch == "effective": expected["effective_configuration"]["simulation"]["seed"] = 46
    else:
        if mismatch == "outcome_clock": row["clock_role"] = "wrong"
        elif mismatch == "outcome_horizon": row["requested_update"] += 1
        elif mismatch == "outcome_seed": row["seed"] = 43
        elif mismatch == "outcome_stop": row["actual_stop_update"] += 1
        elif mismatch == "output_link": row["outputs"][0]["filename"] = "../../model.joblib"
        else: end["output_valid"] = False
        # A new synthetic root carries the changed receipt; never overwrite a receipt/link.
        changed_root = receipt.root.parent / (receipt.root.name + "-tampered")
        changed_root.mkdir()
        (changed_root / "start.json").write_bytes((receipt.root / "start.json").read_bytes())
        o._phase15_publish(changed_root, end)
        expected["termination_sha256"] = sha256((changed_root / "termination.json").read_bytes()).hexdigest()
        # Keep the approved lexical root; replace only record reads in this mismatch test.
        original = readiness._record
        def record(path, pin):
            if path.name == "termination.json": return end
            return original(path, pin)
        monkeypatch.setattr(readiness, "_record", record)
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        readiness.read_snapshot_run(receipt.root, expected=expected, alignment=f.alignment["santa_maria"],
            capability=f.capability, authorized=True, authorization_reference="synthetic tamper read")


@pytest.mark.parametrize("mismatch", ["dtype", "reached", "length", "mask"])
def test_reader_rejects_native_output_mismatches(native_alignment_fixture, monkeypatch, mismatch):
    f = native_alignment_fixture
    original = o._phase15_raster
    changed = []
    def export(destination, name, values, engine, monitor, **kwargs):
        selected = not changed and name.startswith("snapshot_") and (
            name.endswith("_reached.tif") if mismatch == "reached" else name.endswith("_state.tif"))
        if selected:
            changed.append(name)
            if mismatch == "dtype":
                values = values.astype(np.int16)
            elif mismatch == "reached":
                values = values.copy()
                values[0, 0] = 1 - values[0, 0]
        identity = original(destination, name, values, engine, monitor, **kwargs)
        if selected and mismatch == "length":
            identity["bytes"] += 1
        elif selected and mismatch == "mask":
            # Deliberate native-metadata tamper, only in this fresh synthetic fixture.
            with rasterio.Env(**readiness.native.ENV_OPTIONS):
                with rasterio.open(destination / name, "r+") as dst:
                    dst.nodata = 0
            identity["bytes"] = (destination / name).stat().st_size
            identity["sha256"] = sha256((destination / name).read_bytes()).hexdigest()
        return identity
    monkeypatch.setattr(o, "_phase15_raster", export)
    receipt = synthetic_receipt(f, monkeypatch)
    assert len(changed) == 1
    with pytest.raises(ValueError):
        readiness.read_snapshot_run(receipt.root, expected=receipt.expected, alignment=f.alignment["santa_maria"],
            capability=f.capability, authorized=True, authorization_reference="synthetic bad native read")


# D-200/D-201 shared-CA authoring. No test below resolves a production input.
def shared_config(incident, arm, seed=42):
    config = base_config(incident)
    protocol = o.PHASE16_DSWD_SHARED_PROTOCOL
    policy, item = protocol["model_policies"][arm], protocol["incidents"][incident]
    run = f"p16-dswd-shared-v1-{incident.replace('_', '-')}-{arm}-s{seed}-r1"
    config["phase15"].update(schema_version=o.PHASE16_DSWD_SHARED_SCHEMA, enabled=False,
        authorization_reference="synthetic-shared-owner-only", run_id=run, scenario_id=run + "-snapshots")
    config["simulation"].update(seed=seed, max_timesteps=item["max_updates"],
                                ignition_points=deepcopy(item["ignition"]))
    config["ml_model"].update(model_path=policy["model_path"], model_sha256=policy["model_sha256"],
        calibration_manifest_path=policy["manifest_path"], calibration_manifest_sha256=policy["manifest_sha256"],
        inference_mode=policy["inference_mode"])
    if arm == "lr":
        config["ml_model"]["proba_threshold"] = 0.40
    config["snapshot_basis"] = o._phase16_shared_basis(incident, arm)
    return config


@pytest.mark.parametrize("incident", ["airbase", "santa_maria"])
@pytest.mark.parametrize("arm", ["rf", "lr"])
@pytest.mark.parametrize("seed", range(42, 47))
def test_shared_frozen_arms_incidents_and_seeds(incident, arm, seed):
    config = shared_config(incident, arm, seed)
    before = deepcopy(config)
    active = o._phase15_normalize(config)
    assert config == before and config["phase15"]["enabled"] is False
    assert active["snapshot_basis"] == o._phase16_shared_basis(incident, arm)
    assert active["simulation"]["ignition_points"] == ([[512, 512]] if incident == "airbase" else [[636, 554]])
    assert active["ml_model"]["model_sha256"] == before["ml_model"]["model_sha256"]
    assert active["simulation"]["max_timesteps"] == (172 if incident == "airbase" else 424)
    assert active["placeholder_transition"] == {"T_3_to_4": 1, "T_4_to_5": 1, "wind_weight": 0.2}
    if arm == "lr":
        assert active["ml_model"]["proba_threshold"] == 0.40
        assert active["ml_model"]["calibration_manifest_path"] is None


@pytest.mark.parametrize("arm", ["rf", "lr"])
@pytest.mark.parametrize("section,key,value", [
    ("ml_model", "model_sha256", "0" * 64),
    ("ml_model", "model_path", "synthetic-wrong-release.joblib"),
    ("ml_model", "calibration_manifest_sha256", "0" * 64),
    ("ml_model", "calibration_manifest_path", "synthetic-manifest.json"),
    ("ml_model", "inference_mode", "deterministic_threshold"),
    ("ml_model", "threshold", 0.4), ("ml_model", "classification_threshold", 0.4),
    ("ml_model", "inference_threshold", 0.4),
    ("simulation", "ignition_points", [[638, 554]]), ("simulation", "seed", True),
    ("simulation", "seed", 47), ("simulation", "max_timesteps", 300),
    ("wind", "direction_deg", 49.0), ("wind", "speed_kmh", 10.6),
    ("placeholder_transition", "T_3_to_4", 2)])
def test_shared_rejects_policy_and_control_substitutions(arm, section, key, value):
    config = shared_config("santa_maria", arm)
    config[section][key] = value
    with pytest.raises(ValueError):
        o._phase15_normalize(config)


@pytest.mark.parametrize("value", [None, True, 0.39, 0.41])
def test_shared_lr_gate_exact(value):
    config = shared_config("santa_maria", "lr")
    config["ml_model"]["proba_threshold"] = value
    with pytest.raises(ValueError):
        o._phase15_normalize(config)


@pytest.mark.parametrize("make", [lambda: base_config("santa_maria"),
                                  lambda: snapshot_config("santa_maria")])
def test_legacy_ignition_guards_not_relaxed(make):
    config = make()
    assert o._phase15_normalize(config)["simulation"]["ignition_points"] == [[638, 554]]
    config["simulation"]["ignition_points"] = [[636, 554]]
    with pytest.raises(ValueError):
        o._phase15_normalize(config)
    assert o.PHASE15_CANDIDATE_IGNITIONS == [[638, 554]]


def test_shared_disabled_manifest_protocol_and_inherited_wind_anchor():
    code = Path(o.__file__).parent
    manifest = o._phase15_yaml((code / o.PHASE16_DSWD_SHARED_MANIFEST).read_bytes())
    protocol = manifest["dswd_snapshots"]
    assert protocol == o.PHASE16_DSWD_SHARED_PROTOCOL
    assert protocol["enabled"] is False and protocol["execution_authorized"] is False
    assert (protocol["trajectory_count"], protocol["planned_outcome_count"],
            protocol["reporting_group_count"]) == (20, 60, 12)
    document = code.parents[1] / protocol["protocol_evidence"]["path"]
    assert sha256(document.read_bytes()).hexdigest() == protocol["protocol_evidence"]["sha256"]
    assert o._phase16_snapshot_manifest(shared_config("santa_maria", "rf"))["protocol_evidence"] == protocol["protocol_evidence"]
    assert o.PHASE16_WIND_INPUT_BASIS["anchor"]["easting"] == 606750.2784
    assert protocol["incidents"]["santa_maria"]["ignition_amendment"]["global_row_col"] == [[892, 2730]]
    with pytest.raises(PermissionError, match="manifest"):
        o.run_simulation(manifest)


class MixedCutoffEstimator(StubEstimator):
    """Feature-stable probabilities on both sides of the float32 gate boundary."""
    @staticmethod
    def support():
        cutoff = np.float32(0.40)
        return np.asarray([np.nextafter(cutoff, np.float32(0)),
                           cutoff, np.nextafter(cutoff, np.float32(1))], dtype=np.float32)

    def predict_proba(self, features):
        values = np.asarray(features).copy()
        self.calls.append(values)
        category = np.rint(values[:, 0] * np.float32(10)).astype(np.intp) % 3
        positive = self.support()[category]
        return np.column_stack((np.float32(1) - positive, positive))


@pytest.mark.parametrize("seed", range(42, 47))
def test_shared_lr_native_export_prefixes_preserve_trajectory(monkeypatch, tmp_path, seed):
    original = noncalm_chunk_engine
    def lr_engine(value):
        template = original(value)
        config = deepcopy(template.config)
        config["ml_model"].update(inference_mode="stochastic_probability_gate", proba_threshold=0.40)
        environment = deepcopy(template.environment)
        category = np.arange(25).reshape(5, 5) % 3
        environment["slope_risk"] = category.astype(np.float32) / np.float32(10)
        engine = FireAutomata(environment, config)
        engine.model = MixedCutoffEstimator()
        engine.set_ignition([(2, 2)])
        return engine
    probe = lr_engine(seed)
    before = (probe.grid.copy(), probe.ignition_timers.copy(), probe.blazing_timers.copy(),
              deepcopy(probe.rng.bit_generator.state))
    susceptible = probe.grid == 2
    category = np.arange(25).reshape(5, 5) % 3
    probabilities = probe._predict_with_model(
        np.zeros((5, 5), dtype=np.int8), susceptible, np.zeros((5, 5), dtype=np.float32))
    support = MixedCutoffEstimator.support()
    assert support[0] < np.float32(0.40) == support[1] < support[2]
    assert probabilities.dtype == np.float32
    for index, retained in enumerate((np.float32(0), support[1], support[2])):
        population = susceptible & (category == index)
        assert np.any(population)
        np.testing.assert_array_equal(probabilities[population],
                                      np.full(np.count_nonzero(population), retained, dtype=np.float32))
    np.testing.assert_array_equal(probe.grid, before[0])
    np.testing.assert_array_equal(probe.ignition_timers, before[1])
    np.testing.assert_array_equal(probe.blazing_timers, before[2])
    assert probe.rng.bit_generator.state == before[3]
    monkeypatch.setitem(globals(), "noncalm_chunk_engine", lr_engine)
    # Reuse all existing native readback, state/timer/RNG/ordered-call/chunk assertions.
    test_export_enabled_noncalm_chunk_prefixes_are_observational(seed, tmp_path)


@pytest.fixture
def shared_native_fixture(native_alignment_fixture, monkeypatch):
    f = native_alignment_fixture
    protocol = deepcopy(o.PHASE16_DSWD_SHARED_PROTOCOL)
    protocol["input_sha256"] = deepcopy(f.protocol["input_sha256"])
    protocol["incidents"] = deepcopy(f.protocol["incidents"])
    protocol["alignment_policy"].update(parent_shape=[5, 5], parent_transform=list(f.transform)[:6])
    monkeypatch.setattr(o, "PHASE16_DSWD_SHARED_PROTOCOL", protocol)
    f.shared_protocol = protocol
    f.shared_alignment = readiness.collect_alignment(f.paths, capability=f.capability, authorized=True,
        authorization_reference="synthetic new shared seed observations", design=protocol["design"])
    return f


def test_shared_receiving_ineligible_seed_rejected_before_engine(shared_native_fixture):
    f = shared_native_fixture
    f.shared_protocol["incidents"]["santa_maria"]["ignition"] = [[4, 0]]
    with pytest.raises(ValueError):
        readiness.collect_alignment(f.paths, capability=f.capability, authorized=True,
            authorization_reference="synthetic invalid receiving seed", design=f.shared_protocol["design"])


def shared_read(f, receipt, *, alignment=None):
    return readiness.read_snapshot_run(receipt.root, expected=receipt.expected,
        alignment=f.shared_alignment["santa_maria"] if alignment is None else alignment,
        capability=f.capability, authorized=True, authorization_reference="synthetic shared receipt read")


@pytest.mark.parametrize("arm", ["rf", "lr"])
def test_shared_verified_reader_model_outcomes_and_collision(shared_native_fixture, monkeypatch, arm):
    f = shared_native_fixture
    receipt = synthetic_receipt(f, monkeypatch, arm=arm)
    receipt.expected["alignment_sha256"] = f.shared_alignment["santa_maria"]["report_sha256"]
    result = shared_read(f, receipt)
    assert len(result["outcomes"]) == 3
    assert all(row["arm"] == arm and row["outcome"]["arm"] == arm for row in result["outcomes"])
    assert not any(row["accepted_for_cross_simulator_comparison"] for row in result["outcomes"])
    assert readiness.compare_shared_runs(result)["accepted_for_shared_ca_comparison"] is False
    first = receipt.root / "snapshot_h0002_state.tif"
    before = first.read_bytes()
    monitor = SimpleNamespace(check=lambda *a: None)
    with pytest.raises(FileExistsError):
        o._phase15_raster(receipt.root, first.name, np.ones((5, 5), np.int8),
            SimpleNamespace(transform=f.transform, crs=rasterio.crs.CRS.from_epsg(32651)),
            monitor, allowed_outputs=o._phase16_snapshot_outputs(receipt.start["configuration"]["effective"]["snapshot_basis"]))
    assert first.read_bytes() == before


@pytest.mark.parametrize("mismatch", ["model", "manifest", "arm", "seed", "ignition", "output"])
def test_shared_reader_rejects_receipt_outcome_swaps(shared_native_fixture, monkeypatch, mismatch):
    f = shared_native_fixture
    receipt = synthetic_receipt(f, monkeypatch, arm="lr")
    receipt.expected["alignment_sha256"] = f.shared_alignment["santa_maria"]["report_sha256"]
    end = deepcopy(receipt.end)
    if mismatch == "model": end["observed_bindings"]["model"]["observed_sha256"] = o.PHASE15_CANDIDATE_MODEL_SHA256
    elif mismatch == "manifest": end["observed_bindings"]["model"]["manifest"] = {"observed_sha256": "0" * 64}
    elif mismatch == "arm": end["snapshot_series"]["outcomes"][0]["arm"] = "rf"
    elif mismatch == "seed": end["snapshot_series"]["outcomes"][0]["seed"] = 43
    elif mismatch == "ignition": end["snapshot_series"]["outcomes"][0]["ignition_local_row_col"] = [[3, 2]]
    else: end["snapshot_series"]["outcomes"][0]["outputs"][0] = end["outputs"][-1]
    original = readiness._record
    monkeypatch.setattr(readiness, "_record", lambda path, pin:
        end if path.name == "termination.json" else original(path, pin))
    with pytest.raises(ValueError):
        shared_read(f, receipt)


def test_shared_rejects_old_ignition_context_and_pairs_verified_records(shared_native_fixture, monkeypatch):
    f = shared_native_fixture
    records, receipts = [], []
    for arm in ("rf", "lr"):
        receipt = synthetic_receipt(f, monkeypatch, arm=arm)
        receipt.expected["alignment_sha256"] = f.shared_alignment["santa_maria"]["report_sha256"]
        records.append(shared_read(f, receipt))
        receipts.append(receipt)
    diagnostic = readiness.compare_shared_runs(*records)
    assert diagnostic["accepted_for_shared_ca_comparison"] is False
    reader = readiness.read_snapshot_run
    roots_read = []
    def tracked(root, **kwargs):
        roots_read.append(root)
        return reader(root, **kwargs)
    monkeypatch.setattr(readiness, "read_snapshot_run", tracked)
    paired = readiness.read_shared_runs(
        receipts[0].root, expected=receipts[0].expected, alignment=f.shared_alignment["santa_maria"],
        capability=f.capability, authorized=True, authorization_reference="synthetic RF pair read",
        counterpart_root=receipts[1].root, counterpart_expected=receipts[1].expected,
        counterpart_alignment=f.shared_alignment["santa_maria"], counterpart_capability=f.capability,
        counterpart_authorized=True, counterpart_authorization_reference="synthetic LR pair read")
    assert roots_read == [receipt.root for receipt in receipts]
    assert paired["accepted_for_shared_ca_comparison"] is True
    assert paired["accepted_for_cross_simulator_comparison"] is False
    changed = deepcopy(records[1])
    changed["shared_pair"]["common"]["masks"]["evaluation_domain"]["sha256"] = "0" * 64
    changed["shared_pair_sha256"] = o.canonical_metadata_hash(changed["shared_pair"])
    with pytest.raises(ValueError, match="common"):
        readiness.compare_shared_runs(records[0], changed)
    asset = deepcopy(f.shared_alignment["santa_maria"])
    asset["report"]["ignition_global_row_col"] = [[3, 2]]
    asset["report_sha256"] = o.canonical_metadata_hash(asset["report"])
    receipt.expected["alignment_sha256"] = asset["report_sha256"]
    with pytest.raises(ValueError, match="ignition"):
        shared_read(f, receipt, alignment=asset)


@pytest.mark.parametrize("failure", ["export", "publication"])
def test_shared_late_failure_retention_and_orphan_rejection(shared_native_fixture, monkeypatch, failure):
    f = shared_native_fixture
    original = o._phase15_finalize
    def finalize(staging, final):
        if final.name == ("snapshot_h0004_state.tif" if failure == "export" else "termination.json"):
            raise OSError("synthetic shared late failure")
        return original(staging, final)
    monkeypatch.setattr(o, "_phase15_finalize", finalize)
    with pytest.raises(OSError, match="late failure"):
        synthetic_receipt(f, monkeypatch, arm="lr")
    root = f.code / "output/phase15/p16-dswd-shared-v1-santa-maria-lr-s42-r1"
    assert (root / "snapshot_h0002_state.tif").exists()
    assert (root / "snapshot_h0002_reached.tif").exists()
    assert not (root / "termination.json").exists()
    if failure == "publication":
        assert (root / "termination.json.partial").exists()
    start_pin = sha256((root / "start.json").read_bytes()).hexdigest()
    with pytest.raises(FileNotFoundError):
        readiness.read_snapshot_run(root, expected={"review_reference": "synthetic orphan",
            "start_sha256": start_pin, "termination_sha256": "0" * 64},
            alignment={}, capability=f.capability, authorized=True,
            authorization_reference="synthetic orphan rejection")


def shared_reporting_rows():
    rows = []
    for incident in ("airbase", "santa_maria"):
        for arm in ("rf", "lr"):
            basis = o._phase16_shared_basis(incident, arm)
            for horizon in basis["horizons"]:
                for seed in range(42, 47):
                    attribution = {key: deepcopy(basis[key]) for key in
                                   ("design", "arm", "model_policy")}
                    outcome = {**deepcopy(basis), **horizon, "seed": seed, "output_valid": True,
                               "status": "censored", "cumulative_reached_count": seed,
                               "active_cell_count": 1, "actual_capture_update": horizon["requested_update"]}
                    rows.append({**attribution, "incident": incident, "seed": seed,
                        "assumed_minutes_per_update": horizon["assumed_minutes_per_update"],
                        "metrics": {**{key: 0.5 for key in scoring.METRICS},
                                    "confusion_matrix": [[10, 2], [3, 4]]}, "outcome": outcome})
    return rows


@pytest.mark.parametrize("problem", ["none", "missing", "duplicate", "failed", "mixed_arm"])
def test_shared_twelve_groups_without_partial_or_mixed_means(problem):
    rows = shared_reporting_rows()
    if problem == "missing": rows.pop(0)
    elif problem == "duplicate": rows[0] = deepcopy(rows[1])
    elif problem == "failed": rows[0]["outcome"]["output_valid"] = False
    elif problem == "mixed_arm":
        rows[0]["outcome"]["arm"] = "lr"
        with pytest.raises(ValueError, match="attribution"):
            scoring.summarize_five_seed_groups(rows)
        return
    summary = scoring.summarize_five_seed_groups(rows)
    assert len(summary) == 12
    if problem == "none":
        assert all(group["status"] == "complete" and group["n"] == 5 for group in summary)
        assert all(group["mean_sample_sd"]["tn"]["mean"] == 10 for group in summary)
        assert all(group["mean_sample_sd"]["cumulative_reached_count"]["sample_sd"] == 2.5**0.5 for group in summary)
    else:
        incomplete = [group for group in summary if group["status"] == "incomplete"]
        assert len(incomplete) == 1 and incomplete[0]["mean_sample_sd"] is None
    assert all(group["trajectories_independent_across_clocks"] is False for group in summary)


def test_shared_empty_matrix_still_reports_twelve_incomplete_groups():
    groups = scoring.summarize_five_seed_groups([], design=o.PHASE16_DSWD_SHARED_PROTOCOL["design"])
    assert len(groups) == 12
    assert all(row["n"] == 0 and row["status"] == "incomplete" and row["mean_sample_sd"] is None for row in groups)


@pytest.mark.parametrize("arm", ["rf", "lr"])
def test_shared_observer_late_failure_invalidates_all_outcomes(monkeypatch, tmp_path, arm):
    original = tiny_basis
    def attributed(horizons=(1, 2, 3)):
        basis = original(horizons)
        shared = o._phase16_shared_basis("santa_maria", arm)
        basis.update({key: deepcopy(shared[key]) for key in
                      ("design", "arm", "model_policy", "ignition_local_row_col", "ignition_global_row_col")})
        return basis
    monkeypatch.setitem(globals(), "tiny_basis", attributed)
    test_late_export_failure_retains_first_capture_and_invalidates_every_outcome(tmp_path)


@pytest.mark.parametrize("stop", [0, 1, 2])
@pytest.mark.parametrize("arm", ["rf", "lr"])
def test_shared_observer_exact_and_early_inactivity(monkeypatch, stop, arm):
    original = tiny_basis
    def attributed(horizons=(1, 2, 3)):
        basis = original(horizons)
        shared = o._phase16_shared_basis("santa_maria", arm)
        basis.update({key: deepcopy(shared[key]) for key in
                      ("design", "arm", "model_policy", "ignition_local_row_col", "ignition_global_row_col")})
        return basis
    monkeypatch.setitem(globals(), "tiny_basis", attributed)
    test_initial_early_and_exact_horizon_inactivity(stop)

@pytest.mark.parametrize("arm", ["rf", "lr"])
def test_shared_active_horizons_remain_censored(arm):
    basis = tiny_basis()
    shared = o._phase16_shared_basis("santa_maria", arm)
    basis.update({key: deepcopy(shared[key]) for key in
                  ("design", "arm", "model_policy", "ignition_local_row_col", "ignition_global_row_col")})
    series = o._Phase16Snapshots(basis)
    (reason, stop, _), _ = collect(series, Scripted([[4], [4], [4], [4]]))
    assert reason == "timestep_limit" and stop == 3
    assert all(row["arm"] == arm and row["censoring"] and row["status"] == "censored"
               and row["state_counts"]["blazing"] == 1 for row in series.rows)


# D-203: dictionaries are diagnostic declarations, never native-read evidence.
def fabricated_shared_records(seed=42):
    records = []
    for arm in ("rf", "lr"):
        basis = o._phase16_shared_basis("santa_maria", arm)
        run_id = f"fabricated-{arm}"
        pair = {"design": basis["design"], "incident": "santa_maria", "seed": seed,
                "arm": arm, "model_policy": deepcopy(basis["model_policy"]),
                "start_sha256": "1" * 64, "termination_sha256": "2" * 64,
                "review_reference": "invented declaration, not reader evidence",
                "common": {"ignition_global_row_col": basis["ignition_global_row_col"],
                           "masks": {"evaluation_domain": {"sha256": "3" * 64, "count": 1}}},
                "wind": {}, "timing": {}, "source": {}, "runtime": {}, "feature_contract": {},
                "verified": True}
        outcomes = []
        for horizon in basis["horizons"]:
            outcomes.append({"outcome": {
                **horizon, "design": basis["design"], "arm": arm,
                "model_policy": deepcopy(basis["model_policy"]), "incident": "santa_maria",
                "seed": seed, "run_id": run_id, "execution_binding_sha256": "4" * 64,
                "start_record_sha256": pair["start_sha256"],
                "ignition_global_row_col": basis["ignition_global_row_col"],
                "ignition_local_row_col": basis["ignition_local_row_col"],
                "output_valid": True, "status": "complete"}})
        records.append({"schema_version": "dswd_verified_snapshot_run.v1", "run_id": run_id,
                        "binding_sha256": "4" * 64, "outcomes": outcomes, "shared_pair": pair,
                        "shared_pair_sha256": o.canonical_metadata_hash(pair), "verified": True,
                        "accepted_for_shared_ca_comparison": True})
    return records


@pytest.mark.parametrize("coordinated_rehash", [False, True])
def test_fabricated_and_jointly_rehashed_dictionaries_never_accept(monkeypatch, coordinated_rehash):
    records = fabricated_shared_records()
    if coordinated_rehash:
        for record in records:
            record["shared_pair"]["common"]["masks"]["evaluation_domain"] = {
                "sha256": "0" * 64, "count": 999}
            record["shared_pair"]["source"] = {"fabricated_revision": "changed on both arms"}
            record["shared_pair_sha256"] = o.canonical_metadata_hash(record["shared_pair"])
    monkeypatch.setattr(readiness, "read_snapshot_run",
                        lambda *a, **k: pytest.fail("Dictionary diagnostics performed a native read"))
    result = readiness.compare_shared_runs(*records)
    assert result["status"].startswith("DIAGNOSTIC MATCH")
    assert result["accepted_for_shared_ca_comparison"] is False
    assert result["accepted_for_cross_simulator_comparison"] is False


def test_fabricated_records_cannot_replace_explicit_roots():
    first, second = fabricated_shared_records()
    with pytest.raises(TypeError):
        readiness.read_shared_runs(
            first, expected={"review_reference": "synthetic root-type rejection"},
            alignment={}, capability={}, authorized=True, authorization_reference="synthetic first",
            counterpart_root=second, counterpart_expected={}, counterpart_alignment={},
            counterpart_capability={}, counterpart_authorized=True,
            counterpart_authorization_reference="synthetic second")


@pytest.mark.parametrize("seed", [True, 42.0, "42", None, 41, 47])
def test_invalid_shared_seeds_rejected_by_diagnostic_and_accepting_boundary(monkeypatch, tmp_path, seed):
    records = fabricated_shared_records(seed)
    with pytest.raises(ValueError, match="seed"):
        readiness.compare_shared_runs(*records)
    supplied = iter(records)
    calls = []
    def reader(root, **kwargs):
        calls.append(root)
        return next(supplied)
    monkeypatch.setattr(readiness, "read_snapshot_run", reader)
    with pytest.raises(ValueError, match="seed"):
        readiness.read_shared_runs(
            tmp_path / "rf", expected={}, alignment={}, capability={}, authorized=True,
            authorization_reference="synthetic first read", counterpart_root=tmp_path / "lr",
            counterpart_expected={}, counterpart_alignment={}, counterpart_capability={},
            counterpart_authorized=True, counterpart_authorization_reference="synthetic second read")
    assert calls == [tmp_path / "rf", tmp_path / "lr"]


@pytest.mark.parametrize("missing", ["first_authority", "first_reference", "second_authority",
                                    "second_reference", "second_expectations",
                                    "second_alignment", "second_capability"])
def test_pair_requires_each_authority_and_explicit_evidence_before_reads(monkeypatch, tmp_path, missing):
    arguments = dict(expected={}, alignment={}, capability={}, authorized=True,
        authorization_reference="synthetic first", counterpart_root=tmp_path / "lr",
        counterpart_expected={}, counterpart_alignment={}, counterpart_capability={},
        counterpart_authorized=True, counterpart_authorization_reference="synthetic second")
    key = {"first_authority": "authorized", "first_reference": "authorization_reference",
           "second_authority": "counterpart_authorized", "second_reference": "counterpart_authorization_reference",
           "second_expectations": "counterpart_expected", "second_alignment": "counterpart_alignment",
           "second_capability": "counterpart_capability"}[missing]
    arguments[key] = False if missing.endswith("authority") else None
    monkeypatch.setattr(readiness, "read_snapshot_run",
                        lambda *a, **k: pytest.fail("Read began without both authorities/evidence"))
    error = PermissionError if "authority" in missing else ValueError
    with pytest.raises(error):
        readiness.read_shared_runs(tmp_path / "rf", **arguments)


def synthetic_pair_request(f, monkeypatch):
    receipts = []
    for arm in ("rf", "lr"):
        receipt = synthetic_receipt(f, monkeypatch, arm=arm)
        receipt.expected["alignment_sha256"] = f.shared_alignment["santa_maria"]["report_sha256"]
        receipts.append(receipt)
    return receipts[0].root, dict(
        expected=receipts[0].expected, alignment=f.shared_alignment["santa_maria"],
        capability=f.capability, authorized=True, authorization_reference="synthetic reviewed RF read",
        counterpart_root=receipts[1].root, counterpart_expected=receipts[1].expected,
        counterpart_alignment=f.shared_alignment["santa_maria"], counterpart_capability=f.capability,
        counterpart_authorized=True, counterpart_authorization_reference="synthetic reviewed LR read")


def test_missing_counterpart_is_pending_after_real_reader(shared_native_fixture, monkeypatch):
    f = shared_native_fixture
    receipt = synthetic_receipt(f, monkeypatch, arm="rf")
    receipt.expected["alignment_sha256"] = f.shared_alignment["santa_maria"]["report_sha256"]
    result = readiness.read_shared_runs(
        receipt.root, expected=receipt.expected, alignment=f.shared_alignment["santa_maria"],
        capability=f.capability, authorized=True, authorization_reference="synthetic reviewed unpaired RF")
    assert result["status"].startswith("PENDING")
    assert result["accepted_for_shared_ca_comparison"] is False
    assert result["accepted_for_cross_simulator_comparison"] is False


@pytest.mark.parametrize("failed_arm", [0, 1])
def test_reader_failure_propagates_without_pair_acceptance(shared_native_fixture, monkeypatch, failed_arm):
    root, arguments = synthetic_pair_request(shared_native_fixture, monkeypatch)
    original = readiness.read_snapshot_run
    calls = []
    def fail_selected(selected_root, **kwargs):
        index = len(calls)
        calls.append(selected_root)
        if index == failed_arm:
            raise OSError("synthetic reader failure")
        return original(selected_root, **kwargs)
    monkeypatch.setattr(readiness, "read_snapshot_run", fail_selected)
    with pytest.raises(OSError, match="reader failure"):
        readiness.read_shared_runs(root, **arguments)
    assert calls == [root, arguments["counterpart_root"]][:failed_arm + 1]


@pytest.mark.parametrize("arm", ["first", "second"])
def test_pair_rejects_independently_reviewed_receipt_pin_mismatch(shared_native_fixture, monkeypatch, arm):
    root, arguments = synthetic_pair_request(shared_native_fixture, monkeypatch)
    key = "expected" if arm == "first" else "counterpart_expected"
    arguments[key] = deepcopy(arguments[key])
    arguments[key]["start_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        readiness.read_shared_runs(root, **arguments)


def test_shared_summary_cannot_inherit_cross_simulator_acceptance():
    rows = shared_reporting_rows()
    for row in rows:
        row["accepted_for_cross_simulator_comparison"] = True
    groups = scoring.summarize_five_seed_groups(rows)
    assert len(groups) == 12
    assert all(group["n"] == 5 and group["status"] == "complete" for group in groups)
    assert all(group["accepted_for_cross_simulator_comparison"] is False for group in groups)


def test_missing_reader_result_cannot_become_pair_acceptance(monkeypatch, tmp_path):
    supplied = iter([fabricated_shared_records()[0], None])
    monkeypatch.setattr(readiness, "read_snapshot_run", lambda *a, **k: next(supplied))
    with pytest.raises(ValueError, match="Counterpart reader"):
        readiness.read_shared_runs(
            tmp_path / "rf", expected={}, alignment={}, capability={}, authorized=True,
            authorization_reference="synthetic first", counterpart_root=tmp_path / "lr",
            counterpart_expected={}, counterpart_alignment={}, counterpart_capability={},
            counterpart_authorized=True, counterpart_authorization_reference="synthetic second")


# D-245 Route A: source preparation only; no production inputs in these tests.
RECOVERY_CASES = [("rf", 42)] + [("lr", seed) for seed in range(42, 47)]


@pytest.mark.parametrize("incident", ["airbase", "santa_maria"])
@pytest.mark.parametrize("arm", ["rf", "lr"])
@pytest.mark.parametrize("seed", range(42, 47))
def test_recovery_allows_only_six_exact_r2_cases(incident, arm, seed):
    config = shared_config(incident, arm, seed)
    run = config["phase15"]["run_id"][:-2] + "r2"
    config["phase15"].update(run_id=run, scenario_id=run + "-snapshots")
    before = deepcopy(config)
    if incident == "airbase" and (arm, seed) in RECOVERY_CASES:
        active = o._phase15_normalize(config)
        assert active["phase15"]["run_id"] == run
        assert active["phase15"]["scenario_id"] == run + "-snapshots"
        assert active["snapshot_basis"] == config["snapshot_basis"]
    else:
        with pytest.raises(ValueError):
            o._phase15_normalize(config)
    assert config == before and config["phase15"]["enabled"] is False


@pytest.mark.parametrize("mutation", ["r3", "scenario", "seed", "ignition", "cap", "wind", "root"])
def test_recovery_rejects_revision_scenario_and_control_changes(mutation):
    config = shared_config("airbase", "rf")
    run = config["phase15"]["run_id"][:-2] + "r2"
    config["phase15"].update(run_id=run, scenario_id=run + "-snapshots")
    if mutation == "r3": config["phase15"].update(run_id=run[:-1] + "3", scenario_id=run[:-1] + "3-snapshots")
    elif mutation == "scenario": config["phase15"]["scenario_id"] = run + "-wrong"
    elif mutation == "seed": config["simulation"]["seed"] = 42.0
    elif mutation == "ignition": config["simulation"]["ignition_points"] = [[638, 554]]
    elif mutation == "cap": config["simulation"]["max_timesteps"] = 300
    elif mutation == "wind": config["wind"]["direction_deg"] = 61.0
    else: config["output"]["output_dir"] = "output/recovery"
    with pytest.raises(ValueError):
        o._phase15_normalize(config)


@pytest.mark.parametrize("arm,seed", RECOVERY_CASES)
def test_recovery_yaml_only_changes_identity_and_comments(arm, seed):
    folder = Path(o.__file__).parent / "config/phase16_dswd_shared_ca_exec_v1"
    name = f"p16-dswd-shared-v1-airbase-{arm}-s{seed}"
    original = o._phase15_yaml((folder / (name + "-r1.yaml")).read_bytes())
    recovery = o._phase15_yaml((folder / (name + "-r2.yaml")).read_bytes())
    expected = deepcopy(original)
    expected["phase15"].update(run_id=name + "-r2", scenario_id=name + "-r2-snapshots")
    assert o._phase15_json_bytes(recovery) == o._phase15_json_bytes(expected)
    assert recovery["phase15"]["enabled"] is False
    assert recovery["phase15"]["authorization_reference"] == "p16-dswd-shared-ca-v1-execution-r1"
    assert o._phase15_normalize(recovery)["snapshot_basis"] == original["snapshot_basis"]


@pytest.mark.parametrize("arm", ["rf", "lr"])
def test_shared_binding_pins_actual_guard_and_changes_on_drift(monkeypatch, tmp_path, arm):
    code = tmp_path / "Code"
    code.mkdir()
    guard = code / "dswd_inference_guard.py"
    guard.write_bytes(Path(inference_guard.__file__).read_bytes())
    (code / "run_dswd_shared_ca_v1.ps1").write_bytes(
        (Path(o.__file__).parent / "run_dswd_shared_ca_v1.ps1").read_bytes())
    (code / "dswd_snapshot_results.py").write_text("# synthetic source\n", encoding="utf-8")
    monkeypatch.setattr(o, "PHASE15_CODE_DIR", code)
    monkeypatch.setattr(o, "_phase15_source", lambda: {"files": {}, "head": "synthetic", "dirty": True})
    monkeypatch.setattr(o, "_phase15_runtime", lambda: {"runtime": "synthetic"})
    monkeypatch.setattr(o, "_phase16_snapshot_manifest", lambda *a: {"path": "synthetic", "sha256": "a" * 64})
    config = shared_config("airbase", arm)
    # Configuration paths are lexical synthetic declarations; no model is created.
    active = deepcopy(config)
    active["phase15"]["enabled"] = True
    path = code / "execution.yaml"
    first = o._phase15_binding(active, path, "0" * 64)
    key = "Thesis_RF/Code/dswd_inference_guard.py"
    assert first["source"]["files"][key] == sha256(guard.read_bytes()).hexdigest()
    guard.write_text("# changed synthetic guard\n", encoding="utf-8")
    second = o._phase15_binding(active, path, "0" * 64)
    assert o.canonical_metadata_hash(first) != o.canonical_metadata_hash(second)
    assert first["source"]["files"][key] != second["source"]["files"][key]


@pytest.mark.parametrize("arm", ["rf", "lr"])
def test_recovery_native_receipt_and_collision_are_still_guarded(shared_native_fixture, monkeypatch, arm):
    f = shared_native_fixture
    receipt = synthetic_receipt(f, monkeypatch, arm=arm, incident="airbase", revision=2)
    receipt.expected["alignment_sha256"] = f.shared_alignment["airbase"]["report_sha256"]
    result = readiness.read_snapshot_run(receipt.root, expected=receipt.expected,
        alignment=f.shared_alignment["airbase"], capability=f.capability,
        authorized=True, authorization_reference="synthetic recovery read")
    assert result["run_id"].endswith("-r2") and len(result["outcomes"]) == 3
    assert readiness.compare_shared_runs(result)["accepted_for_shared_ca_comparison"] is False
    retained = {p.name: p.read_bytes() for p in receipt.root.iterdir()}
    options = dict(explicitly_authorized=True,
        approval_reference=receipt.start["authorization_reference"],
        expected_binding_sha256=receipt.start["binding"]["calculated_sha256"],
        config_path=Path(receipt.start["configuration"]["path"]))
    config = deepcopy(receipt.start["configuration"]["effective"])
    config["phase15"]["enabled"] = False
    with pytest.raises(PermissionError, match="binding"):
        o.run_simulation(config, **{**options, "expected_binding_sha256": "0" * 64})
    with pytest.raises(FileExistsError):
        o.run_simulation(config, **options)
    (f.code / "dswd_inference_guard.py").write_text("# changed synthetic guard\n", encoding="utf-8")
    monkeypatch.setattr(o, "_phase15_candidate_environment",
                        lambda *a: pytest.fail("Stale guard binding reached environmental startup"))
    with pytest.raises(PermissionError, match="binding"):
        o.run_simulation(config, **options)
    assert retained == {p.name: p.read_bytes() for p in receipt.root.iterdir()}


@pytest.mark.parametrize("difference", ["source", "runtime"])
def test_pairing_never_ignores_source_or_runtime(difference):
    records = fabricated_shared_records()
    records[1]["shared_pair"][difference] = {"revision": "different"}
    records[1]["shared_pair_sha256"] = o.canonical_metadata_hash(records[1]["shared_pair"])
    with pytest.raises(ValueError, match="common"):
        readiness.compare_shared_runs(*records)


def test_real_reader_pair_rejects_old_source_prefix(shared_native_fixture, monkeypatch):
    f = shared_native_fixture
    receipts = [synthetic_receipt(f, monkeypatch, arm=arm, source_revision=revision)
                for arm, revision in (("rf", "recovery-source"), ("lr", "historical-source"))]
    for receipt in receipts:
        receipt.expected["alignment_sha256"] = f.shared_alignment["santa_maria"]["report_sha256"]
    with pytest.raises(ValueError, match="common"):
        readiness.read_shared_runs(receipts[0].root, expected=receipts[0].expected,
            alignment=f.shared_alignment["santa_maria"], capability=f.capability,
            authorized=True, authorization_reference="synthetic recovery source read",
            counterpart_root=receipts[1].root, counterpart_expected=receipts[1].expected,
            counterpart_alignment=f.shared_alignment["santa_maria"], counterpart_capability=f.capability,
            counterpart_authorized=True, counterpart_authorization_reference="synthetic old source read")


@pytest.fixture
def calibrated_prediction_fixture():
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.calibration import CalibratedClassifierCV, _CalibratedClassifier
    from modules.feature_pipeline import predict_positive_probability
    calls = []
    class FixedClassifier(ClassifierMixin, BaseEstimator):
        def predict_proba(self, X):
            calls.append(("estimator", np.asarray(X).copy()))
            return np.tile([0.75, 0.25], (len(X), 1))
    class FixedSigmoid:
        def predict(self, values):
            calls.append(("calibrator", np.asarray(values).copy()))
            return np.asarray(values) * 0.8 + 0.1
    classes = np.asarray([0, 1], dtype=np.int64)
    estimator = FixedClassifier()
    estimator.classes_ = classes.copy()
    owner = CalibratedClassifierCV(estimator=estimator, method="sigmoid")
    owner.classes_ = classes
    owner.feature_names_in_ = np.asarray(CANONICAL_FEATURE_NAMES)
    inner = _CalibratedClassifier(estimator, [FixedSigmoid()], classes=classes, method="sigmoid")
    owner.calibrated_classifiers_ = [inner]
    X = np.zeros((3, len(CANONICAL_FEATURE_NAMES)), dtype=np.float32)
    return SimpleNamespace(owner=owner, inner=inner, X=X, calls=calls,
                           predict=lambda: predict_positive_probability(owner, X))


def run_with_inference_guard(callback, operation):
    previous = sys.getprofile()
    try:
        sys.setprofile(callback)
        return operation()
    finally:
        sys.setprofile(previous)


def test_canonical_guard_import_and_factory_do_not_install_or_read(monkeypatch, tmp_path):
    import importlib
    previous = sys.getprofile()
    def denied(*args, **kwargs):
        pytest.fail("Guard import/factory attempted filesystem access")
    monkeypatch.setattr(Path, "open", denied)
    monkeypatch.setattr(Path, "resolve", denied)
    importlib.reload(inference_guard)
    callback = inference_guard.make_inference_guard(tmp_path, arm="rf")
    assert callable(callback) and sys.getprofile() is previous
    with pytest.raises(ValueError):
        inference_guard.make_inference_guard(tmp_path, arm="other")


def test_actual_sklearn_prediction_guard_preserves_outputs_calls_and_rng(calibrated_prediction_fixture, monkeypatch):
    f = calibrated_prediction_fixture
    expected = f.predict()
    expected_calls = [(name, values.copy()) for name, values in f.calls]
    f.calls.clear()
    classes_before = f.owner.classes_.copy()
    generator = np.random.default_rng(42)
    generator_before = deepcopy(generator.bit_generator.state)
    global_before = np.random.get_state()
    def no_draw(*args, **kwargs):
        pytest.fail("Inference observation consumed an extra RNG draw")
    monkeypatch.setattr(np.random, "random", no_draw)
    callback = inference_guard.make_inference_guard(Path(o.__file__).parent, arm="rf")
    actual = run_with_inference_guard(callback, f.predict)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(f.owner.classes_, classes_before)
    np.testing.assert_array_equal(f.inner.classes, classes_before)
    assert len(f.calls) == len(expected_calls) == 2
    for (name, values), (expected_name, expected_values) in zip(f.calls, expected_calls):
        assert name == expected_name
        np.testing.assert_array_equal(values, expected_values)
    assert generator.bit_generator.state == generator_before
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0] and global_before[2:] == global_after[2:]
    np.testing.assert_array_equal(global_before[1], global_after[1])


def test_actual_prediction_chain_cannot_use_rf_exception_in_lr_context(calibrated_prediction_fixture, tmp_path):
    with pytest.raises(PermissionError, match="Fitting"):
        run_with_inference_guard(inference_guard.make_inference_guard(tmp_path, arm="lr"),
                                 calibrated_prediction_fixture.predict)


@pytest.mark.parametrize("kind", ["encoder", "encoder_transform", "estimator", "calibrator", "partial_fit"])
def test_actual_guard_rejects_fitting_calls(tmp_path, kind):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import _SigmoidCalibration
    class Incremental:
        def partial_fit(self):
            pytest.fail("Training body was entered")
    operations = {
        "encoder": lambda: LabelEncoder().fit([0, 1]),
        "encoder_transform": lambda: LabelEncoder().fit_transform([0, 1]),
        "estimator": lambda: RandomForestClassifier(n_estimators=1).fit([[0], [1]], [0, 1]),
        "calibrator": lambda: _SigmoidCalibration().fit(np.asarray([0., 1.]), np.asarray([0, 1])),
        "partial_fit": lambda: Incremental().partial_fit(),
    }
    with pytest.raises(PermissionError, match="Fitting"):
        run_with_inference_guard(inference_guard.make_inference_guard(tmp_path, arm="rf"), operations[kind])


@pytest.fixture
def label_fit_frame(calibrated_prediction_fixture):
    import sklearn.calibration as cal
    import sklearn.preprocessing._label as labels
    import modules.feature_pipeline as features
    f = calibrated_prediction_fixture
    origin = SimpleNamespace(f_code=features.predict_positive_probability.__code__,
        f_globals=vars(features), f_locals={"model": f.owner, "features": f.X})
    outer = SimpleNamespace(f_code=cal.CalibratedClassifierCV.predict_proba.__code__,
        f_globals=vars(cal), f_locals={"self": f.owner, "X": f.X}, f_lineno=530, f_back=origin)
    caller = SimpleNamespace(f_code=cal._CalibratedClassifier.predict_proba.__code__,
        f_globals=vars(cal), f_locals={"self": f.inner, "X": f.X}, f_lineno=810, f_back=outer)
    frame = SimpleNamespace(f_code=labels.LabelEncoder.fit.__code__, f_globals=vars(labels),
        f_locals={"self": labels.LabelEncoder(), "y": f.inner.classes}, f_back=caller)
    return SimpleNamespace(frame=frame, caller=caller, outer=outer, origin=origin, prediction=f)


@pytest.mark.parametrize("classes,allowed", [
    (np.asarray([0, 1], dtype=np.int8), True),
    (np.asarray([0, 1], dtype=np.uint64), True),
    (np.asarray([0., 1.]), False), (np.asarray([False, True]), False),
    (np.asarray([1, 0]), False), (np.asarray([0, 2]), False), ([0, 1], False)])
def test_canonical_guard_class_representation_is_fail_closed(label_fit_frame, tmp_path, classes, allowed):
    f = label_fit_frame
    f.prediction.inner.classes = f.prediction.owner.classes_ = classes
    f.frame.f_locals["y"] = classes
    callback = inference_guard.make_inference_guard(tmp_path, arm="rf")
    if allowed:
        callback(f.frame, "call", None)
    else:
        with pytest.raises(PermissionError): callback(f.frame, "call", None)


@pytest.mark.parametrize("mismatch", ["arm", "code_path", "globals", "caller", "caller_line", "outer_line",
    "origin", "reused_encoder", "subclass", "inner_type", "outer_type", "outer_globals", "origin_globals",
    "ownership", "model", "feature_identity", "class_identity", "method"])
def test_canonical_guard_rejects_wrong_context(label_fit_frame, tmp_path, mismatch):
    from sklearn.preprocessing import LabelEncoder
    f = label_fit_frame
    arm = "lr" if mismatch == "arm" else "rf"
    if mismatch == "code_path": f.frame.f_code = f.frame.f_code.replace(co_filename="unapproved.py")
    elif mismatch == "globals": f.frame.f_globals = {}
    elif mismatch == "caller": f.caller.f_code = f.frame.f_code
    elif mismatch == "caller_line": f.caller.f_lineno = 726
    elif mismatch == "outer_line": f.outer.f_lineno = 531
    elif mismatch == "origin": f.origin.f_code = f.frame.f_code
    elif mismatch == "reused_encoder": f.frame.f_locals["self"].classes_ = np.asarray([0, 1])
    elif mismatch == "subclass":
        class OtherEncoder(LabelEncoder): pass
        f.frame.f_locals["self"] = OtherEncoder()
    elif mismatch == "inner_type": f.caller.f_locals["self"] = object()
    elif mismatch == "outer_type": f.outer.f_locals["self"] = object()
    elif mismatch == "outer_globals": f.outer.f_globals = {}
    elif mismatch == "origin_globals": f.origin.f_globals = {}
    elif mismatch == "ownership": f.prediction.owner.calibrated_classifiers_ = []
    elif mismatch == "model": f.origin.f_locals["model"] = object()
    elif mismatch == "feature_identity": f.caller.f_locals["X"] = f.prediction.X.copy()
    elif mismatch == "class_identity": f.frame.f_locals["y"] = f.prediction.inner.classes.copy()
    elif mismatch == "method": f.prediction.inner.method = "isotonic"
    with pytest.raises(PermissionError):
        inference_guard.make_inference_guard(tmp_path, arm=arm)(f.frame, "call", None)


@pytest.mark.parametrize("name", ["simulation_main", "_run_legacy_simulation", "collect_alignment",
    "read_snapshot_run", "run_set_c_pilot", "run_model_free_teacher", "draw_figure", "evaluate_anything"])
def test_canonical_guard_retains_project_entry_exclusions(tmp_path, name):
    frame = SimpleNamespace(f_code=SimpleNamespace(co_name=name, co_filename=str(tmp_path / "entry.py")))
    with pytest.raises(PermissionError, match="Excluded project entry"):
        inference_guard.make_inference_guard(tmp_path, arm="lr")(frame, "call", None)


# D-338: actual embedded caller logic; no production startup or child process.
@pytest.fixture(autouse=True)
def forbid_production_metadata(monkeypatch, tmp_path):
    import os
    code = str(Path(o.__file__).parent).replace("\\", "/").lower()
    temporary = str(tmp_path).replace("\\", "/").lower()
    def checked(delegate, path, *args, **kwargs):
        if not isinstance(path, int):
            # Lexical normalization only; denial precedes the metadata delegate.
            value = os.path.abspath(os.fsdecode(path)).replace("\\", "/").lower()
            if not value.startswith(temporary + "/") and (
                    "/thesis_lr/" in value or any(value.startswith(code + "/" + p)
                    for p in ("output/", "processeddata/", "datafiles/", "models/"))):
                raise AssertionError("Synthetic test denied production metadata: " + value)
        return delegate(path, *args, **kwargs)
    real_stat, real_lstat = os.stat, os.lstat
    monkeypatch.setattr(os, "stat", lambda path, *a, **k: checked(real_stat, path, *a, **k))
    monkeypatch.setattr(os, "lstat", lambda path, *a, **k: checked(real_lstat, path, *a, **k))


@pytest.fixture
def actual_caller_contract(monkeypatch, tmp_path):
    import os
    import re
    import warnings
    import modules.feature_pipeline as features
    text = (Path(o.__file__).parent / "run_dswd_shared_ca_v1.ps1").read_text(encoding="utf-8")
    begin, end = "# BEGIN DSWD CALLER CONTRACT", "# END DSWD CALLER CONTRACT"
    assert text.count(begin) == text.count(end) == 1
    source = text.split(begin, 1)[1].split(end, 1)[0]
    body = text.split("$body=@'\n", 1)[1].split("\n'@", 1)[0]
    dependencies = "def require(" + body.split("def require(", 1)[1].split("def sig(", 1)[0]
    source = dependencies + source
    namespace = {"os": os, "sys": sys, "Path": Path, "re": re, "json": json,
                 "warnings": warnings,
                 "SITE": tmp_path / "site", "lex": lambda p: os.path.normcase(os.path.abspath(p))}
    exec(compile(source, "<actual-dswd-caller-contract>", "exec"), namespace)
    monkeypatch.setenv("JOBLIB_MULTIPROCESSING", "0")
    return SimpleNamespace(execute=namespace["selected_dispatch"], namespace=namespace,
                           features=features, source=source)


@pytest.fixture(params=["rf", "lr"])
def caller_case(actual_caller_contract, monkeypatch, tmp_path, request):
    config = shared_config("airbase", request.param)
    run = config["phase15"]["run_id"][:-2] + "r2"
    config["phase15"].update(run_id=run, scenario_id=run + "-snapshots")
    calls, callbacks = [], []
    real_factory = inference_guard.make_inference_guard
    def factory(code, *, arm):
        callback = real_factory(code, arm=arm)
        callbacks.append((arm, callback))
        return callback
    monkeypatch.setattr(inference_guard, "make_inference_guard", factory)
    def startup(received, **kwargs):
        calls.append((received, kwargs, sys.getprofile()))
        assert received is config and received["phase15"]["enabled"] is False
        assert sys.getprofile() is callbacks[-1][1]
    monkeypatch.setattr(o, "run_simulation", startup)
    def binding(active, path, digest):
        return {"source": {"files": {"Thesis_RF/Code/run_dswd_shared_ca_v1.ps1": "c" * 64}},
                "runtime": {"scope": "synthetic"}, "run": {"run_id": active["phase15"]["run_id"]}}
    monkeypatch.setattr(o, "_phase15_binding", binding)
    expected = o.canonical_metadata_hash(binding(config, None, None))
    options = dict(run_id=run, arm=request.param, authority="D-999999", execute=True,
                   expected_config_sha256="a" * 64, disk_sha256="a" * 64,
                   expected_binding_sha256=expected, launcher_sha256="c" * 64)
    state = {}
    def invoke(**overrides):
        return actual_caller_contract.execute(o, inference_guard, actual_caller_contract.features,
            config, tmp_path / (run + ".yaml"), state=state, **{**options, **overrides})
    return SimpleNamespace(config=config, state=state, options=options, callbacks=callbacks,
                           calls=calls, invoke=invoke, contract=actual_caller_contract)


def test_actual_caller_installs_canonical_guard_before_exactly_one_disabled_dispatch(caller_case):
    r = caller_case
    before = deepcopy(r.config)
    active = r.invoke()
    assert active["phase15"]["enabled"] is True
    assert r.config == before and r.config["phase15"]["enabled"] is False
    assert len(r.calls) == len(r.callbacks) == 1
    assert r.callbacks[0][0] == r.options["arm"]
    assert r.calls[0][1] == dict(explicitly_authorized=True,
        approval_reference="p16-dswd-shared-ca-v1-execution-r1",
        expected_binding_sha256=r.options["expected_binding_sha256"],
        config_path=r.calls[0][1]["config_path"])
    assert r.state["guard_entry"] and r.state["guard_completion"] and r.state["guard_removed"]
    assert r.state["dispatch_attempts"] == r.state["dispatch_completions"] == 1
    assert sys.getprofile() is None


@pytest.mark.parametrize("change, message", [
    ({"execute": False}, "authority"), ({"authority": "D-338"}, "authority"),
    ({"authority": "p16-dswd-shared-ca-v1-execution-r1"}, "authority"),
    ({"arm": "opposite"}, "arm"), ({"run_id": "p16-dswd-shared-v1-airbase-rf-s42-r1"}, "Unselected"),
    ({"disk_sha256": "b" * 64}, "Configuration identity"),
    ({"expected_binding_sha256": "0" * 64}, "binding")])
def test_actual_caller_rejects_before_startup(caller_case, change, message):
    r = caller_case
    if change.get("arm") == "opposite":
        change = {"arm": "lr" if r.options["arm"] == "rf" else "rf"}
    with pytest.raises(RuntimeError, match=message):
        r.invoke(**change)
    assert r.calls == r.callbacks == []
    assert sys.getprofile() is None


def test_actual_caller_rejects_legacy_profile_and_environment(caller_case, monkeypatch):
    r = caller_case
    monkeypatch.setenv("JOBLIB_MULTIPROCESSING", "1")
    with pytest.raises(RuntimeError, match="Serialization environment"):
        r.invoke()
    monkeypatch.setenv("JOBLIB_MULTIPROCESSING", "0")
    r.config["phase15"]["schema_version"] = o.PHASE16_DSWD_SCHEMA
    with pytest.raises(RuntimeError, match="Shared trajectory"):
        r.invoke()
    assert not r.calls and not r.callbacks


@pytest.mark.parametrize("failure", ["loss", "callback", "startup"])
def test_actual_caller_never_reinstalls_or_retries(caller_case, monkeypatch, failure):
    r = caller_case
    entered = []
    def fit():
        pytest.fail("Canonical guard allowed prohibited fitting body")
    def startup(config, **kwargs):
        entered.append(sys.getprofile())
        assert entered[-1] is r.callbacks[-1][1]
        if failure == "loss": sys.setprofile(None)
        elif failure == "callback": fit()
        else: raise RuntimeError("synthetic startup failure")
    monkeypatch.setattr(o, "run_simulation", startup)
    expected = PermissionError if failure == "callback" else RuntimeError
    message = {"loss": "callback lost", "callback": "Fitting is excluded", "startup": "synthetic startup"}[failure]
    with pytest.raises(expected, match=message):
        r.invoke()
    assert len(entered) == len(r.callbacks) == 1
    assert r.state["dispatch_attempts"] == 1
    assert not r.state["guard_completion"]
    assert sys.getprofile() is None


def test_actual_caller_rejects_existing_profiler(caller_case):
    def existing(frame, event, arg):
        return None
    previous = sys.getprofile()
    try:
        sys.setprofile(existing)
        with pytest.raises(RuntimeError, match="Pre-existing profiler"):
            caller_case.invoke()
        assert sys.getprofile() is existing
        assert not caller_case.calls and not caller_case.callbacks
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("operation", ["open", "io_open", "native", "resolve", "stat", "lstat", "path_stat", "path_lstat"])
def test_production_io_denial_is_independent_of_canonical_callback(operation):
    import os
    path = Path(o.__file__).parent / "processedData/raster/raster/forbidden-synthetic-probe.tif"
    operations = {"open": lambda: builtins.open(path, "rb"), "io_open": lambda: io.open(path, "rb"),
                  "native": lambda: rasterio.open(path), "resolve": lambda: path.resolve(),
                  "stat": lambda: os.stat(path), "lstat": lambda: os.lstat(path),
                  "path_stat": lambda: path.stat(), "path_lstat": lambda: path.lstat()}
    assert sys.getprofile() is None
    with pytest.raises(AssertionError, match="production I/O|production metadata"):
        operations[operation]()


@pytest.fixture
def caller_binding_case(monkeypatch, tmp_path):
    code = tmp_path / "Code"
    code.mkdir()
    for name in ("dswd_inference_guard.py", "dswd_snapshot_results.py"):
        (code / name).write_text("# synthetic source\n", encoding="utf-8")
    caller = code / "run_dswd_shared_ca_v1.ps1"
    caller.write_bytes((Path(o.__file__).parent / caller.name).read_bytes())
    monkeypatch.setattr(o, "PHASE15_CODE_DIR", code)
    monkeypatch.setattr(o, "_phase15_source", lambda: {"files": {}, "head": "synthetic", "dirty": True})
    monkeypatch.setattr(o, "_phase15_runtime", lambda: {"scope": "synthetic"})
    monkeypatch.setattr(o, "_phase16_snapshot_manifest", lambda *a: {"path": "synthetic", "sha256": "f" * 64})
    return SimpleNamespace(code=code, caller=caller)


@pytest.mark.parametrize("arm", ["rf", "lr"])
def test_complete_caller_binding_drift_includes_text_outside_python(caller_binding_case, arm):
    f = caller_binding_case
    active = shared_config("airbase", arm)
    active["phase15"]["enabled"] = True
    first = o._phase15_binding(active, f.code / "execution.yaml", "a" * 64)
    original = f.caller.read_bytes()
    f.caller.write_bytes(original + b"\n# synthetic whole-caller drift outside body\n")
    second = o._phase15_binding(active, f.code / "execution.yaml", "a" * 64)
    key = "Thesis_RF/Code/run_dswd_shared_ca_v1.ps1"
    assert first["source"]["files"][key] == sha256(original).hexdigest()
    assert second["source"]["files"][key] == sha256(f.caller.read_bytes()).hexdigest()
    assert o.canonical_metadata_hash(first) != o.canonical_metadata_hash(second)


@pytest.mark.parametrize("arm", ["rf", "lr"])
@pytest.mark.parametrize("failure", ["missing", "reparse"])
def test_caller_binding_rejects_missing_or_reparse(caller_binding_case, monkeypatch, arm, failure):
    import stat
    f = caller_binding_case
    active = shared_config("airbase", arm)
    active["phase15"]["enabled"] = True
    if failure == "missing":
        f.caller.unlink()
        error = FileNotFoundError
    else:
        real = Path.lstat
        def lstat(path, *args, **kwargs):
            if path == f.caller:
                return SimpleNamespace(st_mode=stat.S_IFREG, st_file_attributes=0x400)
            return real(path, *args, **kwargs)
        monkeypatch.setattr(Path, "lstat", lstat)
        error = PermissionError
    with pytest.raises(error):
        o._phase15_binding(active, f.code / "execution.yaml", "a" * 64)


@pytest.mark.parametrize("arm", ["rf", "lr"])
def test_stale_caller_binding_cannot_reach_protected_startup(shared_native_fixture, monkeypatch, arm):
    f = shared_native_fixture
    receipt = synthetic_receipt(f, monkeypatch, arm=arm, incident="airbase", revision=2)
    caller = f.code / "run_dswd_shared_ca_v1.ps1"
    caller.write_bytes(caller.read_bytes() + b"\n# synthetic changed caller\n")
    monkeypatch.setattr(o, "_phase15_candidate_environment",
                        lambda *a: pytest.fail("Stale caller reached protected startup"))
    config = deepcopy(receipt.start["configuration"]["effective"])
    config["phase15"]["enabled"] = False
    with pytest.raises(PermissionError, match="binding"):
        o.run_simulation(config, explicitly_authorized=True,
            approval_reference=receipt.start["authorization_reference"],
            expected_binding_sha256=receipt.start["binding"]["calculated_sha256"],
            config_path=Path(receipt.start["configuration"]["path"]))


def test_legacy_binding_never_requires_the_shared_caller(caller_binding_case, monkeypatch):
    f = caller_binding_case
    f.caller.unlink()
    legacy = snapshot_config("santa_maria")
    legacy["phase15"]["enabled"] = True
    binding = o._phase15_binding(legacy, f.code / "legacy.yaml", "a" * 64)
    assert "Thesis_RF/Code/run_dswd_shared_ca_v1.ps1" not in binding["source"]["files"]
    assert "Thesis_RF/Code/dswd_inference_guard.py" not in binding["source"]["files"]


@pytest.mark.parametrize("repeat", [0, 1, 3])
def test_actual_caller_allows_source_matched_repeated_production_warnings(caller_case, monkeypatch, repeat):
    import warnings
    r = caller_case
    estimator = "CalibratedClassifierCV" if r.options["arm"] == "rf" else "LogisticRegression"
    class StubEstimator:
        classes_ = np.asarray([0, 1], dtype=np.int64)
        feature_names_in_ = np.asarray(CANONICAL_FEATURE_NAMES)
        def predict_proba(self, X):
            for unused in range(repeat):
                warnings.warn_explicit("X does not have valid feature names, but " + estimator
                    + " was fitted with feature names", UserWarning,
                    str(r.contract.namespace["SITE"] / "sklearn/utils/validation.py"), 2691)
            return np.tile([0.75, 0.25], (len(X), 1))
    calls = []
    def startup(config, **kwargs):
        calls.append(config)
        for unused in range(2):
            result = r.contract.features.predict_positive_probability(
                StubEstimator(), np.zeros((1, 11), dtype=np.float32))
            assert result.tolist() == [0.25]
    monkeypatch.setattr(o, "run_simulation", startup)
    r.invoke()
    assert len(calls) == 1 and len(r.state["warnings"]) == 2 * repeat
    assert all(w["accepted"] and w["arm"] == r.options["arm"] for w in r.state["warnings"])


def test_actual_caller_retains_and_rejects_unexpected_warning(caller_case, monkeypatch):
    import warnings
    r = caller_case
    def startup(config, **kwargs):
        warnings.warn("synthetic unexpected warning", UserWarning)
    monkeypatch.setattr(o, "run_simulation", startup)
    with pytest.raises(RuntimeError, match="Unexpected production warning"):
        r.invoke()
    assert r.state["dispatch_attempts"] == 1
    assert r.state["dispatch_completions"] == 0
    assert len(r.state["warnings"]) == 1 and not r.state["warnings"][0]["accepted"]
    assert r.state["warnings"][0]["message"] == "synthetic unexpected warning"
    assert sys.getprofile() is None
