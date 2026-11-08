"""Synthetic Airbase execution profile and unchanged F01 routing checks."""
from copy import deepcopy

import pytest

import orchestrator as o
from test_simulation_readiness_contract import candidate_ready, ready, rebind, run, record
from test_simulation_readiness_contract import use_lr_comparison


@pytest.fixture
def airbase_ready(candidate_ready, monkeypatch):
    r = candidate_ready
    config = r.config
    config["phase15"].update(schema_version=o.PHASE16_AIRBASE_SCHEMA,
                            scenario_id=o.PHASE16_AIRBASE_SCENARIO_ID)
    config["wind_input_basis"] = deepcopy(o.PHASE16_AIRBASE_WIND_INPUT_BASIS)
    config["wind"].update(speed_kmh=10.6, direction_deg=o.PHASE16_AIRBASE_GRID_FROM_DEG)
    monkeypatch.setattr(o, "PHASE16_AIRBASE_WINDOW", deepcopy(config["candidate_inputs"]["window"]))
    monkeypatch.setattr(o, "PHASE16_AIRBASE_IGNITIONS", [[0, 0]])
    rebind(r)
    return r


@pytest.mark.parametrize("seed", range(42, 47))
def test_airbase_five_seeds_use_frozen_rf_and_candidate_reader(airbase_ready, seed):
    r = airbase_ready
    r.config["simulation"]["seed"] = seed
    r.config["phase15"]["scenario_id"] = o.PHASE16_AIRBASE_SCENARIO_ID.replace("-s42-", f"-s{seed}-")
    rebind(r)
    assert "manifest" in r.binding["model"]
    assert r.binding["evidence_scope"]["incident"] == "airbase_sanggi_20250124"
    assert r.binding["evidence_scope"]["final_test_eligible"] is False
    assert r.binding["evidence_scope"]["reference_accessed_by_simulation"] is False
    run(r)
    assert record(r)["output_valid"] is True
    assert len(r.reader_calls) == 1


@pytest.mark.parametrize("section,key,value", [
    ("simulation", "seed", 47), ("simulation", "ignition_points", [[638, 554]]),
    ("wind", "speed_kmh", 17.7), ("wind", "direction_deg", 61.0),
    ("wind", "direction_deg", 48.825208),
    ("ml_model", "model_sha256", o.PHASE16_LR_MODEL_SHA256),
    ("ml_model", "calibration_manifest_sha256", None),
    ("ml_model", "inference_mode", "stochastic_probability_gate"),
    ("ml_model", "inference_threshold", 0.40),
    ("phase15", "scenario_id", o.PHASE16_SCENARIO_ID),
    ("candidate_inputs", "window", o.PHASE16_AIRBASE_WINDOW),
    ("wind_input_basis", "source_row", "2025,1,24,12,3.4,60.8"),
])
def test_airbase_rejects_changed_incident_controls(airbase_ready, section, key, value):
    config = deepcopy(airbase_ready.config)
    config[section][key] = value
    with pytest.raises((ValueError, TypeError)):
        o._phase15_normalize(config)


def test_f01_does_not_accept_airbase_window_or_basis(candidate_ready):
    config = deepcopy(candidate_ready.config)
    config["candidate_inputs"]["window"] = deepcopy(o.PHASE16_AIRBASE_WINDOW)
    config["wind_input_basis"] = deepcopy(o.PHASE16_AIRBASE_WIND_INPUT_BASIS)
    with pytest.raises((ValueError, TypeError)):
        o._phase15_normalize(config)


def test_frozen_real_airbase_geometry_has_boundary_margin_and_correct_ignition():
    assert o.PHASE16_AIRBASE_WINDOW["global_origin"] == [820, 1989]
    assert o.PHASE16_AIRBASE_WINDOW["shape"] == [1024, 1024]
    assert o.PHASE16_AIRBASE_IGNITIONS == [[512, 512]]
    assert [820 + 512, 1989 + 512] == [1332, 2501]
    assert min(512, 1023 - 512) > 300


def use_duration90(r):
    r.config["phase15"].update(schema_version=o.PHASE16_AIRBASE_DURATION90_SCHEMA,
        scenario_id=o.PHASE16_AIRBASE_SCENARIO_ID + "-h90-dt1m")
    r.config["simulation"]["max_timesteps"] = 90
    r.config["duration_basis"] = deepcopy(o.PHASE16_AIRBASE_DURATION90_BASIS)


@pytest.mark.parametrize("seed", range(42, 47))
def test_duration90_binds_assumed_clock_without_changing_timers(airbase_ready, seed):
    r = airbase_ready
    use_duration90(r)
    r.config["simulation"]["seed"] = seed
    r.config["phase15"]["scenario_id"] = o.PHASE16_AIRBASE_SCENARIO_ID.replace("-s42-", f"-s{seed}-") + "-h90-dt1m"
    rebind(r)
    assert r.active["simulation"]["max_timesteps"] == 90
    assert r.active["placeholder_transition"]["T_3_to_4"] == r.active["placeholder_transition"]["T_4_to_5"] == 1
    assert r.binding["evidence_scope"]["duration_basis"]["assumed_minutes_per_update"] == 1.0
    assert r.binding["evidence_scope"]["duration_basis"]["mapping_status"].startswith("ASSUMPTION")
    run(r)
    assert record(r)["output_valid"] is True


@pytest.mark.parametrize("section,key,value", [
    ("simulation", "max_timesteps", 89), ("simulation", "max_timesteps", 300),
    ("placeholder_transition", "T_4_to_5", 90),
    ("phase15", "scenario_id", o.PHASE16_AIRBASE_SCENARIO_ID),
    ("duration_basis", "assumed_minutes_per_update", 0.5),
    ("duration_basis", "assumed_minutes_per_update", True),
    ("duration_basis", "reported_duration_minutes", 90.0),
    ("duration_basis", "stopping_interpretation", "simulated fire-out"),
])
def test_duration90_rejects_undeclared_mapping_or_horizon(airbase_ready, section, key, value):
    use_duration90(airbase_ready)
    airbase_ready.config[section][key] = value
    with pytest.raises((ValueError, TypeError)):
        o._phase15_normalize(airbase_ready.config)


def test_original_airbase_profile_remains_max300_without_duration_metadata(airbase_ready):
    r = airbase_ready
    r.config["simulation"]["max_timesteps"] = 90
    with pytest.raises(ValueError):
        o._phase15_normalize(r.config)
    r.config["simulation"]["max_timesteps"] = 300
    r.config["duration_basis"] = deepcopy(o.PHASE16_AIRBASE_DURATION90_BASIS)
    with pytest.raises(ValueError):
        o._phase15_normalize(r.config)


def test_duration90_active_outcome_is_censored_not_declared_fireout(airbase_ready, monkeypatch):
    r = airbase_ready
    use_duration90(r)
    rebind(r)
    monkeypatch.setattr(o.FireAutomata, "step", lambda self: None)
    run(r)
    end = record(r)
    assert end["last_completed_timestep"] == 90
    assert end["termination_reason"] == "timestep_limit"
    assert end["status"] == "censored" and end["active_cell_count"] == 1


def use_airbase_lr_gate(r, monkeypatch):
    config = use_lr_comparison(r, monkeypatch)
    config["phase15"].update(schema_version=o.PHASE16_AIRBASE_LR_GATE_SCHEMA,
        scenario_id=o.PHASE16_AIRBASE_SCENARIO_ID.replace("-rf-", "-lr-") + "-gate040")
    config["wind_input_basis"] = deepcopy(o.PHASE16_AIRBASE_WIND_INPUT_BASIS)
    config["wind"].update(speed_kmh=10.6, direction_deg=o.PHASE16_AIRBASE_GRID_FROM_DEG)
    config["ml_model"].update(inference_mode="stochastic_probability_gate", proba_threshold=0.40)
    return config


@pytest.mark.parametrize("seed", range(42, 47))
def test_airbase_lr_gate_five_seeds_keep_cap300_and_no_duration_rule(airbase_ready, monkeypatch, seed):
    r = airbase_ready
    config = use_airbase_lr_gate(r, monkeypatch)
    config["simulation"]["seed"] = seed
    config["phase15"]["scenario_id"] = config["phase15"]["scenario_id"].replace("-s42-", f"-s{seed}-")
    rebind(r)
    assert "duration_basis" not in r.active and r.active["simulation"]["max_timesteps"] == 300
    assert "manifest" not in r.binding["model"]
    scope = r.binding["evidence_scope"]
    assert scope["incident"] == "airbase_sanggi_20250124"
    assert scope["probability_gate"]["cutoff"] == 0.40
    assert scope["reference_accessed_by_simulation"] is False and scope["final_test_eligible"] is False
    run(r)
    assert record(r)["output_valid"] is True
    assert "manifest" not in record(r)["observed_bindings"]["model"]


@pytest.mark.parametrize("section,key,value", [
    ("simulation", "seed", 47), ("simulation", "max_timesteps", 20),
    ("simulation", "max_timesteps", 90), ("simulation", "max_timesteps", 100),
    ("ml_model", "proba_threshold", 0.30), ("ml_model", "proba_threshold", True),
    ("ml_model", "inference_threshold", 0.40), ("ml_model", "inference_mode", "stochastic_probability"),
    ("ml_model", "model_sha256", o.PHASE15_CANDIDATE_MODEL_SHA256),
    ("ml_model", "model_path", "substitute.joblib"),
    ("ml_model", "calibration_manifest_path", "calibration_manifest.json"),
    ("ml_model", "calibration_manifest_sha256", "0" * 64),
    ("phase15", "schema_version", o.PHASE16_AIRBASE_SCHEMA),
    ("phase15", "schema_version", o.PHASE16_LR_GATE_SCHEMA),
    ("phase15", "scenario_id", o.PHASE16_AIRBASE_SCENARIO_ID),
    ("wind", "direction_deg", o.PHASE16_GRID_FROM_DEG),
    ("placeholder_transition", "T_4_to_5", 90),
    ("candidate_inputs", "window", o.PHASE15_CANDIDATE_WINDOW),
])
def test_airbase_lr_gate_rejects_changed_release_gate_incident_or_horizon(airbase_ready, monkeypatch, section, key, value):
    config = use_airbase_lr_gate(airbase_ready, monkeypatch)
    config[section][key] = value
    with pytest.raises((ValueError, TypeError)):
        o._phase15_normalize(config)


def test_airbase_lr_gate_rejects_duration_metadata(airbase_ready, monkeypatch):
    config = use_airbase_lr_gate(airbase_ready, monkeypatch)
    config["duration_basis"] = deepcopy(o.PHASE16_AIRBASE_DURATION90_BASIS)
    with pytest.raises(ValueError):
        o._phase15_normalize(config)
