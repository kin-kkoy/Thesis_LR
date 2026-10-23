"""Bounded synthetic regressions; the 1024-run campaign is never launched."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import types

import numpy as np
import pytest

import modules.automata_engine as engine_module
import modules.additive_exposure_assessment as assessment
from modules.additive_exposure_teacher import TEACHER_ID
from modules.transition_observer import canonical_metadata_hash, mask_sha256

PREIMAGE_SHA256 = "9a092f9566fbf57eacd92c65839767a10024a196117bf3770efa24c9260e8fdb"


def installed_preimage_module():
    preimage = installed_preimage_bytes()
    module = types.ModuleType("modules._installed_r2_preimage")
    module.__package__ = "modules"
    module.__file__ = engine_module.__file__
    # This source is the installed, uncommitted pre-edit engine, not Git HEAD.
    exec(compile(preimage, "<installed-pre-edit-engine>", "exec"), module.__dict__)
    return module


class RecordingRng:
    def __init__(self, generator):
        self.generator, self.calls = generator, []

    @property
    def bit_generator(self):
        return self.generator.bit_generator

    def random(self, *args, **kwargs):
        self.calls.append((args, dict(kwargs)))
        return self.generator.random(*args, **kwargs)


class RecordingNumpy:
    def __init__(self):
        self.probabilities = []

    def __getattr__(self, name):
        return getattr(np, name)

    def clip(self, *args, **kwargs):
        result = np.clip(*args, **kwargs)
        if isinstance(result, np.ndarray) and result.ndim == 1:
            self.probabilities.append(result.copy())
        return result


def provenance(config, environment):
    domain = environment["burnable_mask"] & ~environment["nodata_mask"]
    return {
        **{key: "synthetic-r2-baseline-regression" for key in (
            "set_id", "experiment_id", "event_id", "scenario_id", "scenario_family_id",
            "run_id", "grid_id", "simulation_valid_mask_id", "mapped_building_mask_id",
            "simulator_id", "simulator_version", "source_revision", "environment_id",
            "execution_id", "capture_source_id")},
        "source_dirty": True, "configuration_hash": canonical_metadata_hash(config),
        "input_hashes": {"synthetic": "a" * 64},
        "authorization": "d013_source_only_in_memory",
        "authorization_reference": "synthetic source regression only",
        "transition_source_type": "authorized_simulated", "seed": config["simulation"]["seed"],
        "rng_bit_generator": "PCG64", "wind_manifest": config["wind"],
        "wind_weight": config["placeholder_transition"]["wind_weight"],
        "simulation_valid_mask_sha256": mask_sha256(domain),
        "mapped_building_mask_sha256": mask_sha256(environment["building_presence"] > 0),
    }


@pytest.mark.parametrize("teacher_id", [assessment.BASELINE_ID, TEACHER_ID])
def test_constructor_order_and_original_generator_forwarding(monkeypatch, teacher_id):
    environment, ignition = assessment.dynamic_fixture("interior", 1)
    created = []
    original = np.random.default_rng
    def track(seed):
        generator = original(seed)
        created.append(generator)
        return generator
    monkeypatch.setattr(np.random, "default_rng", track)
    engine = assessment._AssessmentAutomata(
        environment, assessment.engine_configuration(0, (0, 0)), teacher_id=teacher_id)
    assert len(created) == 1 and engine.rng.generator is created[0]
    assert not engine.rng.calls
    assert engine.p_base.dtype == np.float32
    assert engine.teacher_id == teacher_id
    assert engine.feature_assembler is not None
    engine.set_ignition([ignition])
    assert engine.grid[ignition] == 4
    assert engine.ignition_timers.dtype == engine.blazing_timers.dtype == np.int16
    assert not np.any(engine.ignition_timers) and not np.any(engine.blazing_timers)
    assert (engine.susceptibility_trace is not None) == (teacher_id == TEACHER_ID)


def test_unknown_selection_rejects_before_base_constructor(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Base constructor reached")
    monkeypatch.setattr(engine_module.FireAutomata, "__init__", forbidden)
    with pytest.raises(ValueError, match="Unknown teacher"):
        assessment._AssessmentAutomata({}, {}, teacher_id="unknown")


@pytest.mark.parametrize("model_free,observed", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("seed", [0, 17])
def test_installed_preimage_probability_state_timer_rng_and_observer_regression(
        monkeypatch, model_free, observed, seed):
    old = installed_preimage_module()
    old_numpy, new_numpy = RecordingNumpy(), RecordingNumpy()
    old.np = old_numpy
    monkeypatch.setattr(engine_module, "np", new_numpy)
    environment, ignition = assessment.dynamic_fixture("boundary", 1)
    config = assessment.engine_configuration(seed, (10, 315))
    old_observations, new_observations = [], []
    engines = []
    for cls, observations in ((old.FireAutomata, old_observations),
                              (engine_module.FireAutomata, new_observations)):
        kwargs = {"model_free": model_free}
        if observed:
            kwargs.update(transition_observer=observations.append,
                          transition_provenance=provenance(config, environment))
        engine = cls(deepcopy(environment), deepcopy(config), **kwargs)
        engine.set_ignition([ignition])
        engine.rng = RecordingRng(engine.rng)
        engines.append(engine)
    assert np.array_equal(engines[0].p_base.view(np.uint32), engines[1].p_base.view(np.uint32))
    for _ in range(8):
        for engine in engines:
            engine.step()
        left, right = engines
        for field in ("grid", "ignition_timers", "blazing_timers"):
            assert np.array_equal(getattr(left, field), getattr(right, field))
        assert left.timestep == right.timestep
        assert left.rng.bit_generator.state == right.rng.bit_generator.state
        assert left.rng.calls == right.rng.calls
    assert len(old_numpy.probabilities) == len(new_numpy.probabilities)
    for left, right in zip(old_numpy.probabilities, new_numpy.probabilities):
        assert left.dtype == right.dtype == np.float32
        assert np.array_equal(left.view(np.uint32), right.view(np.uint32))
    if observed:
        assert len(old_observations) == len(new_observations) == 8
        for left, right in zip(old_observations, new_observations):
            assert np.array_equal(left.state_t, right.state_t)
            assert np.array_equal(left.state_t1, right.state_t1)
            assert np.array_equal(left.eligible_mask_t, right.eligible_mask_t)
            assert left.timestep_t == right.timestep_t


@pytest.mark.parametrize("observed,model_free", [(False, False), (True, False), (False, True)])
def test_installed_preimage_failure_checkpoint_policy(monkeypatch, observed, model_free):
    old = installed_preimage_module()
    environment, ignition = assessment.dynamic_fixture("interior", 1)
    config = assessment.engine_configuration(0, (0, 0))
    config["simulation"].pop("emergency_checkpoint_enabled")
    def fail(*_args, **_kwargs):
        raise ValueError("synthetic neighborhood failure")
    monkeypatch.setattr(old, "compute_moore_neighbor_count", fail)
    monkeypatch.setattr(engine_module, "compute_moore_neighbor_count", fail)
    outcomes = []
    for cls in (old.FireAutomata, engine_module.FireAutomata):
        kwargs = {"model_free": model_free}
        if observed:
            kwargs.update(transition_observer=lambda row: None,
                          transition_provenance=provenance(config, environment))
        engine = cls(deepcopy(environment), deepcopy(config), **kwargs)
        engine.set_ignition([ignition])
        calls = []
        engine._save_emergency_checkpoint = lambda: calls.append("attempt") or "synthetic-no-write"
        with pytest.raises(RuntimeError) as caught:
            engine.step()
        outcomes.append((str(caught.value), calls, engine.grid.copy(),
                         engine.ignition_timers.copy(), engine.blazing_timers.copy()))
    assert outcomes[0][:2] == outcomes[1][:2]
    for left, right in zip(outcomes[0][2:], outcomes[1][2:]):
        assert np.array_equal(left, right)
    assert outcomes[0][1] == ([] if observed or model_free else ["attempt"])


def test_installed_preimage_observer_failure_after_state_commit():
    old = installed_preimage_module()
    environment, ignition = assessment.dynamic_fixture("interior", 1)
    config = assessment.engine_configuration(0, (0, 0))
    engines = []
    def fail(row):
        raise LookupError("synthetic observer failure")
    for cls in (old.FireAutomata, engine_module.FireAutomata):
        engine = cls(deepcopy(environment), deepcopy(config), model_free=True,
                     transition_observer=fail, transition_provenance=provenance(config, environment))
        engine.set_ignition([ignition])
        def forbidden():
            raise AssertionError("Checkpoint reached")
        engine._save_emergency_checkpoint = forbidden
        with pytest.raises(LookupError):
            engine.step()
        engines.append(engine)
    assert engines[0].timestep == engines[1].timestep == 1
    assert np.array_equal(engines[0].grid, engines[1].grid)
    assert np.array_equal(engines[0].blazing_timers, engines[1].blazing_timers)
    assert engines[0].rng.bit_generator.state == engines[1].rng.bit_generator.state


def make_engine(teacher_id=TEACHER_ID):
    environment, ignition = assessment.dynamic_fixture("interior", 1)
    engine = assessment._AssessmentAutomata(
        environment, assessment.engine_configuration(0, (0, 0)), teacher_id=teacher_id)
    engine.set_ignition([ignition])
    return engine


def test_transition_capture_is_actual_aligned_and_cleared(monkeypatch):
    engine = make_engine()
    initial = engine.initialization_record()
    for field in ("eligible_opportunities", "probability_sum", "blazing_t",
                  "probability_sum_per_blazing", "new_ignitions", "growth_increment", "rng_draw_calls"):
        assert initial[field] is None
    expected = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80], dtype=np.float32)
    seen = []
    def compute(*, beta32, wind_weighted_score32):
        seen.append(wind_weighted_score32.copy())
        return types.SimpleNamespace(q32=expected)
    monkeypatch.setattr(assessment, "compute_additive_exposure_probability", compute)
    record = engine.step()
    indices, used = engine._transition_capture
    assert np.array_equal(indices, [30, 31, 32, 39, 41, 48, 49, 50])
    assert np.array_equal(used.view(np.uint32), expected.view(np.uint32))
    assert len(seen) == 1 and record["eligible_opportunities"] == 8
    assert record["probability_sum"] == math.fsum(float(p) for p in expected)
    assert record["blazing_t"] == 1 and record["state_counts"]["extinguished"] == 1
    assert record["from_timestep"] == 0 and record["to_timestep"] == 1
    assert engine.rng.calls == [(((9, 9),), {})]
    engine.grid[engine.burnable_mask] = 5
    record = engine.step()
    assert engine._transition_capture is None
    assert record["eligible_opportunities"] == record["probability_sum"] == record["rng_draw_calls"] == 0
    assert record["probability_sum_per_blazing"] is None


def test_assessment_rejects_model_and_checkpoint_boundaries(monkeypatch):
    engine = make_engine()
    calls = []
    monkeypatch.setattr(engine_module.joblib, "load", lambda path: calls.append(path))
    with pytest.raises(RuntimeError, match="cannot attach"):
        engine.model = object()
    with pytest.raises(RuntimeError, match="cannot load an estimator"):
        engine.load_model("synthetic-forbidden-model")
    with pytest.raises(RuntimeError, match="cannot write checkpoints"):
        engine.save_checkpoint("synthetic-forbidden-checkpoint")
    with pytest.raises(RuntimeError, match="cannot load checkpoints"):
        engine.load_checkpoint("synthetic-forbidden-checkpoint")
    assert calls == []


def test_zero_probability_still_draws_and_strict_equality_does_not_ignite(monkeypatch):
    engine = make_engine()
    monkeypatch.setattr(assessment, "compute_additive_exposure_probability",
                        lambda **kw: types.SimpleNamespace(q32=np.zeros(8, dtype=np.float32)))
    class ZeroRng:
        def random(self, *args, **kwargs):
            return np.zeros(args[0], dtype=np.float64)
    engine.rng = assessment._ForwardingRng(ZeroRng())
    record = engine.step()
    assert record["eligible_opportunities"] == 8 and record["rng_draw_calls"] == 1
    assert record["new_ignitions"] == 0


def test_failed_step_mutation_does_not_replace_completed_record(monkeypatch):
    engine = make_engine()
    initial = deepcopy(engine.initialization_record())
    def fail(*_args):
        engine.grid[:] = 5
        engine.ignition_timers[:] = 11
        raise RuntimeError("injected after mutation")
    monkeypatch.setattr(engine, "_model_free_probability", fail)
    with pytest.raises(RuntimeError):
        engine.step()
    assert engine.last_completed == initial
    assert engine.failed_transition["transition_completed"] is False
    assert engine.failed_transition["to_timestep"] is None
    assert engine._transition_capture is None


def active_config():
    config = assessment.disabled_configuration()
    config.update(enabled=True, authorization_reference="synthetic-test-only")
    config["resources"] = {"max_process_rss_bytes": 100, "min_available_physical_memory_bytes": 10,
                           "max_wall_seconds": 100, "observation_interval_seconds": 1}
    return config


def test_authorization_pending_resources_and_no_campaign_on_import(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Runtime/campaign reached")
    monkeypatch.setattr(assessment, "_binding", forbidden)
    monkeypatch.setattr(assessment, "planned_runs", forbidden)
    with pytest.raises(PermissionError):
        assessment.run_assessment(assessment.disabled_configuration())
    config = active_config()
    config["resources"]["max_wall_seconds"] = None
    with pytest.raises(PermissionError):
        assessment.run_assessment(config, execution_authorized=True,
                                  approval_reference="synthetic-test-only")


def test_plan_order_and_denominators_without_running_campaign():
    specs = assessment.planned_runs()
    assert len(specs) == 1024
    assert specs[0] == {"layout": "interior", "material": 1, "wind": [0.0, 0.0], "seed": 0, "teacher": "baseline"}
    assert specs[1]["teacher"] == "candidate" and specs[2]["seed"] == 1
    runs = [{"spec": spec, "status": "not_attempted"} for spec in specs]
    runs[0].update(status="complete", last_completed={"footprint": 1, "inactive_time": 1})
    runs[1]["status"] = "failed"
    result = assessment.summarize(runs)
    assert (result["complete"], result["failed"], result["not_attempted"]) == (1, 1, 1022)
    group = result["groups"][0]
    assert group["seed_only_extinction_frequency"] == 1
    assert group["statistics"]["footprint"]["nearest_rank_p05_p50_p95"] == [1, 1, 1]


def test_cap_failure_and_reporting_failure_stop_without_retry(monkeypatch):
    monkeypatch.setattr(assessment._Resources, "check", lambda *_: None)
    monkeypatch.setattr(assessment, "_runtime_check", lambda: None)
    monkeypatch.setattr(assessment, "_binding", lambda config: {"synthetic": True})
    attempted = []
    def fail(spec, emit, check):
        attempted.append(spec)
        error = assessment.CapInvariantViolation("cap_invariant_violation")
        error.assessment_summary = {"last_completed": {"footprint": 1, "state_timestep": 161}}
        raise error
    monkeypatch.setattr(assessment, "_run_fixture", fail)
    def bad_console(row):
        raise OSError("console unavailable")
    with pytest.raises(assessment.AssessmentFailure) as caught:
        assessment.run_assessment(active_config(), execution_authorized=True,
                                  approval_reference="synthetic-test-only", emit=bad_console)
    report = caught.value.report
    assert len(attempted) == 1
    assert report["original_failure"]["type"] == "CapInvariantViolation"
    assert report["reporting_failures"][0]["type"] == "OSError"
    assert report["summary"]["failed"] == 1 and report["summary"]["not_attempted"] == 1023


def test_step_failure_console_preserves_wrapper_cause_and_completed_record(monkeypatch, capsys):
    monkeypatch.setattr(assessment._Resources, "check", lambda *_: None)
    monkeypatch.setattr(assessment, "_runtime_check", lambda: None)
    monkeypatch.setattr(assessment, "_binding", lambda config: {"synthetic": True})
    spec = {"layout": "interior", "material": 1, "wind": [0.0, 0.0],
            "seed": 0, "teacher": "candidate"}
    monkeypatch.setattr(assessment, "planned_runs", lambda: [spec, {**spec, "seed": 1}])
    monkeypatch.setattr(
        assessment, "compute_additive_exposure_probability",
        lambda **kw: types.SimpleNamespace(q32=np.ones(kw["beta32"].shape, dtype=np.float32)))
    original_count = engine_module.compute_moore_neighbor_count
    calls = []
    def fail_second_step(state):
        calls.append(None)
        if len(calls) == 2:
            raise ValueError("injected underlying step failure")
        return original_count(state)
    monkeypatch.setattr(engine_module, "compute_moore_neighbor_count", fail_second_step)
    with pytest.raises(assessment.AssessmentFailure) as caught:
        assessment.run_assessment(active_config(), execution_authorized=True,
                                  approval_reference="synthetic-test-only", emit=assessment._console)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    retained = next(row for row in records if row["record_type"] == "assessment_failure")
    failure = retained["original_failure"]
    assert failure["type"] == "RuntimeError"
    assert failure["message"] == (
        "In-memory simulation step failed; no emergency checkpoint was "
        "written. Explicit caller action is required before continuing.")
    assert failure["causal_chain"] == [
        {"relation": "cause", "type": "ValueError", "message": "injected underlying step failure"}]
    assert failure["causal_chain_cycle_detected"] is False
    assert failure == caught.value.report["original_failure"]
    last = retained["runs"][0]["last_completed"]
    assert last["transition_completed"] is True and last["to_timestep"] == 1
    assert retained["runs"][0]["incomplete_transition"] == {
        "record_type": "transition", "transition_completed": False,
        "from_timestep": 1, "to_timestep": None}
    assert retained["runs"][1]["status"] == "not_attempted"
    assert len(calls) == 2


def test_failure_chain_context_suppression_cause_priority_and_cycle():
    outer = RuntimeError("outer wrapper")
    context = LookupError("implicit original")
    outer.__context__ = context
    assert assessment._failure_evidence(outer)["causal_chain"] == [
        {"relation": "context", "type": "LookupError", "message": "implicit original"}]
    outer.__suppress_context__ = True
    assert assessment._failure_evidence(outer)["causal_chain"] == []
    cause = ValueError("explicit original")
    outer.__cause__ = cause
    cause.__cause__ = outer
    evidence = assessment._failure_evidence(outer)
    assert evidence["type"] == "RuntimeError" and evidence["message"] == "outer wrapper"
    assert evidence["causal_chain"] == [
        {"relation": "cause", "type": "ValueError", "message": "explicit original"}]
    assert evidence["causal_chain_cycle_detected"] is True


def test_resource_measurement_failure_is_closed(monkeypatch):
    monkeypatch.setattr(assessment, "capture_resource_snapshot",
                        lambda path: types.SimpleNamespace(process_rss_bytes=None,
                            available_memory_bytes=100, free_disk_bytes=100, captured_at_utc="synthetic"))
    resources = assessment._Resources(active_config()["resources"], assessment.time.monotonic())
    with pytest.raises(RuntimeError, match="observation"):
        resources.check("synthetic")



def test_assessment_baseline_matches_installed_preimage(monkeypatch):
    old = installed_preimage_module()
    old_numpy = RecordingNumpy()
    old.np = old_numpy
    environment, ignition = assessment.dynamic_fixture("interior", 5)
    config = assessment.engine_configuration(17, (10, 315))
    baseline = old.FireAutomata(deepcopy(environment), deepcopy(config), model_free=True)
    baseline.rng = RecordingRng(baseline.rng)
    candidate_adapter = assessment._AssessmentAutomata(
        deepcopy(environment), deepcopy(config), teacher_id=assessment.BASELINE_ID)
    for engine in (baseline, candidate_adapter):
        engine.set_ignition([ignition])
    candidate_adapter.initialization_record()
    for _ in range(8):
        old_count = len(old_numpy.probabilities)
        baseline.step()
        candidate_adapter.step()
        capture = candidate_adapter._transition_capture
        if len(old_numpy.probabilities) == old_count:
            assert capture is None
        else:
            assert np.array_equal(old_numpy.probabilities[-1].view(np.uint32),
                                  capture[1].view(np.uint32))
        for field in ("grid", "ignition_timers", "blazing_timers"):
            assert np.array_equal(getattr(baseline, field), getattr(candidate_adapter, field))
        assert baseline.rng.calls == candidate_adapter.rng.calls
        assert baseline.rng.bit_generator.state == candidate_adapter.rng.bit_generator.state


def test_probability_hook_uses_supplied_score_without_wind_or_randomness(monkeypatch):
    engine = make_engine()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Wind or randomness recomputed by probability hook")
    monkeypatch.setattr(engine_module, "compute_wind_weighted_score", forbidden)
    monkeypatch.setattr(engine.rng, "random", forbidden)
    score = np.full((9, 9), np.float32(0.125))
    indices = np.array([1, 7, 11], dtype=np.intp)
    probabilities = engine._model_free_probability(indices, score)
    assert probabilities.shape == (3,) and probabilities.dtype == np.float32
    assert np.array_equal(engine._transition_capture[0], indices)
    with pytest.raises(ValueError, match="one-dimensional"):
        engine._model_free_probability(indices.reshape(1, 3), score)


@pytest.mark.parametrize("inactive_at_cap", [False, True])
def test_actual_run_loop_checks_inactivity_before_cap_failure(monkeypatch, inactive_at_cap):
    class FakeEngine:
        def __init__(self, *_args, **_kwargs):
            self.timestep = 0
            self.failed_transition = None
        def set_ignition(self, points):
            pass
        def initialization_record(self):
            return {"state_timestep": 0}
        def step(self):
            self.timestep += 1
            return {"state_timestep": self.timestep, "transition_completed": True}
        def is_active(self):
            return not (inactive_at_cap and self.timestep == 162)
    monkeypatch.setattr(assessment, "_AssessmentAutomata", FakeEngine)
    spec = assessment.planned_runs()[0]
    records = []
    if inactive_at_cap:
        result = assessment._run_fixture(spec, records.append, lambda boundary: None)
        assert result["last_completed"]["state_timestep"] == 162
    else:
        with pytest.raises(assessment.CapInvariantViolation) as caught:
            assessment._run_fixture(spec, records.append, lambda boundary: None)
        assert caught.value.assessment_summary["last_completed"]["state_timestep"] == 162
    assert len(records) == 163


def test_completed_metadata_survives_after_step_resource_failure(monkeypatch):
    engine = make_engine()
    monkeypatch.setattr(assessment, "_AssessmentAutomata", lambda *a, **kw: engine)
    def check(boundary):
        if boundary == "after_step":
            raise RuntimeError("resource_limit")
    with pytest.raises(RuntimeError) as caught:
        assessment._run_fixture(assessment.planned_runs()[0], lambda record: None, check)
    last = caught.value.assessment_summary["last_completed"]
    assert last["transition_completed"] is True and last["to_timestep"] == 1


def test_summary_nearest_rank_and_no_failed_zero_observations():
    spec = assessment.planned_runs()[0]
    runs = [{"spec": spec, "status": "complete",
             "last_completed": {"footprint": value, "inactive_time": value + 1}}
            for value in range(1, 33)]
    runs.append({"spec": spec, "status": "failed"})
    group = assessment.summarize(runs)["groups"][0]
    assert group["complete"] == 32 and group["failed"] == 1
    assert group["statistics"]["footprint"]["nearest_rank_p05_p50_p95"] == [2, 16, 31]
    assert group["statistics"]["footprint"]["mean"] == 16.5

# Fixed inverse edit plus the before-edit digest losslessly preserve all installed bytes.
_ORIGINAL_EXPRESSION = (
    b"\t\t\t\t\tp_effective = np.clip(\n"
    b"\t\t\t\t\t\tself.p_base.ravel()[susceptible_indices]\n"
    b"\t\t\t\t\t\t* wind_weighted_score.ravel()[susceptible_indices],\n"
    b"\t\t\t\t\t\t0.0,\n"
    b"\t\t\t\t\t\t1.0,\n"
    b"\t\t\t\t\t)"
)
_EXTRACTED_CALL = (
    b"\t\t\t\t\tp_effective = self._model_free_probability(\n"
    b"\t\t\t\t\t\tsusceptible_indices, wind_weighted_score\n"
    b"\t\t\t\t\t)"
)
_INSERTED_HOOK = (
    b"\tdef _model_free_probability(self, susceptible_indices, wind_weighted_score) -> np.ndarray:\n"
    b"\t\t\"\"\"Return probabilities aligned with susceptible_indices; preserve legacy arithmetic.\"\"\"\n"
    b"\t\treturn np.clip(\n"
    b"\t\t\tself.p_base.ravel()[susceptible_indices]\n"
    b"\t\t\t* wind_weighted_score.ravel()[susceptible_indices],\n"
    b"\t\t\t0.0,\n"
    b"\t\t\t1.0,\n"
    b"\t\t)\n"
    b"\n"
)


def installed_preimage_bytes():
    current = Path(engine_module.__file__).read_bytes()
    assert current.count(_EXTRACTED_CALL) == current.count(_INSERTED_HOOK) == 1
    preimage = current.replace(_EXTRACTED_CALL, _ORIGINAL_EXPRESSION, 1).replace(_INSERTED_HOOK, b"", 1)
    assert len(preimage) == 18377
    assert hashlib.sha256(preimage).hexdigest() == PREIMAGE_SHA256
    return preimage
