"""In-memory R2 assessment API. No CLI, file publisher, model or real-input loader."""
from copy import deepcopy
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .automata_engine import FireAutomata
from .ca_state import validate_full_state_grid
from .data_loader import EnvironmentManager
from .wind_convention import WindContract
from .set_c_pilot import capture_resource_snapshot
from .additive_exposure_teacher import (
    COEFFICIENTS, NUMERICAL_ID, TEACHER_ID,
    compute_susceptibility32, compute_additive_exposure_probability,
)

BASELINE_ID = "model-free-fire-automata"
SCHEMA = "additive_exposure_assessment.r2"
WINDS = ((0.0, 0.0), (10.0, 0.0), (10.0, 90.0),
         (10.0, 180.0), (10.0, 270.0), (10.0, 315.0))
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1))
LAYOUTS = ("interior", "boundary", "corridor", "separated_clusters")


def disabled_configuration():
    return {
        "schema_version": SCHEMA, "enabled": False, "authorization_reference": None,
        "candidate_teacher_id": TEACHER_ID, "numerical_contract_id": NUMERICAL_ID,
        "baseline_teacher_id": BASELINE_ID, "susceptibility": dict(COEFFICIENTS),
        "wind_weight": 0.20, "transition_timers": {"T_3_to_4": 1, "T_4_to_5": 1},
        "fixture_specification": "additive_exposure_r2_fixed_fixtures",
        "deterministic_winds": [list(w) for w in WINDS],
        "tolerances": {"interior_q64_max_encoding_distance": 8,
                       "oracle_q32_max_encoding_distance": 1,
                       "actual_q64_to_q32_rounding": "bit_exact",
                       "zero_encoding": "positive_zero"},
        "assessment": {"layout_order": list(LAYOUTS), "material_order": [1, 5],
                       "wind_order": [[0.0, 0.0], [10.0, 315.0]],
                       "seed_first": 0, "seed_last": 31,
                       "teacher_order": ["baseline", "candidate"], "planned_runs": 1024,
                       "maximum_completed_updates": 162, "rng_policy": "default_rng_per_run",
                       "required_bit_generator": "PCG64", "concurrent_runs": 1,
                       "automatic_retry": False},
        "resources": {"max_process_rss_bytes": None,
                      "min_available_physical_memory_bytes": None,
                      "max_wall_seconds": None, "observation_interval_seconds": None},
        "output": {"console_only": True, "persistent_output_enabled": False},
    }


def validate_configuration(config, *, active=False):
    expected = disabled_configuration()
    compared = deepcopy(config)
    if active:
        if compared.get("enabled") is not True:
            raise PermissionError("Assessment is disabled")
        reference = compared.get("authorization_reference")
        if not isinstance(reference, str) or not reference.strip():
            raise PermissionError("Execution approval reference is required")
        limits = compared.get("resources", {})
        for key in expected["resources"]:
            value = limits.get(key)
            numeric = type(value) is int if key.endswith("_bytes") else type(value) in (int, float)
            if not numeric or not math.isfinite(value) or value <= 0:
                raise PermissionError("Resource limits/cadence remain pending or invalid")
        if set(limits) != set(expected["resources"]):
            raise ValueError("Unknown resource controls")
        compared["enabled"] = False
        compared["authorization_reference"] = None
        compared["resources"] = expected["resources"]
    if json.dumps(compared, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("Configuration differs from adopted R2")


def engine_configuration(seed, wind):
    if type(seed) is not int or seed not in range(32) or tuple(wind) not in WINDS:
        raise ValueError("Seed or wind outside the fixed design")
    speed, bearing = wind
    return {
        "simulation": {"seed": seed, "emergency_checkpoint_enabled": False},
        "wind": WindContract(speed, bearing).to_manifest(),
        "placeholder_transition": {**COEFFICIENTS, "wind_weight": 0.20,
                                   "T_3_to_4": 1, "T_4_to_5": 1},
        "ml_model": {"enabled": False, "inference_mode": "stochastic_probability",
                     "threshold": None, "inference_threshold": None, "positive_label": 1},
        "flammability_weights": {"base_weight": 0.5, "slope_weight": 0.3,
                                 "proximity_weight": 0.2},
    }


def prepare_environment(raw):
    """Normal loader construction, four independent writable buffers, no IO."""
    if set(raw) != {"slope", "proximity", "buildings", "materials"}:
        raise ValueError("Exactly four raw fixture arrays are required")
    shape = raw["slope"].shape
    if shape not in ((3, 3), (9, 9)):
        raise ValueError("Only the prescribed fixture sizes are supported")
    if any(not isinstance(a, np.ndarray) or a.dtype != np.float32 or
           a.shape != shape or not np.all(np.isfinite(a)) for a in raw.values()):
        raise ValueError("Fixture arrays must be aligned finite float32")
    buildings, materials = raw["buildings"], raw["materials"]
    if not np.all(np.isin(buildings, (0, 10))) or not np.all(np.isin(materials, range(6))):
        raise ValueError("Invalid categorical fixture encoding")
    if not np.array_equal(buildings == 10, materials > 0):
        raise ValueError("Building/material inconsistency")
    manager = EnvironmentManager({
        "raster_dir": ".", "slope_file": "synthetic_slope",
        "proximity_file": "synthetic_proximity", "buildings_file": "synthetic_buildings",
        "materials_file": "synthetic_materials", "nodata_value": -9999,
    })
    manager.grid_shape = shape
    manager.crs = "EPSG:32651"
    manager.transform = (3.0, 0.0, 0.0, 0.0, -3.0, 3.0 * shape[0])
    for name, array in raw.items():
        setattr(manager, name + "_raw", array.copy())
    manager.build_masks()
    manager.normalize_layers()
    return manager.get_environment()


def uniform_raw(size, material=1, slope=0, proximity=0):
    return {name: np.full((size, size), value, dtype=np.float32)
            for name, value in (("slope", slope), ("proximity", proximity),
                                ("buildings", 10), ("materials", material))}


def dynamic_fixture(layout, material):
    if layout not in LAYOUTS or material not in (1, 5):
        raise ValueError("Unknown dynamic fixture")
    rows, cols = np.indices((9, 9))
    mapped = np.ones((9, 9), dtype=bool)
    ignition = (4, 4)
    if layout == "boundary":
        ignition = (0, 4)
    elif layout == "corridor":
        mapped = rows == 4
    elif layout == "separated_clusters":
        mapped = np.isin(rows, (3, 4, 5)) & np.isin(cols, (0, 1, 2, 6, 7, 8))
        ignition = (4, 1)
    raw = uniform_raw(9, material)
    raw["buildings"][~mapped] = 0
    raw["materials"][~mapped] = 0
    return prepare_environment(raw), ignition


def pattern_offsets(target):
    row, col = target
    if row not in range(3) or col not in range(3):
        raise ValueError("Pattern target outside 3x3 fixture")
    return tuple((dr, dc) for dr, dc in OFFSETS
                 if 0 <= row + dr < 3 and 0 <= col + dc < 3)


def _validate_environment(environment):
    shape = environment["grid_shape"]
    if shape not in ((3, 3), (9, 9)):
        raise ValueError("Assessment geometry outside fixed fixtures")
    for key in ("slope_risk", "proximity_risk"):
        a = environment[key]
        if a.shape != shape or a.dtype != np.float32 or not np.all(np.isfinite(a)):
            raise ValueError("Canonical predictor dtype/shape/finite failure")
    for key in ("burnable_mask", "nodata_mask"):
        if environment[key].shape != shape or environment[key].dtype != bool:
            raise ValueError("Masks must be aligned Boolean arrays")
    building, material = environment["building_presence"], environment["material_class"]
    if building.shape != shape or material.shape != shape:
        raise ValueError("Categorical shape mismatch")
    if material.dtype != np.int8 or not np.all(np.isin(material, range(6))):
        raise ValueError("Invalid canonical material")
    if not np.all(np.isin(building, (0, 1))) or not np.array_equal(building > 0, material > 0):
        raise ValueError("Invalid canonical building/material correspondence")


class _ForwardingRng:
    def __init__(self, generator):
        self.generator = generator
        self.calls = []

    @property
    def bit_generator(self):
        return self.generator.bit_generator

    def random(self, *args, **kwargs):
        self.calls.append((args, dict(kwargs)))
        return self.generator.random(*args, **kwargs)


class _AssessmentAutomata(FireAutomata):
    def __init__(self, environment, config, *, teacher_id):
        # Base construction dispatches _precompute_base_probability before assembler creation.
        if teacher_id not in (BASELINE_ID, TEACHER_ID):
            raise ValueError("Unknown teacher selection")
        self.teacher_id = teacher_id
        self.susceptibility_trace = None
        self._transition_capture = None
        self.last_completed = None
        self.failed_transition = None
        _validate_environment(environment)
        expected = engine_configuration(config["simulation"]["seed"],
                                        (config["wind"]["speed_kmh"], config["wind"]["direction_deg"]))
        if config != expected:
            raise ValueError("Assessment engine configuration must be the fixed R2 configuration")
        super().__init__(environment, config, model_free=True)
        if self.rng.bit_generator.__class__.__name__ != "PCG64":
            raise RuntimeError("Required default_rng backend is PCG64")
        self.rng = _ForwardingRng(self.rng)

    def _precompute_base_probability(self):
        if self.teacher_id == BASELINE_ID:
            return super()._precompute_base_probability()
        self.susceptibility_trace = compute_susceptibility32(
            slope32=self.slope_risk, building32=self.building_presence,
            proximity32=self.proximity_risk, material_risk32=self.material_risk)
        self.p_base = self.susceptibility_trace.beta32

    def _model_free_probability(self, susceptible_indices, wind_weighted_score):
        if susceptible_indices.ndim != 1:
            raise ValueError("Susceptible indices must be a one-dimensional vector")
        if self.teacher_id == BASELINE_ID:
            probabilities = super()._model_free_probability(susceptible_indices, wind_weighted_score)
        else:
            probabilities = compute_additive_exposure_probability(
                beta32=self.p_base.ravel()[susceptible_indices],
                wind_weighted_score32=wind_weighted_score.ravel()[susceptible_indices]).q32
        if probabilities.dtype != np.float32 or probabilities.shape != susceptible_indices.shape:
            raise ValueError("Probability hook must return an aligned float32 vector")
        if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Invalid captured comparison probabilities")
        self._transition_capture = (susceptible_indices.copy(), probabilities.copy())
        return probabilities

    def state_record(self):
        validate_full_state_grid(self.grid)
        footprint = np.isin(self.grid, (3, 4, 5))
        return {"teacher_id": self.teacher_id,
                "numerical_contract_id": NUMERICAL_ID if self.teacher_id == TEACHER_ID else None,
                "state_timestep": self.timestep, "state_counts": self.get_state_counts(),
                "footprint": int(np.count_nonzero(footprint)),
                "boundary_contact": bool(np.any(footprint[0]) or np.any(footprint[-1]) or
                                         np.any(footprint[:, 0]) or np.any(footprint[:, -1]))}

    def initialization_record(self):
        record = {**self.state_record(), "record_type": "initialization",
                  "transition_completed": None, "from_timestep": None, "to_timestep": None,
                  "eligible_opportunities": None, "probability_sum": None, "blazing_t": None,
                  "probability_sum_per_blazing": None, "new_ignitions": None,
                  "growth_increment": None, "rng_draw_calls": None, "cumulative_rng_draw_calls": 0,
                  "inactive_time": None}
        self.last_completed = deepcopy(record)
        return record

    def step(self):
        self._transition_capture = None
        self.failed_transition = None
        before = self.grid.copy()
        timestep = self.timestep
        calls = len(self.rng.calls)
        try:
            validate_full_state_grid(before)
            super().step()
            after = self.state_record()
            if not np.array_equal(self.grid == 1, ~self.burnable_mask):
                raise ValueError("State/domain invariant")
            legal = ((before == self.grid) | ((before == 2) & (self.grid == 3)) |
                     ((before == 3) & (self.grid == 4)) | ((before == 4) & (self.grid == 5)))
            if not np.all(legal):
                raise ValueError("Illegal state transition")
            for timer in (self.ignition_timers, self.blazing_timers):
                if timer.dtype != np.int16 or timer.shape != self.grid_shape or np.any(timer < 0):
                    raise ValueError("Timer invariant")
            count, total = 0, 0.0
            if self._transition_capture is not None:
                indices, probabilities = self._transition_capture
                count, total = len(indices), math.fsum(float(p) for p in probabilities)
            draws = len(self.rng.calls) - calls
            if draws != int(count > 0):
                raise ValueError("RNG draw-call invariant")
            newly = int(np.count_nonzero((before == 2) & (self.grid == 3)))
            growth = after["footprint"] - int(np.count_nonzero(np.isin(before, (3, 4, 5))))
            if growth != newly or newly > count:
                raise ValueError("Footprint/opportunity invariant")
            blazing = int(np.count_nonzero(before == 4))
            record = {**after, "record_type": "transition", "transition_completed": True,
                      "from_timestep": timestep, "to_timestep": self.timestep,
                      "eligible_opportunities": count, "probability_sum": total, "blazing_t": blazing,
                      "probability_sum_per_blazing": total / blazing if blazing else None,
                      "new_ignitions": newly, "growth_increment": growth,
                      "rng_draw_calls": draws, "cumulative_rng_draw_calls": len(self.rng.calls),
                      "inactive_time": self.timestep if not self.is_active() else None}
            self.last_completed = deepcopy(record)
            return record
        except BaseException:
            self.failed_transition = {"record_type": "transition", "transition_completed": False,
                                      "from_timestep": timestep, "to_timestep": None}
            self._transition_capture = None
            raise


class AssessmentFailure(RuntimeError):
    def __init__(self, report):
        super().__init__("Assessment failed; no retry is authorized")
        self.report = report


class CapInvariantViolation(RuntimeError):
    pass


def _run_fixture(spec, emit, check):
    environment, ignition = dynamic_fixture(spec["layout"], spec["material"])
    engine = _AssessmentAutomata(environment, engine_configuration(spec["seed"], spec["wind"]),
                                 teacher_id=BASELINE_ID if spec["teacher"] == "baseline" else TEACHER_ID)
    engine.set_ignition([ignition])
    initial = engine.initialization_record()
    summary = {"initial": initial, "last_completed": initial}
    try:
        emit(initial)
        for _ in range(162):
            check("before_step")
            record = engine.step()
            summary["last_completed"] = deepcopy(record)
            check("after_step")
            emit(record)
            if not engine.is_active():
                return summary
        raise CapInvariantViolation("cap_invariant_violation")
    except BaseException as error:
        error.assessment_summary = summary
        error.incomplete_transition = engine.failed_transition
        raise


class _Resources:
    def __init__(self, limits, started):
        self.limits, self.started = limits, started
        self.samples = []

    def summary(self):
        return {
            "sample_count": len(self.samples),
            "sampled_peak_process_rss_bytes": max(
                (s["process_rss_bytes"] for s in self.samples), default=None),
            "sampled_min_available_memory_bytes": min(
                (s["available_memory_bytes"] for s in self.samples), default=None),
            "sampled_min_free_disk_bytes": min(
                (s["free_disk_bytes"] for s in self.samples), default=None),
            "max_observed_gap_seconds": max(
                (s["gap_seconds"] for s in self.samples if s["gap_seconds"] is not None),
                default=None),
            "decision_elapsed_seconds": time.monotonic() - self.started,
        }

    def check(self, boundary):
        now = time.monotonic()
        if not self.samples or now - self.samples[-1]["elapsed_seconds"] - self.started >= self.limits["observation_interval_seconds"]:
            snapshot = capture_resource_snapshot(Path(__file__).resolve().parents[1])
            values = (snapshot.process_rss_bytes, snapshot.available_memory_bytes, snapshot.free_disk_bytes)
            if any(type(v) is not int or v < 0 for v in values) or values[0] == 0:
                raise RuntimeError("Missing/invalid resource observation")
            self.samples.append({"boundary": boundary, "elapsed_seconds": time.monotonic() - self.started,
                                 "gap_seconds": None if not self.samples else time.monotonic() - self.started - self.samples[-1]["elapsed_seconds"],
                                 "process_rss_bytes": values[0], "available_memory_bytes": values[1],
                                 "free_disk_bytes": values[2], "captured_at_utc": snapshot.captured_at_utc})
        elapsed = time.monotonic() - self.started
        last = self.samples[-1]
        if (last["process_rss_bytes"] > self.limits["max_process_rss_bytes"] or
                last["available_memory_bytes"] < self.limits["min_available_physical_memory_bytes"]):
            raise RuntimeError("resource_limit")
        if elapsed > self.limits["max_wall_seconds"]:
            raise RuntimeError("wall_limit")


def _binding(config):
    root = Path(__file__).resolve().parents[1]
    sources = ("modules/automata_engine.py", "modules/additive_exposure_teacher.py",
               "modules/additive_exposure_assessment.py", "modules/data_loader.py",
               "modules/feature_pipeline.py", "modules/ca_state.py", "modules/wind_convention.py",
               "modules/transition_observer.py", "modules/set_c_pilot.py")
    return {"source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
            "disk_configuration_sha256": hashlib.sha256(
                (root / "config/phase15_additive_exposure_r2.yaml").read_bytes()).hexdigest(),
            "effective_configuration_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "runtime": {"executable": sys.executable, "python": sys.version, "platform": platform.platform(),
                        "packages": {p: metadata.version(p) for p in ("numpy", "scipy", "rasterio", "joblib", "PyYAML")}},
            "teacher_ids": [BASELINE_ID, TEACHER_ID], "numerical_contract_id": NUMERICAL_ID}


def _runtime_check():
    if sys.float_info.rounds != 1:
        raise RuntimeError("Nearest-even runtime requires verification")
    for dtype, integer in ((np.float32, np.uint32), (np.float64, np.uint64)):
        tiny = np.array([1], dtype=integer).view(dtype)
        if np.multiply(tiny, dtype(1), dtype=dtype).view(integer)[0] != 1:
            raise RuntimeError("Gradual underflow unavailable")


def planned_runs():
    return [{"layout": layout, "material": material, "wind": list(wind), "seed": seed, "teacher": teacher}
            for layout in LAYOUTS for material in (1, 5) for wind in ((0.0, 0.0), (10.0, 315.0))
            for seed in range(32) for teacher in ("baseline", "candidate")]


def summarize(runs):
    result = {"planned": len(runs)}
    for status in ("complete", "failed", "not_attempted"):
        result[status] = sum(run["status"] == status for run in runs)
    result["groups"] = []
    for layout in LAYOUTS:
        for material in (1, 5):
            for wind in ((0.0, 0.0), (10.0, 315.0)):
                for teacher in ("baseline", "candidate"):
                    group = [r for r in runs if (r["spec"]["layout"], r["spec"]["material"],
                             tuple(r["spec"]["wind"]), r["spec"]["teacher"]) == (layout, material, wind, teacher)]
                    completed = [r for r in group if r["status"] == "complete"]
                    stats = {}
                    for field in ("footprint", "inactive_time"):
                        values = sorted(r["last_completed"][field] for r in completed)
                        stats[field] = None if not values else {
                            "mean": math.fsum(values) / len(values), "range": [values[0], values[-1]],
                            "nearest_rank_p05_p50_p95": [values[math.ceil(p * len(values)) - 1] for p in (0.05, 0.5, 0.95)]}
                    result["groups"].append({
                        "layout": layout, "material": material, "wind": list(wind), "teacher": teacher,
                        "planned": len(group), "complete": len(completed),
                        "failed": sum(r["status"] == "failed" for r in group),
                        "not_attempted": sum(r["status"] == "not_attempted" for r in group),
                        "seed_only_extinctions": sum(r["last_completed"]["footprint"] == 1 for r in completed),
                        "seed_only_extinction_frequency": None if not completed else
                            sum(r["last_completed"]["footprint"] == 1 for r in completed) / len(completed),
                        "statistics": stats})
    return result


def _console(record):
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def _failure_evidence(error):
    evidence = {"type": type(error).__name__, "message": str(error),
                "causal_chain": [], "causal_chain_cycle_detected": False}
    seen = {id(error)}
    current = error
    while True:
        if current.__cause__ is not None:
            relation, following = "cause", current.__cause__
        elif not current.__suppress_context__:
            relation, following = "context", current.__context__
        else:
            break
        if following is None:
            break
        if id(following) in seen:
            evidence["causal_chain_cycle_detected"] = True
            break
        seen.add(id(following))
        evidence["causal_chain"].append(
            {"relation": relation, "type": type(following).__name__, "message": str(following)})
        current = following
    return evidence


def run_assessment(config, *, execution_authorized=False, approval_reference=None, emit=_console):
    """One explicitly released synthetic campaign; never invoked by ordinary tests."""
    if execution_authorized is not True:
        raise PermissionError("Separate owner execution approval is required")
    started = time.monotonic()
    validate_configuration(config, active=True)
    if approval_reference != config["authorization_reference"]:
        raise PermissionError("Approval reference mismatch")
    resources = _Resources(config["resources"], started)
    report = {"schema_version": SCHEMA, "approval_reference": approval_reference,
              "runs": [{"spec": s, "status": "not_attempted"} for s in planned_runs()],
              "original_failure": None, "reporting_failures": [], "resource_samples": resources.samples,
              "resource_limits": deepcopy(config["resources"]),
              "resource_backend": "set_c_pilot.capture_resource_snapshot; current process; cooperative sampling",
              "qualification": "Exploratory synthetic unit exposure; no physical-validity or RF-fidelity claim"}
    current = None
    try:
        resources.check("before_binding")
        _runtime_check()
        report["binding"] = _binding(config)
        resources.check("after_binding")
        for current in report["runs"]:
            resources.check("before_run")
            current["status"] = "failed"
            summary = _run_fixture(current["spec"],
                                   lambda row: emit({"spec": current["spec"], **row}), resources.check)
            current.update(summary)
            resources.check("after_run")
            current["status"] = "complete"
        report["summary"] = summarize(report["runs"])
        resources.check("completion")
        report["resource_summary"] = resources.summary()
        emit({"record_type": "assessment_summary", **report})
        return report
    except BaseException as error:
        report["original_failure"] = _failure_evidence(error)
        if current is not None and current["status"] == "failed":
            current.update(getattr(error, "assessment_summary", {}))
            current["incomplete_transition"] = getattr(error, "incomplete_transition", None)
        report["summary"] = summarize(report["runs"])
        report["resource_summary"] = resources.summary()
        try:
            emit({"record_type": "assessment_failure", **report})
        except BaseException as reporting_error:
            report["reporting_failures"].append({"type": type(reporting_error).__name__,
                                                 "message": str(reporting_error)})
        raise AssessmentFailure(report) from error
