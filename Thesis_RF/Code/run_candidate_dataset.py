"""Opt-in candidate dataset runner. Importing this module never starts collection."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import yaml

from modules import candidate_dataset_collection as collection
from modules import candidate_dataset_crops as crops
from modules.candidate_dataset_inputs import read_inputs, require_digest, require_text, validate_seeds
from modules import candidate_dataset_release as release
from modules.set_c_pilot import capture_resource_snapshot

CODE = Path(__file__).resolve().parent
SOURCE_FILES = (
    "run_candidate_dataset.py", "modules/__init__.py", "modules/candidate_dataset_release.py",
    "modules/candidate_dataset_collection.py", "modules/candidate_dataset_inputs.py",
    "modules/additive_exposure_teacher.py", "modules/automata_engine.py",
    "modules/feature_pipeline.py", "modules/ca_state.py", "modules/data_loader.py",
    "modules/wind_convention.py", "modules/transition_observer.py", "modules/set_c_collector.py",
    "modules/model_free_teacher.py", "modules/set_c_pilot.py", "modules/candidate_dataset_crops.py",
)


def file_identity(path, check) -> dict:
    h, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            check("identity_read")
            h.update(chunk)
            size += len(chunk)
    return dict(bytes=size, sha256=h.hexdigest())


def execution_payload(config: dict, authority: dict) -> dict:
    return dict(schema_version="candidate_dataset_execution.v1",
                configuration=config, authority=authority)


def publication_payload(execution_sha256: str, authority: dict) -> dict:
    return dict(schema_version="candidate_dataset_publication.v1",
        execution_sha256=execution_sha256,
        approval_reference=authority["publication_approval_reference"],
        destinations=authority["destinations"], custody=authority["custody"],
        allocation_node_budget=authority["allocation_node_budget"])


def validate_authority(config: dict, authority: dict, *, execution_authorized: bool, publication_authorized: bool,
                       expected_execution_sha256: str | None, expected_publication_sha256: str | None) -> tuple:
    if execution_authorized is not True:
        raise PermissionError("Separate owner candidate execution authority required")
    if publication_authorized is not True:
        raise PermissionError("Separate owner candidate publication authority required")
    collection.validate_configuration(config, active=True)
    fields = {"execution_approval_reference", "publication_approval_reference", "source",
              "runtime", "run_bindings", "destinations", "custody", "allocation_node_budget",
              "disk_configuration_sha256"}
    if not isinstance(authority, dict) or set(authority) != fields:
        raise PermissionError("Complete explicit authority inventory required")
    require_digest(authority["disk_configuration_sha256"])
    for key in ("execution_approval_reference", "publication_approval_reference"):
        require_text(authority[key])
    if authority["execution_approval_reference"] != config["authorization_reference"]:
        raise PermissionError("Execution approval mismatch")
    if type(authority["allocation_node_budget"]) is not int or authority["allocation_node_budget"] <= 0:
        raise PermissionError("Unresolved allocation budget")
    source = authority["source"]
    for field in ("source_revision", "execution_id", "grid_id"):
        require_text(source.get(field))
    if type(source.get("source_dirty")) is not bool or not isinstance(authority["runtime"], dict) or not authority["runtime"]:
        raise PermissionError("Actual source dirty state and runtime binding required")
    if set(source["source_sha256"]) != set(SOURCE_FILES):
        raise PermissionError("Exact candidate source inventory required")
    for digest in source["source_sha256"].values():
        require_digest(digest)
    custody, families = authority["custody"], config["inventory"]
    if set(custody) != {"approval_reference", "families"}:
        raise PermissionError("Exact custody inventory required")
    require_text(custody["approval_reference"])
    if set(custody["families"]) != {f["scenario_family_id"] for f in families}:
        raise PermissionError("Explicit per-family final-test eligibility required")
    for f in families:
        assessment = custody["families"][f["scenario_family_id"]]
        if (set(assessment) != {"eligible", "prior_use", "assessment_reference"}
                or type(assessment["eligible"]) is not bool
                or assessment["prior_use"] != f["prior_use"]):
            raise PermissionError("Prior-use/final-test assessment mismatch")
        require_text(assessment["assessment_reference"])
    if "spatial_design" in config:
        if (not source["execution_id"].startswith(crops.CAMPAIGN_ID + "-")
                or custody["families"][families[0]["scenario_family_id"]]["eligible"]):
            raise PermissionError("Cropped identity or F01 exclusion changed")
    plan = collection.planned_runs(families)
    if set(authority["run_bindings"]) != {r["run_id"] for r in plan}:
        raise PermissionError("All 364 independently delivered run bindings required")
    for digest in authority["run_bindings"].values():
        require_digest(digest)
    destinations = authority["destinations"]
    if set(destinations) != {"evidence", "package"}:
        raise PermissionError("Exact evidence and package destinations required")
    for value in destinations.values():
        require_text(value)
        if value.replace("/", "\\").startswith("\\\\") or not Path(value).is_absolute():
            raise PermissionError("Explicit local absolute destinations required")
    require_digest(expected_execution_sha256)
    require_digest(expected_publication_sha256)
    actual = release.digest_bytes(release.json_bytes(execution_payload(config, authority)))
    pub = publication_payload(actual, authority)
    if (actual != expected_execution_sha256
            or release.digest_bytes(release.json_bytes(pub)) != expected_publication_sha256):
        raise PermissionError("Independently delivered execution/publication binding mismatch")
    return plan, pub


class Resources:
    """Cooperative observations; no background sampler or blocked-call interruption."""
    def __init__(self, limits, destinations, started, *, probe=capture_resource_snapshot,
                 clock=time.monotonic) -> None:
        self.limits, self.destinations = limits, tuple(map(Path, destinations))
        self.started, self.probe, self.clock = started, probe, clock
        self.samples, self.failure, self.last_observation = [], None, None

    def check(self, boundary: str) -> None:
        if self.failure is not None:
            raise RuntimeError("Resource failure is latched") from self.failure
        try:
            elapsed = self.clock() - self.started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError("Invalid elapsed clock")
            frequent = {"identity_read", "row_reconciliation", "grouping", "component_adequacy",
                "allocation_search", "publication_write", "publication_readback",
                "transition_reconciliation", "run_reconciliation", "master_identity",
                "publication_run_membership"}
            if (boundary in frequent and self.last_observation is not None
                    and elapsed - self.last_observation < self.limits["observation_interval_seconds"]):
                if elapsed > self.limits["max_wall_seconds"]:
                    raise RuntimeError("Candidate wall limit")
                return
            for destination in self.destinations:
                s = self.probe(destination)
                values = (s.process_rss_bytes, s.available_memory_bytes, s.free_disk_bytes)
                if any(type(v) is not int or v <= 0 for v in values):
                    raise RuntimeError("Missing/invalid resource observation")
                elapsed = self.clock() - self.started
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise RuntimeError("Invalid elapsed clock after resource observation")
                self.last_observation = elapsed
                self.samples.append(dict(boundary=boundary, elapsed_seconds=elapsed,
                    destination=str(destination), rss_bytes=values[0],
                    available_bytes=values[1], free_disk_bytes=values[2]))
                breaches = []
                if values[0] > self.limits["max_process_rss_bytes"]:
                    breaches.append("rss")
                if values[1] < self.limits["min_available_physical_memory_bytes"]:
                    breaches.append("available_memory")
                if values[2] < self.limits["min_free_disk_bytes"]:
                    breaches.append("free_disk")
                if elapsed > self.limits["max_wall_seconds"]:
                    breaches.append("wall")
                if breaches:
                    raise RuntimeError("Candidate resource limits: " + ",".join(breaches))
        except BaseException as error:
            self.failure = error
            raise


def verify_sources_runtime(authority: dict, check) -> dict:
    for relative, digest in authority["source"]["source_sha256"].items():
        path = CODE / relative
        if path.is_symlink() or path.is_junction() or not path.is_file():
            raise PermissionError("Ordinary pinned source required")
        if relative == "run_candidate_dataset.py":
            module = sys.modules[__name__]
        else:
            name = relative[:-3].replace("/", ".")
            if name.endswith(".__init__"):
                name = name[:-9]
            module = sys.modules.get(name)
        if module is None or Path(module.__file__).resolve() != path:
            raise PermissionError("Actual project import origin mismatch: " + relative)
        if file_identity(path, check)["sha256"] != digest:
            raise PermissionError("Actual source identity mismatch: " + relative)
    import importlib
    runtime = dict(python=list(sys.version_info[:3]), executable=str(Path(sys.executable).resolve()),
        executable_sha256=file_identity(sys.executable, check)["sha256"],
        venv_configuration=str(Path(sys.prefix) / "pyvenv.cfg"),
        venv_configuration_sha256=file_identity(Path(sys.prefix) / "pyvenv.cfg", check)["sha256"])
    for name in ("numpy", "scipy", "pandas", "rasterio", "joblib", "yaml"):
        module = importlib.import_module(name)
        runtime[name] = dict(version=str(module.__version__), origin=str(Path(module.__file__).resolve()))
    runtime["gdal"] = importlib.import_module("rasterio").__gdal_version__
    if runtime != authority["runtime"]:
        raise PermissionError("Actual runtime binding mismatch")
    return runtime


def _validate_crop_extent(binding: dict, batch) -> None:
    """Validate bound crop metadata even when the batch contains no observations."""
    source, interpretation = binding.get("source"), binding.get("interpretation")
    execution_id = source.get("execution_id") if isinstance(source, dict) else None
    identified = isinstance(execution_id, str) and execution_id.startswith(crops.CAMPAIGN_ID)
    if not identified and not (isinstance(interpretation, dict) and "window" in interpretation):
        return  # Legacy full-map bindings retain the unchanged reconciler's checks.
    spec = binding.get("spec")
    ordinal = spec.get("family_ordinal") if isinstance(spec, dict) else None
    window = interpretation.get("window") if isinstance(interpretation, dict) else None
    origin, parent = binding.get("global_origin"), binding.get("full_grid_shape")
    if (not isinstance(execution_id, str) or not execution_id.startswith(crops.CAMPAIGN_ID + "-")
            or type(ordinal) is not int or not 1 <= ordinal <= len(crops.FAMILIES)
            or type(window) is not list or len(window) != 4
            or any(type(n) is not int for n in window)
            or any(n < 0 for n in window[:2]) or any(n <= 0 for n in window[2:])
            or any(type(v) is not list or len(v) != 2 or any(type(n) is not int for n in v)
                   for v in (origin, parent))):
        raise ValueError("Missing or malformed crop extent metadata")
    expected = crops.frozen_design()["family_windows"][ordinal - 1]
    if (spec.get("scenario_family_id") != expected["scenario_family_id"]
            or window != expected["global_origin"] + expected["shape"]
            or origin != window[:2] or parent != list(crops.PARENT_SHAPE)
            or any(o + n > p for o, n, p in zip(window[:2], window[2:], parent, strict=True))):
        raise ValueError("Inconsistent crop extent metadata")
    row_origin, col_origin, height, width = window
    for row, col in zip(batch.global_rows, batch.global_cols, strict=True):
        if not (row_origin <= row < row_origin + height
                and col_origin <= col < col_origin + width):
            raise ValueError("Observation outside bound crop extent")


class EvidenceStore:
    """Exclusively allocated, retained evidence; no abort cleanup or retry."""
    def __init__(self, root: Path, *, allocated=None) -> None:
        self.root, self.streams = Path(root), {}
        self.rows, self.run_rows, self.transitions, self.last_order = 0, {}, {}, None
        self.root.mkdir()
        if allocated is not None:
            allocated()
        try:
            for name in ("transitions", "runs"):
                self.streams[name] = (self.root / (name + ".jsonl")).open("x+b")
        except BaseException as error:
            try:
                self.close()
            except BaseException as closing:
                error.add_note("Secondary evidence-close failure: " + repr(closing))
            raise

    def append(self, name: str, record: dict) -> None:
        data = release.json_bytes(record) + b"\n"
        stream = self.streams[name]
        offset = stream.seek(0, 2)
        if stream.write(data) != len(data):
            raise OSError("Short evidence write")
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(offset)
        if stream.read(len(data)) != data:
            raise OSError("Evidence readback mismatch")
        stream.seek(0, 2)

    def persist(self, binding: dict, snapshot: tuple) -> None:
        for batch in snapshot:
            _validate_crop_extent(binding, batch)
            envelope, rows = release.reconcile_batch(batch, binding)
            run_id = binding["spec"]["run_id"]
            before = self.transitions.get(run_id, 0)
            if envelope["timestep_t"] != before:
                raise ValueError("Missing or repeated completed transition")
            if self.rows + len(rows) > collection.MAX_PACKAGE_ROWS:
                raise RuntimeError("Package row cap exceeded; no truncation")
            for row in rows:
                key = tuple(row["order"])
                if self.last_order is not None and key <= self.last_order:
                    raise ValueError("Collection order changed")
                self.last_order = key
            self.append("transitions", dict(binding=binding, envelope=envelope, rows=rows))
            self.rows += len(rows)
            self.run_rows[run_id] = self.run_rows.get(run_id, 0) + len(rows)
            self.transitions[run_id] = before + 1

    def close(self) -> None:
        failures = []
        for name, stream in list(self.streams.items()):
            try:
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException as error:
                failures.append(f"{name}.flush: {type(error).__name__}: {error}")
            try:
                stream.close()
            except BaseException as error:
                failures.append(f"{name}.close: {type(error).__name__}: {error}")
        self.streams.clear()
        if failures:
            raise OSError("Evidence finalization failures: " + "; ".join(failures))


def persist_and_acknowledge(engine, store: EvidenceStore) -> None:
    snapshot = engine.batches
    if type(snapshot) is not tuple or not snapshot:
        raise ValueError("Nonempty committed snapshot required")
    store.persist(engine.binding, snapshot)
    # Identity prevents stale acknowledgement; persistence is established by the store separately.
    engine.acknowledge_batches(snapshot)


_RECONCILIATION_CHUNK_ROWS = 4096


def _reconcile_retained_chunk(retained, envelope, binding, start, stop) -> int:
    """Validate a nonpersisted view; only scalar accounting leaves this frame."""
    selected = retained[start:stop]
    columns = [release.decode_row(r["csv"]) for r in selected]
    arrays = [np.array([v[0] for v in columns], dtype=np.float32).reshape(-1, 11),
        np.array([v[1] for v in columns], dtype=np.int8),
        np.array([r["payload"]["cell_row"] for r in selected], dtype=np.int64),
        np.array([r["payload"]["cell_col"] for r in selected], dtype=np.int64),
        np.array([r["observation_id"].encode("ascii") for r in selected], dtype="S64"),
        np.array([r["q32_bits"] for r in selected], dtype=np.uint32)]
    for array in arrays:
        array.setflags(write=False)
    view = dict(envelope, bindings=envelope["bindings"][start:stop])
    batch = collection.CandidateBatch(*arrays, envelope_bytes=release.json_bytes(view))
    _validate_crop_extent(binding, batch)
    _, rebuilt = release.reconcile_batch(batch, binding)
    if rebuilt != selected:
        raise ValueError("Retained transition/row reconciliation failed")
    return sum(v[1] for v in columns)


def load_completed_evidence(root: Path, config: dict, authority: dict, check) -> tuple:
    rows, transitions, runs, row_counts, positive_counts, prior_states = [], {}, [], {}, {}, {}
    with (root / "transitions.jsonl").open("rb") as stream:
        for line in stream:
            check("transition_reconciliation")
            item = strict_json(line)
            if release.json_bytes(item) + b"\n" != line:
                raise ValueError("Noncanonical or partial transition record")
            binding, envelope, retained = item["binding"], item["envelope"], item["rows"]
            spec = binding["spec"]
            if release.digest_bytes(release.json_bytes(binding)) != authority["run_bindings"][spec["run_id"]]:
                raise ValueError("Retained run binding mismatch")
            count = len(retained)
            if len(envelope["bindings"]) != count:
                raise ValueError("Comparison envelope/run/count mismatch")
            positives = 0
            for start in range(0, max(1, count), _RECONCILIATION_CHUNK_ROWS):
                positives += _reconcile_retained_chunk(retained, envelope, binding,
                    start, min(start + _RECONCILIATION_CHUNK_ROWS, count))
            run_id = spec["run_id"]
            if envelope["timestep_t"] != transitions.get(run_id, 0):
                raise ValueError("Retained transition/row reconciliation failed")
            if run_id in prior_states and envelope["state_t_sha256"] != prior_states[run_id]:
                raise ValueError("Completed transition state-hash chain is discontinuous")
            if len(rows) + count > collection.MAX_PACKAGE_ROWS:
                raise ValueError("Retained package exceeds cap")
            prior_states[run_id] = envelope["state_t1_sha256"]
            transitions[run_id] = transitions.get(run_id, 0) + 1
            rows.extend(retained)
            row_counts[run_id] = row_counts.get(run_id, 0) + count
            positive_counts[run_id] = positive_counts.get(run_id, 0) + positives
            # Raw bytes survive chunking; record-only owners retire before the next read.
            del line, item, binding, envelope, retained, spec
    with (root / "runs.jsonl").open("rb") as stream:
        for line in stream:
            check("run_reconciliation")
            item = strict_json(line)
            if release.json_bytes(item) + b"\n" != line:
                raise ValueError("Noncanonical or partial run record")
            runs.append(item)
    plan = collection.planned_runs(config["inventory"])
    if len(runs) != 2 * len(plan):
        raise ValueError("Every initialization and inactive termination is required")
    for i, spec in enumerate(plan):
        initial, final = runs[2 * i:2 * i + 2]
        total = row_counts.get(spec["run_id"], 0)
        before, after = initial["completed"]["state_counts"], final["completed"]["state_counts"]
        names = {"non_burnable", "not_yet_burning", "ignited", "blazing", "extinguished"}
        shape = (config["spatial_design"]["family_windows"][spec["family_ordinal"] - 1]["shape"]
                 if "spatial_design" in config else config["grid"]["shape"])
        cells = shape[0] * shape[1]
        seeds = len(config["inventory"][spec["family_ordinal"] - 1]["ignition_points"])
        positives = positive_counts.get(spec["run_id"], 0)
        if (set(before) != names or set(after) != names
                or any(type(v) is not int or v < 0 for counts in (before, after) for v in counts.values())
                or sum(before.values()) != cells or sum(after.values()) != cells
                or before["blazing"] != seeds or before["ignited"] != 0 or before["extinguished"] != 0
                or after["non_burnable"] != before["non_burnable"]
                or after["extinguished"] != seeds + positives
                or after["not_yet_burning"] != before["not_yet_burning"] - positives):
            raise ValueError("Inactive state/label conservation mismatch")
        if (initial["kind"] != "initialization" or initial["spec"] != spec
                or initial["completed"]["state_timestep"] != 0
                or initial["completed"]["committed_rows"] != 0
                or final["kind"] != "inactive" or final["spec"] != spec
                or final["completed"]["state_counts"]["ignited"] != 0
                or final["completed"]["state_counts"]["blazing"] != 0
                or final["completed"]["state_timestep"] != transitions.get(spec["run_id"], 0)
                or not 1 <= final["completed"]["state_timestep"] <= collection.MAX_UPDATES
                or final["completed"]["committed_rows"] != total
                or total > collection.MAX_RUN_ROWS):
            raise ValueError("Run completeness/cap/count mismatch")
    release.validate_rows(rows, check)
    return rows, runs


def strict_json(data: bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


class CandidateFailure(RuntimeError):
    def __init__(self, report: dict) -> None:
        super().__init__("Candidate dataset incomplete; retain custodian evidence")
        self.report = report


def _capture_resource_report(report, guard, config, coverage):
    try:
        report["resources"] = deepcopy(guard.samples)
        times = [sample["elapsed_seconds"] for sample in report["resources"]]
        report["resource_policy"] = dict(enforcement="cooperative_boundaries_only",
            nominal_interval_seconds=config["resources"]["observation_interval_seconds"],
            max_actual_gap_seconds=max((b - a for a, b in zip([0.0] + times, times)), default=0.0),
            coverage_boundary=coverage, resource_sample_count=len(times),
            last_resource_boundary=report["resources"][-1]["boundary"] if times else None,
            startup_exclusion="module/interpreter startup before main; no continuous enforcement")
    except BaseException as error:
        report.pop("resource_policy", None)
        report["finalization_failures"].append("resource-report capture: " + repr(error))


def _read_family_crop(config, family):
    window = config["spatial_design"]["family_windows"][family["ordinal"] - 1]
    prepared = read_inputs({k: v["path"] for k, v in config["inputs"].items()},
        {k: v["sha256"] for k, v in config["inputs"].items()}, expected_grid=config["grid"],
        configured_nodata=config["configured_nodata"], capability=config["reader_capability"],
        window=window["global_origin"] + window["shape"])
    validate_seeds(family["ignition_points"], prepared, family["coordinate_system"])
    return prepared


def inspect_crop_inputs(config, check):
    """Read-only inventory, callable only with separate real-input authorization.

    No context/CA construction, RNG, output files or outcome-based selection.
    """
    collection.validate_configuration(config, active=True)
    if "spatial_design" not in config:
        raise ValueError("A frozen cropped campaign is required")
    entries = []
    for family, window in zip(config["inventory"], config["spatial_design"]["family_windows"], strict=True):
        check("before_crop_inventory")
        prepared = _read_family_crop(config, family)
        crops.validate_preparation(config["spatial_design"],
            dict(family_ordinal=family["ordinal"], scenario_family_id=family["scenario_family_id"]), prepared)
        entry = crops.inventory_entry(window, prepared)
        entry["local_ignition_points"] = [list(p) for p in validate_seeds(
            family["ignition_points"], prepared, family["coordinate_system"])]
        entries.append(entry)
        del prepared
        check("after_crop_inventory")
    return dict(spatial_design=deepcopy(config["spatial_design"]), windows=entries,
                geometric_overlaps=crops.overlapping_windows(config["spatial_design"]),
                qualification="Overlap is not independence; observed global blocks still govern grouping")


def _family_context(config, family, check):
    prepared = _read_family_crop(config, family)
    check("after_input_preparation")
    context = collection.build_candidate_context(prepared, config)
    check("after_candidate_context")
    return context


def _check_family_bindings(config, authority, plan, context):
    for spec in plan:
        binding = collection.execution_binding(config, context.prepared, spec, authority["source"],
                                               static_context=context)
        if release.digest_bytes(release.json_bytes(binding)) != authority["run_bindings"][spec["run_id"]]:
            raise PermissionError("Run binding mismatch before allocation/RNG")


def run(config: dict, authority: dict, *, execution_authorized: bool = False, publication_authorized: bool = False,
        expected_execution_sha256: str | None = None, expected_publication_sha256: str | None = None,
        started: float | None = None, emit=None) -> dict:
    started = time.monotonic() if started is None else started
    report = dict(complete=False, committed=None, allocation_status="not_started", original_failure=None,
                  finalization_failures=[], resources=[], evidence=None, last_completed=None,
                  incomplete_transition=None, diagnostics_source=None)
    store = engine = guard = original = engine_handoff = engine_failure_latch = None
    try:
        plan, pub_binding = validate_authority(config, authority,
            execution_authorized=execution_authorized, publication_authorized=publication_authorized,
            expected_execution_sha256=expected_execution_sha256,
            expected_publication_sha256=expected_publication_sha256)
        evidence, destination = (Path(authority["destinations"][k]) for k in ("evidence", "package"))
        if (evidence == destination or evidence in destination.parents or destination in evidence.parents
                or any(os.path.lexists(p) for p in (evidence, destination))
                or any(not p.parent.is_dir() or p.parent.is_symlink() or p.parent.is_junction()
                       for p in (evidence, destination))
                or evidence.parent.stat().st_dev != destination.parent.stat().st_dev):
            raise ValueError("Absent distinct same-volume destinations with existing ordinary parents required")
        guard = Resources(config["resources"], (evidence, destination), started)
        def check(boundary):
            if boundary == "allocation_search":
                report["allocation_status"] = "in_progress"
            guard.check(boundary)
            if boundary == "allocation_completed":
                report["allocation_status"] = "allocated"
        check("before_preflight")
        verify_sources_runtime(authority, check)
        cropped = "spatial_design" in config
        if cropped:
            # First pass checks every commitment before a journal or run generator exists.
            for family in config["inventory"]:
                static_context = _family_context(config, family, check)
                family_plan = [p for p in plan if p["family_ordinal"] == family["ordinal"]]
                _check_family_bindings(config, authority, family_plan, static_context)
                static_context = None
            prepared = None
            active_family = None
        else:
            prepared = read_inputs({k: v["path"] for k, v in config["inputs"].items()},
                {k: v["sha256"] for k, v in config["inputs"].items()}, expected_grid=config["grid"],
                configured_nodata=config["configured_nodata"], capability=config["reader_capability"])
            check("after_input_preparation")
            for family in config["inventory"]:
                validate_seeds(family["ignition_points"], prepared, family["coordinate_system"])
            static_context = collection.build_candidate_context(prepared, config)
            prepared = static_context.prepared
            check("after_candidate_context")
            _check_family_bindings(config, authority, plan, static_context)
        check("before_evidence_allocation")
        store = EvidenceStore(evidence, allocated=lambda: report.update(evidence=str(evidence)))
        for spec in plan:
            check("before_run")
            if cropped and active_family != spec["family_ordinal"]:
                # Release the previous family before reading/owning the next window.
                static_context = prepared = None
                family = config["inventory"][spec["family_ordinal"] - 1]
                static_context = _family_context(config, family, check)
                family_plan = [p for p in plan if p["family_ordinal"] == family["ordinal"]]
                _check_family_bindings(config, authority, family_plan, static_context)
                prepared = static_context.prepared
                active_family = family["ordinal"]
            engine = collection.CandidateAutomata(prepared, config, spec, authority["source"],
                execution_authorized=True, approval_reference=config["authorization_reference"],
                expected_binding_sha256=authority["run_bindings"][spec["run_id"]],
                static_context=static_context)
            engine_handoff = engine.validate_handoff
            engine_failure_latch = engine.latch_handoff_failure
            store.append("runs", dict(kind="initialization", spec=spec,
                                      completed=deepcopy(engine.last_completed)))
            if emit is not None:
                emit(dict(kind="initialization", spec=deepcopy(spec), completed=deepcopy(engine.last_completed)))
            engine_handoff(expected_context=static_context)
            check("after_initialization")
            while engine.is_active():
                if engine.timestep >= collection.MAX_UPDATES:
                    raise RuntimeError("Active run reached update cap; no censoring or substitution")
                check("before_step")
                engine.step()
                persist_and_acknowledge(engine, store)
                if emit is not None:
                    emit(dict(kind="completed_transition", spec=deepcopy(spec), completed=deepcopy(engine.last_completed)))
                engine_handoff(expected_context=static_context)
                check("after_completed_transition")
            if (engine.batches or engine.row_count != store.run_rows.get(spec["run_id"], 0)
                    or engine.timestep != store.transitions.get(spec["run_id"], 0)):
                raise ValueError("Inactive run accounting mismatch")
            store.append("runs", dict(kind="inactive", spec=spec,
                                      completed=deepcopy(engine.last_completed)))
            if emit is not None:
                emit(dict(kind="inactive", spec=deepcopy(spec), completed=deepcopy(engine.last_completed)))
            engine_handoff(expected_context=static_context)
            check("after_inactive_run")
            report["last_completed"], report["incomplete_transition"] = (
                deepcopy(engine.last_completed), deepcopy(engine.incomplete_transition))
            report["diagnostics_source"] = dict(run_id=spec["run_id"], boundary="successful_retirement")
            engine_handoff = engine_failure_latch = engine = None
            gc.collect()
            check("after_engine_retirement")
        store.close()
        if cropped:
            static_context = prepared = None
        rows, runs = load_completed_evidence(evidence, config, authority, check)
        items = release.components(rows, config["complete_event_ids"],
                                   authority["custody"]["families"], check)
        master_hash = hashlib.sha256(release.HEADER)
        for row in rows:
            check("master_identity")
            master_hash.update(row["csv"].encode("ascii"))
        dataset_sha256 = master_hash.hexdigest()
        verify_sources_runtime(authority, check)
        provenance = dict(binding=pub_binding, dataset_sha256=dataset_sha256,
            teacher_id=config["teacher_id"], numerical_contract_id=config["numerical_contract_id"],
            configuration=config, execution_authority=authority,
            run_records=runs, evidence_files={name: file_identity(evidence / name, check)
                for name in ("transitions.jsonl", "runs.jsonl")})
        report["committed"] = release.publish(rows, items,
            payload=evidence / "payload", destination=destination, authority=provenance, check=check,
            publication_authorized=True, expected_binding_sha256=expected_publication_sha256)
        report["allocation_status"] = "allocated"
        if emit is not None:
            emit(dict(kind="package_committed", package=deepcopy(report["committed"])))
        check("after_commit_reporting")
        report["complete"] = True
    except BaseException as error:
        original = error
        if engine_failure_latch is not None:
            try:
                engine_failure_latch()
            except BaseException as latching:
                report["finalization_failures"].append("engine failure latch: " + repr(latching))
        if isinstance(error, release.AllocationIncomplete) or (
                report["allocation_status"] == "in_progress"
                and not isinstance(error, release.AllocationInfeasible)):
            report["allocation_status"] = "allocation_incomplete"
        elif isinstance(error, release.AllocationInfeasible):
            report["allocation_status"] = "allocation_infeasible"
        report["original_failure"] = dict(type=type(error).__name__, message=str(error),
            traceback="".join(traceback.format_exception(error)))
    finally:
        if engine is not None:
            try:
                report["last_completed"], report["incomplete_transition"] = (
                    deepcopy(engine.last_completed), deepcopy(engine.incomplete_transition))
                report["diagnostics_source"] = dict(run_id=spec["run_id"], boundary="live_engine_finalization")
            except BaseException as error:
                report["finalization_failures"].append("completed-state capture: " + repr(error))
        if store is not None:
            try:
                store.close()
            except BaseException as error:
                report["finalization_failures"].append(repr(error))
        if guard is not None:
            _capture_resource_report(report, guard, config, "before_outcome_publication")
        if report["finalization_failures"]:
            report["complete"] = False
        if report["evidence"] is not None:
            report["report_stage"] = "outcome_snapshot"
            report["outcome_snapshot"] = dict(coverage_boundary="before_outcome_publication",
                resource_sample_count=len(report["resources"]), not_final_completion_evidence=True,
                excludes=["its own publication incidents", "later resource observations",
                          "later finalization/reporting incidents"])
            snapshot_written = False
            try:
                release.write_new(Path(report["evidence"]) / "outcome.json",
                    [release.json_bytes(report) + b"\n"], lambda _: None)
                snapshot_written = True
            except BaseException as error:
                report["finalization_failures"].append("outcome publication: " + repr(error))
            if snapshot_written and guard is not None:
                try:
                    guard.check("after_outcome_publication")
                except BaseException as error:
                    report["finalization_failures"].append("final resource observation: " + repr(error))
        if guard is not None:
            _capture_resource_report(report, guard, config, "after_last_attempted_resource_observation")
        report["report_stage"] = "final_report"
        if report["finalization_failures"]:
            report["complete"] = False
    if not report["complete"]:
        raise CandidateFailure(report) from original
    return report


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError("Duplicate or non-string YAML key")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)



def print_final_report(report: dict) -> None:
    try:
        print(release.json_bytes(report).decode("ascii"), flush=True)
    except BaseException as error:
        report["complete"] = False
        report["finalization_failures"].append("console reporting: " + repr(error))
        try:
            sys.stderr.write(release.json_bytes(report).decode("ascii") + "\n")
            sys.stderr.flush()
        except BaseException:
            pass
        raise CandidateFailure(report) from error

def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--authorize-execution", action="store_true")
    parser.add_argument("--authorize-publication", action="store_true")
    parser.add_argument("--expected-execution-sha256", required=True)
    parser.add_argument("--expected-publication-sha256", required=True)
    args = parser.parse_args(argv)
    authority = strict_json(Path(args.authority).read_bytes())
    data = Path(args.config).read_bytes()
    if release.digest_bytes(data) != authority["disk_configuration_sha256"]:
        raise PermissionError("Disk configuration identity mismatch")
    config = yaml.load(data.decode("utf-8"), Loader=UniqueLoader)
    try:
        report = run(config, authority, execution_authorized=args.authorize_execution,
            publication_authorized=args.authorize_publication,
            expected_execution_sha256=args.expected_execution_sha256,
            expected_publication_sha256=args.expected_publication_sha256,
            started=started, emit=lambda record: print(release.json_bytes(record).decode("ascii"), flush=True))
    except CandidateFailure as error:
        try:
            print_final_report(error.report)
        except BaseException as reporting:
            error.add_note("Failure report incident: " + repr(reporting))
        raise
    print_final_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
