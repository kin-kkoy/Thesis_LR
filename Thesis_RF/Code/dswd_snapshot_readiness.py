"""Explicit-scope DSWD native readiness reads; no launcher or import-time artifact I/O.

Callers need separate owner read authority. Reviewed packet hashes are supplied
out of band; embedded path strings are declarations, never discovery targets.
No model, manifest, row dataset, LR file, or referenced evidence is opened.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import warnings

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.env import get_gdal_config
from rasterio.transform import Affine
from rasterio.windows import Window

try:
    from . import orchestrator as runner
    from . import dswd_snapshot_results as scoring
    from .modules import candidate_dataset_inputs as native
    from .modules.ca_state import STATE_CODES, validate_full_state_grid
    from .modules.transition_observer import canonical_metadata_hash, mask_sha256
except ImportError:
    import orchestrator as runner
    import dswd_snapshot_results as scoring
    from modules import candidate_dataset_inputs as native
    from modules.ca_state import STATE_CODES, validate_full_state_grid
    from modules.transition_observer import canonical_metadata_hash, mask_sha256


def alignment_read_whitelist() -> dict:
    """Six lexical paths only; does not resolve, stat, scan or read them."""
    root = runner.PHASE15_CODE_DIR
    folder = root / "processedData/raster/raster"
    paths = {name: folder / filename for name, filename in (
        ("slope", "stack_slope_final.tif"), ("proximity", "stack_proximity.tif"),
        ("buildings", "stack_buildings.tif"), ("materials", "stack_materials.tif"))}
    paths.update({name + "_reference": root / item["reference_path"]
                  for name, item in runner.PHASE16_DSWD_PROTOCOL["incidents"].items()})
    return paths


def _same(actual, expected, message):
    if runner._phase15_json_bytes(actual) != runner._phase15_json_bytes(expected):
        raise ValueError(message)


def _authority(authorized, reference):
    if authorized is not True:
        raise PermissionError("Separate explicit artifact-read authority required")
    native.require_text(reference)


def _ordinary(path):
    path = Path(path)
    if (not path.is_absolute() or ".." in path.parts
            or str(path).replace("/", "\\").startswith("\\\\")):
        raise ValueError("Explicit absolute local path without traversal required")
    runner._phase15_no_reparse(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Ordinary local file required")
    return path


def _digest(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _capability(capability):
    expected = {"policy_id": native.POLICY_ID, "controls_verified": True,
                "windowed_reads_verified": True,
                "runtime": {"rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__},
                "environment_options": native.ENV_OPTIONS, "open_options": native.OPEN_OPTIONS}
    if not isinstance(capability, dict):
        raise PermissionError("Reviewed native-reader capability required")
    _same({k: capability.get(k) for k in expected}, expected, "Reader capability/runtime differs")
    for key in ("verification_reference", "window_verification_reference"):
        native.require_text(capability.get(key))


def _valid(raw, mask, nodata):
    result = (mask != 0) & np.isfinite(raw) & (raw != -9999)
    if nodata is not None and np.isfinite(nodata):
        result &= raw != nodata
    return result


def _read_native(path, pin, grid, *, capability, window=None):
    """Restricted embedded-only native read, with before/after whole-file pins."""
    _capability(capability)
    path = _ordinary(path)
    native.require_digest(pin)
    if _digest(path) != pin:
        raise ValueError("Native input/output SHA-256 differs")
    before = path.stat()
    transform = Affine(*grid["transform"][:6])
    expected_shape = tuple(grid["shape"])
    if window is not None:
        if (len(window) != 4 or any(type(v) is not int for v in window)
                or any(v < 0 for v in window[:2]) or any(v <= 0 for v in window[2:])
                or any(o + n > total for o, n, total in zip(window[:2], window[2:], expected_shape))):
            raise ValueError("Exact integer crop outside native grid")
        row, col, height, width = window
        selection = Window(col, row, width, height)
        shape = (height, width)
        local_transform = transform * Affine.translation(col, row)
    else:
        selection, shape, local_transform = None, expected_shape, transform
    with rasterio.Env(**native.ENV_OPTIONS), warnings.catch_warnings():
        warnings.simplefilter("error")
        if any(get_gdal_config(k, normalize=False) != v for k, v in native.ENV_OPTIONS.items()):
            raise RuntimeError("Restricted native controls not effective")
        with rasterio.open(path, mode="r", driver="GTiff", sharing=False,
                           **native.OPEN_OPTIONS) as src:
            flags = sorted(flag.name for flag in src.mask_flag_enums[0])
            if (src.driver != "GTiff" or src.mode != "r" or src.count != 1 or src.subdatasets
                    or src.options.get("GEOREF_SOURCES") != "INTERNAL"
                    or [Path(p).absolute() for p in src.files] != [path.absolute()]
                    or flags not in (["all_valid"], ["nodata"], ["per_dataset"])
                    or src.shape != expected_shape or src.crs != CRS.from_user_input(grid["crs"])
                    or src.transform != transform):
                raise ValueError("Native metadata/geometry/embedded-only contract differs")
            native._geometry(tuple(src.shape), src.crs, src.transform)
            if selection is not None and src.window_transform(selection) != local_transform:
                raise ValueError("Native crop transform differs")
            raw, mask = src.read(1, window=selection), src.read_masks(1, window=selection)
            if (raw.shape != shape or raw.dtype.kind not in "iuf" or mask.shape != shape
                    or mask.dtype != np.uint8
                    or (flags == ["all_valid"] and not np.all(mask != 0))):
                raise ValueError("Native values/mask/shape differs")
            nodata = src.nodata
            metadata = {"dtype": raw.dtype.name,
                        "nodata": None if nodata is None else
                                  float(nodata) if np.isfinite(nodata) else str(nodata),
                        "mask_flags": flags, "native_mask_sha256": mask_sha256(mask != 0),
                        "valid_sha256": mask_sha256(_valid(raw, mask, nodata)),
                        "parent_shape": list(src.shape), "crs": src.crs.to_string(),
                        "parent_transform": list(src.transform),
                        "crop_shape": list(shape), "crop_transform": list(local_transform)}
        if any(get_gdal_config(k, normalize=False) != v for k, v in native.ENV_OPTIONS.items()):
            raise RuntimeError("Restricted native controls changed")
    after = path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or _digest(path) != pin):
        raise ValueError("Native file changed during read")
    return raw, mask, nodata, metadata


def compare_lr_packet(report: dict, packet: dict | None, reviewed_sha256: str | None) -> dict:
    """Compare a separately custodian-reviewed packet; never follow its references."""
    fields = ("incident", "input_sha256", "reference_sha256", "parent_grid", "crop_origin",
              "shape", "crs", "transform", "native_metadata", "validity_policy",
              "building_encoding", "reference_encoding", "masks", "reference_positive",
              "mask_digest_specification", "ignition_global_row_col")
    if packet is None or reviewed_sha256 is None:
        return {"status": "NEEDS EVIDENCE", "accepted": False}
    native.require_digest(reviewed_sha256)
    if canonical_metadata_hash(packet) != reviewed_sha256:
        raise ValueError("LR packet differs from independently reviewed identity")
    required = fields + ("confirmation_reference", "reader_reference", "reader_source_sha256",
                         "simulation_configuration_reference")
    missing = [key for key in required if key not in packet or packet[key] is None
               or (isinstance(packet[key], str) and not packet[key].strip())]
    if missing:
        return {"status": "NEEDS EVIDENCE", "accepted": False, "missing": missing}
    for key in ("confirmation_reference", "reader_reference", "simulation_configuration_reference"):
        native.require_text(packet.get(key))
    native.require_digest(packet.get("reader_source_sha256"))
    _same({k: packet[k] for k in fields}, {k: report[k] for k in fields},
          "RF/LR native identities, metadata or spatial masks differ")
    return {"status": "MATCH_REVIEWED_PACKET", "accepted": True,
            "packet_sha256": reviewed_sha256,
            "qualification": "Packet identity/equality; LR observations require custodian provenance review"}


def collect_alignment(paths: dict, *, capability: dict, authorized=False,
                      authorization_reference=None, lr_packets=None, lr_packet_pins=None,
                      design=None) -> dict:
    """Later-authorized six-TIFF observations, exact crops, no scoring or writes."""
    _authority(authorized, authorization_reference)
    whitelist = alignment_read_whitelist()
    if set(paths) != set(whitelist) or any(Path(paths[k]) != whitelist[k] for k in whitelist):
        raise ValueError("Exactly the six approved alignment paths required")
    if design not in (None, runner.PHASE16_DSWD_SHARED_PROTOCOL["design"]):
        raise ValueError("Unknown alignment design")
    shared = design is not None
    protocol = runner.PHASE16_DSWD_SHARED_PROTOCOL if shared else runner.PHASE16_DSWD_PROTOCOL
    grid = {"shape": protocol["alignment_policy"]["parent_shape"],
            "crs": protocol["alignment_policy"]["crs"],
            "transform": list(Affine(*protocol["alignment_policy"]["parent_transform"][:6]))}
    if len({(_ordinary(p).stat().st_dev, _ordinary(p).stat().st_ino)
            for p in paths.values()}) != 6:
        raise ValueError("Alignment sources alias one another")
    result = {}
    for incident, item in protocol["incidents"].items():
        window = item["crop_origin"] + item["shape"]
        raw, masks, nodata, metadata = {}, {}, {}, {}
        for name in native.LAYERS:
            raw[name], masks[name], nodata[name], metadata[name] = _read_native(
                paths[name], protocol["input_sha256"][name], grid,
                capability=capability, window=window)
        prepared = native.prepare_arrays(
            raw, masks, nodata, configured_nodata=-9999, crs=grid["crs"],
            transform=Affine(*item["crop_transform"]), input_hashes=protocol["input_sha256"],
            interpretation={"policy_id": native.POLICY_ID, "layers": metadata},
            global_origin=tuple(item["crop_origin"]), full_grid_shape=tuple(grid["shape"]))
        reference, ref_mask, ref_nodata, ref_metadata = _read_native(
            paths[incident + "_reference"], item["reference_sha256"], grid,
            capability=capability, window=window)
        ref_valid = _valid(reference, ref_mask, ref_nodata)
        if not np.all(np.isin(reference[ref_valid], (0, 1))):
            raise ValueError("Reference encoding must be native valid 0/1")
        env = prepared.environment["burnable_mask"]
        mapped = prepared.environment["building_presence"] > 0
        building_valid = _valid(raw["buildings"], masks["buildings"], nodata["buildings"])
        domain = env & mapped & ref_valid & building_valid
        mask_arrays = {"environmental_valid": env, "mapped_building": mapped,
                       "eligible_building": prepared.receiving_mask, "evaluation_domain": domain}
        coverage = {"total_in_crop": int(np.count_nonzero(ref_valid & (reference == 1))),
                    "included": int(np.count_nonzero(domain & (reference == 1)))}
        coverage["excluded"] = coverage["total_in_crop"] - coverage["included"]
        _same([int(np.count_nonzero(domain)), coverage["included"], coverage["total_in_crop"]],
              [item["expected_domain_cells"], item["expected_reference_positive_included"],
               item["expected_reference_positive_total"]], "Historical domain expectations differ")
        report = {"schema_version": "dswd_native_alignment.v1", "incident": incident,
                  "input_sha256": deepcopy(protocol["input_sha256"]),
                  "reference_sha256": item["reference_sha256"], "parent_grid": deepcopy(grid),
                  "crop_origin": deepcopy(item["crop_origin"]), "shape": deepcopy(item["shape"]),
                  "ignition_global_row_col": [[r + item["crop_origin"][0], c + item["crop_origin"][1]]
                                               for r, c in item["ignition"]],
                  "crs": grid["crs"], "transform": deepcopy(item["crop_transform"]),
                  "native_metadata": {**metadata, "reference": ref_metadata},
                  "validity_policy": protocol["alignment_policy"]["validity"],
                  "building_encoding": deepcopy(protocol["alignment_policy"]["building_encoding"]),
                  "reference_encoding": deepcopy(protocol["alignment_policy"]["reference_encoding"]),
                  "masks": {name: {"sha256": mask_sha256(mask),
                                    "count": int(np.count_nonzero(mask))}
                            for name, mask in mask_arrays.items()},
                  "reference_positive": coverage,
                  "reference_positive_mask_sha256": mask_sha256(ref_valid & (reference == 1)),
                  "buildings_valid_sha256": mask_sha256(building_valid),
                  "reference_valid_sha256": mask_sha256(ref_valid),
                  "mask_digest_specification": "transition_observer.mask_sha256: shape|bool|C-order uint8",
                  "reader": {"policy_id": native.POLICY_ID, "capability": deepcopy(capability),
                             "authorization_reference": authorization_reference},
                  "qualification": item["qualification"]}
        if shared:
            points = [[r + item["crop_origin"][0], c + item["crop_origin"][1]] for r, c in item["ignition"]]
            # Atomic native receiving validation; no engine or estimator is constructed.
            native.validate_seeds(points, prepared, "grid_row_col")
            report.update(design=design, ignition_receiving_verified=True)
            parity = {"status": "NOT APPLICABLE: shared CA, counterpart pending", "accepted": False}
        else:
            parity = compare_lr_packet(report, (lr_packets or {}).get(incident),
                                       (lr_packet_pins or {}).get(incident))
        arrays = {"reference": reference, "reference_valid": ref_valid,
                  "buildings": raw["buildings"], "buildings_valid": building_valid,
                  "environment": env.astype(np.uint8), "environment_valid": np.ones(env.shape, bool)}
        arrays = {k: native._readonly(v) for k, v in arrays.items()}
        result[incident] = {"report": report, "report_sha256": canonical_metadata_hash(report),
                            "arrays": arrays, "parity": parity,
                            "lr_packet": deepcopy((lr_packets or {}).get(incident))}
    return result


def _record(path, pin):
    path = _ordinary(path)
    native.require_digest(pin)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Record exceeds bounded metadata scope")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != pin:
        raise ValueError("Record SHA-256 differs")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    record = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if runner._phase15_json_bytes(record) != raw:
        raise ValueError("Expected canonical guarded record bytes")
    return record


def _counts(state):
    return {name: int(np.count_nonzero(state == code)) for name, code in STATE_CODES.items()}


def _observation(row, counts, update):
    reached = sum(counts[k] for k in ("ignited", "blazing", "extinguished"))
    active = counts["ignited"] + counts["blazing"]
    for key, value in (("state_counts", counts), ("cumulative_reached_count", reached),
                       ("active_cell_count", active), ("completed_update", update)):
        _same(row[key], value, "State/per-update observation differs")


def _alignment_arrays(alignment, incident, expected_pin):
    report, arrays = alignment["report"], alignment["arrays"]
    if (report["incident"] != incident or canonical_metadata_hash(report) != expected_pin
            or alignment["report_sha256"] != expected_pin):
        raise ValueError("Reviewed alignment report identity differs")
    env, mapped = arrays["environment"] == 1, arrays["buildings_valid"] & (arrays["buildings"] == 10)
    domain = env & mapped & arrays["reference_valid"]
    for name, mask in (("environmental_valid", env), ("mapped_building", mapped),
                       ("eligible_building", env & mapped), ("evaluation_domain", domain)):
        _same({"sha256": mask_sha256(mask), "count": int(np.count_nonzero(mask))},
              report["masks"][name], "Observed alignment mask differs")
    for key, mask in (("reference_positive_mask_sha256",
                       arrays["reference_valid"] & (arrays["reference"] == 1)),
                      ("buildings_valid_sha256", arrays["buildings_valid"]),
                      ("reference_valid_sha256", arrays["reference_valid"])):
        _same(mask_sha256(mask), report[key], "Observed alignment array differs")
    return arrays


def read_snapshot_run(root, *, expected: dict, alignment: dict, capability: dict,
                      authorized=False, authorization_reference=None) -> dict:
    """Read only fixed names under an explicit root, requiring a reviewed final receipt.

    expected supplies independent start/termination pins, full reviewed binding,
    effective configuration, alignment report pin and nonempty review_reference.
    It grants no execution authority. No payload path is dereferenced.
    """
    _authority(authorized, authorization_reference)
    native.require_text(expected["review_reference"])
    root = Path(root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("Explicit versioned absolute run root required")
    runner._phase15_no_reparse(root)
    start = _record(root / "start.json", expected["start_sha256"])
    # Absence of this final receipt rejects orphan outcomes without reading partials.
    end = _record(root / "termination.json", expected["termination_sha256"])
    required_start = {"schema_version", "run_id", "scenario_id", "destination", "started_at_utc",
                      "clock_boundary", "authorization_reference", "invocation", "binding",
                      "configuration", "source", "runtime", "model", "inputs", "scenario",
                      "resource_policy", "initial_resource_observations", "output_policy"}
    required_end = {"schema_version", "run_id", "start_record_sha256", "binding_sha256", "ended_at_utc",
                    "status", "termination_reason", "exit_code", "last_completed_timestep", "failed_step",
                    "state_counts", "active_cell_count", "decision_elapsed_seconds", "clock_policy",
                    "resource_observations", "violations", "observed_bindings", "output_valid", "outputs",
                    "snapshot_series", "error"}
    if not required_start <= set(start) or not required_end <= set(end):
        raise ValueError("Incomplete final run records")
    partial = _ordinary(root / "termination.json.partial")
    if not os.path.samefile(root / "termination.json", partial):
        raise ValueError("Final termination record is not its retained publication link")
    binding, config = start["binding"]["payload"], start["configuration"]["effective"]
    _same(binding, expected["binding"], "Independent execution binding differs")
    _same(config, expected["effective_configuration"], "Reviewed effective configuration differs")
    digest = canonical_metadata_hash(binding)
    for value in (start["binding"]["expected_sha256"], start["binding"]["calculated_sha256"],
                  end["binding_sha256"]):
        _same(value, digest, "Binding link differs")
    _same(end["start_record_sha256"], expected["start_sha256"], "Start record link differs")
    _same(start["configuration"], {**binding["configuration"], "effective": config},
          "Configuration link differs")
    _same(canonical_metadata_hash(config), binding["configuration"]["activated_effective_sha256"],
          "Effective configuration digest differs")
    runner._phase15_digest(binding["configuration"]["disk_sha256"])
    disabled = deepcopy(config)
    disabled["phase15"]["enabled"] = False
    _same(runner._phase15_normalize(disabled), config, "Effective frozen profile differs")
    if not runner._phase16_is_snapshot(config):
        raise ValueError("DSWD snapshot profile required")
    incident, seed = config["snapshot_basis"]["incident"], config["simulation"]["seed"]
    shared = runner._phase16_is_shared(config)
    protocol = runner.PHASE16_DSWD_SHARED_PROTOCOL if shared else runner.PHASE16_DSWD_PROTOCOL
    item = protocol["incidents"][incident]
    if (root.name != config["phase15"]["run_id"] or Path(binding["run"]["destination"]) != root
            or start["schema_version"] != "phase15_run_start.v1"
            or end["schema_version"] != "phase15_run_termination.v1"
            or binding["schema_version"] != "phase15_execution_binding.v1"):
        raise ValueError("Run root/schema differs")
    for key in ("run_id", "scenario_id", "destination"):
        _same(start[key], binding["run"][key], "Start/run identity differs")
    _same(end["run_id"], root.name, "Termination/run identity differs")
    _same(binding["run"]["run_id"], root.name, "Binding/run identity differs")
    _same(binding["run"]["scenario_id"], config["phase15"]["scenario_id"], "Scenario/configuration differs")
    _same(binding["approval_reference"], config["phase15"]["authorization_reference"], "Configuration approval differs")
    _same(binding["run"]["destination"],
          (Path(config["output"]["output_dir"]) / config["phase15"]["run_id"]).as_posix(),
          "Configuration destination differs")
    _same(start["authorization_reference"], binding["approval_reference"], "Approval link differs")
    for key in ("source", "runtime"):
        _same(start[key], binding[key], "Source/runtime binding differs")
    _same(binding["snapshot_basis"], config["snapshot_basis"], "Snapshot basis binding differs")
    _same(binding["candidate_inputs"], config["candidate_inputs"], "Candidate binding differs")
    _same(start["model"], {**binding["model"], "observed_sha256": None, "interface_validated": None},
          "Start/model declaration differs")
    _same(start["inputs"], {k: {**v, "observed_sha256": None} for k, v in binding["inputs"].items()},
          "Start/input declarations differ")
    _same(start["scenario"], {"seed": seed, "wind": config["wind"],
          "ignition_points": config["simulation"]["ignition_points"], "converted_ignitions": None,
          "grid": None, "timing": config["placeholder_transition"], "max_timesteps": item["max_updates"]},
          "Start/scenario declarations differ")
    _same(start["resource_policy"], config["phase15"]["resources"], "Resource-policy declaration differs")
    _same(end["exit_code"], 0, "Successful integer exit code required")
    if (end["output_valid"] is not True or end["exit_code"] != 0 or end["error"] is not None
            or end["violations"] or end["failed_step"] is not None
            or end["termination_reason"] not in ("inactive", "timestep_limit")):
        raise ValueError("Failed/incomplete trajectory cannot supply outcomes")
    stop, reason = end["last_completed_timestep"], end["termination_reason"]
    if (type(stop) is not int or not 0 <= stop <= item["max_updates"]
            or (reason == "timestep_limit" and stop != item["max_updates"])
            or end["status"] != ("complete" if reason == "inactive" else "censored")):
        raise ValueError("Trajectory stop/status differs")
    observed = end["observed_bindings"]
    for name in native.LAYERS:
        _same(binding["inputs"][name]["sha256"], protocol["input_sha256"][name],
              "Frozen input identity differs")
        _same(observed["inputs"][name]["observed_sha256"], binding["inputs"][name]["sha256"],
              "Observed input identity differs")
    model = {"path": config["ml_model"]["model_path"], "sha256": config["ml_model"]["model_sha256"]}
    if not shared or config["snapshot_basis"]["arm"] == "rf":
        model["manifest"] = {"path": config["ml_model"]["calibration_manifest_path"],
                             "sha256": runner.PHASE15_CANDIDATE_MANIFEST_SHA256}
    _same(binding["model"], model, "Frozen model binding differs")
    if (observed["model"]["observed_sha256"] != model["sha256"]
            or observed["model"]["interface_validated"] is not True):
        raise ValueError("Observed model declarations differ")
    if "manifest" in model:
        _same(observed["model"]["manifest"]["observed_sha256"], model["manifest"]["sha256"],
              "Observed calibration manifest differs")
    elif "manifest" in observed["model"]:
        raise ValueError("LR requires calibration manifest absence")

    _same(observed["converted_ignitions"], config["simulation"]["ignition_points"], "Ignition link differs")
    arrays = _alignment_arrays(alignment, incident, expected["alignment_sha256"])
    report = alignment["report"]
    if shared:
        _same(report.get("design"), protocol["design"], "Alignment design/ignition context differs")
        _same(report.get("ignition_receiving_verified"), True, "Native receiving eligibility unresolved")
        _same(report["ignition_global_row_col"], config["snapshot_basis"]["ignition_global_row_col"],
              "Alignment actual ignition differs; historical894 is not892 evidence")
        parity = {"status": "NOT APPLICABLE: shared CA", "accepted": False}
    else:
        parity = compare_lr_packet(report, alignment.get("lr_packet"), expected.get("lr_packet_sha256"))
    _same(report["input_sha256"], {k: binding["inputs"][k]["sha256"] for k in native.LAYERS},
          "Alignment/input identity differs")
    _same(report["reference_sha256"], item["reference_sha256"], "Reference identity differs")
    _same(report["shape"], item["shape"], "Alignment crop differs")
    _same(report["crop_origin"], item["crop_origin"], "Alignment origin differs")
    parent = config["candidate_inputs"]["expected_grid"]
    _same(report["parent_grid"]["shape"], parent["shape"], "Alignment parent shape differs")
    if (CRS.from_user_input(report["parent_grid"]["crs"]) != CRS.from_user_input(parent["crs"])
            or Affine(*report["parent_grid"]["transform"][:6]) != Affine(*parent["transform"][:6])):
        raise ValueError("Alignment parent CRS/affine differs")
    _same(report["crs"], protocol["alignment_policy"]["crs"], "Alignment CRS differs")
    _same(report["transform"], item["crop_transform"], "Alignment affine differs")
    grid = {"shape": item["shape"], "crs": report["crs"], "transform": item["crop_transform"]}
    _same(observed["grid"], {"shape": item["shape"], "crs": CRS.from_user_input(grid["crs"]).to_wkt(),
                           "transform": list(Affine(*item["crop_transform"]))[:6]}, "Observed run grid differs")
    series = end["snapshot_series"]
    _same(series["basis"], config["snapshot_basis"], "Series basis differs")
    if (series["schema_version"] != "phase16_dswd_snapshot_series.v1"
            or series["completed_updates"] != stop or series["failed_step"] is not None):
        raise ValueError("Series completion differs")
    curve = series["per_update"]
    if len(curve) != stop + 1:
        raise ValueError("Per-update chronology is incomplete")
    previous = None
    for update, row in enumerate(curve):
        counts = row["state_counts"]
        if (set(counts) != set(STATE_CODES) or
                any(type(v) is not int or v < 0 for v in counts.values())
                or sum(counts.values()) != int(np.prod(item["shape"]))):
            raise ValueError("Per-update state counts invalid")
        _observation(row, counts, update)
        reached = row["cumulative_reached_count"]
        new = 0 if previous is None else reached - previous
        _same(row["new_ignition_count"], new, "New ignition/cumulative chronology differs")
        if new < 0 or (update < stop and row["active_cell_count"] == 0):
            raise ValueError("Nonabsorbing or decreasing chronology")
        previous = reached
    _same({k: series["initial"][k] for k in curve[0]}, curve[0], "Initial observation differs")
    _same(series["initial"]["initial_reached_count"], curve[0]["cumulative_reached_count"],
          "Initial reached count differs")
    _same(series["initial"]["initial_ignition_count"], curve[0]["active_cell_count"],
          "Initial ignition count differs")
    _same(end["state_counts"], curve[-1]["state_counts"], "Final counts differ")
    _same(end["active_cell_count"], curve[-1]["active_cell_count"], "Final active count differs")
    if (reason == "inactive") != (curve[-1]["active_cell_count"] == 0):
        raise ValueError("Trajectory inactivity/censoring differs")
    rows = series["outcomes"]
    basis = config["snapshot_basis"]["horizons"]
    if len(rows) != len(basis):
        raise ValueError("Missing or extra horizon outcomes")
    filenames = {"final_state.tif", "simulation_environment_valid.tif"}
    for row in basis:
        filenames.update(f"snapshot_h{row['requested_update']:04d}_{kind}.tif" for kind in ("state", "reached"))
    identities = {record["filename"]: record for record in end["outputs"]}
    if len(identities) != len(end["outputs"]) or set(identities) != filenames:
        raise ValueError("Output inventory differs")
    _same(start["output_policy"], {"allowlist": sorted(runner._phase16_snapshot_outputs(config["snapshot_basis"])),
                                  "resume": False}, "Output policy differs")
    def output(name, dtype):
        identity = identities[name]
        if (identity["verified"] is not True or type(identity["bytes"]) is not int
                or identity["bytes"] <= 0):
            raise ValueError("Output verification/length invalid")
        final = _ordinary(root / name)
        staging = _ordinary(root / (name + ".partial"))
        if final.stat().st_size != identity["bytes"] or not os.path.samefile(final, staging):
            raise ValueError("Output length/publication link differs")
        raw, mask, nodata, metadata = _read_native(final, identity["sha256"], grid, capability=capability)
        if (raw.dtype != np.dtype(dtype) or nodata is not None
                or metadata["mask_flags"] != ["all_valid"] or not np.all(mask != 0)):
            raise ValueError("Snapshot native dtype/nodata/mask differs")
        for key, value in (("shape", item["shape"]), ("dtype", dtype),
                           ("transform", list(Affine(*item["crop_transform"]))[:6]),
                           ("crs", CRS.from_user_input(grid["crs"]).to_wkt())):
            _same(identity[key], value, "Export geometry/dtype declaration differs")
        return raw, metadata
    environment, _ = output("simulation_environment_valid.tif", "uint8")
    if not np.array_equal(environment, arrays["environment"]):
        raise ValueError("Run environment differs from all-four-layer alignment")
    _same(observed["mask_identities"]["environment_sha256"],
          report["masks"]["environmental_valid"]["sha256"], "Observed environment mask differs")
    _same(observed["mask_identities"]["building_sha256"],
          report["masks"]["mapped_building"]["sha256"], "Observed building mask differs")
    _same(observed["mask_identities"]["receiving_sha256"],
          report["masks"]["eligible_building"]["sha256"], "Observed eligibility mask differs")
    final, _ = output("final_state.tif", "int8")
    validate_full_state_grid(final)
    _same(_counts(final), end["state_counts"], "Final native counts differ")
    results = []
    for row, prescribed in zip(rows, basis):
        capture = min(row["requested_update"], stop)
        inactive = capture == stop and reason == "inactive"
        links = {**prescribed, "incident": incident, "seed": seed, "run_id": root.name,
                 "execution_binding_sha256": digest, "start_record_sha256": expected["start_sha256"],
                 "configuration_sha256": binding["configuration"]["activated_effective_sha256"],
                 "duration_basis": config["snapshot_basis"]["duration_basis"],
                 "actual_capture_update": capture, "horizon_outcome_update": capture,
                 "actual_stop_update": stop, "trajectory_stop_update": stop,
                 "absorbing_carry_forward": capture < row["requested_update"],
                 "status": "complete" if inactive else "censored", "output_valid": True,
                 "termination_reason": "inactive" if inactive else "timestep_limit", "censoring": not inactive}
        if shared:
            links.update({key: config["snapshot_basis"][key] for key in
                          ("design", "arm", "model_policy", "ignition_local_row_col", "ignition_global_row_col")})
        _same({k: row[k] for k in links}, links, "Outcome identity/clock/horizon/chronology differs")
        state_name = f"snapshot_h{row['requested_update']:04d}_state.tif"
        reached_name = f"snapshot_h{row['requested_update']:04d}_reached.tif"
        _same(row["outputs"], [identities[state_name], identities[reached_name]], "Outcome/output link differs")
        state, state_metadata = output(state_name, "int8")
        reached, _ = output(reached_name, "uint8")
        validate_full_state_grid(state)
        if not np.array_equal(reached, np.isin(state, (3, 4, 5)).astype(np.uint8)):
            raise ValueError("Reached export differs from native states 3/4/5")
        if capture == stop and not np.array_equal(state, final):
            raise ValueError("Absorbing/final snapshot differs")
        _observation({**row, "completed_update": capture}, _counts(state), capture)
        _same(row["state_counts"], curve[capture]["state_counts"], "Snapshot/per-update counts differ")
        local = {**arrays, "prediction": state, "prediction_valid": np.ones(state.shape, bool), "reached": reached}
        native_metadata = {"prediction": {k: state_metadata[k] for k in ("dtype", "nodata", "mask_flags")},
                           "environment": {"dtype": "uint8", "nodata": None, "mask_flags": ["all_valid"]}}
        for key in ("reference", "buildings"):
            native_metadata[key] = {k: report["native_metadata"][key][k] for k in ("dtype", "nodata", "mask_flags")}
        result = scoring.evaluate_snapshot_arrays(
            local, row, incident, native_metadata=native_metadata,
            **({"design": protocol["design"], "arm": config["snapshot_basis"]["arm"]} if shared else {}))
        result["parity_status"] = parity["status"]
        result["accepted_for_cross_simulator_comparison"] = parity["accepted"]
        results.append(result)
    # Detect record substitution during native reads without opening arbitrary references.
    _record(root / "start.json", expected["start_sha256"])
    _record(root / "termination.json", expected["termination_sha256"])
    result = {"schema_version": "dswd_verified_snapshot_run.v1", "run_id": root.name,
              "binding_sha256": digest, "alignment_sha256": expected["alignment_sha256"],
              "outcomes": results, "execution_authorized": False}
    if shared:
        result["shared_pair"] = {
            "design": protocol["design"], "incident": incident, "seed": seed,
            "arm": config["snapshot_basis"]["arm"], "model_policy": deepcopy(config["snapshot_basis"]["model_policy"]),
            "start_sha256": expected["start_sha256"], "termination_sha256": expected["termination_sha256"],
            "review_reference": expected["review_reference"],
            "common": {key: deepcopy(report[key]) for key in
                       ("input_sha256", "reference_sha256", "parent_grid", "crop_origin", "shape",
                        "ignition_global_row_col", "crs", "transform", "validity_policy", "building_encoding",
                        "reference_encoding", "native_metadata", "masks", "reference_positive", "reference_positive_mask_sha256",
                        "buildings_valid_sha256", "reference_valid_sha256")},
            "wind": deepcopy(config["wind"]), "timing": deepcopy(config["placeholder_transition"]),
            "source": deepcopy(binding["source"]), "runtime": deepcopy(binding["runtime"]),
            "feature_contract": deepcopy(config["rf_contract"])}
        result["shared_pair_sha256"] = canonical_metadata_hash(result["shared_pair"])
        result["shared_pair_status"] = "PENDING: verified counterpart required"
    return result


def compare_shared_runs(first: dict, counterpart: dict | None = None) -> dict:
    """Diagnostic declaration comparison only; dictionaries cannot prove native reads."""
    def declared(record):
        if record.get("schema_version") != "dswd_verified_snapshot_run.v1":
            raise ValueError("Verified snapshot run required")
        pair = record["shared_pair"]
        if type(pair["seed"]) is not int or pair["seed"] not in runner.PHASE16_DSWD_SHARED_PROTOCOL["seeds"]:
            raise ValueError("Shared seed must be an integer in42–46")
        _same(canonical_metadata_hash(pair), record["shared_pair_sha256"], "Shared verification record changed")
        native.require_text(pair["review_reference"])
        for key in ("start_sha256", "termination_sha256"):
            runner._phase15_digest(pair[key])
        runner._phase15_digest(record["binding_sha256"])
        basis = runner._phase16_shared_basis(pair["incident"], pair["arm"])
        _same(pair["design"], basis["design"], "Shared design differs")
        _same(pair["model_policy"], basis["model_policy"], "Shared frozen arm policy differs")
        _same(pair["common"]["ignition_global_row_col"], basis["ignition_global_row_col"],
              "Shared actual ignition differs")
        rows = record["outcomes"]
        if len(rows) != 3:
            raise ValueError("Three verified shared outcomes required")
        for row, horizon in zip(rows, basis["horizons"]):
            outcome = row["outcome"]
            expected_links = {**horizon, "design": pair["design"], "arm": pair["arm"],
                              "model_policy": pair["model_policy"], "incident": pair["incident"],
                              "seed": pair["seed"], "run_id": record["run_id"],
                              "execution_binding_sha256": record["binding_sha256"],
                              "start_record_sha256": pair["start_sha256"],
                              "ignition_global_row_col": basis["ignition_global_row_col"],
                              "ignition_local_row_col": basis["ignition_local_row_col"]}
            _same({key: outcome[key] for key in expected_links}, expected_links, "Verified shared outcome differs")
            if outcome["output_valid"] is not True or outcome["status"] not in ("complete", "censored"):
                raise ValueError("Failed shared trajectory")
        return pair
    left = declared(first)
    if counterpart is None:
        return {"status": "PENDING: verified counterpart required", "accepted_for_shared_ca_comparison": False,
                "accepted_for_cross_simulator_comparison": False}
    right = declared(counterpart)
    if left["arm"] == right["arm"] or {left["arm"], right["arm"]} != {"rf", "lr"}:
        raise ValueError("Distinct RF and LR counterparts required")
    for key in ("design", "incident", "seed", "common", "wind", "timing", "feature_contract", "source", "runtime"):
        _same(left[key], right[key], "Shared-arm common inputs/geometry/ignition/masks/domain differ")
    if first["run_id"] == counterpart["run_id"]:
        raise ValueError("Distinct versioned shared run identities required")
    return {"status": "DIAGNOSTIC MATCH: native reader verification not established",
            "accepted_for_shared_ca_comparison": False,
            "accepted_for_cross_simulator_comparison": False,
            "run_ids": [first["run_id"], counterpart["run_id"]],
            "binding_sha256": [first["binding_sha256"], counterpart["binding_sha256"]]}


def read_shared_runs(root, *, expected: dict, alignment: dict, capability: dict,
                     authorized=False, authorization_reference=None,
                     counterpart_root=None, counterpart_expected=None,
                     counterpart_alignment=None, counterpart_capability=None,
                     counterpart_authorized=False, counterpart_authorization_reference=None) -> dict:
    """Accept only results obtained here through each explicitly authorized native reader.

    Each arm needs its own root, independently reviewed expectations and alignment/
    capability evidence. No supplied record, flag or digest replaces those reads.
    """
    _authority(authorized, authorization_reference)
    evidence = (expected, alignment, capability)
    if counterpart_root is not None:
        _authority(counterpart_authorized, counterpart_authorization_reference)
        evidence += (counterpart_expected, counterpart_alignment, counterpart_capability)
    elif (any(value is not None for value in
              (counterpart_expected, counterpart_alignment, counterpart_capability,
               counterpart_authorization_reference)) or counterpart_authorized is not False):
        raise ValueError("Explicit counterpart root required with counterpart evidence")
    if any(not isinstance(value, dict) for value in evidence):
        raise ValueError("Explicit reviewed expectations, alignment and capability required for each arm")
    first = read_snapshot_run(
        root, expected=expected, alignment=alignment, capability=capability,
        authorized=authorized, authorization_reference=authorization_reference)
    if not isinstance(first, dict):
        raise ValueError("First reader did not return a complete run record")
    if counterpart_root is None:
        return compare_shared_runs(first)
    counterpart = read_snapshot_run(
        counterpart_root, expected=counterpart_expected, alignment=counterpart_alignment,
        capability=counterpart_capability, authorized=counterpart_authorized,
        authorization_reference=counterpart_authorization_reference)
    if not isinstance(counterpart, dict):
        raise ValueError("Counterpart reader did not return a complete run record")
    result = compare_shared_runs(first, counterpart)
    # Both records are local reader returns; dictionary-only diagnostics never accept.
    return {**result, "status": "MATCH", "accepted_for_shared_ca_comparison": True}
